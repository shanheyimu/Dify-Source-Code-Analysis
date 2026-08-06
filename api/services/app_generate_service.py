from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Generator, Mapping
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from ..configs import dify_config
from ..core.app.apps.advanced_chat.app_generator import AdvancedChatAppGenerator
from ..core.app.apps.agent_app.app_generator import AgentAppGenerator
from ..core.app.apps.agent_chat.app_generator import AgentChatAppGenerator
from ..core.app.apps.chat.app_generator import ChatAppGenerator
from ..core.app.apps.completion.app_generator import CompletionAppGenerator
from ..core.app.apps.message_based_app_generator import MessageBasedAppGenerator
from ..core.app.apps.workflow.app_generator import WorkflowAppGenerator
from ..core.app.entities.app_invoke_entities import InvokeFrom
from ..core.app.features.rate_limiting import RateLimit
from ..core.app.features.rate_limiting.rate_limit import rate_limit_context
from ..core.app.layers.pause_state_persist_layer import PauseStateLayerConfig
from ..core.db import session_factory
from ..enums.quota_type import QuotaType
from ..extensions.otel import AppGenerateHandler, trace_span
from ..models.model import Account, App, AppMode, EndUser
from ..models.workflow import Workflow, WorkflowRun
from ..services.errors.app import QuotaExceededError, WorkflowIdFormatError, WorkflowNotFoundError
from ..services.errors.llm import InvokeRateLimitError
from ..services.quota_service import QuotaService, unlimited
from ..services.workflow_service import WorkflowService
from ..tasks.app_generate.workflow_execute_task import AppExecutionParams, workflow_based_app_execution_task

logger = logging.getLogger(__name__)

SSE_TASK_START_FALLBACK_MS = 200

if TYPE_CHECKING:
    from ..controllers.console.app.workflow import LoopNodeRunPayload


class AppGenerateService:
    """
    所有生成型接口的"统一总闸"。

    web / console / service_api / openapi / mcp 等所有蓝图里的
    chat / completion / workflow / agent 生成请求，最终都汇聚到这里的 generate()。
    职责：护栏（配额 + 限流）→ 按 AppMode 策略分发 → 对应 AppGenerator。
    """

    @staticmethod
    def _build_streaming_task_on_subscribe(start_task: Callable[[], None]) -> Callable[[], None]:
        """
        Build a subscription callback that coordinates when the background task starts.

        - streams transport: start immediately (events are durable; late subscribers can replay).
        - pubsub/sharded transport: start on first subscribe, with a short fallback timer so the task
          still runs if the client never connects.
        """
        started = False
        lock = threading.Lock()

        def _try_start() -> bool:
            nonlocal started
            with lock:
                if started:
                    return True
                try:
                    start_task()
                except Exception:
                    logger.exception("Failed to enqueue streaming task")
                    return False
                started = True
                return True

        channel_type = dify_config.PUBSUB_REDIS_CHANNEL_TYPE
        if channel_type == "streams":
            # With Redis Streams, we can safely start right away; consumers can read past events.
            _try_start()

            # Keep return type Callable[[], None] consistent while allowing an extra (no-op) call.
            def _on_subscribe_streams() -> None:
                _try_start()

            return _on_subscribe_streams

        # Pub/Sub modes (at-most-once): subscribe-gated start with a tiny fallback.
        timer = threading.Timer(SSE_TASK_START_FALLBACK_MS / 1000.0, _try_start)
        timer.daemon = True
        timer.start()

        def _on_subscribe() -> None:
            if _try_start():
                timer.cancel()

        return _on_subscribe

    @classmethod
    @trace_span(AppGenerateHandler)  # OTel 链路追踪埋点（observability）
    def generate(
        cls,
        app_model: App,
        user: Account | EndUser,
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        *,
        session: Session,          # * 之后的参数是"强制关键字参数"，调用方必须按名字传
        streaming: bool = True,
        root_node_id: str | None = None,
    ):
        """
        App Content Generate
        :param app_model: app model
        :param user: user
        :param args: args
        :param invoke_from: invoke from
        :param streaming: streaming
        :return:
        """
        # 先过护栏（配额 + 限流），再分发到具体 Generator。
        # action 是延迟执行的闭包：拿到 rate_limit 和 request_id 之后才真正分发。
        return cls._run_with_guardrails(
            app_model=app_model,
            streaming=streaming,
            action=lambda rate_limit, request_id: cls._dispatch_generate(
                app_model=app_model,
                user=user,
                args=args,
                invoke_from=invoke_from,
                streaming=streaming,
                root_node_id=root_node_id,
                session=session,
                rate_limit=rate_limit,
                request_id=request_id,
            ),
        )

    @classmethod
    def _run_with_guardrails(
        cls,
        *,
        app_model: App,
        streaming: bool,
        action: Callable[[RateLimit, str], Any],
    ):
        """
        护栏层：所有生成请求进入具体 Generator 之前，先做两道检查：
        ① 配额预留（计费租户）  ② 应用级并发限流
        """
        # ① 配额预留：预留一次执行额度，成功后才 commit
        quota_charge = unlimited()
        # 只有开启计费才检查
        if dify_config.BILLING_ENABLED:
            try:
                quota_charge = QuotaService.reserve(QuotaType.WORKFLOW, app_model.tenant_id)
            except QuotaExceededError:
                raise InvokeRateLimitError(f"Workflow execution quota limit reached for tenant {app_model.tenant_id}")

        # ② 应用级限流：限制同一应用的同时执行数（0=不限）
        max_active_request = cls._get_max_active_requests(app_model)
        rate_limit = RateLimit(app_model.id, max_active_request)
        request_id = RateLimit.gen_request_key()
        try:
            request_id = rate_limit.enter(request_id)  # 占用一个并发名额
            quota_charge.commit()
            return action(rate_limit, request_id)      # 执行真正的分发
        except Exception:
            quota_charge.refund()                      # 失败：退还配额
            if streaming:
                rate_limit.exit(request_id)            # 流式：异常时主动释放名额
            raise
        finally:
            if not streaming:
                rate_limit.exit(request_id)            # 阻塞式：无论如何释放名额

    @classmethod
    def _dispatch_generate(
        cls,
        *,
        app_model: App,
        user: Account | EndUser,
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: bool,
        root_node_id: str | None,
        session: Session,
        rate_limit: RateLimit,
        request_id: str,
    ):
        """
        策略分发：按应用类型（AppMode）选对应的 Generator。

        两类分法：
        - 简单型（CHAT / COMPLETION / AGENT*）：直接同步调用 Generator，结果即时返回
        - 重型（WORKFLOW / ADVANCED_CHAT）：流式走 Celery 异步 + SSE，阻塞式直接同步跑
        """
        # 兼容：Agent 应用在会话模式下会被当作 AGENT_CHAT 处理
        # 运行前，先判断这个 Agent 是否开启了会话能力；
        # 如果开启了，就把它临时当成 AGENT_CHAT 来处理，然后根据最终模式 (effective_mode) 分发到不同的生成逻辑。
        effective_mode = (
            AppMode.AGENT_CHAT
            if app_model.is_agent_with_session(session=session) and app_model.mode != AppMode.AGENT_CHAT
            else app_model.mode
        )
        match effective_mode:
            case AppMode.COMPLETION:
                # 简单型：同步调 Generator，convert_to_event_stream 把 dict/流统一转成事件流
                return rate_limit.generate(
                    CompletionAppGenerator.convert_to_event_stream(
                        CompletionAppGenerator().generate(
                            session=session,
                            app_model=app_model,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=streaming,
                        ),
                    ),
                    request_id=request_id,
                )
            case AppMode.AGENT_CHAT:
                return rate_limit.generate(
                    AgentChatAppGenerator.convert_to_event_stream(
                        AgentChatAppGenerator().generate(
                            session=session,
                            app_model=app_model,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=streaming,
                        ),
                    ),
                    request_id,
                )
            case AppMode.AGENT:
                return rate_limit.generate(
                    AgentAppGenerator.convert_to_event_stream(
                        AgentAppGenerator().generate(
                            app_model=app_model,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            session=session,
                            streaming=streaming,
                        ),
                    ),
                    request_id,
                )
            case AppMode.CHAT:
                return rate_limit.generate(
                    ChatAppGenerator.convert_to_event_stream(
                        ChatAppGenerator().generate(
                            session=session,
                            app_model=app_model,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=streaming,
                        ),
                    ),
                    request_id=request_id,
                )
            case AppMode.ADVANCED_CHAT:
                workflow_id = args.get("workflow_id")
                workflow = cls._get_workflow(app_model, invoke_from, workflow_id, session=session)

                if streaming:
                    # 流式模式：先建立 SSE 连接，等第一个订阅者接入时才提交 Celery 任务（懒启动）
                    # 序列化执行参数 → 存入 payload_json，交给 Celery worker 执行
                    with rate_limit_context(rate_limit, request_id):
                        payload = AppExecutionParams.new(
                            app_model=app_model,
                            workflow=workflow,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=True,
                            call_depth=0,
                            workflow_run_id=str(uuid.uuid4()),
                        )
                        payload_json = payload.model_dump_json()

                    def on_subscribe():
                        workflow_based_app_execution_task.delay(payload_json)

                    on_subscribe = cls._build_streaming_task_on_subscribe(on_subscribe)
                    generator = AdvancedChatAppGenerator()
                    return rate_limit.generate(
                        generator.convert_to_event_stream(
                            generator.retrieve_events(
                                AppMode.ADVANCED_CHAT,
                                payload.workflow_run_id,
                                on_subscribe=on_subscribe,
                            ),
                        ),
                        request_id=request_id,
                    )

                # Blocking mode: run synchronously and return JSON instead of SSE
                # Keep behaviour consistent with WORKFLOW blocking branch.
                pause_config = PauseStateLayerConfig(
                    session_factory=session_factory.get_session_maker(),
                    state_owner_user_id=workflow.created_by,
                )
                advanced_generator = AdvancedChatAppGenerator()
                return rate_limit.generate(
                    advanced_generator.convert_to_event_stream(
                        advanced_generator.generate(
                            app_model=app_model,
                            workflow=workflow,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            workflow_run_id=str(uuid.uuid4()),
                            streaming=False,
                            pause_state_config=pause_config,
                            session=session,
                        )
                    ),
                    request_id=request_id,
                )
            case AppMode.WORKFLOW:
                workflow_id = args.get("workflow_id")
                workflow = cls._get_workflow(app_model, invoke_from, workflow_id, session=session)
                if streaming:
                    # 流式模式：同 ADVANCED_CHAT，SSE 懒启动 + Celery 异步执行
                    with rate_limit_context(rate_limit, request_id):
                        payload = AppExecutionParams.new(
                            app_model=app_model,
                            workflow=workflow,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=True,
                            call_depth=0,
                            root_node_id=root_node_id,
                            workflow_run_id=str(uuid.uuid4()),
                        )
                        payload_json = payload.model_dump_json()

                    def on_subscribe():
                        workflow_based_app_execution_task.delay(payload_json)

                    on_subscribe = cls._build_streaming_task_on_subscribe(on_subscribe)
                    return rate_limit.generate(
                        WorkflowAppGenerator.convert_to_event_stream(
                            MessageBasedAppGenerator.retrieve_events(
                                AppMode.WORKFLOW,
                                payload.workflow_run_id,
                                on_subscribe=on_subscribe,
                            ),
                        ),
                        request_id,
                    )

                pause_config = PauseStateLayerConfig(
                    session_factory=session_factory.get_session_maker(),
                    state_owner_user_id=workflow.created_by,
                )
                return rate_limit.generate(
                    WorkflowAppGenerator.convert_to_event_stream(
                        WorkflowAppGenerator().generate(
                            app_model=app_model,
                            workflow=workflow,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=False,
                            root_node_id=root_node_id,
                            call_depth=0,
                            pause_state_config=pause_config,
                        ),
                    ),
                    request_id,
                )
            case _:
                raise ValueError(f"Invalid app mode {app_model.mode}")

    @staticmethod
    def _get_max_active_requests(app: App) -> int:
        """
        Get the maximum number of active requests allowed for an app.

        Returns the smaller value between app's custom limit and global config limit.
        A value of 0 means infinite (no limit).

        Args:
            app: The App model instance

        Returns:
            The maximum number of active requests allowed
        """
        app_limit = app.max_active_requests or dify_config.APP_DEFAULT_ACTIVE_REQUESTS
        config_limit = dify_config.APP_MAX_ACTIVE_REQUESTS

        # Filter out infinite (0) values and return the minimum, or 0 if both are infinite
        limits = [limit for limit in [app_limit, config_limit] if limit > 0]
        return min(limits) if limits else 0

    @classmethod
    def generate_single_iteration(
        cls,
        app_model: App,
        user: Account,
        node_id: str,
        args: Any,
        *,
        session: Session,
        streaming: bool = True,
    ):
        """
        单节点调试：只跑工作流里的某个"迭代节点"（console 编辑器调试用）。
        node_id 指定要调试的节点，其余节点不执行。
        """
        match app_model.mode:
            case AppMode.COMPLETION | AppMode.CHAT | AppMode.AGENT_CHAT:
                raise ValueError(f"Invalid app mode {app_model.mode}")
            case AppMode.ADVANCED_CHAT:
                workflow = cls._get_workflow(app_model, InvokeFrom.DEBUGGER, session=session)
                return AdvancedChatAppGenerator.convert_to_event_stream(
                    AdvancedChatAppGenerator().single_iteration_generate(
                        app_model=app_model,
                        workflow=workflow,
                        node_id=node_id,
                        user=user,
                        args=args,
                        streaming=streaming,
                        session=session,
                    )
                )
            case AppMode.WORKFLOW:
                workflow = cls._get_workflow(app_model, InvokeFrom.DEBUGGER, session=session)
                return AdvancedChatAppGenerator.convert_to_event_stream(
                    WorkflowAppGenerator().single_iteration_generate(
                        app_model=app_model,
                        workflow=workflow,
                        node_id=node_id,
                        user=user,
                        args=args,
                        streaming=streaming,
                        session=session,
                    )
                )
            case AppMode.CHANNEL | AppMode.RAG_PIPELINE:
                raise ValueError(f"Invalid app mode {app_model.mode}")
            case _:
                raise ValueError(f"Invalid app mode {app_model.mode}")

    @classmethod
    def generate_single_loop(
        cls,
        app_model: App,
        user: Account,
        node_id: str,
        args: LoopNodeRunPayload,
        *,
        session: Session,
        streaming: bool = True,
    ):
        """
        单节点调试：只跑工作流里的某个"循环节点"（console 编辑器调试用）。
        """
        match app_model.mode:
            case AppMode.COMPLETION | AppMode.CHAT | AppMode.AGENT_CHAT:
                raise ValueError(f"Invalid app mode {app_model.mode}")
            case AppMode.ADVANCED_CHAT:
                workflow = cls._get_workflow(app_model, InvokeFrom.DEBUGGER, session=session)
                return AdvancedChatAppGenerator.convert_to_event_stream(
                    AdvancedChatAppGenerator().single_loop_generate(
                        app_model=app_model,
                        workflow=workflow,
                        node_id=node_id,
                        user=user,
                        args=args,
                        streaming=streaming,
                        session=session,
                    )
                )
            case AppMode.WORKFLOW:
                workflow = cls._get_workflow(app_model, InvokeFrom.DEBUGGER, session=session)
                return AdvancedChatAppGenerator.convert_to_event_stream(
                    WorkflowAppGenerator().single_loop_generate(
                        app_model=app_model,
                        workflow=workflow,
                        node_id=node_id,
                        user=user,
                        args=args,
                        streaming=streaming,
                        session=session,
                    )
                )
            case AppMode.CHANNEL | AppMode.RAG_PIPELINE:
                raise ValueError(f"Invalid app mode {app_model.mode}")
            case _:
                raise ValueError(f"Invalid app mode {app_model.mode}")

    @classmethod
    def generate_more_like_this(
        cls,
        app_model: App,
        user: Account | EndUser,
        message_id: str,
        invoke_from: InvokeFrom,
        *,
        session: Session,
        streaming: bool = True,
    ) -> Mapping | Generator:
        """
        Generate more like this
        :param app_model: app model
        :param user: user
        :param message_id: message id
        :param invoke_from: invoke from
        :param streaming: streaming
        :return:
        """
        return CompletionAppGenerator().generate_more_like_this(
            session=session,
            app_model=app_model,
            message_id=message_id,
            user=user,
            invoke_from=invoke_from,
            stream=streaming,
        )

    @classmethod
    def _get_workflow(
        cls,
        app_model: App,
        invoke_from: InvokeFrom,
        workflow_id: str | None = None,
        *,
        session: Session,
    ) -> Workflow:
        """
        Get workflow
        :param app_model: app model
        :param invoke_from: invoke from
        :param workflow_id: optional workflow id to specify a specific version
        :return:
        """
        workflow_service = WorkflowService()

        # If workflow_id is specified, get the specific workflow version
        if workflow_id:
            try:
                _ = uuid.UUID(workflow_id)
            except ValueError:
                raise WorkflowIdFormatError(f"Invalid workflow_id format: '{workflow_id}'. ")
            workflow = workflow_service.get_published_workflow_by_id(
                app_model=app_model, workflow_id=workflow_id, session=session
            )
            if not workflow:
                raise WorkflowNotFoundError(f"Workflow not found with id: {workflow_id}")
            return workflow

        if invoke_from == InvokeFrom.DEBUGGER:
            # fetch draft workflow by app_model
            workflow = workflow_service.get_draft_workflow(app_model=app_model, session=session)

            if not workflow:
                raise ValueError("Workflow not initialized")
        else:
            # fetch published workflow by app_model
            workflow = workflow_service.get_published_workflow(app_model=app_model, session=session)

            if not workflow:
                raise ValueError("Workflow not published")

        return workflow

    @classmethod
    def get_response_generator(
        cls,
        app_model: App,
        workflow_run: WorkflowRun,
    ):
        """
        按工作流运行记录从 Redis Topic 重放事件流（SSE 断线重连/恢复场景用）。
        """
        if workflow_run.status.is_ended():
            # TODO(QuantumGhost): handled the ended scenario.
            pass

        generator = AdvancedChatAppGenerator()

        return generator.convert_to_event_stream(
            generator.retrieve_events(AppMode(app_model.mode), workflow_run.id),
        )
