import contextvars
import logging
import threading
import uuid
from collections.abc import Generator, Mapping
from typing import Any, Literal, overload

from flask import Flask, copy_current_request_context, current_app
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .....configs import dify_config
from .....core.app.app_config.easy_ui_based_app.model_config.converter import ModelConfigConverter
from .....core.app.app_config.features.file_upload.manager import FileUploadConfigManager
from .....core.app.apps.base_app_queue_manager import AppQueueManager, PublishFrom
from .....core.app.apps.completion.app_config_manager import CompletionAppConfigManager
from .....core.app.apps.completion.app_runner import CompletionAppRunner
from .....core.app.apps.completion.generate_response_converter import CompletionAppGenerateResponseConverter
from .....core.app.apps.exc import GenerateTaskStoppedError
from .....core.app.apps.message_based_app_generator import MessageBasedAppGenerator
from .....core.app.apps.message_based_app_queue_manager import MessageBasedAppQueueManager
from .....core.app.entities.app_invoke_entities import CompletionAppGenerateEntity, InvokeFrom
from .....core.db.session_factory import session_factory
from .....core.helper.trace_id_helper import extract_trace_session_id_from_args
from .....core.ops.ops_trace_manager import TraceQueueManager
from .....extensions.ext_database import db
from .....factories import file_factory
from graphon.model_runtime.errors.invoke import InvokeAuthorizationError
from .....models import Account, App, AppModelConfig, Conversation, EndUser, Message
from .....models.model import load_annotation_reply_config
from .....services.errors.app import MoreLikeThisDisabledError
from .....services.errors.message import MessageNotExistsError

logger = logging.getLogger(__name__)


class CompletionAppGenerator(MessageBasedAppGenerator):
    """
    "文本生成"应用的生成器（AppMode.COMPLETION）。

    由 AppGenerateService._dispatch_generate 按应用类型分发到这里。
    职责：编排一次单轮文本生成的完整流程——
    配置加载 → 实体构建 → 落库记录 → 启动 worker 线程执行 runner → 返回响应流。
    对比 ChatAppGenerator：completion 无多轮会话（conversation 始终为 None），更简单。
    """
    @overload
    def generate(
        self,
        app_model: App,
        user: Account | EndUser,
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: Literal[True],
        *,
        session: Session,
    ) -> Generator[str | Mapping[str, Any], None, None]: ...

    @overload
    def generate(
        self,
        app_model: App,
        user: Account | EndUser,
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: Literal[False],
        *,
        session: Session,
    ) -> Mapping[str, Any]: ...

    @overload
    def generate(
        self,
        app_model: App,
        user: Account | EndUser,
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: bool = False,
        *,
        session: Session,
    ) -> Mapping[str, Any] | Generator[str | Mapping[str, Any], None, None]: ...

    def generate(
        self,
        app_model: App,
        user: Account | EndUser,
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: bool = True,
        *,
        session: Session,
    ) -> Mapping[str, Any] | Generator[str | Mapping[str, Any], None, None]:
        """
        Generate App response.

        :param app_model: App
        :param user: account or end user
        :param args: request args
        :param invoke_from: invoke from source
        :param streaming: is stream
        """
        query = args["query"]
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        query = query.replace("\x00", "")
        inputs = args["inputs"]

        # get conversation
        conversation = None  # completion 是单轮生成，没有会话概念，始终为 None

        # 加载应用配置：从 App/AppModelConfig 表读出模型、提示词、变量等配置
        app_model_config = self._get_app_model_config(
            app_model=app_model,
            conversation=conversation,
            session=session,
        )

        # 调试模式才允许覆盖模型配置（override），正常请求禁止
        override_model_config_dict = None
        if args.get("model_config"):
            if invoke_from != InvokeFrom.DEBUGGER:
                raise ValueError("Only in App debug mode can override model config")

            # validate config
            override_model_config_dict = CompletionAppConfigManager.config_validate(
                tenant_id=app_model.tenant_id, config=args.get("model_config", {}), session=session
            )

        annotation_reply = (
            None if override_model_config_dict else load_annotation_reply_config(session, app_model_config.app_id)
        )

        # parse files
        # TODO(QuantumGhost): Move file parsing logic to the API controller layer
        # for better separation of concerns.
        #
        # For implementation reference, see the `_parse_file` function and
        # `DraftWorkflowNodeRunApi` class which handle this properly.
        with self._bind_file_access_scope(tenant_id=app_model.tenant_id, user=user, invoke_from=invoke_from):
            files = args["files"] if args.get("files") else []
            file_extra_config = FileUploadConfigManager.convert(
                override_model_config_dict or app_model_config.to_dict(annotation_reply=annotation_reply)
            )
            if file_extra_config:
                # 把前端传来的文件 dict 转成内部的 File 对象（校验类型/大小/上传方式）
                file_objs = file_factory.build_from_mappings(
                    mappings=files,
                    tenant_id=app_model.tenant_id,
                    config=file_extra_config,
                    access_controller=self._file_access_controller,
                )
            else:
                file_objs = []

            # 把 DB 配置转成运行实体 CompletionAppConfig（含变量定义、模型配置、功能开关）
            app_config = CompletionAppConfigManager.get_app_config(
                app_model=app_model,
                app_model_config=app_model_config,
                override_config_dict=override_model_config_dict,
                annotation_reply=annotation_reply,
            )

            # get tracing instance
            trace_manager = TraceQueueManager(
                app_id=app_model.id, user_id=user.id if isinstance(user, Account) else user.session_id
            )

            # 构建"运行实体"：贯穿本次生成的唯一数据对象（后续 runner / 队列都靠它传参）
            application_generate_entity = CompletionAppGenerateEntity(
                task_id=str(uuid.uuid4()),
                app_config=app_config,
                model_conf=ModelConfigConverter.convert(app_config),  # 模型配置 → 带凭证的运行配置
                file_upload_config=file_extra_config,
                inputs=self._prepare_user_inputs(
                    user_inputs=inputs, variables=app_config.variables, tenant_id=app_model.tenant_id
                ),
                query=query,
                files=list(file_objs),
                user_id=user.id,
                stream=streaming,
                invoke_from=invoke_from,
                extras={
                    **extract_trace_session_id_from_args(args),
                },
                trace_manager=trace_manager,
            )

            # 落库：创建 Conversation（单轮也会建）+ Message 记录
            (conversation, message) = self._init_generate_records(
                application_generate_entity,
                session=session,
            )

            # 队列管理器：worker 线程产出的事件通过它发布（附带会话/消息定位信息）
            queue_manager = MessageBasedAppQueueManager(
                task_id=application_generate_entity.task_id,
                user_id=application_generate_entity.user_id,
                invoke_from=application_generate_entity.invoke_from,
                conversation_id=conversation.id,
                app_mode=conversation.mode,
                message_id=message.id,
            )

            context = contextvars.copy_context()

            # 新线程执行真正的生成：复制当前请求上下文（Flask context + contextvars）
            @copy_current_request_context
            def worker_with_context():
                return context.run(
                    self._generate_worker,
                    flask_app=current_app._get_current_object(),  # type: ignore
                    application_generate_entity=application_generate_entity,
                    queue_manager=queue_manager,
                    message_id=message.id,
                )

            worker_thread = threading.Thread(target=worker_with_context)

            worker_thread.start()  # ★ 关键：worker 线程跑生成，主线程立即返回响应流

            # 主线程：从队列读事件 → dict（阻塞）或 Generator（流式）
            response = self._handle_response(
                application_generate_entity=application_generate_entity,
                queue_manager=queue_manager,
                conversation=conversation,
                message=message,
                user=user,
                stream=streaming,
            )

            return CompletionAppGenerateResponseConverter.convert(response=response, invoke_from=invoke_from)

    def _generate_worker(
        self,
        flask_app: Flask,
        application_generate_entity: CompletionAppGenerateEntity,
        queue_manager: AppQueueManager,
        message_id: str,
    ):
        """
        Generate worker in a new thread.
        :param flask_app: Flask app
        :param application_generate_entity: application generate entity
        :param queue_manager: queue manager
        :param message_id: message ID
        :return:
        """
        # 在新线程里重建 Flask 应用上下文（worker 线程没有请求上下文，需要手动重建）
        with flask_app.app_context():
            try:
                # get message
                message = self._get_message(message_id)

                # 真正执行：CompletionAppRunner.run() 拼 Prompt、调 LLM，
                # 结果以事件形式发布到 queue_manager（主线程读取后转 SSE/JSON）
                runner = CompletionAppRunner()
                with session_factory.create_session() as session:
                    runner.run(
                        application_generate_entity=application_generate_entity,
                        queue_manager=queue_manager,
                        message=message,
                        session=session,
                    )
            except GenerateTaskStoppedError:
                pass  # 用户点了停止：正常结束，不发布错误
            except InvokeAuthorizationError:
                # API Key 无效：发布错误事件（主线程转成 401 响应）
                queue_manager.publish_error(
                    InvokeAuthorizationError("Incorrect API key provided"), PublishFrom.APPLICATION_MANAGER
                )
            except ValidationError as e:
                logger.exception("Validation Error when generating")
                queue_manager.publish_error(e, PublishFrom.APPLICATION_MANAGER)
            except ValueError as e:
                if dify_config.DEBUG:
                    logger.exception("Error when generating")
                queue_manager.publish_error(e, PublishFrom.APPLICATION_MANAGER)
            except Exception as e:
                logger.exception("Unknown Error when generating")
                queue_manager.publish_error(e, PublishFrom.APPLICATION_MANAGER)
            finally:
                db.session.close()

    def generate_more_like_this(
        self,
        app_model: App,
        message_id: str,
        user: Account | EndUser,
        invoke_from: InvokeFrom,
        stream: bool = True,
        *,
        session: Session,
    ) -> Mapping | Generator[Mapping | str, None, None]:
        """
        生成"更多类似回答"（More Like This，前端"换一个"按钮）。

        以某条历史消息为输入，调高 temperature 重新生成，得到相似但不重复的回答。
        流程与 generate() 几乎一致，区别：基于已有 message 反推配置、并强制 temperature=0.9。
        """
        # 按消息 ID + 归属（应用/来源/用户）找到历史消息，防止越权访问
        stmt = select(Message).where(
            Message.id == message_id,
            Message.app_id == app_model.id,
            Message.from_source == ("api" if isinstance(user, EndUser) else "console"),
            Message.from_end_user_id == (user.id if isinstance(user, EndUser) else None),
            Message.from_account_id == (user.id if isinstance(user, Account) else None),
        )
        message = session.scalar(stmt)

        if not message:
            raise MessageNotExistsError()

        # 检查应用是否开启了"更多类似"功能
        current_app_model_config = (
            session.get(AppModelConfig, app_model.app_model_config_id) if app_model.app_model_config_id else None
        )
        if not current_app_model_config:
            raise MoreLikeThisDisabledError()

        more_like_this = current_app_model_config.more_like_this_dict

        if not current_app_model_config.more_like_this or more_like_this.get("enabled", False) is False:
            raise MoreLikeThisDisabledError()

        # 用历史消息所在的配置生成新回答（而非当前配置）
        conversation = session.get(Conversation, message.conversation_id) if message.conversation_id else None
        app_model_config = (
            session.get(AppModelConfig, conversation.app_model_config_id)
            if conversation and conversation.app_model_config_id
            else None
        )
        if not app_model_config:
            raise ValueError("Message app_model_config is None")
        annotation_reply = load_annotation_reply_config(session, app_model_config.app_id)
        override_model_config_dict = app_model_config.to_dict(annotation_reply=annotation_reply)
        model_dict = override_model_config_dict["model"]
        completion_params = model_dict.get("completion_params", {})
        completion_params["temperature"] = 0.9  # ★ 提高随机性，让"类似但不同"
        model_dict["completion_params"] = completion_params
        override_model_config_dict["model"] = model_dict

        with self._bind_file_access_scope(tenant_id=app_model.tenant_id, user=user, invoke_from=invoke_from):
            # parse files
            file_extra_config = FileUploadConfigManager.convert(override_model_config_dict)
            if file_extra_config:
                file_objs = file_factory.build_from_mappings(
                    mappings=message.message_files_with_session(session=session),
                    tenant_id=app_model.tenant_id,
                    config=file_extra_config,
                    access_controller=self._file_access_controller,
                )
            else:
                file_objs = []

            # convert to app config
            app_config = CompletionAppConfigManager.get_app_config(
                app_model=app_model,
                app_model_config=app_model_config,
                override_config_dict=override_model_config_dict,
                annotation_reply=annotation_reply,
            )

            # init application generate entity
            application_generate_entity = CompletionAppGenerateEntity(
                task_id=str(uuid.uuid4()),
                app_config=app_config,
                model_conf=ModelConfigConverter.convert(app_config),
                inputs=message.inputs_with_session(session=session),
                query=message.query,
                files=list(file_objs),
                user_id=user.id,
                stream=stream,
                invoke_from=invoke_from,
                extras={},
            )

            # init generate records
            (conversation, message) = self._init_generate_records(
                application_generate_entity,
                session=session,
            )

            # init queue manager
            queue_manager = MessageBasedAppQueueManager(
                task_id=application_generate_entity.task_id,
                user_id=application_generate_entity.user_id,
                invoke_from=application_generate_entity.invoke_from,
                conversation_id=conversation.id,
                app_mode=conversation.mode,
                message_id=message.id,
            )

            context = contextvars.copy_context()

            # new thread with request context
            @copy_current_request_context
            def worker_with_context():
                return context.run(
                    self._generate_worker,
                    flask_app=current_app._get_current_object(),  # type: ignore
                    application_generate_entity=application_generate_entity,
                    queue_manager=queue_manager,
                    message_id=message.id,
                )

            worker_thread = threading.Thread(target=worker_with_context)

            worker_thread.start()

            # return response or stream generator
            response = self._handle_response(
                application_generate_entity=application_generate_entity,
                queue_manager=queue_manager,
                conversation=conversation,
                message=message,
                user=user,
                stream=stream,
            )

            return CompletionAppGenerateResponseConverter.convert(response=response, invoke_from=invoke_from)
