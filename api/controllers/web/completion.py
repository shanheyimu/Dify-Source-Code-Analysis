import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from werkzeug.exceptions import BadRequest, InternalServerError, NotFound

import services
from ...controllers.common.fields import GeneratedAppResponse, SimpleResultResponse
from ...controllers.common.schema import register_response_schema_models, register_schema_models
from ...controllers.console.app.wraps import with_session
from ...controllers.web import web_ns
from ...controllers.web.error import (
    AgentNotPublishedError,
    AppUnavailableError,
    CompletionRequestError,
    ConversationCompletedError,
    NotChatAppError,
    NotCompletionAppError,
    ProviderModelCurrentlyNotSupportError,
    ProviderNotInitializeError,
    ProviderQuotaExceededError,
)
from ...controllers.web.error import InvokeRateLimitError as InvokeRateLimitHttpError
from ...controllers.web.wraps import WebApiResource
from ...core.app.apps.agent_app.errors import AgentAppNotPublishedError
from ...core.app.entities.app_invoke_entities import InvokeFrom
from ...core.errors.error import (
    ModelCurrentlyNotSupportError,
    ProviderTokenNotInitError,
    QuotaExceededError,
)
from graphon.model_runtime.errors.invoke import InvokeError
from libs import helper
from ...libs.helper import uuid_value
from ...models.model import App, AppMode, EndUser
from ...services.app_generate_service import AppGenerateService
from ...services.app_task_service import AppTaskService
from ...services.conversation_service import ConversationService
from ...services.errors.llm import InvokeRateLimitError

logger = logging.getLogger(__name__)


def _resolve_agent_app_streaming(*, app_mode: AppMode, response_mode: str | None) -> bool:
    """Agent App runtime is SSE-only until backend blocking runs are supported."""
    if app_mode != AppMode.AGENT:
        return response_mode == "streaming"
    if response_mode == "blocking":
        raise BadRequest("Agent App only supports streaming response mode.")
    return True


class CompletionMessagePayload(BaseModel):
    inputs: dict[str, Any] = Field(
        description="Input variables for the completion",
    )
    query: str = Field(default="", description="Query text for completion")
    files: list[dict[str, Any]] | None = Field(
        default=None,
        description="Files to be processed",
    )
    response_mode: Literal["blocking", "streaming"] | None = Field(
        default=None, description="Response mode: blocking or streaming"
    )
    retriever_from: str = Field(default="web_app", description="Source of retriever")


class ChatMessagePayload(BaseModel):
    inputs: dict[str, Any] = Field(
        description="Input variables for the chat",
    )
    query: str = Field(description="User query/message")
    files: list[dict[str, Any]] | None = Field(
        default=None,
        description="Files to be processed",
    )
    response_mode: Literal["blocking", "streaming"] | None = Field(
        default=None, description="Response mode: blocking or streaming"
    )
    conversation_id: str | None = Field(default=None, description="Conversation ID")
    parent_message_id: str | None = Field(default=None, description="Parent message ID")
    retriever_from: str = Field(default="web_app", description="Source of retriever")

    @field_validator("conversation_id", "parent_message_id")
    @classmethod
    def validate_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return uuid_value(value)


register_schema_models(web_ns, CompletionMessagePayload, ChatMessagePayload)
register_response_schema_models(web_ns, GeneratedAppResponse, SimpleResultResponse)


# 单次文本生成接口：POST /api/completion-messages
# 对应"文本生成应用"（AppMode.COMPLETION），一次请求返回一段完整文本，不涉及多轮会话。
@web_ns.route("/completion-messages")
class CompletionApi(WebApiResource):
    @web_ns.doc("Create Completion Message")
    @web_ns.doc(description="Create a completion message for text generation applications.")
    @web_ns.expect(web_ns.models[CompletionMessagePayload.__name__])
    @web_ns.doc(
        responses={
            200: "Success",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "App Not Found",
            500: "Internal Server Error",
        }
    )
    # 以上 @web_ns.* 装饰器只用于生成 OpenAPI/Swagger 文档，不影响业务逻辑。
    @web_ns.response(200, "Success", web_ns.models[GeneratedAppResponse.__name__])
    # @with_session 注入数据库会话 session（装饰器注入，同 WebApiResource 注入 app_model/end_user 一样）
    # session: 数据库会话 | app_model: 当前应用 | end_user: 当前最终用户（均由装饰器注入）
    @with_session
    def post(self, session: Session, app_model: App, end_user: EndUser):
        # 类型检查：本接口只服务"文本生成"应用，其他类型直接拒绝
        if app_model.mode != AppMode.COMPLETION:
            raise NotCompletionAppError()

        # payload 校验：用 Pydantic 模型校验并解析请求体（约等于 Spring 的 @Validated DTO）
        payload = CompletionMessagePayload.model_validate(web_ns.payload or {})
        # 转成普通 dict 传给 Service 层，排除值为 None 的字段
        args = payload.model_dump(exclude_none=True)

        # 由请求体决定响应方式：streaming=流式(SSE) / blocking=一次性 JSON
        streaming = payload.response_mode == "streaming"
        # 单次生成不自动生成会话名称（chat 接口才有会话概念）
        args["auto_generate_name"] = False

        try:
            # 核心调用：所有生成型接口的统一入口
            # 内部会做 配额预留 + 限流（护栏层）→ 按 AppMode 策略分发 → 对应 AppGenerator
            # 返回 dict（阻塞）或 Generator（流式）
            response = AppGenerateService.generate(
                session=session,
                app_model=app_model,
                user=end_user,
                args=args,
                invoke_from=InvokeFrom.WEB_APP,  # 标记请求来源（影响事件处理/权限语义）
                streaming=streaming,
            )

            # response-contract:ignore compact_generate_response
            # 响应包装：dict → JSON；Generator → SSE 流（text/event-stream）
            return helper.compact_generate_response(response)
        # 异常翻译链：把 Service 层的业务异常翻译成 HTTP 错误码（对应 Spring 的 @ExceptionHandler）
        except services.errors.conversation.ConversationNotExistsError:
            raise NotFound("Conversation Not Exists.")  # 会话不存在 → 404
        except services.errors.conversation.ConversationCompletedError:
            raise ConversationCompletedError()  # 会话已结束 → 400
        except services.errors.app_model_config.AppModelConfigBrokenError:
            logger.exception("App model config broken.")
            raise AppUnavailableError()  # 应用配置损坏 → 400
        except AgentAppNotPublishedError:
            raise AgentNotPublishedError()  # Agent 应用未发布 → 400
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)  # 模型没配 API Key → 400
        except QuotaExceededError:
            raise ProviderQuotaExceededError()  # 额度用尽 → 400
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()  # 当前模型不支持 → 400
        except InvokeError as e:
            raise CompletionRequestError(e.description)  # LLM 调用失败 → 400
        except ValueError as e:
            raise e  # 业务逻辑自身的参数错误，原样抛出
        except Exception as e:
            logger.exception("internal server error.")
            raise InternalServerError()  # 兜底：未知异常 → 500


@web_ns.route("/completion-messages/<string:task_id>/stop")
class CompletionStopApi(WebApiResource):
    @web_ns.doc("Stop Completion Message")
    @web_ns.doc(description="Stop a running completion message task.")
    @web_ns.doc(params={"task_id": {"description": "Task ID to stop", "type": "string", "required": True}})
    @web_ns.doc(
        responses={
            200: "Success",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Task Not Found",
            500: "Internal Server Error",
        }
    )
    @web_ns.response(200, "Success", web_ns.models[SimpleResultResponse.__name__])
    def post(self, app_model: App, end_user: EndUser, task_id: str):
        if app_model.mode != AppMode.COMPLETION:
            raise NotCompletionAppError()

        AppTaskService.stop_task(
            task_id=task_id,
            invoke_from=InvokeFrom.WEB_APP,
            user_id=end_user.id,
            app_mode=AppMode.value_of(app_model.mode),
        )

        return SimpleResultResponse(result="success").model_dump(mode="json"), 200


@web_ns.route("/chat-messages")
class ChatApi(WebApiResource):
    @web_ns.doc("Create Chat Message")
    @web_ns.doc(description="Create a chat message for conversational applications.")
    @web_ns.expect(web_ns.models[ChatMessagePayload.__name__])
    @web_ns.doc(
        responses={
            200: "Success",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "App Not Found",
            500: "Internal Server Error",
        }
    )
    @web_ns.response(200, "Success", web_ns.models[GeneratedAppResponse.__name__])
    @with_session
    def post(self, session: Session, app_model: App, end_user: EndUser):
        app_mode = AppMode.value_of(app_model.mode)
        if app_mode not in {AppMode.CHAT, AppMode.AGENT_CHAT, AppMode.ADVANCED_CHAT, AppMode.AGENT}:
            raise NotChatAppError()

        payload = ChatMessagePayload.model_validate(web_ns.payload or {})
        args = payload.model_dump(exclude_none=True)

        streaming = _resolve_agent_app_streaming(app_mode=app_mode, response_mode=payload.response_mode)
        args["auto_generate_name"] = False

        try:
            # Eagerly validate conversation to avoid hanging on invalid conversation_id
            if payload.conversation_id:
                ConversationService.get_conversation(
                    app_model=app_model,
                    conversation_id=payload.conversation_id,
                    user=end_user,
                    session=session,
                )

            response = AppGenerateService.generate(
                session=session,
                app_model=app_model,
                user=end_user,
                args=args,
                invoke_from=InvokeFrom.WEB_APP,
                streaming=streaming,
            )

            # response-contract:ignore compact_generate_response
            return helper.compact_generate_response(response)
        except services.errors.conversation.ConversationNotExistsError:
            raise NotFound("Conversation Not Exists.")
        except services.errors.conversation.ConversationCompletedError:
            raise ConversationCompletedError()
        except services.errors.app_model_config.AppModelConfigBrokenError:
            logger.exception("App model config broken.")
            raise AppUnavailableError()
        except AgentAppNotPublishedError:
            raise AgentNotPublishedError()
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)
        except QuotaExceededError:
            raise ProviderQuotaExceededError()
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()
        except InvokeRateLimitError as ex:
            raise InvokeRateLimitHttpError(ex.description)
        except InvokeError as e:
            raise CompletionRequestError(e.description)
        except ValueError as e:
            raise e
        except Exception as e:
            logger.exception("internal server error.")
            raise InternalServerError()


@web_ns.route("/chat-messages/<string:task_id>/stop")
class ChatStopApi(WebApiResource):
    @web_ns.doc("Stop Chat Message")
    @web_ns.doc(description="Stop a running chat message task.")
    @web_ns.doc(params={"task_id": {"description": "Task ID to stop", "type": "string", "required": True}})
    @web_ns.doc(
        responses={
            200: "Success",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Task Not Found",
            500: "Internal Server Error",
        }
    )
    @web_ns.response(200, "Success", web_ns.models[SimpleResultResponse.__name__])
    def post(self, app_model: App, end_user: EndUser, task_id: str):
        app_mode = AppMode.value_of(app_model.mode)
        if app_mode not in {AppMode.CHAT, AppMode.AGENT_CHAT, AppMode.ADVANCED_CHAT, AppMode.AGENT}:
            raise NotChatAppError()

        AppTaskService.stop_task(
            task_id=task_id,
            invoke_from=InvokeFrom.WEB_APP,
            user_id=end_user.id,
            app_mode=app_mode,
        )

        return SimpleResultResponse(result="success").model_dump(mode="json"), 200
