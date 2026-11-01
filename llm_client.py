import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAI

# 显式按文件位置定位 .env：无论从哪个目录启动服务都能读到
# （load_dotenv 默认只找当前工作目录，换目录启动会导致读不到 Key）
load_dotenv(Path(__file__).resolve().parent / ".env")

logger = logging.getLogger("llm_client")

_client = None


class LLMRequestError(Exception):
    """调用大模型失败。

    message 为面向用户的安全提示；kind 用于诊断分类：
    auth（密钥错误/未配置）、network（连接失败/超时）、empty（空回复）。
    原始异常通过 __cause__ 保留。
    """

    def __init__(self, message, kind):
        super().__init__(message)
        self.kind = kind


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise LLMRequestError("网络或密钥错误", "auth") from RuntimeError(
                "LLM_API_KEY not set. Please configure it in .env"
            )
        # base_url 直接取 .env 中的 LLM_BASE_URL（如 https://api.deepseek.com/v1），
        # SDK 会自动在其后拼接 /chat/completions，最终请求完整地址
        # https://api.deepseek.com/v1/chat/completions，不需要手动拼路径
        base_url = os.getenv("LLM_BASE_URL")
        _client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("LLM client initialized (base_url=%s)", base_url)
    return _client


def chat(messages, tools=None):
    """通用对话入口：发送完整消息列表，可选附带 OpenAI 格式 tools 定义。

    返回 {"content": str|None, "tool_calls": [{"id", "name", "arguments"}, ...]}；
    失败时抛出 LLMRequestError（kind: auth/network/empty）。
    """
    client = get_client()
    model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
    try:
        kwargs = {"model": model, "max_tokens": 1024, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = client.chat.completions.create(**kwargs)
    except AuthenticationError as exc:
        logger.error("LLM authentication failed (model=%s)", model)
        raise LLMRequestError("网络或密钥错误", "auth") from exc
    except APIConnectionError as exc:
        # 连接被拒、DNS 失败、超时（APITimeoutError 是其子类）都走这里
        logger.error("LLM connection failed (model=%s)", model)
        raise LLMRequestError("网络或密钥错误", "network") from exc
    message = response.choices[0].message
    tool_calls = [
        {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
        for tc in (message.tool_calls or [])
    ]
    if not message.content and not tool_calls:
        logger.error("LLM returned empty content (model=%s)", model)
        raise LLMRequestError("大模型返回内容为空", "empty")
    logger.info(
        "LLM responded (%d chars, %d tool_calls, model=%s)",
        len(message.content or ""),
        len(tool_calls),
        model,
    )
    return {"content": message.content, "tool_calls": tool_calls}


def ask_llm(question: str) -> str:
    """单轮问答（无工具），供 /rag 等旧流程使用。"""
    return chat([{"role": "user", "content": question}])["content"]
