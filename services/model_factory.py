"""统一模型工厂：所有厂商均通过 OpenAI 兼容协议调用。

不再使用任何厂商私有 SDK（DashScope 原生接口、Tongyi SDK 等）。
厂商（OFFICEMATE_LLM_PROVIDER）只决定三件事：
1. 读取哪个 API Key 环境变量
2. 默认 base_url（OpenAI 兼容端点）
3. 页面候选模型列表

支持的厂商：
- dashscope   : 阿里云百炼（OpenAI 兼容模式，Qwen / DeepSeek / 通义系全都能调）
- openai      : OpenAI 官方或任意 OpenAI 兼容网关
- deepseek    : DeepSeek 官方 API
- zhipu       : 智谱 AI（OpenAI 兼容端点）
- ollama      : 本地 Ollama（OpenAI 兼容端点，无需密钥）
- siliconflow : 硅基流动（OpenAI 兼容，可选）

Base URL 优先级：OFFICEMATE_{PROVIDER}_BASE_URL > OFFICEMATE_BASE_URL > 内置默认
"""

import logging
from typing import Optional

import config_data as config

logger = logging.getLogger(__name__)


def _resolve_kwargs(model: str, provider: str, require_key: bool = True) -> dict:
    """组装 OpenAI 兼容客户端的公共参数。"""
    base_url = config.get_base_url(provider)
    api_key = config.get_api_key(provider)
    kwargs = {"model": model, "max_retries": 3}
    if api_key:
        kwargs["api_key"] = api_key
    elif provider == "ollama":
        kwargs["api_key"] = "ollama"  # 本地服务不校验密钥，占位即可
    elif require_key:
        raise RuntimeError(
            f"未配置厂商 {provider} 的 API Key（{config._API_KEY_ENV_VARS.get(provider)}）。"
        )
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def create_chat_model(model_name: Optional[str] = None):
    """创建对话模型（OpenAI 兼容协议，支持页面级模型覆盖）。"""
    from langchain_openai import ChatOpenAI

    model = model_name or config.chat_model_name
    return ChatOpenAI(**_resolve_kwargs(model, config.llm_provider))


def create_embeddings():
    """创建主 Embedding 模型（OpenAI 兼容协议）。"""
    from langchain_openai import OpenAIEmbeddings

    provider = config.llm_provider
    kwargs = _resolve_kwargs(config.embedding_model_name, provider)
    kwargs.pop("max_retries", None)
    # 非官方 OpenAI 端点不收 token 数组，直接发原始字符串
    kwargs["check_embedding_ctx_length"] = False
    return OpenAIEmbeddings(**kwargs)


def create_embeddings_secondary():
    """创建第二个 Embedding 模型（多 Embedding 融合用，未配置返回 None）。"""
    if not config.embedding2_model_name:
        return None
    from langchain_openai import OpenAIEmbeddings

    provider = config.llm_provider
    kwargs = _resolve_kwargs(config.embedding2_model_name, provider, require_key=False)
    kwargs["check_embedding_ctx_length"] = False
    return OpenAIEmbeddings(**kwargs)


def get_provider_info() -> dict:
    """返回当前厂商信息。"""
    return {
        "provider": config.llm_provider,
        "embedding_model": config.embedding_model_name,
        "chat_model": config.chat_model_name,
        "embedding2_model": config.embedding2_model_name or "(未启用)",
        "base_url": config.get_base_url() or "(默认)",
    }