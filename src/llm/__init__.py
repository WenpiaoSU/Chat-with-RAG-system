# -*- coding: utf-8 -*-
"""
LLM 模块

提供大语言模型调用接口，支持多种提供商（OpenAI、Xinference）。
"""

from .base import BaseLLM, LLMResponse
from .openai_llm import OpenAILLM
from .xinference_llm import XinferenceLLM, XinferenceClient, XinferenceEmbeddingLLM
from .llm_factory import (
    LLMFactory,
    get_llm,
    create_openai_llm,
    create_xinference_llm,
)

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "OpenAILLM",
    "XinferenceLLM",
    "XinferenceClient",
    "XinferenceEmbeddingLLM",
    "LLMFactory",
    "get_llm",
    "create_openai_llm",
    "create_xinference_llm",
]
