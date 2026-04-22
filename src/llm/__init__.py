# -*- coding: utf-8 -*-
"""
LLM 模块

提供大语言模型调用接口，支持多种提供商（OpenAI 等）。
"""

from .base import BaseLLM, LLMResponse
from .openai_llm import OpenAILLM

__all__ = ["BaseLLM", "LLMResponse", "OpenAILLM"]
