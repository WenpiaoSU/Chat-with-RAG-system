# -*- coding: utf-8 -*-
"""
LLM 工厂类

提供统一的 LLM 实例创建接口，根据配置自动选择合适的 LLM 实现。
支持 OpenAI API 和 Xinference 本地部署两种方式。
"""

import logging
from typing import Any, Dict, Optional, Type

from .base import BaseLLM
from .openai_llm import OpenAILLM
from .xinference_llm import XinferenceLLM

logger = logging.getLogger(__name__)

_llm_instance_cache: Dict[str, BaseLLM] = {}


class LLMFactory:
    """LLM 工厂类

    根据配置创建和管理 LLM 实例，支持多种提供商。
    """

    _PROVIDER_MAP: Dict[str, Type[BaseLLM]] = {
        "openai": OpenAILLM,
        "xinference": XinferenceLLM,
    }

    @classmethod
    def create_from_settings(cls, **overrides: Any) -> BaseLLM:
        """从配置文件创建 LLM 实例

        根据 settings.llm.provider 配置自动选择合适的 LLM 实现。

        Args:
            **overrides: 要覆盖的配置参数

        Returns:
            BaseLLM: LLM 实例

        Raises:
            ValueError: 不支持的 provider
        """
        from ..configs.settings import get_settings

        settings = get_settings()
        provider = settings.llm.provider.lower()

        if provider not in cls._PROVIDER_MAP:
            raise ValueError(
                f"不支持的 LLM provider: {provider}。"
                f"支持的 provider: {list(cls._PROVIDER_MAP.keys())}"
            )

        llm_cls = cls._PROVIDER_MAP[provider]

        if provider == "openai":
            config = settings.llm.openai
            llm = llm_cls(
                model_name=overrides.get("model_name", config.model),
                api_key=overrides.get("api_key", config.api_key or None),
                api_base=overrides.get("api_base", config.api_base),
                temperature=overrides.get("temperature", config.temperature),
                max_tokens=overrides.get("max_tokens", config.max_tokens),
                streaming=overrides.get("streaming", config.streaming),
                **overrides,
            )
        elif provider == "xinference":
            config = settings.llm.xinference
            llm = llm_cls(
                endpoint=overrides.get("endpoint", config.endpoint),
                model_uid=overrides.get("model_uid", config.model_uid),
                temperature=overrides.get("temperature", config.temperature),
                max_tokens=overrides.get("max_tokens", config.max_tokens),
                streaming=overrides.get("streaming", config.streaming),
                timeout=overrides.get("timeout", config.timeout),
                **overrides,
            )
        else:
            raise ValueError(f"不支持的 provider: {provider}")

        return llm

    @classmethod
    def create(
        cls,
        provider: str,
        **kwargs: Any,
    ) -> BaseLLM:
        """直接指定 provider 创建 LLM 实例

        Args:
            provider: 提供商名称 (openai | xinference)
            **kwargs: 提供商特定的配置参数

        Returns:
            BaseLLM: LLM 实例

        Raises:
            ValueError: 不支持的 provider
        """
        provider = provider.lower()

        if provider not in cls._PROVIDER_MAP:
            raise ValueError(
                f"不支持的 LLM provider: {provider}。"
                f"支持的 provider: {list(cls._PROVIDER_MAP.keys())}"
            )

        llm_cls = cls._PROVIDER_MAP[provider]

        if provider == "openai":
            llm = llm_cls(
                model_name=kwargs.get("model_name", "gpt-4o-mini"),
                api_key=kwargs.get("api_key"),
                api_base=kwargs.get("api_base", "https://api.openai.com/v1"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000),
                streaming=kwargs.get("streaming", False),
                timeout=kwargs.get("timeout"),
                max_retries=kwargs.get("max_retries", 3),
            )
        elif provider == "xinference":
            llm = llm_cls(
                endpoint=kwargs.get("endpoint", "http://localhost:9997"),
                model_uid=kwargs.get("model_uid", "qwen2.5-7b-instruct"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000),
                streaming=kwargs.get("streaming", False),
                timeout=kwargs.get("timeout", 120.0),
                max_retries=kwargs.get("max_retries", 3),
            )
        else:
            raise ValueError(f"不支持的 provider: {provider}")

        return llm

    @classmethod
    def get_instance(cls, cache_key: str = "default") -> BaseLLM:
        """获取 LLM 单例实例

        相同 cache_key 的调用返回同一个实例。

        Args:
            cache_key: 缓存键，默认为 "default"

        Returns:
            BaseLLM: LLM 实例
        """
        if cache_key not in _llm_instance_cache:
            _llm_instance_cache[cache_key] = cls.create_from_settings()

        return _llm_instance_cache[cache_key]

    @classmethod
    def set_instance(cls, llm: BaseLLM, cache_key: str = "default") -> None:
        """设置 LLM 单例实例

        Args:
            llm: LLM 实例
            cache_key: 缓存键
        """
        _llm_instance_cache[cache_key] = llm

    @classmethod
    def clear_cache(cls) -> None:
        """清空 LLM 实例缓存"""
        _llm_instance_cache.clear()
        logger.info("LLM 实例缓存已清空")

    @classmethod
    def get_available_providers(cls) -> list:
        """获取支持的提供商列表

        Returns:
            list: 提供商名称列表
        """
        return list(cls._PROVIDER_MAP.keys())

    @classmethod
    def register_provider(cls, name: str, llm_cls: Type[BaseLLM]) -> None:
        """注册自定义 LLM 提供商

        Args:
            name: 提供商名称
            llm_cls: LLM 类
        """
        if not issubclass(llm_cls, BaseLLM):
            raise TypeError(f"{llm_cls} 必须继承自 BaseLLM")

        cls._PROVIDER_MAP[name.lower()] = llm_cls
        logger.info(f"已注册 LLM provider: {name}")

    @classmethod
    def get_current_provider(cls) -> str:
        """获取当前配置的 provider

        Returns:
            str: provider 名称
        """
        from ..configs.settings import get_settings

        settings = get_settings()
        return settings.llm.provider.lower()


def get_llm(**kwargs: Any) -> BaseLLM:
    """快捷函数：获取 LLM 实例

    这是一个便捷函数，等同于 LLMFactory.get_instance()。

    Args:
        **kwargs: 要覆盖的配置参数

    Returns:
        BaseLLM: LLM 实例
    """
    if kwargs:
        return LLMFactory.create_from_settings(**kwargs)
    return LLMFactory.get_instance()


def create_openai_llm(**kwargs: Any) -> OpenAILLM:
    """快捷函数：创建 OpenAI LLM 实例

    Args:
        **kwargs: 配置参数

    Returns:
        OpenAILLM: OpenAI LLM 实例
    """
    return LLMFactory.create("openai", **kwargs)


def create_xinference_llm(**kwargs: Any) -> XinferenceLLM:
    """快捷函数：创建 Xinference LLM 实例

    Args:
        **kwargs: 配置参数

    Returns:
        XinferenceLLM: Xinference LLM 实例
    """
    return LLMFactory.create("xinference", **kwargs)
