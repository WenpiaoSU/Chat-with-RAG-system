# -*- coding: utf-8 -*-
"""
OpenAI LLM 实现

基于 LangChain 的 OpenAI API 调用实现，支持同步/异步、流式/非流式调用。
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI 大语言模型
    
    使用 LangChain 的 ChatOpenAI 接口调用 OpenAI API。
    """
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        streaming: bool = False,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI LLM
        
        Args:
            model_name: 模型名称，默认 gpt-4o-mini
            api_key: OpenAI API Key，若为 None 则从环境变量 OPENAI_API_KEY 读取
            api_base: API 基础地址，默认 https://api.openai.com/v1
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            streaming: 是否启用流式输出
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            **kwargs: 其他参数
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs,
        )
        
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._client = self._create_client()
    
    def _create_client(self) -> ChatOpenAI:
        """创建 LangChain ChatOpenAI 客户端
        
        Returns:
            ChatOpenAI: LangChain 客户端实例
        """
        client_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
            "max_retries": self.max_retries,
        }
        
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        
        if self.timeout:
            client_kwargs["request_timeout"] = self.timeout
        
        client_kwargs.update(self._extra_params)
        
        return ChatOpenAI(**client_kwargs)
    
    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """同步调用 LLM 生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数（如 stop、response_format 等）
            
        Returns:
            LLMResponse: 包含生成结果的响应对象
        """
        try:
            messages = [HumanMessage(content=prompt)]
            response = self._client.invoke(messages, **kwargs)
            
            return LLMResponse(
                content=response.content,
                raw_response=response,
                usage={
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                    "total_tokens": response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                } if hasattr(response, "usage_metadata") else None,
                model=self.model_name,
                finish_reason=None,
            )
        except Exception as e:
            logger.error(f"OpenAI LLM invoke error: {e}")
            raise
    
    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """异步调用 LLM 生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            LLMResponse: 包含生成结果的响应对象
        """
        try:
            messages = [HumanMessage(content=prompt)]
            response = await self._client.ainvoke(messages, **kwargs)
            
            return LLMResponse(
                content=response.content,
                raw_response=response,
                usage={
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                    "total_tokens": response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                } if hasattr(response, "usage_metadata") else None,
                model=self.model_name,
                finish_reason=None,
            )
        except Exception as e:
            logger.error(f"OpenAI LLM async invoke error: {e}")
            raise
    
    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """流式调用 LLM（同步版本）
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Yields:
            str: 逐步生成的文本片段
        """
        if not self.streaming:
            yield self.invoke(prompt, **kwargs).content
            return
        
        try:
            messages = [HumanMessage(content=prompt)]
            for chunk in self._client.stream(messages, **kwargs):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"OpenAI LLM stream error: {e}")
            raise
    
    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """流式调用 LLM（异步版本）
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Yields:
            str: 逐步生成的文本片段
        """
        if not self.streaming:
            response = await self.ainvoke(prompt, **kwargs)
            yield response.content
            return
        
        try:
            messages = [HumanMessage(content=prompt)]
            async for chunk in self._client.astream(messages, **kwargs):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"OpenAI LLM async stream error: {e}")
            raise
    
    def invoke_with_messages(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """使用消息列表调用 LLM
        
        支持 System、User、Assistant 等多种消息类型。
        
        Args:
            messages: LangChain 消息列表
            **kwargs: 其他参数
            
        Returns:
            LLMResponse: 包含生成结果的响应对象
        """
        try:
            response = self._client.invoke(messages, **kwargs)
            
            return LLMResponse(
                content=response.content,
                raw_response=response,
                usage={
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                    "total_tokens": response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                } if hasattr(response, "usage_metadata") else None,
                model=self.model_name,
                finish_reason=None,
            )
        except Exception as e:
            logger.error(f"OpenAI LLM invoke_with_messages error: {e}")
            raise
    
    def get_num_tokens(self, text: str) -> int:
        """计算文本的 token 数量
        
        使用 tiktoken 计算文本的 token 数。
        
        Args:
            text: 输入文本
            
        Returns:
            int: token 数量
        """
        try:
            return self._client.get_num_tokens(text)
        except Exception:
            return len(text) // 4
    
    def get_token_count(self, messages: List[BaseMessage]) -> int:
        """计算消息列表的 token 总数
        
        Args:
            messages: LangChain 消息列表
            
        Returns:
            int: token 总数
        """
        return self._client.get_token_count(messages)
