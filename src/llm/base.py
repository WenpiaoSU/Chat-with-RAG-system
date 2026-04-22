# -*- coding: utf-8 -*-
"""
LLM 抽象基类

定义 LLM 接口规范，所有具体 LLM 实现需继承此类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional


@dataclass
class LLMResponse:
    """LLM 响应数据结构
    
    Attributes:
        content: 生成的文本内容
        raw_response: 原始响应对象（提供商特定）
        usage: Token 使用统计信息
        model: 使用的模型名称
        finish_reason: 生成结束原因（如 stop、length 等）
    """
    content: str
    raw_response: Optional[Any] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None


class BaseLLM(ABC):
    """大语言模型抽象基类"""
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        streaming: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化 LLM
        
        Args:
            model_name: 模型名称
            temperature: 生成温度，范围 [0, 2]，值越大随机性越强
            max_tokens: 最大生成 token 数
            streaming: 是否启用流式输出
            **kwargs: 其他提供商特定参数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self._extra_params = kwargs
    
    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """同步调用 LLM 生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数（如 stop、response_format 等）
            
        Returns:
            LLMResponse: 包含生成结果的响应对象
        """
        pass
    
    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """异步调用 LLM 生成文本
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            LLMResponse: 包含生成结果的响应对象
        """
        pass
    
    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """流式调用 LLM（同步版本）
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Yields:
            str: 逐步生成的文本片段
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support sync streaming")
    
    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """流式调用 LLM（异步版本）
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Yields:
            str: 逐步生成的文本片段
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support async streaming")
    
    @abstractmethod
    def get_num_tokens(self, text: str) -> int:
        """计算文本的 token 数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: token 数量
        """
        pass
    
    def batch_invoke(self, prompts: List[str], **kwargs: Any) -> List[LLMResponse]:
        """批量同步调用 LLM
        
        Args:
            prompts: 输入提示词列表
            **kwargs: 其他参数
            
        Returns:
            List[LLMResponse]: 响应列表
        """
        return [self.invoke(prompt, **kwargs) for prompt in prompts]
    
    async def batch_ainvoke(self, prompts: List[str], **kwargs: Any) -> List[LLMResponse]:
        """批量异步调用 LLM
        
        Args:
            prompts: 输入提示词列表
            **kwargs: 其他参数
            
        Returns:
            List[LLMResponse]: 响应列表
        """
        import asyncio
        return await asyncio.gather(*[self.ainvoke(prompt, **kwargs) for prompt in prompts])
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, temperature={self.temperature})"
