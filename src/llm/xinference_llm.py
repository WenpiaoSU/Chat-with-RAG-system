# -*- coding: utf-8 -*-
"""
Xinference LLM 实现

基于 Xinference 本地部署的大语言模型调用实现。
支持同步/异步、流式/非流式调用，与 OpenAI API 兼容的接口设计。
"""

import asyncio
import logging
import requests
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class XinferenceLLM(BaseLLM):
    """Xinference 本地大语言模型"""
    def __init__(
        self,
        endpoint: str = "http://localhost:9997",
        model_uid: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        streaming: bool = False,
        timeout: Optional[float] = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """初始化 Xinference LLM

        Args:
            endpoint: Xinference 服务地址
            model_uid: 模型 UID（启动 Xinference 时指定的模型标识）
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            streaming: 是否启用流式输出
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            **kwargs: 其他参数
        """
        super().__init__(
            model_name=model_uid,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs,
        )

        self.endpoint = endpoint.rstrip("/")
        self.model_uid = model_uid
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _get_client(self) -> "XinferenceClient":
        """获取或创建 Xinference 客户端

        Returns:
            XinferenceClient: 客户端实例
        """
        if self._client is None:
            self._client = XinferenceClient(
                endpoint=self.endpoint,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """同步调用 LLM 生成文本

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数（如 stop、response_format 等）

        Returns:
            LLMResponse: 包含生成结果的响应对象
        """
        client = self._get_client()

        request_kwargs = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if "stop" in kwargs:
            request_kwargs["stop"] = kwargs["stop"]

        try:
            response = client.chat_completions(
                model_uid=self.model_uid,
                prompt=prompt,
                **request_kwargs,
            )

            return LLMResponse(
                content=response["content"],
                raw_response=response,
                usage=response.get("usage"),
                model=self.model_name,
                finish_reason=response.get("finish_reason"),
            )
        except Exception as e:
            logger.error(f"Xinference LLM invoke error: {e}")
            raise

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """异步调用 LLM 生成文本

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数

        Returns:
            LLMResponse: 包含生成结果的响应对象
        """
        client = self._get_client()

        request_kwargs = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if "stop" in kwargs:
            request_kwargs["stop"] = kwargs["stop"]

        try:
            response = await client.achat_completions(
                model_uid=self.model_uid,
                prompt=prompt,
                **request_kwargs,
            )

            return LLMResponse(
                content=response["content"],
                raw_response=response,
                usage=response.get("usage"),
                model=self.model_name,
                finish_reason=response.get("finish_reason"),
            )
        except Exception as e:
            logger.error(f"Xinference LLM async invoke error: {e}")
            raise

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """流式调用 LLM（同步版本）

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数

        Yields:
            str: 逐步生成的文本片段
        """
        client = self._get_client()

        request_kwargs = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        try:
            for chunk in client.stream_chat(
                model_uid=self.model_uid,
                prompt=prompt,
                **request_kwargs,
            ):
                if chunk.get("delta"):
                    yield chunk["delta"]
        except Exception as e:
            logger.error(f"Xinference LLM stream error: {e}")
            raise

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """流式调用 LLM（异步版本）

        Args:
            prompt: 输入提示词
            **kwargs: 其他参数

        Yields:
            str: 逐步生成的文本片段
        """
        client = self._get_client()

        request_kwargs = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        try:
            async for chunk in client.astream_chat(
                model_uid=self.model_uid,
                prompt=prompt,
                **request_kwargs,
            ):
                if chunk.get("delta"):
                    yield chunk["delta"]
        except Exception as e:
            logger.error(f"Xinference LLM async stream error: {e}")
            raise

    def get_num_tokens(self, text: str) -> int:
        """计算文本的 token 数量

        使用粗略估算：中文约 1.5 tokens/字符，英文约 0.25 tokens/字符。

        Args:
            text: 输入文本

        Returns:
            int: token 数量
        """
        if not text:
            return 0

        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars

        estimated_tokens = chinese_chars * 1.5 + other_chars * 0.25
        return int(estimated_tokens)

    @classmethod
    def from_settings(cls, **overrides) -> "XinferenceLLM":
        """从配置创建实例

        Args:
            **overrides: 要覆盖的配置参数

        Returns:
            XinferenceLLM: 实例对象
        """
        from ..configs.settings import get_settings

        settings = get_settings()
        xinference_config = settings.llm.xinference

        return cls(
            endpoint=overrides.get("endpoint", xinference_config.endpoint),
            model_uid=overrides.get("model_uid", xinference_config.model_uid),
            temperature=overrides.get("temperature", xinference_config.temperature),
            max_tokens=overrides.get("max_tokens", xinference_config.max_tokens),
            streaming=overrides.get("streaming", xinference_config.streaming),
            timeout=overrides.get("timeout", xinference_config.timeout),
            **overrides,
        )


class XinferenceClient:
    """Xinference HTTP 客户端

    提供与 Xinference 服务通信的能力，支持 chat completions 接口。
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:9997",
        timeout: Optional[float] = 120.0,
        max_retries: int = 3,
    ) -> None:
        """初始化 Xinference 客户端

        Args:
            endpoint: Xinference 服务地址
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _make_request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Any:
        """发送 HTTP 请求

        Args:
            method: HTTP 方法
            path: 请求路径
            json_data: JSON 请求体
            stream: 是否流式响应

        Returns:
            响应内容或迭代器
        """
        

        url = f"{self.endpoint}{path}"
        headers = {"Content-Type": "application/json"}

        for attempt in range(self.max_retries):
            try:
                if stream:
                    response = requests.post(
                        url,
                        json=json_data,
                        headers=headers,
                        timeout=self.timeout,
                        stream=True,
                    )
                    response.raise_for_status()
                    return response.iter_lines()
                else:
                    response = requests.post(
                        url,
                        json=json_data,
                        headers=headers,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Xinference 请求失败 (尝试 {attempt + 1}): {e}")

    async def _make_async_request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """异步发送 HTTP 请求

        Args:
            method: HTTP 方法
            path: 请求路径
            json_data: JSON 请求体

        Returns:
            响应内容
        """
        import aiohttp

        url = f"{self.endpoint}{path}"
        headers = {"Content-Type": "application/json"}

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=json_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        response.raise_for_status()
                        return await response.json()

            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Xinference 异步请求失败 (尝试 {attempt + 1}): {e}")

    def chat_completions(
        self,
        model_uid: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """同步 chat completions 调用

        Args:
            model_uid: 模型 UID
            prompt: 输入提示词
            temperature: 生成温度
            max_tokens: 最大 token 数
            stop: 停止词列表
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 响应结果
        """
        payload: Dict[str, Any] = {
            "model": model_uid,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stop:
            payload["stop"] = stop

        payload.update(kwargs)

        response = self._make_request("POST", "/v1/chat/completions", payload)

        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")

        usage = response.get("usage", {})
        if usage:
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return {
            "content": content,
            "finish_reason": message.get("finish_reason"),
            "usage": usage,
            "raw": response,
        }

    async def achat_completions(
        self,
        model_uid: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """异步 chat completions 调用

        Args:
            model_uid: 模型 UID
            prompt: 输入提示词
            temperature: 生成温度
            max_tokens: 最大 token 数
            stop: 停止词列表
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 响应结果
        """
        payload: Dict[str, Any] = {
            "model": model_uid,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stop:
            payload["stop"] = stop

        payload.update(kwargs)

        response = await self._make_async_request(
            "POST", "/v1/chat/completions", payload
        )

        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")

        usage = response.get("usage", {})
        if usage:
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return {
            "content": content,
            "finish_reason": message.get("finish_reason"),
            "usage": usage,
            "raw": response,
        }

    def stream_chat(
        self,
        model_uid: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """同步流式 chat completions 调用

        Args:
            model_uid: 模型 UID
            prompt: 输入提示词
            temperature: 生成温度
            max_tokens: 最大 token 数
            stop: 停止词列表
            **kwargs: 其他参数

        Yields:
            Dict[str, Any]: 流式响应片段
        """
        payload: Dict[str, Any] = {
            "model": model_uid,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if stop:
            payload["stop"] = stop

        payload.update(kwargs)

        lines = self._make_request("POST", "/v1/chat/completions", payload, stream=True)

        for line in lines:
            if not line:
                continue

            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break

                import json
                try:
                    chunk_data = json.loads(data)
                    delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield {"delta": delta, "raw": chunk_data}
                except json.JSONDecodeError:
                    continue

    async def astream_chat(
        self,
        model_uid: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """异步流式 chat completions 调用

        Args:
            model_uid: 模型 UID
            prompt: 输入提示词
            temperature: 生成温度
            max_tokens: 最大 token 数
            stop: 停止词列表
            **kwargs: 其他参数

        Yields:
            Dict[str, Any]: 流式响应片段
        """
        import aiohttp
        import json

        payload: Dict[str, Any] = {
            "model": model_uid,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if stop:
            payload["stop"] = stop

        payload.update(kwargs)

        url = f"{self.endpoint}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode("utf-8").strip()

                    if not line:
                        continue

                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data)
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                yield {"delta": delta, "raw": chunk_data}
                        except json.JSONDecodeError:
                            continue


class XinferenceEmbeddingLLM:
    """Xinference Embedding 模型封装

    用于在 Xinference 平台上部署的 Embedding 模型调用。
    提供与标准 Embedding 接口兼容的方法。
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:9997",
        model_uid: str = "bge-large-zh",
        batch_size: int = 32,
        dimension: int = 1024,
        timeout: Optional[float] = 60.0,
    ) -> None:
        """初始化 Xinference Embedding

        Args:
            endpoint: Xinference 服务地址
            model_uid: Embedding 模型 UID
            batch_size: 批处理大小
            dimension: 向量维度
            timeout: 请求超时时间
        """
        self.endpoint = endpoint.rstrip("/")
        self.model_uid = model_uid
        self.batch_size = batch_size
        self.dimension = dimension
        self.timeout = timeout

    def embed(self, texts: List[str]) -> List[List[float]]:
        """获取文本的向量表示

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        import requests

        url = f"{self.endpoint}/v1/embeddings"
        headers = {"Content-Type": "application/json"}

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            payload = {
                "model": self.model_uid,
                "input": batch,
            }

            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                embeddings = [item["embedding"] for item in result.get("data", [])]
                all_embeddings.extend(embeddings)

            except Exception as e:
                logger.error(f"Xinference Embedding 请求失败: {e}")
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """获取单个查询的向量表示

        Args:
            text: 查询文本

        Returns:
            List[float]: 向量
        """
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """获取多个文档的向量表示

        Args:
            texts: 文档列表

        Returns:
            List[List[float]]: 向量列表
        """
        return self.embed(texts)
