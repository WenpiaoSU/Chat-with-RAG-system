# -*- coding: utf-8 -*-
"""
重排序模块

基于 BGE-Reranker 模型的交叉注意力重排序实现。
通过精细的 query-document 交互打分，提高检索结果的排序质量。
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

# 设置模型缓存目录
os.environ.setdefault("MODEL_CACHE_DIR", "/hot_disk_1T/data/swp/huggingface")

import numpy as np

from .base import BaseReranker, RetrievalResult

logger = logging.getLogger(__name__)


class BGEReranker(BaseReranker):
    """BGE-Reranker 重排序器

    基于 BAAI/BGE-Reranker 模型实现的交叉注意力重排序。
    该模型通过将查询和文档拼接，利用交叉注意力机制精细评估相关性。

    Model Architecture:
        Input: [CLS] + query + [SEP] + passage + [SEP]
        Process: Cross-attention between query and passage tokens
        Output: 0~1 的相关性分数

    Features:
        - 交叉注意力机制
        - 精细的相关性打分
        - 支持批量处理
        - 自动设备选择（CUDA/CPU）
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: Optional[str] = None,
        top_k: int = 5,
        max_length: int = 512,
        batch_size: int = 8,
        cache_folder: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 BGE-Reranker

        Args:
            model_name: HuggingFace 模型名称或本地路径
            device: 运行设备，默认为 cuda（如果可用）
            top_k: 返回的 top-k 结果数
            max_length: 最大序列长度
            batch_size: 批处理大小
            cache_folder: 模型缓存目录
            **kwargs: 其他参数
        """
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        super().__init__(
            model_name=model_name,
            device=device,
            top_k=top_k,
            max_length=max_length,
            **kwargs,
        )

        self.batch_size = batch_size
        self.cache_folder = cache_folder or self._get_default_cache_dir()

        self._tokenizer = None
        self._model = None
        self._is_loaded = False

    @staticmethod
    def _get_default_cache_dir() -> str:
        """获取默认的模型缓存目录

        优先级：环境变量 MODEL_CACHE_DIR > 环境变量 HF_HOME > 默认路径

        Returns:
            str: 缓存目录路径
        """
        env_cache = os.environ.get("MODEL_CACHE_DIR")
        if env_cache:
            return env_cache

        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return hf_home

        return os.path.expanduser("~/.cache/huggingface")

    def load_model(self) -> Any:
        """加载 BGE-Reranker 模型和分词器

        Returns:
            Any: 模型实例
        """
        if self._model is not None:
            return self._model

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "请安装 transformers 和 torch: pip install transformers torch"
            )

        logger.info(f"正在加载 BGE-Reranker 模型: {self.model_name}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_folder,
            )
        except Exception:
            logger.warning(
                f"无法从 HuggingFace 加载模型 {self.model_name}，"
                "可能需要登录或网络连接。"
            )
            raise

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_folder,
        ).to(self.device)

        self._model.eval()

        self._is_loaded = True

        logger.info(
            f"BGE-Reranker 模型加载完成: {self.model_name}, "
            f"device={self.device}"
        )

        return self._model

    def _ensure_model_loaded(self) -> None:
        """确保模型已加载"""
        if not self._is_loaded:
            self.load_model()

    def rerank(
        self,
        query: str,
        results: Union[List[RetrievalResult], List[Any]],
        top_k: Optional[int] = None,
        return_scores: bool = False,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """对检索结果进行重排序

        Args:
            query: 查询文本
            results: 初步检索结果列表
            top_k: 返回的结果数量
            return_scores: 是否返回原始分数
            **kwargs: 其他参数

        Returns:
            List[RetrievalResult]: 重排序后的结果列表
        """
        self._ensure_model_loaded()

        if not results:
            return []

        k = top_k or self.top_k

        retrieval_results = self._ensure_retrieval_results(results)

        query_doc_pairs = [
            (query, result.content) for result in retrieval_results
        ]

        scores = self._compute_scores(query_doc_pairs)

        for i, result in enumerate(retrieval_results):
            result.score = scores[i]
            result.score_raw = scores[i]

        retrieval_results.sort(key=lambda x: x.score, reverse=True)

        for i, result in enumerate(retrieval_results):
            result.rank = i + 1

        reranked_results = retrieval_results[:k]

        logger.debug(f"BGE-Reranker 完成重排序: {len(results)} -> {len(reranked_results)}")

        return reranked_results

    def _compute_scores(self, query_doc_pairs: List[tuple]) -> List[float]:
        """计算 query-document 对的相关性分数

        Args:
            query_doc_pairs: (query, document) 元组列表

        Returns:
            List[float]: 相关性分数列表
        """
        self._ensure_model_loaded()

        all_scores = []

        for i in range(0, len(query_doc_pairs), self.batch_size):
            batch_pairs = query_doc_pairs[i : i + self.batch_size]
            batch_scores = self._score_batch(batch_pairs)
            all_scores.extend(batch_scores)

        return all_scores

    def _score_batch(self, query_doc_pairs: List[tuple]) -> List[float]:
        """批量计算分数

        Args:
            query_doc_pairs: 批量 (query, document) 元组列表

        Returns:
            List[float]: 批量相关性分数
        """
        import torch
        from transformers import AutoTokenizer

        queries = [pair[0] for pair in query_doc_pairs]
        documents = [pair[1] for pair in query_doc_pairs]

        inputs = self._tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

            if logits.shape[-1] == 1:
                scores = torch.sigmoid(logits).squeeze(-1)
            else:
                scores = torch.softmax(logits, dim=-1)[:, 1]

        return scores.cpu().tolist()

    def score(
        self,
        query: str,
        documents: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """计算查询与文档的相关性分数

        Args:
            query: 查询文本
            documents: 文档列表
            **kwargs: 其他参数

        Returns:
            List[float]: 相关性分数列表
        """
        self._ensure_model_loaded()

        pairs = [(query, doc) for doc in documents]
        return self._compute_scores(pairs)

    async def arerank(
        self,
        query: str,
        results: Union[List[RetrievalResult], List[Any]],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """异步重排序

        Args:
            query: 查询文本
            results: 初步检索结果列表
            top_k: 返回的结果数量
            **kwargs: 其他参数

        Returns:
            List[RetrievalResult]: 重排序后的结果列表
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.rerank, query, results, top_k
        )

    @classmethod
    def from_settings(cls, **overrides) -> "BGEReranker":
        """从配置创建实例

        Args:
            **overrides: 要覆盖的配置参数

        Returns:
            BGEReranker: 实例对象
        """
        from ..configs.settings import get_settings

        settings = get_settings()
        reranker_config = settings.reranker

        return cls(
            model_name=overrides.get("model_name", reranker_config.model_name),
            device=overrides.get("device", reranker_config.device),
            top_k=overrides.get("top_k", reranker_config.top_k),
            max_length=overrides.get("max_length", reranker_config.max_length),
            **overrides,
        )

    def __repr__(self) -> str:
        return (
            f"BGEReranker("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"top_k={self.top_k})"
        )


class SimpleReranker(BaseReranker):
    """简单的基于分数的重排序器

    当没有 BGE-Reranker 模型时，作为轻量级的备选方案。
    根据已有分数进行简单排序。
    """

    def __init__(
        self,
        top_k: int = 5,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化简单重排序器

        Args:
            top_k: 返回的 top-k 结果数
            normalize: 是否归一化分数
            **kwargs: 其他参数
        """
        super().__init__(
            model_name="simple",
            device="cpu",
            top_k=top_k,
            **kwargs,
        )
        self.normalize = normalize

    def load_model(self) -> None:
        """无需加载模型"""
        pass

    def rerank(
        self,
        query: str,
        results: Union[List[RetrievalResult], List[Any]],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """基于已有分数排序

        Args:
            query: 查询文本（未使用）
            results: 初步检索结果列表
            top_k: 返回的结果数量
            **kwargs: 其他参数

        Returns:
            List[RetrievalResult]: 排序后的结果列表
        """
        k = top_k or self.top_k

        retrieval_results = self._ensure_retrieval_results(results)

        if not retrieval_results:
            return []

        if self.normalize:
            max_score = max(r.score for r in retrieval_results)
            min_score = min(r.score for r in retrieval_results)
            score_range = max_score - min_score

            if score_range > 0:
                for r in retrieval_results:
                    r.score = (r.score - min_score) / score_range

        retrieval_results.sort(key=lambda x: x.score, reverse=True)

        for i, result in enumerate(retrieval_results):
            result.rank = i + 1

        return retrieval_results[:k]
