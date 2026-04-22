# -*- coding: utf-8 -*-
"""
语义检索模块

基于向量数据库（Chroma HNSW 索引）的语义相似度检索实现。
支持余弦相似度、欧氏距离等多种距离度量。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import BaseRetriever, RetrievalResult, RetrievalResults

logger = logging.getLogger(__name__)


class SemanticSearch(BaseRetriever):
    """基于 HNSW 向量检索的语义检索器

    利用 Chroma 等向量数据库的 HNSW（Hierarchical Navigable Small World）索引，实现高效的近似最近邻搜索。

    Features:
        - HNSW 近似最近邻检索
        - 多种距离度量（cosine/euclidean/l2）
        - 元数据过滤
        - 可配置的 top-k 和分数阈值
        - 自动 embedding 生成
    """

    def __init__(
        self,
        vectorstore: Any = None,
        embedder: Optional[Any] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        distance_metric: str = "cosine",
        search_type: str = "similarity",
        **kwargs: Any,
    ) -> None:
        """初始化语义检索器

        Args:
            vectorstore: 向量存储实例（如 ChromaVectorStore）
            embedder: 嵌入模型实例
            top_k: 默认返回的 top-k 结果数
            score_threshold: 相似度分数阈值（0-1）
            distance_metric: 距离度量方式
            search_type: 搜索类型（similarity/mmr）
            **kwargs: 其他配置参数
        """
        super().__init__(
            vectorstore=vectorstore,
            embedder=embedder,
            top_k=top_k,
            score_threshold=score_threshold,
            **kwargs,
        )

        self.distance_metric = distance_metric
        self.search_type = search_type

        if vectorstore is not None:
            self.distance_metric = getattr(
                vectorstore, "distance_metric", distance_metric
            )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """执行语义相似度检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter: 元数据过滤条件
            score_threshold: 相似度分数阈值
            **kwargs: 其他参数

        Returns:
            RetrievalResults: 检索结果集
        """
        start_time = time.time()

        k = top_k or self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        if self.vectorstore is None:
            logger.warning("未配置向量存储，执行空检索")
            return RetrievalResults(
                query=query,
                metadata={"strategy": "semantic", "top_k": k},
            )

        try:
            # 将用户输入的文本转换为向量表示
            if self.embedder is not None and hasattr(self.embedder, "encode_query"):
                query_embedding = self.embedder.encode_query(query)
            elif self.embedder is not None:
                query_embedding = self.embedder.encode([query])[0]
            else:
                query_embedding = query
            # 调用向量数据库检索
            search_results = self.vectorstore.similarity_search_with_score(
                query=query_embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
            # 结果去重
            retrieval_results: List[RetrievalResult] = []
            seen_contents: set = set()

            for doc, raw_score in search_results:
                content = getattr(doc, "page_content", None) or str(doc)
                metadata = dict(getattr(doc, "metadata", {}) or {})

                if content in seen_contents:
                    continue
                seen_contents.add(content)
                # 分数归一化
                normalized_score = self._normalize_score(raw_score)
                # 过滤掉低于阈值的低质量结果
                if threshold is not None and normalized_score < threshold:
                    continue

                result = RetrievalResult(
                    content=content,
                    metadata=metadata,
                    score=normalized_score,
                    score_raw=raw_score,
                    source=metadata.get("source", "semantic"),
                    document_id=metadata.get("id") or metadata.get("doc_id"),
                )
                retrieval_results.append(result)

            retrieval_results = self._normalize_results(retrieval_results, top_k=k)

            query_time = (time.time() - start_time) * 1000

            return RetrievalResults(
                results=retrieval_results,
                query=query,
                total_count=len(retrieval_results),
                query_time_ms=query_time,
                metadata={
                    "strategy": "semantic",
                    "distance_metric": self.distance_metric,
                    "top_k": k,
                    "score_threshold": threshold,
                },
            )

        except Exception as e:
            logger.error(f"语义检索失败: {e}")
            return RetrievalResults(
                query=query,
                metadata={"strategy": "semantic", "error": str(e)},
            )

    def _normalize_score(self, raw_score: float) -> float:
        """将原始距离分数归一化为相似度分数

        Args:
            raw_score: 原始分数（可能是距离值）

        Returns:
            float: 归一化的相似度分数（0-1）
        """
        if self.distance_metric == "cosine":
            return 1.0 - raw_score if raw_score <= 1.0 else raw_score
        elif self.distance_metric in ("euclidean", "l2"):
            return 1.0 / (1.0 + raw_score)
        else:
            return 1.0 / (1.0 + raw_score)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """便捷方法：执行相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤
            **kwargs: 其他参数

        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        results = self.retrieve(query=query, top_k=k, filter=filter, **kwargs)
        return results.results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievalResults]:
        """批量检索

        Args:
            queries: 查询文本列表
            top_k: 返回结果数量
            filter: 元数据过滤条件
            **kwargs: 其他参数

        Returns:
            List[RetrievalResults]: 每个查询的检索结果列表
        """
        return [
            self.retrieve(query=q, top_k=top_k, filter=filter, **kwargs)
            for q in queries
        ]

    def get_all_documents(
        self,
        limit: int = 1000,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """获取集合中的所有文档

        Args:
            limit: 返回数量限制
            filter: 元数据过滤条件

        Returns:
            List[RetrievalResult]: 所有文档的检索结果
        """
        if self.vectorstore is None:
            return []

        try:
            if hasattr(self.vectorstore, "_collection"):
                collection = self.vectorstore._collection
                results = collection.get(
                    limit=limit,
                    where=self._build_where_clause(filter) if filter else None,
                    include=["documents", "metadatas", "embeddings"],
                )

                retrieval_results = []
                for i, doc_text in enumerate(results.get("documents", [])):
                    metadata = results["metadatas"][i] if results.get("metadatas") else {}
                    retrieval_results.append(
                        RetrievalResult(
                            content=doc_text,
                            metadata=metadata,
                            score=1.0,
                            source=metadata.get("source", "unknown"),
                            document_id=results["ids"][i] if results.get("ids") else None,
                        )
                    )
                return retrieval_results
        except Exception as e:
            logger.error(f"获取文档失败: {e}")

        return []

    def _build_where_clause(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """构建 Chroma 的 where 子句

        Args:
            filter_dict: 过滤条件字典

        Returns:
            Dict[str, Any]: Chroma where 子句
        """
        where = {}
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                where[key] = value
            elif isinstance(value, list):
                where[key] = {"$in": value}
            else:
                where[key] = value
        return where

    def count_documents(self) -> int:
        """获取集合中的文档数量

        Returns:
            int: 文档数量
        """
        if self.vectorstore is None:
            return 0

        if hasattr(self.vectorstore, "count"):
            return self.vectorstore.count
        return 0

    def __repr__(self) -> str:
        return (
            f"SemanticSearch("
            f"top_k={self.top_k}, "
            f"distance_metric={self.distance_metric}, "
            f"has_vectorstore={self.vectorstore is not None})"
        )
