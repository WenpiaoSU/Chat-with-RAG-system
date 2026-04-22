# -*- coding: utf-8 -*-
"""
混合检索模块

结合语义检索与关键词检索，通过 RRF（Reciprocal Rank Fusion）算法
进行多策略融合，实现更全面、更准确的检索结果。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from .base import BaseRetriever, RetrievalResult, RetrievalResults
from .semantic_search import SemanticSearch
from .keyword_search import BM25Search

logger = logging.getLogger(__name__)


class RRFusion:
    """RRF 倒数排名融合算法

    Reciprocal Rank Fusion (RRF) 是一种简单而有效的多策略结果融合方法，
    通过考虑各文档在不同检索策略中的排名来计算最终得分。

    Algorithm:
        RRF_score(d) = Σ(weight_i / (k + rank_i(d)))

    其中：
        - weight_i: 策略 i 的权重
        - rank_i(d): 文档 d 在策略 i 中的排名
        - k: 平滑参数（通常为 60）

    特点：
        - 无需训练，简单高效
        - 对排名差异敏感，对分数差异不敏感
        - 可以灵活组合多个检索策略
    """

    def __init__(
        self,
        k: int = 60,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """初始化 RRF 融合器

        Args:
            k: RRF 平滑参数（通常 60），值越大，排名靠后的文档影响越大
            weights: 各检索策略的权重字典，如 {"semantic": 0.7, "bm25": 0.3}
        """
        self.k = k
        self.weights = weights or {}

    def fuse(
        self,
        results_list: List[RetrievalResults],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """执行 RRF 融合

        Args:
            results_list: 各检索策略的结果列表
            top_k: 返回的 top-k 结果数

        Returns:
            List[RetrievalResult]: 融合后的排序结果
        """
        if not results_list:
            return []

        score_map: Dict[str, float] = {}
        rank_map: Dict[str, int] = {}
        doc_info: Dict[str, RetrievalResult] = {}

        for results in results_list:
            if results is None or not results.results:
                continue

            strategy = results.metadata.get("strategy", "unknown")
            weight = self.weights.get(strategy, 1.0)

            for result in results.results:
                doc_key = self._get_doc_key(result)

                if doc_key not in rank_map:
                    rank_map[doc_key] = 0

                rank_map[doc_key] += 1

                if result.score_raw is not None:
                    position = result.rank
                else:
                    position = rank_map[doc_key]

                rrf_score = weight / (self.k + position)

                score_map[doc_key] = score_map.get(doc_key, 0.0) + rrf_score
                doc_info[doc_key] = result

        fused_results: List[RetrievalResult] = []
        seen_contents: set = set()

        for doc_key, rrf_score in score_map.items():
            result = doc_info[doc_key]
            content = result.content

            if content in seen_contents:
                continue
            seen_contents.add(content)

            fused_result = RetrievalResult(
                content=result.content,
                metadata=result.metadata.copy(),
                score=rrf_score,
                score_raw=rrf_score,
                source=result.source,
                document_id=result.document_id,
            )
            fused_results.append(fused_result)

        fused_results.sort(key=lambda x: x.score, reverse=True)

        for i, result in enumerate(fused_results):
            result.rank = i + 1

        if top_k is not None:
            fused_results = fused_results[:top_k]

        return fused_results

    def _get_doc_key(self, result: RetrievalResult) -> str:
        """生成文档唯一标识键

        Args:
            result: 检索结果

        Returns:
            str: 文档唯一键
        """
        if result.document_id:
            return f"doc_{result.document_id}"

        return f"content_{hash(result.content)}"

    def fuse_two(
        self,
        results_a: RetrievalResults,
        results_b: RetrievalResults,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """融合两个检索结果集

        Args:
            results_a: 第一个检索结果集
            results_b: 第二个检索结果集
            weight_a: 第一个结果集的权重
            weight_b: 第二个结果集的权重
            top_k: 返回结果数

        Returns:
            List[RetrievalResult]: 融合后的结果
        """
        self.weights["default_a"] = weight_a
        self.weights["default_b"] = weight_b

        results_a.metadata["strategy"] = "default_a"
        results_b.metadata["strategy"] = "default_b"

        return self.fuse([results_a, results_b], top_k=top_k)


class HybridSearch(BaseRetriever):
    """混合检索器

    结合语义检索（HNSW）和关键词检索（BM25）两种策略，使用 RRF 算法进行结果融合，并支持可选的重排序步骤。
    """

    def __init__(
        self,
        vectorstore: Any = None,
        embedder: Optional[Any] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        enable_semantic: bool = True,
        enable_keyword: bool = True,
        enable_rerank: bool = False,
        reranker: Optional[Any] = None,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        **kwargs: Any,
    ) -> None:
        """初始化混合检索器

        Args:
            vectorstore: 向量存储实例
            embedder: 嵌入模型实例
            top_k: 默认返回的 top-k 结果数
            score_threshold: 相似度分数阈值
            semantic_weight: 语义检索权重
            keyword_weight: 关键词检索权重
            enable_semantic: 是否启用语义检索
            enable_keyword: 是否启用关键词检索
            enable_rerank: 是否启用重排序
            reranker: 重排序器实例
            rrf_k: RRF 平滑参数
            bm25_k1: BM25 k1 参数
            bm25_b: BM25 b 参数
            **kwargs: 其他配置参数
        """
        super().__init__(
            vectorstore=vectorstore,
            embedder=embedder,
            top_k=top_k,
            score_threshold=score_threshold,
            **kwargs,
        )

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.enable_semantic = enable_semantic
        self.enable_keyword = enable_keyword
        self.enable_rerank = enable_rerank
        self.reranker = reranker
        self.rrf_k = rrf_k

        self._semantic_retriever: Optional[SemanticSearch] = None
        self._bm25_retriever: Optional[BM25Search] = None
        self._fusion = RRFusion(k=rrf_k)

        self._semantic_top_k = max(top_k * 2, 10)
        self._bm25_top_k = max(top_k * 2, 10)

        if self.enable_semantic and vectorstore is not None:
            self._semantic_retriever = SemanticSearch(
                vectorstore=vectorstore,
                embedder=embedder,
                top_k=self._semantic_top_k,
                score_threshold=score_threshold,
            )

        if self.enable_keyword:
            self._bm25_retriever = BM25Search(
                top_k=self._bm25_top_k,
                score_threshold=score_threshold,
                k1=bm25_k1,
                b=bm25_b,
            )

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """添加文档到检索器

        同时添加到语义检索和关键词检索。

        Args:
            texts: 文档文本列表
            metadatas: 元数据列表
            ids: 文档 ID 列表
            **kwargs: 其他参数

        Returns:
            bool: 是否添加成功
        """
        super().add_documents(texts, metadatas, ids, **kwargs)

        success = True

        if self._semantic_retriever is not None:
            result = self._semantic_retriever.add_documents(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                **kwargs,
            )
            if not result:
                success = False

        if self._bm25_retriever is not None:
            self._bm25_retriever.index_corpus(
                texts=self._documents,
                metadatas=self._metadata,
                ids=self._document_ids,
            )

        return success

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        enable_semantic: Optional[bool] = None,
        enable_keyword: Optional[bool] = None,
        enable_rerank: Optional[bool] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """执行混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter: 元数据过滤条件
            score_threshold: 相似度分数阈值
            enable_semantic: 是否启用语义检索（覆盖默认值）
            enable_keyword: 是否启用关键词检索（覆盖默认值）
            enable_rerank: 是否启用重排序（覆盖默认值）
            **kwargs: 其他参数

        Returns:
            RetrievalResults: 检索结果集
        """
        start_time = time.time()

        k = top_k or self.top_k
        use_semantic = enable_semantic if enable_semantic is not None else self.enable_semantic
        use_keyword = enable_keyword if enable_keyword is not None else self.enable_keyword
        use_rerank = enable_rerank if enable_rerank is not None else self.enable_rerank

        results_list: List[RetrievalResults] = []
        semantic_results: Optional[RetrievalResults] = None
        keyword_results: Optional[RetrievalResults] = None

        if use_semantic and self._semantic_retriever is not None:
            try:
                semantic_results = self._semantic_retriever.retrieve(
                    query=query,
                    top_k=self._semantic_top_k,
                    filter=filter,
                    score_threshold=score_threshold,
                    **kwargs,
                )
                if semantic_results.results:
                    results_list.append(semantic_results)
            except Exception as e:
                logger.error(f"语义检索失败: {e}")

        if use_keyword and self._bm25_retriever is not None:
            try:
                if self._bm25_retriever.indexed_doc_count > 0:
                    keyword_results = self._bm25_retriever.retrieve(
                        query=query,
                        top_k=self._bm25_top_k,
                        score_threshold=score_threshold,
                        **kwargs,
                    )
                    if keyword_results.results:
                        results_list.append(keyword_results)
            except Exception as e:
                logger.error(f"关键词检索失败: {e}")

        if not results_list:
            return RetrievalResults(
                query=query,
                metadata={
                    "strategy": "hybrid",
                    "error": "no retrieval strategy available",
                    "semantic_enabled": use_semantic,
                    "keyword_enabled": use_keyword,
                },
            )

        self._fusion.weights = {
            "semantic": self.semantic_weight,
            "bm25": self.keyword_weight,
        }

        if len(results_list) == 1:
            fused_results = results_list[0].results
        else:
            fused_results = self._fusion.fuse(results_list, top_k=k * 2)

        if use_rerank and self.reranker is not None:
            try:
                fused_results = self.reranker.rerank(
                    query=query,
                    results=fused_results,
                    top_k=k,
                )
            except Exception as e:
                logger.error(f"重排序失败: {e}")

        fused_results = self._normalize_results(fused_results, top_k=k)

        query_time = (time.time() - start_time) * 1000

        return RetrievalResults(
            results=fused_results,
            query=query,
            total_count=len(fused_results),
            query_time_ms=query_time,
            metadata={
                "strategy": "hybrid",
                "semantic_weight": self.semantic_weight,
                "keyword_weight": self.keyword_weight,
                "semantic_enabled": use_semantic,
                "keyword_enabled": use_keyword,
                "rerank_enabled": use_rerank,
                "top_k": k,
                "score_threshold": score_threshold,
                "rrf_k": self.rrf_k,
            },
            semantic_results=semantic_results,
            keyword_results=keyword_results,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """便捷方法：执行混合搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            **kwargs: 其他参数

        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        results = self.retrieve(query=query, top_k=k, **kwargs)
        return results.results

    def semantic_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """仅执行语义检索

        Args:
            query: 查询文本
            top_k: 返回结果数
            **kwargs: 其他参数

        Returns:
            RetrievalResults: 语义检索结果
        """
        if self._semantic_retriever is None:
            return RetrievalResults(
                query=query,
                metadata={"strategy": "semantic_only", "error": "semantic retriever not configured"},
            )

        return self._semantic_retriever.retrieve(query=query, top_k=top_k or self.top_k, **kwargs)

    def keyword_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """仅执行关键词检索

        Args:
            query: 查询文本
            top_k: 返回结果数
            **kwargs: 其他参数

        Returns:
            RetrievalResults: 关键词检索结果
        """
        if self._bm25_retriever is None:
            return RetrievalResults(
                query=query,
                metadata={"strategy": "keyword_only", "error": "bm25 retriever not configured"},
            )

        return self._bm25_retriever.retrieve(query=query, top_k=top_k or self.top_k, **kwargs)

    def set_weights(
        self,
        semantic_weight: float,
        keyword_weight: float,
    ) -> None:
        """动态调整检索权重

        Args:
            semantic_weight: 语义检索权重
            keyword_weight: 关键词检索权重
        """
        total = semantic_weight + keyword_weight
        if total > 0:
            self.semantic_weight = semantic_weight / total
            self.keyword_weight = keyword_weight / total
        else:
            self.semantic_weight = 0.5
            self.keyword_weight = 0.5

        logger.info(
            f"调整检索权重: semantic={self.semantic_weight:.2f}, "
            f"keyword={self.keyword_weight:.2f}"
        )

    def enable_reranker(self, reranker: Any) -> None:
        """启用重排序器

        Args:
            reranker: 重排序器实例
        """
        self.reranker = reranker
        self.enable_rerank = True
        logger.info("已启用重排序器")

    def disable_reranker(self) -> None:
        """禁用重排序器"""
        self.reranker = None
        self.enable_rerank = False
        logger.info("已禁用重排序器")

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        stats = super().get_stats()

        stats.update({
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
            "semantic_enabled": self.enable_semantic,
            "keyword_enabled": self.enable_keyword,
            "rerank_enabled": self.enable_rerank,
            "rrf_k": self.rrf_k,
        })

        if self._bm25_retriever:
            stats["bm25_indexed_docs"] = self._bm25_retriever.indexed_doc_count

        return stats

    def __repr__(self) -> str:
        return (
            f"HybridSearch("
            f"semantic_weight={self.semantic_weight}, "
            f"keyword_weight={self.keyword_weight}, "
            f"rerank={self.enable_rerank}, "
            f"top_k={self.top_k})"
        )
