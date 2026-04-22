# -*- coding: utf-8 -*-
"""
检索模块

提供多种检索策略的实现：
- 语义检索：基于向量相似度的语义匹配
- 关键词检索：基于 BM25 算法的关键词匹配
- 混合检索：语义检索 + 关键词检索 + RRF 融合
- 查询改写：扩展、分解、假设生成等策略
- 重排序：基于交叉注意力的精细打分排序

模块结构：
- base: 检索器基类和通用数据结构
- semantic_search: 基于 HNSW 的语义检索
- keyword_search: 基于 BM25 的关键词检索
- hybrid_search: 混合检索与 RRF 融合
- query_rewriter: 查询改写模块
- reranker: BGE-Reranker 重排序
"""

from .base import (
    BaseRetriever,
    BaseReranker,
    RetrievalResult,
    RetrievalResults,
    RewriteResult,
    RewriteStrategy,
)
from .semantic_search import SemanticSearch
from .keyword_search import BM25Search
from .hybrid_search import HybridSearch, RRFusion
from .query_rewriter import QueryRewriter
from .reranker import BGEReranker, SimpleReranker

__all__ = [
    # 基类与数据结构
    "BaseRetriever",
    "BaseReranker",
    "RetrievalResult",
    "RetrievalResults",
    "RewriteResult",
    "RewriteStrategy",
    # 检索器实现
    "SemanticSearch",
    "BM25Search",
    "HybridSearch",
    "RRFusion",
    # 查询改写与重排序
    "QueryRewriter",
    "BGEReranker",
    "SimpleReranker",
]
