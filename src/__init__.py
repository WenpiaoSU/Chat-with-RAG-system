# -*- coding: utf-8 -*-
"""
Chat-with-RAG-system

基于 LangChain 的本地知识库 RAG 智能问答系统。
"""

__version__ = "0.1.0"

from src.configs.settings import settings, get_settings

# Data Process
from src.data_process import (
    DataPipeline,
    build_pipeline,
    LoaderFactory,
    ChineseRecursiveTextSplitter,
    ChineseTextCleaner,
)

# Storage
from src.storage import (
    ChromaVectorStore,
    ChromaConfig,
    KnowledgeBaseManager,
    KnowledgeBaseFactory,
)

# Retrieval
from src.retrieval import (
    BaseRetriever,
    BaseReranker,
    RetrievalResult,
    RetrievalResults,
    RewriteResult,
    RewriteStrategy,
    SemanticSearch,
    BM25Search,
    HybridSearch,
    RRFusion,
    QueryRewriter,
    BGEReranker,
    SimpleReranker,
)

__all__ = [
    "settings",
    "get_settings",
    # Data Process
    "DataPipeline",
    "build_pipeline",
    "LoaderFactory",
    "ChineseRecursiveTextSplitter",
    "ChineseTextCleaner",
    # Storage
    "ChromaVectorStore",
    "ChromaConfig",
    "KnowledgeBaseManager",
    "KnowledgeBaseFactory",
    # Retrieval
    "BaseRetriever",
    "BaseReranker",
    "RetrievalResult",
    "RetrievalResults",
    "RewriteResult",
    "RewriteStrategy",
    "SemanticSearch",
    "BM25Search",
    "HybridSearch",
    "RRFusion",
    "QueryRewriter",
    "BGEReranker",
    "SimpleReranker",
]
