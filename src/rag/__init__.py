# -*- coding: utf-8 -*-
"""
RAG 模块

提供检索增强生成（RAG）功能，包括向量存储、检索链、提示词管理等。
"""

from .prompts import (
    PromptManager,
    PromptTemplates,
    get_default_prompt_manager,
)
from .rag_chain import (
    ChromaVectorStore,
    RAGChain,
    RAGResponse,
    RetrievalResult,
    VectorStore,
)

__all__ = [
    "PromptManager",
    "PromptTemplates",
    "get_default_prompt_manager",
    "ChromaVectorStore",
    "RAGChain",
    "RAGResponse",
    "RetrievalResult",
    "VectorStore",
]
