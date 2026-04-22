# -*- coding: utf-8 -*-
"""
Embedding 模块

提供文本嵌入功能，支持多种 Embedding 模型（BGE 等）。
"""

from .base import BaseEmbedding, EmbeddingResult
from .bge_embedding import BGEEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingResult",
    "BGEEmbedding",
]
