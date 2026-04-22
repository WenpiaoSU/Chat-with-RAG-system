# -*- coding: utf-8 -*-
"""
向量存储抽象基类

定义向量存储的标准接口规范。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class SearchResult:
    """检索结果数据结构

    Attributes:
        content: 文档内容
        metadata: 文档元数据
        score: 相似度分数
        distance: 距离度量值
        document_id: 文档 ID
    """
    content: str
    metadata: Dict[str, Any]
    score: float
    distance: Optional[float] = None
    document_id: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """向量存储配置基类

    Attributes:
        persist_directory: 持久化存储目录
        collection_name: 集合名称
        distance_metric: 距离度量方式（cosine/euclidean/manhattan）
    """
    persist_directory: str = "./data/vector_db"
    collection_name: str = "default"
    distance_metric: str = "cosine"


@dataclass
class AddResult:
    """添加文档结果

    Attributes:
        success: 是否成功
        ids: 生成的文档 ID 列表
        count: 添加的文档数量
        error: 错误信息（如果有）
    """
    success: bool
    ids: List[str] = field(default_factory=list)
    count: int = 0
    error: Optional[str] = None


class BaseVectorStore(ABC):
    """向量存储抽象基类

    定义向量存储的标准接口，包括：
    - 添加文档（文本 + 向量）
    - 相似度检索
    - 文档管理（删除、更新）
    - 集合管理
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine",
        **kwargs,
    ) -> None:
        """初始化向量存储

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            distance_metric: 距离度量方式
            **kwargs: 其他配置参数
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric
        self._extra_params = kwargs

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        embeddings: Optional[List[np.ndarray]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> AddResult:
        """添加文本及其嵌入向量到存储

        Args:
            texts: 文本列表
            embeddings: 对应的嵌入向量列表
            metadatas: 元数据列表
            ids: 指定的文档 ID 列表
            **kwargs: 其他参数

        Returns:
            AddResult: 添加结果
        """
        ...

    @abstractmethod
    def similarity_search(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """执行相似度检索

        Args:
            query: 查询文本或嵌入向量
            k: 返回结果数量
            filter: 元数据过滤条件
            score_threshold: 相似度阈值（0-1）
            **kwargs: 其他参数

        Returns:
            List[SearchResult]: 检索结果列表
        """
        ...

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Tuple[Any, float]]:
        """执行相似度检索并返回原始分数

        Args:
            query: 查询文本或嵌入向量
            k: 返回结果数量
            filter: 元数据过滤条件
            **kwargs: 其他参数

        Returns:
            List[Tuple[Any, float]]: (Document, score) 元组列表
        """
        ...

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bool:
        """删除文档

        Args:
            ids: 文档 ID 列表
            filter: 元数据过滤条件
            **kwargs: 其他参数

        Returns:
            bool: 是否删除成功
        """
        ...

    @abstractmethod
    def update_document(
        self,
        id: str,
        text: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bool:
        """更新文档

        Args:
            id: 文档 ID
            text: 新文本内容
            embedding: 新嵌入向量
            metadata: 新元数据
            **kwargs: 其他参数

        Returns:
            bool: 是否更新成功
        """
        ...

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        ...

    @abstractmethod
    def exists(self, collection_name: Optional[str] = None) -> bool:
        """检查集合是否存在

        Args:
            collection_name: 集合名称，默认为当前实例的集合

        Returns:
            bool: 是否存在
        """
        ...

    @abstractmethod
    def persist(self) -> bool:
        """持久化存储

        Returns:
            bool: 是否持久化成功
        """
        ...

    @abstractmethod
    def reset(self) -> bool:
        """重置/清空集合

        Returns:
            bool: 是否重置成功
        """
        ...

    def _normalize_score(self, distance: float) -> float:
        """将距离转换为相似度分数

        Args:
            distance: 距离值

        Returns:
            float: 相似度分数（0-1）
        """
        if self.distance_metric == "cosine":
            return 1.0 - distance
        elif self.distance_metric == "euclidean":
            return 1.0 / (1.0 + distance)
        else:
            return 1.0 / (1.0 + distance)

    def _compute_distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """计算两个向量的距离

        Args:
            embedding1: 第一个向量
            embedding2: 第二个向量

        Returns:
            float: 距离值
        """
        if self.distance_metric == "cosine":
            return 1.0 - np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
            )
        elif self.distance_metric == "euclidean":
            return float(np.linalg.norm(embedding1 - embedding2))
        else:
            return float(np.sum(np.abs(embedding1 - embedding2)))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"collection={self.collection_name}, "
            f"persist_dir={self.persist_directory})"
        )
