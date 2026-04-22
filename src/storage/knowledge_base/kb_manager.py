# -*- coding: utf-8 -*-
"""
知识库管理器

提供知识库的完整管理功能，包括文档添加、删除、检索和更新。
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.storage.vectorstore.base import AddResult, SearchResult
from src.storage.vectorstore.chroma_store import ChromaVectorStore, ChromaConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """文档记录

    Attributes:
        id: 文档唯一标识
        content: 文档内容
        metadata: 元数据
        created_at: 创建时间
        updated_at: 更新时间
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeBaseStats:
    """知识库统计信息

    Attributes:
        collection_name: 集合名称
        document_count: 文档数量
        total_chunks: 块数量
        last_updated: 最后更新时间
        sources: 文档来源列表
    """
    collection_name: str
    document_count: int
    total_chunks: int
    last_updated: Optional[datetime] = None
    sources: List[str] = field(default_factory=list)


class KnowledgeBaseManager:
    """知识库管理器

    提供知识库的完整管理功能：
    - 文档的增删改查
    - 向量化存储
    - 集合管理
    - 统计信息
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str = "./data/vector_db",
        distance_metric: str = "cosine",
        embedder: Optional[Any] = None,
        enable_parent_child: bool = False,
        **kwargs,
    ) -> None:
        """初始化知识库管理器

        Args:
            collection_name: 集合名称
            persist_directory: 持久化存储目录
            distance_metric: 距离度量
            embedder: 嵌入模型实例
            enable_parent_child: 是否启用父子文档策略
            **kwargs: 其他参数
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric
        self._embedder = embedder
        self.enable_parent_child = enable_parent_child

        self._vectorstore = self._create_vectorstore()

        self._document_registry: Dict[str, DocumentRecord] = {}

    def _create_vectorstore(self) -> ChromaVectorStore:
        """创建向量存储实例"""
        return ChromaVectorStore(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            distance_metric=self.distance_metric,
            embedding_function=self._embedder,
        )

    @property
    def vectorstore(self) -> ChromaVectorStore:
        """获取向量存储实例"""
        return self._vectorstore

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        source: Optional[str] = None,
        **kwargs,
    ) -> AddResult:
        """添加文档到知识库

        Args:
            texts: 文本列表
            metadatas: 元数据列表
            ids: 文档 ID 列表
            source: 文档来源
            **kwargs: 其他参数

        Returns:
            AddResult: 添加结果
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        for i, meta in enumerate(metadatas):
            meta["source"] = source or meta.get("source", "unknown")
            meta["added_at"] = datetime.now().isoformat()

            if ids and i < len(ids):
                meta["doc_id"] = ids[i]
            else:
                meta["doc_id"] = str(uuid.uuid4())

        result = self._vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

        if result.success:
            for i, (text, meta) in enumerate(zip(texts, metadatas)):
                doc_id = result.ids[i] if i < len(result.ids) else str(uuid.uuid4())
                self._document_registry[doc_id] = DocumentRecord(
                    id=doc_id,
                    content=text,
                    metadata=meta,
                )

        return result

    def add_documents_from_files(
        self,
        file_paths: List[str],
        show_progress: bool = True,
        **kwargs,
    ) -> AddResult:
        """从文件添加文档

        使用 data_process 模块的加载器和分割器处理文件。

        Args:
            file_paths: 文件路径列表
            show_progress: 是否显示进度
            **kwargs: 其他参数

        Returns:
            AddResult: 添加结果
        """
        from src.data_process import build_pipeline, LoaderFactory

        pipeline = build_pipeline(**kwargs)
        all_texts = []
        all_metadatas = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(file_paths, desc="处理文件")
            except ImportError:
                iterator = file_paths
        else:
            iterator = file_paths

        for file_path in iterator:
            try:
                docs = pipeline.process(file_path=file_path)

                for doc in docs:
                    all_texts.append(doc.page_content)
                    meta = dict(doc.metadata)
                    meta["source"] = str(file_path)
                    all_metadatas.append(meta)

            except Exception as e:
                logger.error(f"处理文件失败: {file_path}, error: {e}")
                continue

        if not all_texts:
            return AddResult(success=False, error="没有成功处理任何文档")

        return self.add_documents(
            texts=all_texts,
            metadatas=all_metadatas,
        )

    def search(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """检索文档

        Args:
            query: 查询文本或向量
            k: 返回结果数量
            filter: 元数据过滤条件
            score_threshold: 相似度阈值
            **kwargs: 其他参数

        Returns:
            List[SearchResult]: 检索结果列表
        """
        return self._vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
            score_threshold=score_threshold,
            **kwargs,
        )

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
        success = self._vectorstore.delete(ids=ids, filter=filter, **kwargs)

        if success and ids:
            for doc_id in ids:
                self._document_registry.pop(doc_id, None)

        return success

    def delete_by_source(self, source: str) -> int:
        """根据来源删除文档

        Args:
            source: 文档来源

        Returns:
            int: 删除的文档数量
        """
        results = self._vectorstore.similarity_search_with_score(
            query="dummy",
            k=self._vectorstore.count,
        )

        ids_to_delete = []
        for doc, _ in results:
            if doc.metadata.get("source") == source:
                ids_to_delete.append(doc.metadata.get("id"))

        if ids_to_delete:
            self.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def update_document(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bool:
        """更新文档

        Args:
            id: 文档 ID
            text: 新文本内容
            metadata: 新元数据
            **kwargs: 其他参数

        Returns:
            bool: 是否更新成功
        """
        if metadata is None:
            metadata = {}

        metadata["updated_at"] = datetime.now().isoformat()

        success = self._vectorstore.update_document(
            id=id,
            text=text,
            metadata=metadata,
            **kwargs,
        )

        if success and id in self._document_registry:
            self._document_registry[id].content = text
            self._document_registry[id].metadata = metadata
            self._document_registry[id].updated_at = datetime.now()

        return success

    def get_stats(self) -> KnowledgeBaseStats:
        """获取知识库统计信息

        Returns:
            KnowledgeBaseStats: 统计信息
        """
        stats = self._vectorstore.get_collection_stats()
        sources = set()

        if self._vectorstore.count > 0:
            try:
                results = self._vectorstore._collection.get(limit=1000, include=["metadatas"])
                for meta in results.get("metadatas", []):
                    if meta and "source" in meta:
                        sources.add(meta["source"])
            except Exception as e:
                logger.warning(f"获取来源列表失败: {e}")

        return KnowledgeBaseStats(
            collection_name=self.collection_name,
            document_count=len(self._document_registry),
            total_chunks=stats.get("count", 0),
            sources=list(sources),
        )

    def exists(self) -> bool:
        """检查知识库是否存在

        Returns:
            bool: 是否存在
        """
        return self._vectorstore.exists()

    def list_collections(self) -> List[str]:
        """列出所有集合

        Returns:
            List[str]: 集合名称列表
        """
        return self._vectorstore.list_collections()

    def reset(self, confirm: bool = False) -> bool:
        """重置知识库

        Args:
            confirm: 必须设置为 True 才能执行

        Returns:
            bool: 是否重置成功
        """
        if not confirm:
            logger.warning("重置操作需要 confirm=True")
            return False

        success = self._vectorstore.reset()
        if success:
            self._document_registry.clear()

        return success

    def persist(self) -> bool:
        """持久化知识库

        Returns:
            bool: 是否持久化成功
        """
        return self._vectorstore.persist()


class KnowledgeBaseFactory:
    """知识库工厂

    根据配置创建知识库管理器实例。
    """

    @staticmethod
    def create(
        collection_name: str = "default",
        embedder: Optional[Any] = None,
        **kwargs,
    ) -> KnowledgeBaseManager:
        """创建知识库管理器

        Args:
            collection_name: 集合名称
            embedder: 嵌入模型
            **kwargs: 其他参数

        Returns:
            KnowledgeBaseManager: 知识库管理器实例
        """
        return KnowledgeBaseManager(
            collection_name=collection_name,
            embedder=embedder,
            **kwargs,
        )

    @staticmethod
    def create_from_config(
        config: ChromaConfig,
        embedder: Optional[Any] = None,
    ) -> KnowledgeBaseManager:
        """从配置创建知识库管理器

        Args:
            config: Chroma 配置
            embedder: 嵌入模型

        Returns:
            KnowledgeBaseManager: 知识库管理器实例
        """
        return KnowledgeBaseManager(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            distance_metric=config.distance_metric,
            embedder=embedder,
        )
