# -*- coding: utf-8 -*-
"""
Chroma 向量存储实现

基于 Chroma 数据库的向量存储，支持持久化和多种距离度量。
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.storage.vectorstore.base import (
    AddResult,
    BaseVectorStore,
    SearchResult,
    VectorStoreConfig,
)

logger = logging.getLogger(__name__)


class ChromaConfig(VectorStoreConfig):
    """Chroma 向量库配置

    Attributes:
        persist_directory: 持久化存储路径
        collection_name: 集合名称
        distance_metric: 距离度量（cosine/euclidean/l2）
        allow_reset: 是否允许重置集合
    """

    def __init__(
        self,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "default",
        distance_metric: str = "cosine",
        allow_reset: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            persist_directory=persist_directory,
            collection_name=collection_name,
            distance_metric=distance_metric,
        )
        self.allow_reset = allow_reset


class ChromaVectorStore(BaseVectorStore):
    """Chroma 向量存储

    基于 Chroma 数据库实现，支持：
    - 持久化存储
    - HNSW 近似最近邻检索
    - 多种距离度量（cosine/euclidean/l2）
    - 元数据过滤
    - 集合管理
    """

    METADATA_KEYS_BLACKLIST = {"allowed_record_count", "total_record_count"}

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        distance_metric: str = "cosine",
        allow_reset: bool = False,
        embedding_function: Optional[Any] = None,
        client: Optional[Any] = None,
        client_settings: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """初始化 Chroma 向量存储

        Args:
            collection_name: 集合名称
            persist_directory: 持久化存储目录
            distance_metric: 距离度量
                - "cosine": 余弦距离
                - "euclidean": 欧氏距离
                - "l2": L2 距离
            allow_reset: 是否允许重置集合
            embedding_function: 嵌入函数（用于自动生成向量）
            client: Chroma 客户端实例
            client_settings: 客户端配置
            **kwargs: 其他参数
        """
        super().__init__(
            collection_name=collection_name,
            persist_directory=persist_directory,
            distance_metric=distance_metric,
        )

        self._embedding_function = embedding_function
        self._client = client
        self._client_settings = client_settings
        self._collection = None
        self._allow_reset = allow_reset

        self._initialize()

    def _initialize(self) -> None:
        """初始化 Chroma 客户端和集合"""
        import chromadb
        from chromadb.config import Settings

        if self._client is not None:
            self._client = self._client
        else:
            persist_path = self.persist_directory or "./data/vector_db"

            if persist_path:
                Path(persist_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=persist_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=self._allow_reset,
                    ),
                )
            else:
                self._client = chromadb.Client()

        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        """获取或创建集合"""
        import chromadb.api

        metadata = None
        if self.distance_metric:
            chroma_distance = self._to_chroma_distance(self.distance_metric)
            metadata = {"hnsw:space": chroma_distance}

        try:
            collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"获取集合失败: {e}")
            self._client.reset()
            collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata,
            )

        return collection

    @staticmethod
    def _to_chroma_distance(distance_metric: str) -> str:
        """转换距离度量到 Chroma 格式"""
        mapping = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "l2": "l2",
            "manhattan": "l1",
            "l1": "l1",
        }
        return mapping.get(distance_metric.lower(), "cosine")

    @property
    def collection(self) -> Any:
        """获取 Chroma 集合对象"""
        return self._collection

    @property
    def count(self) -> int:
        """获取集合中文档数量"""
        return self._collection.count()

    def add_texts(
        self,
        texts: List[str],
        embeddings: Optional[List[np.ndarray]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> AddResult:
        """添加文本及其嵌入向量

        Args:
            texts: 文本列表
            embeddings: 嵌入向量列表
            metadatas: 元数据列表
            ids: 文档 ID 列表
            **kwargs: 其他参数

        Returns:
            AddResult: 添加结果
        """
        try:
            if not texts:
                return AddResult(success=True, count=0)

            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            if embeddings is None:
                if self._embedding_function is None:
                    raise ValueError(
                        "必须提供 embeddings 或 embedding_function"
                    )
                embeddings = self._embedding_function.embed_documents(texts)

            embeddings = [
                emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                for emb in embeddings
            ]

            if metadatas is None:
                metadatas = [{} for _ in texts]

            metadatas = [
                {k: v for k, v in meta.items()
                 if k not in self.METADATA_KEYS_BLACKLIST and v is not None}
                for meta in metadatas
            ]

            self._collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

            logger.info(f"成功添加 {len(texts)} 个文档到集合 {self.collection_name}")

            return AddResult(
                success=True,
                ids=ids,
                count=len(texts),
            )

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return AddResult(
                success=False,
                error=str(e),
            )

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
            score_threshold: 相似度阈值
            **kwargs: 其他参数

        Returns:
            List[SearchResult]: 检索结果列表
        """
        results_with_score = self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            **kwargs,
        )

        search_results = []
        for doc, score in results_with_score:
            if score_threshold is not None:
                if self.distance_metric == "cosine":
                    if score < score_threshold:
                        continue
                else:
                    if score > score_threshold:
                        continue

            search_results.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=self._normalize_score(score),
                distance=score,
                document_id=doc.metadata.get("id"),
            ))

        return search_results

    def similarity_search_with_score(
        self,
        query: Union[str, np.ndarray],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Tuple[Any, float]]:
        """执行相似度检索并返回分数

        Args:
            query: 查询文本或嵌入向量
            k: 返回结果数量
            filter: 元数据过滤条件
            **kwargs: 其他参数

        Returns:
            List[Tuple[Document, float]]: (Document, score) 元组列表
        """
        from langchain_core.documents import Document

        if isinstance(query, str):
            if self._embedding_function is None:
                raise ValueError(
                    "字符串查询需要 embedding_function"
                )
            query_embedding = self._embedding_function.embed_query(query)
        else:
            query_embedding = query.tolist() if isinstance(query, np.ndarray) else list(query)

        where_clause = self._build_where_clause(filter) if filter else None

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

        if not results or not results.get("documents"):
            return []

        documents = []
        for i, doc_text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            metadata["id"] = results["ids"][0][i] if results["ids"] else None

            doc = Document(
                page_content=doc_text,
                metadata=metadata,
            )

            distance = results["distances"][0][i] if results.get("distances") else 0.0
            documents.append((doc, distance))

        return documents

    def _build_where_clause(
        self,
        filter_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建 Chroma 的 where 子句"""
        where = {}
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                where[key] = value
            elif isinstance(value, list):
                where[key] = {"$in": value}
            else:
                where[key] = value
        return where

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
        try:
            if ids:
                self._collection.delete(ids=ids)
            elif filter:
                where = self._build_where_clause(filter)
                results = self._collection.get(where=where, include=[])
                if results and results.get("ids"):
                    self._collection.delete(ids=results["ids"])
            else:
                logger.warning("删除操作需要提供 ids 或 filter")
                return False

            logger.info(f"成功删除文档")
            return True

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

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
        try:
            if embedding is None:
                if self._embedding_function is None:
                    raise ValueError("需要提供 embedding")
                embedding = self._embedding_function.embed_query(text)

            embedding = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)

            if metadata is None:
                metadata = {}

            self._collection.update(
                ids=[id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
            )

            logger.info(f"成功更新文档: {id}")
            return True

        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False

    def upsert(
        self,
        texts: List[str],
        embeddings: Optional[List[np.ndarray]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> AddResult:
        """更新或插入文档

        Args:
            texts: 文本列表
            embeddings: 嵌入向量列表
            metadatas: 元数据列表
            ids: 文档 ID 列表
            **kwargs: 其他参数

        Returns:
            AddResult: 操作结果
        """
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            if embeddings is None:
                if self._embedding_function is None:
                    raise ValueError("需要 embeddings 或 embedding_function")
                embeddings = self._embedding_function.embed_documents(texts)

            embeddings = [
                emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                for emb in embeddings
            ]

            if metadatas is None:
                metadatas = [{} for _ in texts]

            self._collection.upsert(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

            return AddResult(success=True, ids=ids, count=len(texts))

        except Exception as e:
            logger.error(f"Upsert 失败: {e}")
            return AddResult(success=False, error=str(e))

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "collection_name": self.collection_name,
            "count": self.count,
            "distance_metric": self.distance_metric,
            "persist_directory": self.persist_directory,
        }

    def exists(self, collection_name: Optional[str] = None) -> bool:
        """检查集合是否存在

        Args:
            collection_name: 集合名称

        Returns:
            bool: 是否存在
        """
        name = collection_name or self.collection_name
        try:
            self._client.get_collection(name)
            return True
        except Exception:
            return False

    def list_collections(self) -> List[str]:
        """列出所有集合

        Returns:
            List[str]: 集合名称列表
        """
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"列出集合失败: {e}")
            return []

    def persist(self) -> bool:
        """持久化存储

        Chroma 的 PersistentClient 会自动持久化，
        此方法仅作接口兼容。

        Returns:
            bool: 始终返回 True
        """
        logger.debug("Chroma 自动持久化，无需手动调用")
        return True

    def reset(self) -> bool:
        """重置/清空集合

        Returns:
            bool: 是否重置成功
        """
        try:
            if not self._allow_reset:
                logger.warning("allow_reset=False，禁止重置集合")
                return False

            self._client.reset()
            self._collection = self._get_or_create_collection()
            logger.info(f"成功重置集合: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"重置集合失败: {e}")
            return False

    @classmethod
    def from_params(
        cls,
        collection_name: str = "default",
        persist_directory: str = "./data/vector_db",
        distance_metric: str = "cosine",
        **kwargs,
    ) -> "ChromaVectorStore":
        """从参数创建实例

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            distance_metric: 距离度量
            **kwargs: 其他参数

        Returns:
            ChromaVectorStore: 实例
        """
        return cls(
            collection_name=collection_name,
            persist_directory=persist_directory,
            distance_metric=distance_metric,
            **kwargs,
        )

    def __del__(self) -> None:
        """析构时自动持久化"""
        try:
            if self._client is not None:
                pass
        except Exception:
            pass
