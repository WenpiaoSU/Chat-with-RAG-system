# -*- coding: utf-8 -*-
"""
检索器基类和通用数据结构

定义检索结果、检索器基类和重排序器基类的标准接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np


@dataclass
class RetrievalResult:
    """单条检索结果

    Attributes:
        content: 文档内容文本
        metadata: 文档元数据（如来源、页码等）
        score: 相似度分数（归一化到 0-1）
        rank: 在结果集中的排名（从 1 开始）
        source: 文档来源标识
        document_id: 文档唯一标识
        score_raw: 原始分数（未经归一化）
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0
    source: str = "unknown"
    document_id: Optional[str] = None
    score_raw: Optional[float] = None

    def __post_init__(self) -> None:
        """后处理：确保元数据包含基本字段"""
        if "source" not in self.metadata:
            self.metadata["source"] = self.source
        if self.document_id and "id" not in self.metadata:
            self.metadata["id"] = self.document_id

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            Dict[str, Any]: 结果字典
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
            "document_id": self.document_id,
        }


@dataclass
class RetrievalResults:
    """检索结果集合

    包含检索操作的完整结果，包括查询、耗时、统计信息等。

    Attributes:
        results: 检索结果列表
        query: 原始查询文本
        total_count: 匹配到的总文档数
        query_time_ms: 检索耗时（毫秒）
        metadata: 附加元数据（如使用的检索策略、权重等）
        semantic_results: 语义检索结果（混合检索时使用）
        keyword_results: 关键词检索结果（混合检索时使用）
    """
    results: List[RetrievalResult] = field(default_factory=list)
    query: str = ""
    total_count: int = 0
    query_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_results: Optional[List[RetrievalResult]] = None
    keyword_results: Optional[List[RetrievalResult]] = None

    def __post_init(self) -> None:
        """后处理：更新总数和排名"""
        self.total_count = len(self.results)
        for i, result in enumerate(self.results):
            if result.rank == 0:
                result.rank = i + 1

    def __len__(self) -> int:
        """返回结果数量"""
        return len(self.results)

    def __iter__(self):
        """支持迭代"""
        return iter(self.results)

    def __getitem__(self, index: int) -> RetrievalResult:
        """支持索引访问"""
        return self.results[index]

    def top_k(self, k: int) -> List[RetrievalResult]:
        """获取 top-k 结果

        Args:
            k: 返回的结果数量

        Returns:
            List[RetrievalResult]: top-k 结果列表
        """
        return self.results[:k]

    def filter(
        self,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        source: Optional[str] = None,
    ) -> "RetrievalResults":
        """过滤检索结果

        Args:
            min_score: 最低分数阈值
            max_score: 最高分数阈值
            source: 按来源过滤

        Returns:
            RetrievalResults: 过滤后的结果集
        """
        filtered = []
        for r in self.results:
            if min_score is not None and r.score < min_score:
                continue
            if max_score is not None and r.score > max_score:
                continue
            if source is not None and r.source != source:
                continue
            filtered.append(r)

        return RetrievalResults(
            results=filtered,
            query=self.query,
            metadata={**self.metadata, "filtered": True},
        )

    def get_context(self, separator: str = "\n\n") -> str:
        """获取合并的上下文文本

        Args:
            separator: 结果之间的分隔符

        Returns:
            str: 合并后的上下文字符串
        """
        return separator.join(r.content for r in self.results)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            Dict[str, Any]: 结果字典
        """
        return {
            "query": self.query,
            "total_count": self.total_count,
            "query_time_ms": self.query_time_ms,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }


class RewriteStrategy(Enum):
    """查询改写策略枚举"""
    EXPANSION = "expansion"          # 查询扩展：添加同义词、相关词
    DECOMPOSITION = "decomposition"  # 查询分解：将复杂问题拆分为简单子问题
    HYPOTHESIS = "hypothesis"       # 假设生成：生成假设性答案并用于检索
    BACKOFF = "backoff"             # 降级策略：简化查询以提高召回率
    SPECIFIC = "specific"           # 具体化：使查询更具体以提高精度


@dataclass
class RewriteResult:
    """查询改写结果

    Attributes:
        original_query: 原始查询文本
        rewritten_queries: 改写后的查询列表
        strategy: 使用的改写策略
        metadata: 附加元数据
    """
    original_query: str
    rewritten_queries: List[str] = field(default_factory=list)
    strategy: RewriteStrategy = RewriteStrategy.EXPANSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rewritten_queries)

    def __iter__(self):
        return iter(self.rewritten_queries)


class EmbedderProtocol(Protocol):
    """嵌入函数协议

    定义 Embedder 需要实现的接口，供检索器使用。
    """

    def embed_query(self, query: str) -> np.ndarray:
        """编码查询文本

        Args:
            query: 查询文本

        Returns:
            np.ndarray: 查询向量
        """
        ...

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """批量编码文档

        Args:
            texts: 文档列表

        Returns:
            List[np.ndarray]: 文档向量列表
        """
        ...

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码文本

        Args:
            texts: 文本或文本列表
            **kwargs: 其他参数

        Returns:
            np.ndarray: 嵌入向量
        """
        ...


class BaseRetriever(ABC):
    """检索器抽象基类

    定义检索器的基本接口，所有具体检索实现都应继承此类。
    支持语义检索、关键词检索和混合检索等多种策略。
    """

    def __init__(
        self,
        vectorstore: Any = None,
        embedder: Optional[Any] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """初始化检索器

        Args:
            vectorstore: 向量存储实例（如 ChromaVectorStore）
            embedder: 嵌入模型实例
            top_k: 默认返回的 top-k 结果数
            score_threshold: 相似度分数阈值
            **kwargs: 其他配置参数
        """
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._extra_params = kwargs
        self._documents: List[str] = []
        self._document_ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """执行检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter: 元数据过滤条件
            **kwargs: 其他参数

        Returns:
            RetrievalResults: 检索结果集
        """
        pass

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """添加文档到检索器

        Args:
            texts: 文档文本列表
            metadatas: 元数据列表
            ids: 文档 ID 列表
            **kwargs: 其他参数

        Returns:
            bool: 是否添加成功
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        self._documents.extend(texts)
        self._metadata.extend(metadatas)

        if ids:
            self._document_ids.extend(ids)
        else:
            import uuid
            self._document_ids.extend([str(uuid.uuid4()) for _ in texts])

        if self.vectorstore is not None:
            result = self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                **kwargs,
            )
            return result.success

        return True

    def _build_retrieval_result(
        self,
        doc: Any,
        score: float,
        rank: int,
        source: str = "unknown",
    ) -> RetrievalResult:
        """构建检索结果对象

        Args:
            doc: 文档对象或内容
            score: 相似度分数
            rank: 排名
            source: 来源标识

        Returns:
            RetrievalResult: 检索结果对象
        """
        content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
        metadata = getattr(doc, "metadata", None) or {}

        if isinstance(metadata, dict) and "source" not in metadata:
            metadata["source"] = source

        doc_id = metadata.get("id") or metadata.get("doc_id")

        return RetrievalResult(
            content=content,
            metadata=metadata,
            score=score,
            rank=rank,
            source=source,
            document_id=doc_id,
            score_raw=score,
        )

    def _normalize_results(
        self,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """归一化和排序结果

        Args:
            results: 原始结果列表
            top_k: 截取数量

        Returns:
            List[RetrievalResult]: 归一化后的结果
        """
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        for i, r in enumerate(sorted_results):
            r.rank = i + 1

        if top_k is not None:
            sorted_results = sorted_results[:top_k]

        return sorted_results

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return {
            "total_documents": len(self._documents),
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "has_vectorstore": self.vectorstore is not None,
            "has_embedder": self.embedder is not None,
        }


class BaseReranker(ABC):
    """重排序器抽象基类

    定义文档重排序的标准接口，基于交叉注意力等机制对初步检索结果进行精细排序。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cpu",
        top_k: int = 5,
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        """初始化重排序器

        Args:
            model_name: 模型名称
            device: 运行设备
            top_k: 返回的 top-k 结果数
            max_length: 最大序列长度
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.max_length = max_length
        self._extra_params = kwargs
        self._model = None

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: Union[List[RetrievalResult], List[Any]],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """对检索结果进行重排序

        Args:
            query: 查询文本
            results: 初步检索结果列表
            top_k: 返回的结果数量
            **kwargs: 其他参数

        Returns:
            List[RetrievalResult]: 重排序后的结果列表
        """
        pass

    def _ensure_retrieval_results(
        self,
        results: Union[List[RetrievalResult], List[Any]],
    ) -> List[RetrievalResult]:
        """确保结果是 RetrievalResult 类型

        Args:
            results: 输入结果列表

        Returns:
            List[RetrievalResult]: RetrievalResult 列表
        """
        converted = []
        for i, r in enumerate(results):
            if isinstance(r, RetrievalResult):
                converted.append(r)
            elif hasattr(r, "page_content"):
                converted.append(
                    RetrievalResult(
                        content=r.page_content,
                        metadata=getattr(r, "metadata", {}),
                        score=getattr(r, "score", 0.0),
                        rank=i + 1,
                    )
                )
            elif isinstance(r, dict):
                converted.append(
                    RetrievalResult(
                        content=r.get("content", r.get("page_content", "")),
                        metadata=r.get("metadata", {}),
                        score=r.get("score", 0.0),
                        rank=i + 1,
                    )
                )
            else:
                converted.append(
                    RetrievalResult(
                        content=str(r),
                        rank=i + 1,
                    )
                )
        return converted

    @abstractmethod
    def load_model(self) -> Any:
        """加载重排序模型

        Returns:
            Any: 模型实例
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """获取重排序器统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "top_k": self.top_k,
            "max_length": self.max_length,
            "model_loaded": self._model is not None,
        }
