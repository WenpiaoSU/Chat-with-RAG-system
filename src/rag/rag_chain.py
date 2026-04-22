# -*- coding: utf-8 -*-
"""
RAG Chain 实现

提供检索增强生成的核心逻辑，包括检索、问答、对话等。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..embedding.base import BaseEmbedding
from ..llm.base import BaseLLM, LLMResponse
from ..llm.llm_factory import LLMFactory
from .prompts import PromptManager, get_default_prompt_manager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果
    
    Attributes:
        content: 文档内容
        metadata: 文档元数据
        score: 相似度分数
        index: 文档索引
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    index: int = 0


@dataclass
class RAGResponse:
    """RAG 响应
    
    Attributes:
        answer: 生成的回答
        sources: 使用的源文档
        question: 原始问题
        retrieved_docs: 检索到的文档列表
        metadata: 额外元数据
    """
    answer: str
    sources: List[str]
    question: str
    retrieved_docs: List[RetrievalResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """向量存储接口
    
    定义向量存储的标准接口，支持 Chroma、FAISS 等。
    """
    
    def __init__(
        self,
        embedding_model: BaseEmbedding,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
    ) -> None:
        """初始化向量存储
        
        Args:
            embedding_model: 嵌入模型
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加文档到向量存储
        
        Args:
            texts: 文档文本列表
            metadatas: 元数据列表
            ids: 文档ID列表
            
        Returns:
            List[str]: 文档ID列表
        """
        raise NotImplementedError()
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """相似度检索
        
        Args:
            query: 查询文本
            k: 返回文档数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        raise NotImplementedError()
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[RetrievalResult, float]]:
        """带分数的相似度检索
        
        Args:
            query: 查询文本
            k: 返回文档数量
            
        Returns:
            List[Tuple[RetrievalResult, float]]: 检索结果和分数列表
        """
        raise NotImplementedError()
    
    def delete(self, ids: List[str]) -> None:
        """删除文档
        
        Args:
            ids: 文档ID列表
        """
        raise NotImplementedError()
    
    def count(self) -> int:
        """获取文档数量
        
        Returns:
            int: 文档数量
        """
        raise NotImplementedError()


class ChromaVectorStore(VectorStore):
    """Chroma 向量存储实现"""
    
    def __init__(
        self,
        embedding_model: BaseEmbedding,
        collection_name: str = "default",
        persist_directory: str = "./data/vector_db",
    ) -> None:
        """初始化 Chroma 向量存储
        
        Args:
            embedding_model: 嵌入模型
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        super().__init__(
            embedding_model=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        self._collection = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """初始化 Chroma 客户端"""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")
        
        client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        
        self._client = client
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加文档到向量存储"""
        if not texts:
            return []
        
        embeddings = self.embedding_model.encode(texts)
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        self._collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """相似度检索"""
        query_embedding = self.embedding_model.encode_query(query)
        
        where_clause = filter_metadata if filter_metadata else None
        
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
        
        retrieval_results = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0] if results["metadatas"] else [{}],
                results["distances"][0] if results["distances"] else [0.0],
            )):
                retrieval_results.append(RetrievalResult(
                    content=doc,
                    metadata=metadata or {},
                    score=1.0 - distance,
                    index=i,
                ))
        
        return retrieval_results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[RetrievalResult, float]]:
        """带分数的相似度检索"""
        results = self.similarity_search(query, k)
        return [(r, r.score) for r in results]
    
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        self._collection.delete(ids=ids)
    
    def count(self) -> int:
        """获取文档数量"""
        return self._collection.count()


class RAGChain:
    """RAG Chain 核心类
    
    整合检索和生成模块，实现检索增强生成。
    
    Attributes:
        vectorstore: 向量存储
        embedding_model: 嵌入模型
        llm: 大语言模型
        prompt_manager: 提示词管理器
        
    Example:
        >>> rag_chain = RAGChain(
        ...     vectorstore=vectorstore,
        ...     embedding_model=embedder,
        ...     llm=llm
        ... )
        >>> response = rag_chain.invoke("什么是大模型？")
        >>> print(response.answer)
    """
    
    def __init__(
        self,
        vectorstore: VectorStore,
        embedding_model: BaseEmbedding,
        llm: BaseLLM,
        prompt_manager: Optional[PromptManager] = None,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> None:
        """初始化 RAG Chain
        
        Args:
            vectorstore: 向量存储实例
            embedding_model: 嵌入模型
            llm: 大语言模型
            prompt_manager: 提示词管理器
            top_k: 检索文档数量
            score_threshold: 分数阈值
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.llm = llm
        self.prompt_manager = prompt_manager or get_default_prompt_manager()
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """检索相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        results = self.vectorstore.similarity_search(
            query=query,
            k=self.top_k,
        )
        
        filtered_results = [
            r for r in results if r.score >= self.score_threshold
        ]
        
        return filtered_results
    
    def _format_context(self, results: List[RetrievalResult]) -> str:
        """格式化检索结果为上下文
        
        Args:
            results: 检索结果列表
            
        Returns:
            str: 格式化的上下文字符串
        """
        if not results:
            return "没有找到相关的文档信息。"
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", "未知来源")
            page = result.metadata.get("page", "")
            
            header = f"[文档 {i}] 来源: {source}"
            if page:
                header += f", 页码: {page}"
            
            context_parts.append(f"{header}\n{result.content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_answer(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """生成回答
        
        Args:
            query: 查询文本
            context: 上下文
            history: 对话历史
            
        Returns:
            str: 生成的回答
        """
        messages = self.prompt_manager.build_qa_prompt(
            context=context,
            question=query,
            history=history,
        )
        
        if hasattr(self.llm, "invoke_with_messages"):
            response = self.llm.invoke_with_messages(messages)
            return response.content
        else:
            prompt_text = messages[-1].content
            response = self.llm.invoke(prompt_text)
            return response.content
    
    def invoke(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> RAGResponse:
        """执行 RAG 查询
        
        Args:
            query: 用户查询
            history: 对话历史
            
        Returns:
            RAGResponse: RAG 响应
        """
        retrieved_docs = self.retrieve(query)
        
        context = self._format_context(retrieved_docs)
        
        answer = self._generate_answer(query, context, history)
        
        sources = [
            result.metadata.get("source", "未知")
            for result in retrieved_docs
        ]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            question=query,
            retrieved_docs=retrieved_docs,
            metadata={
                "num_retrieved": len(retrieved_docs),
                "retrieval_scores": [r.score for r in retrieved_docs],
            },
        )
    
    def stream(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """流式执行 RAG 查询
        
        Args:
            query: 用户查询
            history: 对话历史
            
        Yields:
            str: 逐步生成的回答片段
        """
        retrieved_docs = self.retrieve(query)
        context = self._format_context(retrieved_docs)
        
        messages = self.prompt_manager.build_qa_prompt(
            context=context,
            question=query,
            history=history,
        )
        
        if hasattr(self.llm, "invoke_with_messages") and hasattr(self.llm, "stream"):
            for chunk in self.llm.stream(messages[-1].content):
                yield chunk
        elif hasattr(self.llm, "stream"):
            for chunk in self.llm.stream(messages[-1].content):
                yield chunk
        else:
            response = self._generate_answer(query, context, history)
            yield response
    
    async def ainvoke(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> RAGResponse:
        """异步执行 RAG 查询
        
        Args:
            query: 用户查询
            history: 对话历史
            
        Returns:
            RAGResponse: RAG 响应
        """
        import asyncio
        
        retrieved_docs = await asyncio.to_thread(self.retrieve, query)
        
        context = self._format_context(retrieved_docs)
        
        messages = self.prompt_manager.build_qa_prompt(
            context=context,
            question=query,
            history=history,
        )
        
        if hasattr(self.llm, "invoke_with_messages"):
            response = await self.llm.ainvoke(
                messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
            )
            answer = response.content
        else:
            answer = await asyncio.to_thread(
                self._generate_answer, query, context, history
            )
        
        sources = [
            result.metadata.get("source", "未知")
            for result in retrieved_docs
        ]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            question=query,
            retrieved_docs=retrieved_docs,
            metadata={
                "num_retrieved": len(retrieved_docs),
                "retrieval_scores": [r.score for r in retrieved_docs],
            },
        )
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """添加文档到向量存储
        
        Args:
            texts: 文档文本列表
            metadatas: 元数据列表
            
        Returns:
            List[str]: 文档ID列表
        """
        return self.vectorstore.add_documents(texts, metadatas)
    
    @classmethod
    def from_defaults(
        cls,
        embedding_model: Optional[BaseEmbedding] = None,
        llm: Optional[BaseLLM] = None,
        vectorstore: Optional[VectorStore] = None,
        **kwargs,
    ) -> "RAGChain":
        """从配置创建 RAG Chain
        
        自动加载默认组件。
        
        Args:
            embedding_model: 嵌入模型（可选）
            llm: 大语言模型（可选）
            vectorstore: 向量存储（可选）
            **kwargs: 其他参数
            
        Returns:
            RAGChain: RAG Chain 实例
        """
        from ..configs.settings import get_settings
        
        settings = get_settings()
        
        if embedding_model is None:
            from ..embedding import BGEEmbedding
            embedding_model = BGEEmbedding.from_settings()
        
        if llm is None:
            from ..llm.llm_factory import LLMFactory
            llm = LLMFactory.create_from_settings()
        
        if vectorstore is None:
            vectorstore = ChromaVectorStore(
                embedding_model=embedding_model,
                collection_name=settings.vectorstore.chroma.collection_name,
                persist_directory=settings.vectorstore.chroma.persist_directory,
            )
        
        return cls(
            vectorstore=vectorstore,
            embedding_model=embedding_model,
            llm=llm,
            top_k=settings.retrieval.top_k,
            score_threshold=settings.retrieval.score_threshold,
            **kwargs,
        )
