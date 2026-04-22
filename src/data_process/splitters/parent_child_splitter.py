# -*- coding: utf-8 -*-
"""
父子文档分割器

实现父-子文档分割策略，兼顾精确检索和完整上下文。
"""

import logging
from typing import Any, List, Optional

from langchain_core.documents import Document

from src.data_process.splitters.base import BaseSplitter
from src.data_process.splitters.recursive_splitter import ChineseRecursiveTextSplitter

logger = logging.getLogger(__name__)


class ParentChildSplitter(BaseSplitter):
    """父子文档分割器

    策略：
    - 先将文档按较大块（parent chunk）分割
    - 再将每个大块按较小块（child chunk）分割
    - 检索时使用小块，检索结果关联到大块

    适用于需要精确检索但同时保留完整上下文的场景。
    """

    def __init__(
        self,
        parent_splitter: Optional[BaseSplitter] = None,
        child_splitter: Optional[BaseSplitter] = None,
        parent_chunk_size: int = 1500,
        parent_chunk_overlap: int = 150,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 20,
        **kwargs: Any,
    ) -> None:
        """初始化父子分割器

        Args:
            parent_splitter: 父块分割器（默认 ChineseRecursiveTextSplitter）
            child_splitter: 子块分割器（默认 ChineseRecursiveTextSplitter）
            parent_chunk_size: 父块大小
            parent_chunk_overlap: 父块重叠
            child_chunk_size: 子块大小
            child_chunk_overlap: 子块重叠
            **kwargs: 其他参数
        """
        super().__init__(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            **kwargs
        )

        if parent_splitter is None:
            parent_splitter = ChineseRecursiveTextSplitter(
                chunk_size=parent_chunk_size,
                chunk_overlap=parent_chunk_overlap,
            )

        if child_splitter is None:
            child_splitter = ChineseRecursiveTextSplitter(
                chunk_size=child_chunk_size,
                chunk_overlap=child_chunk_overlap,
            )

        self._parent_splitter = parent_splitter
        self._child_splitter = child_splitter

    def split_text(self, text: str) -> List[str]:
        """分割文本（返回子块）

        Args:
            text: 待分割文本

        Returns:
            List[str]: 子块列表
        """
        # 分割为父块
        parent_chunks = self._parent_splitter.split_text(text)

        # 对每个父块分割为子块
        all_child_chunks = []
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            child_chunks = self._child_splitter.split_text(parent_chunk)
            for child_idx, child_chunk in enumerate(child_chunks):
                all_child_chunks.append(child_chunk)

        return all_child_chunks

    def split_documents(
        self,
        documents: List[Document],
        **kwargs,
    ) -> List[Document]:
        """分割文档，返回子文档列表

        Args:
            documents: 文档列表

        Returns:
            List[Document]: 分割后的子文档列表，包含父文档信息
        """
        child_docs = []

        for doc in documents:
            # 父块分割
            parent_chunks = self._parent_splitter.split_text(doc.page_content)

            for parent_idx, parent_chunk in enumerate(parent_chunks):
                # 子块分割
                child_chunks = self._child_splitter.split_text(parent_chunk)

                for child_idx, child_chunk in enumerate(child_chunks):
                    # 构建子文档元数据
                    child_metadata = dict(doc.metadata)
                    child_metadata.update({
                        "parent_index": parent_idx,      # 记录属于哪个父块
                        "parent_content": parent_chunk,  # 存储父块完整内容
                        "child_index": child_idx,        # 在父块中的位置
                        "source": doc.metadata.get("source", ""),
                    })
                    # 创建子文档
                    child_docs.append(
                        Document(page_content=child_chunk, metadata=child_metadata)
                    )

        return child_docs

    def get_parent_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """从子文档恢复父文档

        Args:
            documents: 子文档列表

        Returns:
            List[Document]: 父文档列表
        """
        parent_map = {}

        for doc in documents:
            source = doc.metadata.get("source", "")
            parent_idx = doc.metadata.get("parent_index", 0)
            parent_content = doc.metadata.get("parent_content", "")

            key = (source, parent_idx)
            if key not in parent_map:
                parent_metadata = {
                    "source": source,
                    "parent_index": parent_idx,
                    "child_count": 0,     # 子块数量
                }
                parent_map[key] = Document(
                    page_content=parent_content,
                    metadata=parent_metadata,
                )
            parent_map[key].metadata["child_count"] += 1

        return list(parent_map.values())


class SimpleParentSplitter(BaseSplitter):
    """简化版父子分割器

    仅分割父块，不进行子块分割。
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_chunk_overlap: int = 150,
        **kwargs: Any,
    ) -> None:
        """初始化

        Args:
            parent_chunk_size: 父块大小
            parent_chunk_overlap: 父块重叠
            **kwargs: 其他参数
        """
        super().__init__(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            **kwargs
        )
        self._splitter = ChineseRecursiveTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )

    def split_text(self, text: str) -> List[str]:
        """分割文本

        Args:
            text: 待分割文本

        Returns:
            List[str]: 文本块列表
        """
        return self._splitter.split_text(text)
