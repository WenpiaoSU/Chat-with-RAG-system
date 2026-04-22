# -*- coding: utf-8 -*-
"""
文本分割器基类

定义所有文本分割器的统一接口。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.text_splitter import TextSplitter


class BaseSplitter(ABC, TextSplitter):
    """文本分割器抽象基类

    继承自 LangChain 的 TextSplitter，提供统一的分割接口。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs,
    ) -> None:
        """初始化分割器

        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠大小
            **kwargs: 其他参数
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """分割单个文本

        Args:
            text: 待分割文本

        Returns:
            List[str]: 分割后的文本块列表
        """
        ...

    def split_documents(
        self,
        documents: List[Document],
        **kwargs,
    ) -> List[Document]:
        """分割文档列表

        Args:
            documents: 文档列表
            **kwargs: 其他参数

        Returns:
            List[Document]: 分割后的文档列表
        """
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        split_texts = []
        for text in texts:
            chunks = self.split_text(text)
            split_texts.extend(chunks)

        return [
            Document(page_content=chunk, metadata=metadata)
            for chunk, metadata in zip(split_texts, metadatas)
        ]
