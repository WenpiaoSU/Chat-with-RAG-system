# -*- coding: utf-8 -*-
"""
Markdown 标题结构分割器

基于 Markdown 标题层级进行语义分割。
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.text_splitter import MarkdownTextSplitter

from src.data_process.splitters.base import BaseSplitter

logger = logging.getLogger(__name__)


class MarkdownHeaderSplitter(BaseSplitter):
    """Markdown 标题分割器

    按 Markdown 标题层级（# ## ### 等）进行文档分割，
    保留标题作为分割后每个块的前缀。
    """

    def __init__(
        self,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        return_each_line: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化 Markdown 标题分割器

        Args:
            headers_to_split_on: 需要作为分割点的标题列表，
                格式为 [(separator, pattern), ...]
                如 [("#", "标题"), ("##", "小标题")]
            return_each_line: 是否返回每一行
            **kwargs: 其他参数（如 chunk_size, chunk_overlap）
        """
        super().__init__(**kwargs)

        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "标题 1"),
                ("##", "标题 2"),
                ("###", "标题 3"),
                ("####", "标题 4"),
                ("#####", "标题 5"),
                ("######", "标题 6"),
            ]

        self.headers_to_split_on = headers_to_split_on
        self.return_each_line = return_each_line

        self._header_pattern = "|".join(
            f"(?P<{pattern}>{re.escape(sep)}.+)"
            for sep, pattern in headers_to_split_on
        )
        self._split_headers = [
            sep for sep, _ in headers_to_split_on
        ]

    def split_text(self, text: str) -> List[str]:
        """按标题分割 Markdown 文本

        Args:
            text: Markdown 文本

        Returns:
            List[str]: 分割后的文本块列表
        """
        if self.return_each_line:
            return self._split_by_lines(text)
        return self._split_by_headers(text)

    def _split_by_headers(self, text: str) -> List[str]:
        """按标题层级分割"""
        import re

        header_pattern = "^(" + "|".join(
            re.escape(h) for h in self._split_headers
        ) + r")\s+(.+)$"

        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_headers = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_chunk.append(line)
                continue

            match = re.match(header_pattern, stripped)
            if match:
                header_level = len(match.group(1))
                header_text = match.group(2).strip()
                current_headers = current_headers[: header_level - 1]
                current_headers.append(header_text)

            if current_chunk and self._should_split_chunk(
                "\n".join(current_chunk), current_headers
            ):
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []

            current_chunk.append(line)

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _split_by_lines(self, text: str) -> List[str]:
        """按行分割（简单实现）"""
        return [line for line in text.split("\n") if line.strip()]

    def _should_split_chunk(self, chunk: str, headers: List[str]) -> bool:
        """判断是否需要分割当前块"""
        if not headers:
            return len(chunk) >= self._chunk_size
        return len(chunk) >= self._chunk_size * 2

    def split_documents(
        self,
        documents: List[Document],
        **kwargs,
    ) -> List[Document]:
        """分割文档列表，保留元数据"""
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata = dict(doc.metadata)
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(chunks)
                split_docs.append(Document(page_content=chunk, metadata=metadata))
        return split_docs


class MarkdownHeaderTextSplitter(BaseSplitter):
    """基于标题级别的 Markdown 分割器

    使用 LangChain 内置的 MarkdownTextSplitter，
    并增加中文支持。
    """

    def __init__(
        self,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
        strip_headers: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化

        Args:
            headers_to_split_on: 分割标题列表
            strip_headers: 是否移除标题
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self._splitter = MarkdownTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=strip_headers,
        )

    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        return self._splitter.split_text(text)

    def split_documents(
        self,
        documents: List[Document],
        **kwargs,
    ) -> List[Document]:
        """分割文档"""
        return self._splitter.split_documents(documents)
