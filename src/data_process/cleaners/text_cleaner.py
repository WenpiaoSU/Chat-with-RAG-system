# -*- coding: utf-8 -*-
"""
文本清洗工具

提供文档文本的规范化、清理、去重等处理功能。
"""

import logging
import re
import unicodedata
from typing import Any, Callable, List, Optional, Set

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TextCleaner:
    """文本清洗工具类

    提供多种文本清洗方法，支持链式调用和自定义处理函数。
    """

    DEFAULT_SPECIAL_CHARS = r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f\ufeff]"

    DEFAULT_WHITESPACE_CHARS = r"[ \t]+"

    def __init__(
        self,
        remove_extra_whitespace: bool = True,
        remove_special_chars: bool = False,
        strip_html: bool = True,
        normalize_unicode: bool = True,
        lowercase: bool = False,
        special_chars_pattern: Optional[str] = None,
        custom_cleaners: Optional[List[Callable[[str], str]]] = None,
    ) -> None:
        """初始化文本清洗器

        Args:
            remove_extra_whitespace: 去除多余空白（多个空格合并为一个）
            remove_special_chars: 去除特殊控制字符
            strip_html: 去除 HTML 标签
            normalize_unicode: 标准化 Unicode（NFKC）
            lowercase: 转为小写
            special_chars_pattern: 自定义特殊字符正则模式
            custom_cleaners: 自定义清洗函数列表
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_special_chars = remove_special_chars
        self.strip_html = strip_html
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase
        self.special_chars_pattern = special_chars_pattern or self.DEFAULT_SPECIAL_CHARS
        self.custom_cleaners = custom_cleaners or []

    def clean(self, text: str) -> str:
        """执行所有配置的清洗操作

        Args:
            text: 原始文本

        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""

        cleaned = text

        if self.normalize_unicode:
            cleaned = self._normalize_unicode(cleaned)

        if self.strip_html:
            cleaned = self._strip_html(cleaned)

        if self.remove_special_chars:
            cleaned = self._remove_special_chars(cleaned)

        if self.remove_extra_whitespace:
            cleaned = self._remove_extra_whitespace(cleaned)

        if self.lowercase:
            cleaned = cleaned.lower()

        for custom_cleaner in self.custom_cleaners:
            cleaned = custom_cleaner(cleaned)

        return cleaned

    def _normalize_unicode(self, text: str) -> str:
        """Unicode NFKC 标准化"""
        return unicodedata.normalize("NFKC", text)

    def _strip_html(self, text: str) -> str:
        """去除 HTML 标签"""
        try:
            import bleach
            return bleach.clean(text, tags=[], strip=True)
        except ImportError:
            html_pattern = re.compile(r"<[^>]+>")
            return html_pattern.sub(" ", text)

    def _remove_special_chars(self, text: str) -> str:
        """去除控制字符和特殊字符"""
        pattern = re.compile(self.special_chars_pattern)
        return pattern.sub(" ", text)

    def _remove_extra_whitespace(self, text: str) -> str:
        """去除多余空白"""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean_document(self, doc: Document) -> Document:
        """清洗单个文档

        Args:
            doc: 原始文档

        Returns:
            Document: 清洗后的文档
        """
        cleaned_content = self.clean(doc.page_content)
        return Document(page_content=cleaned_content, metadata=dict(doc.metadata))

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """批量清洗文档

        Args:
            documents: 文档列表

        Returns:
            List[Document]: 清洗后的文档列表
        """
        return [self.clean_document(doc) for doc in documents]


class ChineseTextCleaner(TextCleaner):
    """中文文本专用清洗器

    针对中文文档优化，包含中文标点转换、全角半角转换等功能。
    """

    def __init__(
        self,
        convert_fullwidth_to_halfwidth: bool = True,
        normalize_chinese_quotes: bool = True,
        normalize_chinese_punctuation: bool = True,
        **kwargs,
    ) -> None:
        """初始化中文清洗器

        Args:
            convert_fullwidth_to_halfwidth: 全角转半角
            normalize_chinese_quotes: 规范化中文引号
            normalize_chinese_punctuation: 规范化中文标点
            **kwargs: 其他 TextCleaner 参数
        """
        super().__init__(**kwargs)
        self.convert_fullwidth_to_halfwidth = convert_fullwidth_to_halfwidth
        self.normalize_chinese_quotes = normalize_chinese_quotes
        self.normalize_chinese_punctuation = normalize_chinese_punctuation

    def clean(self, text: str) -> str:
        """执行中文优化清洗"""
        if not text:
            return ""

        cleaned = text

        if self.normalize_unicode:
            cleaned = self._normalize_unicode(cleaned)

        if self.strip_html:
            cleaned = self._strip_html(cleaned)

        if self.normalize_chinese_quotes:
            cleaned = self._normalize_chinese_quotes(cleaned)

        if self.convert_fullwidth_to_halfwidth:
            cleaned = self._convert_fullwidth_to_halfwidth(cleaned)

        if self.normalize_chinese_punctuation:
            cleaned = self._normalize_chinese_punctuation(cleaned)

        if self.remove_special_chars:
            cleaned = self._remove_special_chars(cleaned)

        if self.remove_extra_whitespace:
            cleaned = self._remove_extra_whitespace(cleaned)

        for custom_cleaner in self.custom_cleaners:
            cleaned = custom_cleaner(cleaned)

        return cleaned

    def _normalize_chinese_quotes(self, text: str) -> str:
        """规范化中文引号"""
        quote_map = {
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
            "「": '"',
            "」": '"',
            "『": "'",
            "』": "'",
        }
        for old, new in quote_map.items():
            text = text.replace(old, new)
        return text

    def _convert_fullwidth_to_halfwidth(self, text: str) -> str:
        """全角转半角"""
        result = []
        for char in text:
            inside_range = (
                0xFF01 <= ord(char) <= 0xFF5E
            )
            if inside_range:
                result.append(chr(ord(char) - 0xFEE0))
            elif char == "　":
                result.append(" ")
            else:
                result.append(char)
        return "".join(result)

    def _normalize_chinese_punctuation(self, text: str) -> str:
        """规范化中文标点"""
        punctuation_map = {
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "《": "<",
            "》": ">",
            "——": "--",
            "…": "...",
        }
        for old, new in punctuation_map.items():
            text = text.replace(old, new)
        return text


class DuplicateRemover:
    """文档去重工具"""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        use_embedding: bool = False,
        embedder: Optional[Any] = None,
    ) -> None:
        """初始化去重器

        Args:
            similarity_threshold: 相似度阈值，超过此值认为重复
            use_embedding: 是否使用嵌入计算相似度
            embedder: 嵌入模型（use_embedding=True 时需要）
        """
        self.similarity_threshold = similarity_threshold
        self.use_embedding = use_embedding
        self._embedder = embedder

    def deduplicate_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """去除重复文档

        Args:
            documents: 文档列表

        Returns:
            List[Document]: 去重后的文档列表
        """
        if not documents:
            return []

        if self.use_embedding and self._embedder:
            return self._deduplicate_by_embedding(documents)
        return self._deduplicate_by_text(documents)

    def _deduplicate_by_text(self, documents: List[Document]) -> List[Document]:
        """基于文本内容去重"""
        seen: Set[str] = set()
        unique_docs = []

        for doc in documents:
            content_hash = self._compute_hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    def _deduplicate_by_embedding(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """基于嵌入相似度去重"""
        if not self._embedder:
            return documents

        texts = [doc.page_content for doc in documents]
        try:
            embeddings = self._embedder.embed_documents(texts)
        except Exception as e:
            logger.warning(f"嵌入计算失败，使用文本去重: {e}")
            return self._deduplicate_by_text(documents)

        unique_docs = [documents[0]]
        unique_embeddings = [embeddings[0]]

        for i, (doc, embedding) in enumerate(zip(documents[1:], embeddings[1:])):
            is_duplicate = False
            for ue in unique_embeddings:
                sim = self._cosine_similarity(embedding, ue)
                if sim >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)
                unique_embeddings.append(embedding)

        return unique_docs

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _compute_hash(text: str) -> str:
        """计算文本哈希值"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
