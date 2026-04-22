# -*- coding: utf-8 -*-
"""
语义分割器

基于句子嵌入进行语义感知的文档分割。
"""

import logging
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.text_splitter import TextSplitter

from src.data_process.splitters.base import BaseSplitter

logger = logging.getLogger(__name__)


class SemanticTextSplitter(BaseSplitter):
    """基于语义相似度的文本分割器

    通过计算相邻句子/段落之间的语义相似度，
    在相似度低于阈值时进行分割，保留语义完整性。
    """

    def __init__(
        self,
        buffer_size: int = 1,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        embedder: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """初始化语义分割器

        Args:
            buffer_size: 句子缓冲大小
            breakpoint_threshold_type: 断点阈值类型
                - "percentile": 按百分比
                - "standard_deviation": 按标准差倍数
                - "absolute": 绝对值
            breakpoint_threshold_amount: 阈值数值
            embedder: Embedding 模型实例（需有 embed_documents 方法）
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self._embedder = embedder

    def split_text(self, text: str) -> List[str]:
        """基于语义相似度分割文本

        Args:
            text: 待分割文本

        Returns:
            List[str]: 分割后的文本块列表
        """
        # 分割为句子
        sentences = self._split_into_sentences(text)
        # 如果句子数量小于等于2，则直接返回原始文本
        if len(sentences) <= 2:
            return [text]
        # 获取句子嵌入
        embeddings = self._get_embeddings(sentences)
        if embeddings is None:
            return self._fallback_split(sentences)
        # 计算相邻句子间的余弦距离
        distances = self._compute_cosine_distances(embeddings)
        # 找出断点位置
        breakpoints = self._find_breakpoints(distances)
        # 根据断点创建文本块
        chunks = self._create_chunks(sentences, breakpoints)
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        import re

        sentence_endings = r"(?<=[。！？；\?!.])[\s\n]*"
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_embeddings(self, sentences: List[str]) -> Optional[List[List[float]]]:
        """获取句子嵌入"""
        if self._embedder is None:
            return None

        try:
            embeddings = self._embedder.embed_documents(sentences)
            return embeddings
        except Exception as e:
            logger.warning(f"获取嵌入失败，使用降级分割: {e}")
            return None

    def _compute_cosine_distances(self, embeddings: List[List[float]]) -> List[float]:
        """计算相邻句子间的余弦距离"""
        import numpy as np

        distances = []
        for i in range(len(embeddings) - 1):
            e1 = np.array(embeddings[i])
            e2 = np.array(embeddings[i + 1])

            e1_norm = e1 / np.linalg.norm(e1)
            e2_norm = e2 / np.linalg.norm(e2)
            cosine_sim = np.dot(e1_norm, e2_norm)
            distance = 1 - cosine_sim
            distances.append(distance)

        return distances

    def _find_breakpoints(self, distances: List[float]) -> List[int]:
        """根据阈值找出断点位置"""
        import numpy as np

        if not distances:
            return []

        threshold = self._calculate_threshold(distances)
        breakpoints = []

        for i, d in enumerate(distances):
            if d >= threshold:
                breakpoints.append(i + 1)

        return breakpoints

    def _calculate_threshold(self, distances: List[float]) -> float:
        """计算断点阈值"""
        import numpy as np

        if self.breakpoint_threshold_type == "percentile":
            return np.percentile(distances, self.breakpoint_threshold_amount)
        elif self.breakpoint_threshold_type == "standard_deviation":
            mean = np.mean(distances)
            std = np.std(distances)
            return mean + (std * self.breakpoint_threshold_amount)
        elif self.breakpoint_threshold_type == "absolute":
            return self.breakpoint_threshold_amount
        else:
            return np.percentile(distances, self.breakpoint_threshold_amount)

    def _create_chunks(
        self,
        sentences: List[str],
        breakpoints: List[int],
    ) -> List[str]:
        """根据断点创建文本块"""
        if not breakpoints:
            return ["\n".join(sentences)]

        chunks = []
        start = 0

        for bp in breakpoints:
            chunk = sentences[start:bp]
            chunks.append("\n".join(chunk))
            start = bp

        if start < len(sentences):
            chunks.append("\n".join(sentences[start:]))

        return [c.strip() for c in chunks if c.strip()]

    def _fallback_split(self, sentences: List[str]) -> List[str]:
        """降级策略：使用固定大小分割"""
        chunks = []
        for i in range(0, len(sentences), self.buffer_size * 2):
            chunk = sentences[i:i + self.buffer_size * 2]
            chunks.append("\n".join(chunk))
        return chunks


class NLTKTextSplitter(BaseSplitter):
    """基于 NLTK 的句子分割器

    使用 NLTK 库进行句子级别的分割。
    """

    def __init__(
        self,
        language: str = "chinese",
        **kwargs: Any,
    ) -> None:
        """初始化 NLTK 句子分割器

        Args:
            language: 语言设置（支持 chinese, english 等）
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.language = language

    def split_text(self, text: str) -> List[str]:
        """使用 NLTK 分割文本为句子

        Args:
            text: 待分割文本

        Returns:
            List[str]: 句子列表
        """
        try:
            import nltk
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            import nltk
            nltk.download("punkt", quiet=True)
            if self.language == "chinese":
                nltk.download("punkt_tab", quiet=True)

        if self.language == "chinese":
            return self._split_chinese(text)
        return self._split_english(text)

    def _split_chinese(self, text: str) -> List[str]:
        """中文分句"""
        import re

        import nltk
        from nltk.tokenize import PunktSentenceTokenizer

        sentence_endings = "。！？；"
        pattern = f"[{sentence_endings}]"
        sentences = []

        for part in re.split(pattern, text):
            if part.strip():
                if re.search(pattern, part):
                    sentences.append(part.strip() + part[-1])
                else:
                    if sentences:
                        sentences[-1] += part
                    else:
                        sentences.append(part)

        return [s.strip() for s in sentences if s.strip()]

    def _split_english(self, text: str) -> List[str]:
        """英文分句"""
        import nltk

        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
