# -*- coding: utf-8 -*-
"""
关键词检索模块

基于 Okapi BM25 算法的关键词检索实现。
使用 Jieba 分词器处理中文文本，支持停用词过滤。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseRetriever, RetrievalResult, RetrievalResults

logger = logging.getLogger(__name__)

DEFAULT_CJK_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "里", "为", "什么", "他", "来", "用", "能", "而", "把", "被",
    "从", "以", "可以", "这个", "那个", "如果", "因为", "所以", "但是", "虽然",
    "以及", "对于", "关于", "而且", "或者", "还是", "只是", "可能", "应该", "已经",
    "之后", "之前", "然后", "最后", "现在", "当时", "以后", "以前", "其中", "之间",
}


class BM25Search(BaseRetriever):
    """基于 Okapi BM25 算法的关键词检索器

    Algorithm:
        BM25_score(D, Q) = Σ IDF(qi) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D|/avgdl))

    Features:
        - Okapi BM25 算法实现
        - Jieba 中文分词支持
        - 停用词过滤
        - 可调参数：k1, b
    """

    def __init__(
        self,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        k1: float = 1.5,
        b: float = 0.75,
        use_stopwords: bool = True,
        stopwords: Optional[Set[str]] = None,
        tokenizer: str = "jieba",
        **kwargs: Any,
    ) -> None:
        """初始化 BM25 检索器

        Args:
            top_k: 默认返回的 top-k 结果数
            score_threshold: 相似度分数阈值
            k1: BM25 词频饱和参数（通常 1.2-2.0）
            b: BM25 文档长度归一化参数（通常 0.75）
            use_stopwords: 是否使用停用词过滤
            stopwords: 自定义停用词集合
            tokenizer: 分词器类型（jieba/simple）
            **kwargs: 其他配置参数
        """
        super().__init__(
            vectorstore=None,
            embedder=None,
            top_k=top_k,
            score_threshold=score_threshold,
            **kwargs,
        )

        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.use_stopwords = use_stopwords

        self._stopwords: Set[str] = stopwords or DEFAULT_CJK_STOPWORDS
        self._corpus: List[List[str]] = []
        self._corpus_texts: List[str] = []
        self._corpus_ids: List[str] = []
        self._corpus_metadata: List[Dict[str, Any]] = []

        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: List[Dict[str, int]] = []
        self._doc_count: int = 0
        self._idf: Dict[str, float] = {}
        self._is_indexed: bool = False

    def _init_jieba(self) -> Any:
        """初始化 Jieba 分词器"""
        try:
            import jieba
            jieba.setLogLevel(logging.WARNING)
            return jieba
        except ImportError:
            raise ImportError(
                "请安装 Jieba 分词库: pip install jieba"
            )

    def _tokenize(self, text: str) -> List[str]:
        """分词处理

        Args:
            text: 输入文本

        Returns:
            List[str]: 分词结果列表
        """
        if not text:
            return []

        if self.tokenizer == "jieba":
            jieba = self._init_jieba()
            tokens = list(jieba.cut(text))
        else:
            tokens = list(text)

        if self.use_stopwords:
            tokens = [t for t in tokens if t.strip() and t not in self._stopwords]
        else:
            tokens = [t for t in tokens if t.strip()]

        return tokens

    def _calculate_idf(self) -> None:
        """计算所有词项的 IDF 值"""
        idf: Dict[str, float] = {}
        total_docs = self._doc_count

        for doc_freq in self._doc_freqs:
            for word, freq in doc_freq.items():
                idf[word] = idf.get(word, 0) + 1

        for word in idf:
            idf[word] = max(
                0.0,
                (total_docs - idf[word] + 0.5) / (idf[word] + 0.5)
            )

        self._idf = idf

    def _get_doc_freq(self, word: str) -> int:
        """获取词项的文档频率

        Args:
            word: 词项

        Returns:
            int: 包含该词项的文档数
        """
        return sum(1 for doc in self._doc_freqs if word in doc)

    def index_corpus(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> bool:
        """构建 BM25 索引

        Args:
            texts: 文档文本列表
            metadatas: 元数据列表
            ids: 文档 ID 列表

        Returns:
            bool: 是否索引成功
        """
        if not texts:
            return False

        self._corpus_texts = texts
        self._corpus_ids = ids or [str(i) for i in range(len(texts))]
        self._corpus_metadata = metadatas or [{} for _ in texts]

        self._corpus = []
        self._doc_freqs = []
        self._doc_lengths = []

        total_length = 0

        for text in texts:
            tokens = self._tokenize(text)
            self._corpus.append(tokens)
            self._doc_lengths.append(len(tokens))
            total_length += len(tokens)

            doc_freq: Dict[str, int] = {}
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
            self._doc_freqs.append(doc_freq)

        self._doc_count = len(texts)
        self._avg_doc_length = total_length / max(self._doc_count, 1)

        self._calculate_idf()
        self._is_indexed = True

        logger.info(f"BM25 索引构建完成: {self._doc_count} 篇文档")

        return True

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """添加文档并更新索引

        Args:
            texts: 文档文本列表
            metadatas: 元数据列表
            ids: 文档 ID 列表
            **kwargs: 其他参数

        Returns:
            bool: 是否添加成功
        """
        super().add_documents(texts, metadatas, ids, **kwargs)

        return self.index_corpus(
            texts=self._documents,
            metadatas=self._metadata,
            ids=self._document_ids,
        )

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算单个文档的 BM25 分数

        Args:
            query_tokens: 查询分词列表
            doc_idx: 文档索引

        Returns:
            float: BM25 分数
        """
        doc_len = self._doc_lengths[doc_idx]
        doc_freqs = self._doc_freqs[doc_idx]

        score = 0.0

        for token in query_tokens:
            if token not in doc_freqs:
                continue

            tf = doc_freqs[token]
            idf = self._idf.get(token, 0.0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_length, 1))

            score += idf * (numerator / max(denominator, 1e-8))

        return score

    def _score_corpus(self, query_tokens: List[str]) -> List[tuple]:
        """对整个语料库评分

        Args:
            query_tokens: 查询分词列表

        Returns:
            List[tuple]: (文档索引, BM25分数) 列表
        """
        scores = []
        for i in range(self._doc_count):
            score = self._bm25_score(query_tokens, i)
            scores.append((i, score))
        return scores

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """执行 BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter: 元数据过滤条件（暂不支持）
            score_threshold: 分数阈值
            **kwargs: 其他参数

        Returns:
            RetrievalResults: 检索结果集
        """
        start_time = time.time()

        if not self._is_indexed:
            logger.warning("BM25 索引未构建，执行空检索")
            return RetrievalResults(
                query=query,
                metadata={"strategy": "bm25", "error": "index not built"},
            )

        k = top_k or self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return RetrievalResults(
                query=query,
                metadata={"strategy": "bm25", "note": "no valid tokens in query"},
            )

        scores = self._score_corpus(query_tokens)
        scores.sort(key=lambda x: x[1], reverse=True)

        max_score = scores[0][1] if scores else 1.0

        retrieval_results: List[RetrievalResult] = []

        for idx, raw_score in scores:
            if raw_score <= 0:
                continue

            if threshold is not None:
                normalized_score = raw_score / max_score
                if normalized_score < threshold:
                    continue
            else:
                normalized_score = raw_score / max_score

            doc_text = self._corpus_texts[idx]
            metadata = dict(self._corpus_metadata[idx]) if idx < len(self._corpus_metadata) else {}

            result = RetrievalResult(
                content=doc_text,
                metadata=metadata,
                score=normalized_score,
                score_raw=raw_score,
                source=metadata.get("source", "bm25"),
                document_id=self._corpus_ids[idx] if idx < len(self._corpus_ids) else None,
            )
            retrieval_results.append(result)

        retrieval_results = self._normalize_results(retrieval_results, top_k=k)

        query_time = (time.time() - start_time) * 1000

        return RetrievalResults(
            results=retrieval_results,
            query=query,
            total_count=len(retrieval_results),
            query_time_ms=query_time,
            metadata={
                "strategy": "bm25",
                "query_tokens": query_tokens,
                "top_k": k,
                "score_threshold": threshold,
                "k1": self.k1,
                "b": self.b,
                "indexed_docs": self._doc_count,
            },
        )

    def search(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[RetrievalResult]:
        """便捷方法：执行关键词搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            **kwargs: 其他参数

        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        results = self.retrieve(query=query, top_k=k, **kwargs)
        return results.results

    def get_relevant_terms(
        self,
        query: str,
        top_n: int = 10,
    ) -> List[tuple]:
        """获取查询的相关词项及其 IDF 值

        Args:
            query: 查询文本
            top_n: 返回的词项数量

        Returns:
            List[tuple]: (词项, IDF分数) 列表
        """
        tokens = self._tokenize(query)
        terms = []

        for token in tokens:
            idf = self._idf.get(token, 0.0)
            terms.append((token, idf))

        terms.sort(key=lambda x: x[1], reverse=True)
        return terms[:top_n]

    def clear_index(self) -> None:
        """清空 BM25 索引"""
        self._corpus = []
        self._corpus_texts = []
        self._corpus_ids = []
        self._corpus_metadata = []
        self._doc_lengths = []
        self._doc_freqs = []
        self._doc_count = 0
        self._idf = {}
        self._is_indexed = False

        self._documents = []
        self._document_ids = []
        self._metadata = []

        logger.info("BM25 索引已清空")

    @property
    def indexed_doc_count(self) -> int:
        """获取已索引的文档数量"""
        return self._doc_count

    def __repr__(self) -> str:
        return (
            f"BM25Search("
            f"top_k={self.top_k}, "
            f"k1={self.k1}, "
            f"b={self.b}, "
            f"indexed_docs={self._doc_count}, "
            f"tokenizer={self.tokenizer})"
        )
