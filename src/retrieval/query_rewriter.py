# -*- coding: utf-8 -*-
"""
查询改写模块

提供多种查询改写策略，支持查询扩展、分解、假设生成等功能，以提高检索的召回率和准确性。
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import RewriteResult, RewriteStrategy

logger = logging.getLogger(__name__)


class QueryRewriter:
    """查询改写器

    基于 LLM 实现多种查询改写策略，改善检索效果。

    Strategies:
        - EXPANSION: 查询扩展，添加同义词和相关概念
        - DECOMPOSITION: 查询分解，将复杂问题拆分为简单子问题
        - HYPOTHESIS: 假设生成，生成可能的答案用于检索
        - BACKOFF: 降级策略，简化查询以提高召回率
        - SPECIFIC: 具体化，使查询更加具体
    """

    EXPANSION_PROMPT = """你是一个信息检索专家。请根据给定的查询，生成3个不同的查询变体，
以帮助从不同角度检索相关信息。每个查询变体应该：
1. 使用不同的表达方式
2. 可能包含相关的同义词或术语
3. 保持查询的核心意图

原始查询: {query}

要求：直接输出查询列表，每行一个，不需要编号或解释。"""

    DECOMPOSITION_PROMPT = """你是一个信息检索专家。请将复杂查询分解为多个简单的子查询。

原始查询: {query}

分解规则：
1. 每个子查询应该是独立的、简单的问题
2. 子查询之间应该互不重叠
3. 覆盖查询的所有关键方面

要求：直接输出子查询列表，每行一个，不需要编号或解释。"""

    HYPOTHESIS_PROMPT = """你是一个信息检索专家。请根据查询，生成3个可能的答案或结论，
然后将这些作为检索目标。这些假设性的答案将帮助我们找到最相关的信息。

原始查询: {query}

生成规则：
1. 每个假设应该是可能的答案或结论
2. 使用不同的角度和观点
3. 保持与查询的相关性

要求：直接输出假设列表，每行一个，不需要编号或解释。"""

    BACKOFF_PROMPT = """你是一个信息检索专家。请将查询简化为核心关键词或概念，
以便在模糊匹配时获得更好的召回率。

原始查询: {query}

简化规则：
1. 保留最重要的关键词
2. 移除修饰性的词语
3. 提取核心概念

要求：直接输出简化后的查询列表，每行一个，不需要编号或解释。"""

    SPECIFIC_PROMPT = """你是一个信息检索专家。请将宽泛的查询变得更加具体，
以提高检索的精确度。

原始查询: {query}

具体化规则：
1. 添加具体的限定条件
2. 指定上下文或范围
3. 使用更精确的术语

要求：直接输出具体化后的查询列表，每行一个，不需要编号或解释。"""

    def __init__(
        self,
        llm: Optional[Any] = None,
        timeout: float = 3.0,
        enable_fallback: bool = True,
        cache: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化查询改写器

        Args:
            llm: 大语言模型实例
            timeout: 改写超时时间（秒）
            enable_fallback: 超时后是否降级使用原始查询
            cache: 是否缓存改写结果
            **kwargs: 其他配置参数
        """
        self.llm = llm
        self.timeout = timeout
        self.enable_fallback = enable_fallback
        self.cache_enabled = cache
        self._cache: Dict[str, RewriteResult] = {}
        self._extra_params = kwargs

    def _call_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM 生成改写结果

        支持多种 LLM 接口：
        1. agenerate/generate (LangChain 标准接口)
        2. ainvoke/invoke (OpenAI/LangChain Chat 接口)

        Args:
            prompt: 提示词

        Returns:
            Optional[str]: LLM 输出文本
        """
        if self.llm is None:
            return None

        try:
            import asyncio

            async def generate():
                # 优先使用 LangChain 标准接口
                if hasattr(self.llm, "agenerate"):
                    response = await self.llm.agenerate([prompt])
                    return response.generations[0][0].text
                elif hasattr(self.llm, "generate"):
                    return self.llm.generate([prompt]).generations[0][0].text
                # 支持 invoke/ainvoke 接口 (OpenAI/LangChain Chat)
                elif hasattr(self.llm, "ainvoke"):
                    response = await self.llm.ainvoke(prompt)
                    return self._extract_content(response)
                elif hasattr(self.llm, "invoke"):
                    response = self.llm.invoke(prompt)
                    return self._extract_content(response)
                return None

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(asyncio.wait_for(generate(), timeout=self.timeout))
                return result
            except asyncio.TimeoutError:
                logger.warning(f"LLM 调用超时 ({self.timeout}s)")
                return None
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None

    def _extract_content(self, response: Any) -> str:
        """从 LLM 响应中提取文本内容

        支持多种响应格式：
        - LangChain ChatResponse: response.content
        - 自定义 LLMResponse: response.content
        - 字符串直接返回

        Args:
            response: LLM 响应对象

        Returns:
            str: 文本内容
        """
        if response is None:
            return ""

        # 字符串直接返回
        if isinstance(response, str):
            return response

        # LangChain ChatResponse 或自定义 LLMResponse
        if hasattr(response, "content"):
            return response.content

        # 尝试作为字典处理
        if isinstance(response, dict) and "content" in response:
            return response["content"]

        # 兜底：转为字符串
        return str(response)

    def _parse_queries(self, text: str) -> List[str]:
        """解析 LLM 输出，提取查询列表

        Args:
            text: LLM 输出文本

        Returns:
            List[str]: 查询列表
        """
        if not text:
            return []

        lines = text.strip().split("\n")
        queries = []

        for line in lines:
            line = line.strip()
            line = line.lstrip("0123456789.-*、。,)）")

            if line and len(line) > 2:
                queries.append(line)

        return queries

    def _check_cache(self, query: str, strategy: RewriteStrategy) -> Optional[RewriteResult]:
        """检查缓存

        Args:
            query: 查询文本
            strategy: 改写策略

        Returns:
            Optional[RewriteResult]: 缓存的改写结果
        """
        if not self.cache_enabled:
            return None

        cache_key = f"{strategy.value}:{query}"
        return self._cache.get(cache_key)

    def _save_cache(self, result: RewriteResult, strategy: RewriteStrategy) -> None:
        """保存到缓存

        Args:
            result: 改写结果
            strategy: 改写策略
        """
        if self.cache_enabled:
            cache_key = f"{strategy.value}:{result.original_query}"
            self._cache[cache_key] = result

    def rewrite(
        self,
        query: str,
        strategy: Optional[RewriteStrategy] = None,
    ) -> RewriteResult:
        """执行查询改写

        Args:
            query: 原始查询文本
            strategy: 改写策略，若为 None 则根据查询复杂度自动选择

        Returns:
            RewriteResult: 改写结果
        """
        if strategy is None:
            strategy = self._auto_select_strategy(query)

        cached = self._check_cache(query, strategy)
        if cached:
            return cached

        if strategy == RewriteStrategy.EXPANSION:
            return self.expand(query)
        elif strategy == RewriteStrategy.DECOMPOSITION:
            return self.decompose(query)
        elif strategy == RewriteStrategy.HYPOTHESIS:
            return self.generate_hypothesis(query)
        elif strategy == RewriteStrategy.BACKOFF:
            return self.backoff(query)
        elif strategy == RewriteStrategy.SPECIFIC:
            return self.specific(query)
        else:
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy=strategy,
            )

    def _auto_select_strategy(self, query: str) -> RewriteStrategy:
        """自动选择改写策略

        Args:
            query: 查询文本

        Returns:
            RewriteStrategy: 推荐的策略
        """
        question_words = {"吗", "呢", "怎么", "如何", "为什么", "什么", "哪", "是不是"}
        query_chars = set(query)

        if question_words & query_chars:
            if len(query) > 20:
                return RewriteStrategy.DECOMPOSITION
            elif "或" in query or "或者" in query:
                return RewriteStrategy.DECOMPOSITION
            else:
                return RewriteStrategy.EXPANSION
        elif len(query) > 30:
            return RewriteStrategy.SPECIFIC
        else:
            return RewriteStrategy.EXPANSION

    def expand(self, query: str) -> RewriteResult:
        """查询扩展

        生成多个不同表达方式的查询变体。

        Args:
            query: 原始查询

        Returns:
            RewriteResult: 扩展后的查询列表
        """
        cached = self._check_cache(query, RewriteStrategy.EXPANSION)
        if cached:
            return cached

        prompt = self.EXPANSION_PROMPT.format(query=query)
        response = self._call_llm(prompt)

        if response is None:
            if self.enable_fallback:
                return RewriteResult(
                    original_query=query,
                    rewritten_queries=[query],
                    strategy=RewriteStrategy.EXPANSION,
                    metadata={"fallback": True},
                )
            return RewriteResult(
                original_query=query,
                rewritten_queries=[],
                strategy=RewriteStrategy.EXPANSION,
            )

        queries = self._parse_queries(response)

        if not queries and self.enable_fallback:
            queries = [query]

        result = RewriteResult(
            original_query=query,
            rewritten_queries=queries if queries else [query],
            strategy=RewriteStrategy.EXPANSION,
        )

        self._save_cache(result, RewriteStrategy.EXPANSION)
        return result

    def decompose(self, query: str) -> RewriteResult:
        """查询分解

        将复杂问题拆分为多个简单的子问题。

        Args:
            query: 原始查询

        Returns:
            RewriteResult: 分解后的子查询列表
        """
        cached = self._check_cache(query, RewriteStrategy.DECOMPOSITION)
        if cached:
            return cached

        prompt = self.DECOMPOSITION_PROMPT.format(query=query)
        response = self._call_llm(prompt)

        if response is None:
            if self.enable_fallback:
                return RewriteResult(
                    original_query=query,
                    rewritten_queries=[query],
                    strategy=RewriteStrategy.DECOMPOSITION,
                    metadata={"fallback": True},
                )
            return RewriteResult(
                original_query=query,
                rewritten_queries=[],
                strategy=RewriteStrategy.DECOMPOSITION,
            )

        queries = self._parse_queries(response)

        if not queries and self.enable_fallback:
            queries = [query]

        result = RewriteResult(
            original_query=query,
            rewritten_queries=queries if queries else [query],
            strategy=RewriteStrategy.DECOMPOSITION,
        )

        self._save_cache(result, RewriteStrategy.DECOMPOSITION)
        return result

    def generate_hypothesis(self, query: str) -> RewriteResult:
        """假设生成

        生成可能的答案作为检索目标。

        Args:
            query: 原始查询

        Returns:
            RewriteResult: 假设性答案列表
        """
        cached = self._check_cache(query, RewriteStrategy.HYPOTHESIS)
        if cached:
            return cached

        prompt = self.HYPOTHESIS_PROMPT.format(query=query)
        response = self._call_llm(prompt)

        if response is None:
            if self.enable_fallback:
                return RewriteResult(
                    original_query=query,
                    rewritten_queries=[query],
                    strategy=RewriteStrategy.HYPOTHESIS,
                    metadata={"fallback": True},
                )
            return RewriteResult(
                original_query=query,
                rewritten_queries=[],
                strategy=RewriteStrategy.HYPOTHESIS,
            )

        hypotheses = self._parse_queries(response)

        if not hypotheses and self.enable_fallback:
            hypotheses = [query]

        result = RewriteResult(
            original_query=query,
            rewritten_queries=hypotheses if hypotheses else [query],
            strategy=RewriteStrategy.HYPOTHESIS,
        )

        self._save_cache(result, RewriteStrategy.HYPOTHESIS)
        return result

    def backoff(self, query: str) -> RewriteResult:
        """降级/简化查询

        简化查询以提高召回率。

        Args:
            query: 原始查询

        Returns:
            RewriteResult: 简化后的查询列表
        """
        cached = self._check_cache(query, RewriteStrategy.BACKOFF)
        if cached:
            return cached

        prompt = self.BACKOFF_PROMPT.format(query=query)
        response = self._call_llm(prompt)

        if response is None:
            keywords = self._simple_keyword_extraction(query)
            return RewriteResult(
                original_query=query,
                rewritten_queries=keywords,
                strategy=RewriteStrategy.BACKOFF,
                metadata={"method": "simple_extraction"},
            )

        queries = self._parse_queries(response)

        if not queries:
            keywords = self._simple_keyword_extraction(query)
            queries = keywords

        result = RewriteResult(
            original_query=query,
            rewritten_queries=queries if queries else [query],
            strategy=RewriteStrategy.BACKOFF,
        )

        self._save_cache(result, RewriteStrategy.BACKOFF)
        return result

    def specific(self, query: str) -> RewriteResult:
        """查询具体化

        使查询更加具体以提高精确度。

        Args:
            query: 原始查询

        Returns:
            RewriteResult: 具体化后的查询列表
        """
        cached = self._check_cache(query, RewriteStrategy.SPECIFIC)
        if cached:
            return cached

        prompt = self.SPECIFIC_PROMPT.format(query=query)
        response = self._call_llm(prompt)

        if response is None:
            if self.enable_fallback:
                return RewriteResult(
                    original_query=query,
                    rewritten_queries=[query],
                    strategy=RewriteStrategy.SPECIFIC,
                    metadata={"fallback": True},
                )
            return RewriteResult(
                original_query=query,
                rewritten_queries=[],
                strategy=RewriteStrategy.SPECIFIC,
            )

        queries = self._parse_queries(response)

        if not queries and self.enable_fallback:
            queries = [query]

        result = RewriteResult(
            original_query=query,
            rewritten_queries=queries if queries else [query],
            strategy=RewriteStrategy.SPECIFIC,
        )

        self._save_cache(result, RewriteStrategy.SPECIFIC)
        return result

    def _simple_keyword_extraction(self, query: str) -> List[str]:
        """简单关键词提取（无 LLM 时的降级方案）

        Args:
            query: 查询文本

        Returns:
            List[str]: 关键词列表
        """
        stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
        }

        words = []
        current = ""

        for char in query:
            if "\u4e00" <= char <= "\u9fff":
                if current:
                    words.append(current)
                    current = ""
                words.append(char)
            else:
                current += char

        if current:
            words.append(current)

        keywords = [w for w in words if len(w) > 1 and w not in stopwords]

        return keywords if keywords else [query]

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        logger.info("查询改写缓存已清空")

    def multi_strategy(
        self,
        query: str,
        strategies: Optional[List[RewriteStrategy]] = None,
    ) -> Dict[RewriteStrategy, RewriteResult]:
        """多策略改写

        对同一查询应用多种改写策略。

        Args:
            query: 原始查询
            strategies: 策略列表，默认为 [EXPANSION, DECOMPOSITION, HYPOTHESIS]

        Returns:
            Dict[RewriteStrategy, RewriteResult]: 各策略的改写结果
        """
        if strategies is None:
            strategies = [
                RewriteStrategy.EXPANSION,
                RewriteStrategy.DECOMPOSITION,
                RewriteStrategy.HYPOTHESIS,
            ]

        results = {}
        for strategy in strategies:
            results[strategy] = self.rewrite(query, strategy=strategy)

        return results

    def get_all_variants(self, query: str) -> List[str]:
        """获取所有改写变体

        Args:
            query: 原始查询

        Returns:
            List[str]: 所有改写查询的合并列表（去重）
        """
        multi_results = self.multi_strategy(query)

        variants = set()
        variants.add(query)

        for result in multi_results.values():
            variants.update(result.rewritten_queries)

        return list(variants)

    def __repr__(self) -> str:
        return (
            f"QueryRewriter("
            f"llm={self.llm is not None}, "
            f"timeout={self.timeout}, "
            f"cache={self.cache_enabled})"
        )
