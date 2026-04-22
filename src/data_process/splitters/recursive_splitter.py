# -*- coding: utf-8 -*-
"""
中文递归字符文本分割器

基于 LangChain 的 RecursiveCharacterTextSplitter，针对中文优化分隔符。
"""

import logging
import re
from typing import Any, List, Optional

from langchain_core.text_splitter import RecursiveCharacterTextSplitter

from src.data_process.splitters.base import BaseSplitter

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    """从文本末尾开始分割，保持分隔符"""
    if separator:
        if keep_separator:
            pattern = f"({separator})"
            splits_with_sep = re.split(pattern, text)
            splits = ["".join(pair) for pair in zip(splits_with_sep[0::2], splits_with_sep[1::2])]
            if len(splits_with_sep) % 2 == 1:
                splits.append(splits_with_sep[-1])
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(BaseSplitter):
    """中文递归字符文本分割器

    针对中文文档优化的递归分割器，按优先级尝试以下分隔符：
    1. 段落分隔符（\\n\\n）
    2. 换行符（\\n）
    3. 中文句末标点（。！？）
    4. 英文句末标点（. ! ?）
    5. 分号（；;）
    6. 逗号（，,）
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化中文递归分割器

        Args:
            separators: 自定义分隔符列表
            keep_separator: 是否保留分隔符
            is_separator_regex: 分隔符是否为正则表达式
            **kwargs: 其他参数（如 chunk_size, chunk_overlap）
        """
        super().__init__(**kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            r"\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s",
        ]
        self._is_separator_regex = is_separator_regex
        self._keep_separator = keep_separator

    def split_text(self, text: str) -> List[str]:
        """递归分割文本

        Args:
            text: 待分割文本

        Returns:
            List[str]: 分割后的文本块列表
        """
        final_chunks = []
        separator = self._separators[-1]
        new_separators = []

        for i, sep in enumerate(self._separators):
            _sep = sep if self._is_separator_regex else re.escape(sep)
            if sep == "":
                separator = sep
                break
            if re.search(_sep, text):
                separator = sep
                new_separators = self._separators[i + 1:]
                break

        _sep = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _sep, self._keep_separator)

        good_splits = []
        sep_for_merge = "" if self._keep_separator else separator

        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, sep_for_merge)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, sep_for_merge)
            final_chunks.extend(merged)

        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip()
        ]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归分割的核心逻辑

        Args:
            text: 待分割文本
            separators: 当前层级的分隔符列表

        Returns:
            List[str]: 分割后的文本块列表
        """
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            _sep = sep if self._is_separator_regex else re.escape(sep)
            if sep == "":
                separator = sep
                break
            if re.search(_sep, text):
                separator = sep
                new_separators = separators[i + 1:]
                break

        _sep = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _sep, self._keep_separator)

        good_splits = []
        sep_for_merge = "" if self._keep_separator else separator

        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, sep_for_merge)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, sep_for_merge)
            final_chunks.extend(merged)

        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip()
        ]
