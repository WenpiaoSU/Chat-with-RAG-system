# -*- coding: utf-8 -*-
"""
文档加载器基类

定义所有文档加载器的统一接口。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document


class BaseLoader(ABC):
    """文档加载器抽象基类

    所有自定义加载器需继承此类并实现 `load` 方法。
    """

    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = None,
        **kwargs,
    ) -> None:
        """初始化加载器

        Args:
            file_path: 文件路径
            encoding: 文件编码，默认自动检测
            **kwargs: 其他加载参数
        """
        self.file_path = Path(file_path)
        self.encoding = encoding or "utf-8"
        self.kwargs = kwargs

        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

    @abstractmethod
    def load(self) -> List[Document]:
        """加载文档并返回 Document 列表

        Returns:
            List[Document]: 文档列表
        """
        ...

    def load_and_split(self, splitter: "BaseSplitter") -> List[Document]:
        """加载文档并使用指定分割器分割

        Args:
            splitter: 文本分割器实例

        Returns:
            List[Document]: 分割后的文档列表
        """
        docs = self.load()
        return splitter.split_documents(docs)
