# -*- coding: utf-8 -*-
"""
代码文档加载器

支持常见编程语言源码文件的加载。
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from src.data_process.loaders.base import BaseLoader


class CodeLoader(BaseLoader):
    """代码文件加载器

    将源代码文件作为普通文本加载，保留语言信息和基本元数据。
    支持 .py、.js、.java、.cpp、.go、.rs、.ts、.txt 等格式。
    """

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sql": "sql",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".ps1": "powershell",
        ".bat": "batch",
        ".txt": "text",
    }

    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = "utf-8",
        language: Optional[str] = None,
        **kwargs,
    ) -> None:
        """初始化代码加载器

        Args:
            file_path: 代码文件路径
            encoding: 文件编码
            language: 强制指定语言（不自动检测）
            **kwargs: 其他参数
        """
        super().__init__(file_path, encoding, **kwargs)
        self.language = language or self._detect_language()

    def load(self) -> List[Document]:
        """加载代码文件

        Returns:
            List[Document]: 文档列表
        """
        with open(self.file_path, "r", encoding=self.encoding) as f:
            content = f.read()

        line_count = content.count("\n") + 1

        metadata = {
            "source": str(self.file_path),
            "file_name": self.file_path.name,
            "file_type": "code",
            "language": self.language,
            "line_count": line_count,
        }

        return [Document(page_content=content, metadata=metadata)]

    def _detect_language(self) -> str:
        """根据文件扩展名检测编程语言

        Returns:
            str: 语言标识符
        """
        ext = self.file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(ext, "text")
