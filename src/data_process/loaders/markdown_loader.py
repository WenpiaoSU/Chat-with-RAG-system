# -*- coding: utf-8 -*-
"""
Markdown 文档加载器

加载 .md/.markdown 文件，保留基本元数据。
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from src.data_process.loaders.base import BaseLoader


class MarkdownLoader(BaseLoader):
    """Markdown 文档加载器

    基于 TextLoader 实现，将 Markdown 文件加载为 Document。
    保留文件名、标题等基本元数据。
    """

    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = "utf-8",
        metadata_from_metadata: bool = True,
        **kwargs,
    ) -> None:
        """初始化 Markdown 加载器

        Args:
            file_path: Markdown 文件路径
            encoding: 文件编码
            metadata_from_metadata: 是否提取 YAML front matter 作为元数据
            **kwargs: 其他参数
        """
        super().__init__(file_path, encoding, **kwargs)
        self.metadata_from_metadata = metadata_from_metadata

    def load(self) -> List[Document]:
        """加载 Markdown 文档

        Returns:
            List[Document]: 文档列表
        """
        import re

        with open(self.file_path, "r", encoding=self.encoding) as f:
            content = f.read()

        metadata = {
            "source": str(self.file_path),
            "file_name": self.file_path.name,
            "file_type": "markdown",
        }

        if self.metadata_from_metadata:
            front_matter = self._extract_front_matter(content)
            if front_matter:
                metadata.update(front_matter)
                content = front_matter["content"]

        metadata["title"] = self._extract_title(content)

        return [Document(page_content=content, metadata=metadata)]

    def _extract_front_matter(self, content: str) -> Optional[dict]:
        """提取 YAML front matter 元数据

        Args:
            content: 文件内容

        Returns:
            Optional[dict]: 解析后的元数据字典
        """
        import re

        pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(pattern, content, re.DOTALL)

        if not match:
            return None

        try:
            import yaml
            fm_text = match.group(1)
            fm_data = yaml.safe_load(fm_text)
            fm_data["content"] = content[match.end() :]
            return fm_data
        except yaml.YAMLError:
            return None

    def _extract_title(self, content: str) -> str:
        """提取文档标题

        优先从 front matter 获取，否则从第一个 # 标题获取。

        Args:
            content: 文档内容

        Returns:
            str: 标题文本
        """
        import re

        pattern = r"^#\s+(.+)$"
        for line in content.split("\n"):
            match = re.match(pattern, line.strip())
            if match:
                return match.group(1).strip()
        return self.file_path.stem
