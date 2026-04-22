# -*- coding: utf-8 -*-
"""
文档加载器工厂

根据文件扩展名自动选择对应的加载器。
"""

from pathlib import Path
from typing import Dict, List, Type, Optional, Any

from src.data_process.loaders.base import BaseLoader
from src.data_process.loaders.pdf_loader import PDFLoader
from src.data_process.loaders.markdown_loader import MarkdownLoader
from src.data_process.loaders.docx_loader import DocxLoader
from src.data_process.loaders.image_loader import ImageLoader
from src.data_process.loaders.ppt_loader import PPTLoader
from src.data_process.loaders.csv_loader import CSVLoader
from src.data_process.loaders.code_loader import CodeLoader


class LoaderFactory:
    """文档加载器工厂

    根据文件扩展名自动选择对应的加载器。
    """

    EXTENSION_MAP: Dict[str, Type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".docx": DocxLoader,
        ".doc": DocxLoader,
        ".jpg": ImageLoader,
        ".jpeg": ImageLoader,
        ".png": ImageLoader,
        ".gif": ImageLoader,
        ".bmp": ImageLoader,
        ".pptx": PPTLoader,
        ".ppt": PPTLoader,
        ".csv": CSVLoader,
        ".py": CodeLoader,
        ".js": CodeLoader,
        ".java": CodeLoader,
        ".cpp": CodeLoader,
        ".c": CodeLoader,
        ".go": CodeLoader,
        ".rs": CodeLoader,
        ".ts": CodeLoader,
        ".txt": CodeLoader,
    }

    @classmethod
    def register(cls, extension: str, loader_cls: Type[BaseLoader]) -> None:
        """注册自定义加载器

        Args:
            extension: 文件扩展名（如 .pdf）
            loader_cls: 加载器类
        """
        cls.EXTENSION_MAP[extension.lower()] = loader_cls

    @classmethod
    def create(cls, file_path: str, **kwargs) -> BaseLoader:
        """创建加载器实例

        Args:
            file_path: 文件路径
            **kwargs: 传递给加载器的参数

        Returns:
            BaseLoader: 加载器实例

        Raises:
            ValueError: 不支持的文件类型
        """
        ext = Path(file_path).suffix.lower()

        if ext not in cls.EXTENSION_MAP:
            supported = list(cls.EXTENSION_MAP.keys())
            raise ValueError(
                f"不支持的文件类型: {ext}。支持的类型: {supported}"
            )

        loader_cls = cls.EXTENSION_MAP[ext]
        return loader_cls(file_path, **kwargs)

    @classmethod
    def batch_create(
        cls, file_paths: List[str], **kwargs
    ) -> List[BaseLoader]:
        """批量创建加载器

        Args:
            file_paths: 文件路径列表
            **kwargs: 传递给加载器的参数

        Returns:
            List[BaseLoader]: 加载器实例列表
        """
        return [cls.create(fp, **kwargs) for fp in file_paths]

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """获取所有支持的文件扩展名

        Returns:
            List[str]: 扩展名列表
        """
        return list(cls.EXTENSION_MAP.keys())

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """检查文件类型是否支持

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否支持
        """
        ext = Path(file_path).suffix.lower()
        return ext in cls.EXTENSION_MAP
