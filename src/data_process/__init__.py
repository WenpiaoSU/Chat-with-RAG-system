# -*- coding: utf-8 -*-
"""
数据处理模块

包含文档加载、文本分割、数据清洗等流水线组件。

架构设计：
- loaders/: 文档加载器（支持 PDF、Word、图片、PPT、CSV、Markdown 等格式）
- splitters/: 文本分割器（递归分割、语义分割、父子分割、Markdown 标题分割）
- cleaners/: 文本清洗工具
- data_pipeline.py: 自动化数据处理流水线
"""

from src.data_process.loaders.base import BaseLoader
from src.data_process.loaders.loader_factory import LoaderFactory
from src.data_process.loaders.pdf_loader import PDFLoader, UnstructuredPDFLoader
from src.data_process.loaders.markdown_loader import MarkdownLoader
from src.data_process.loaders.docx_loader import DocxLoader
from src.data_process.loaders.image_loader import ImageLoader
from src.data_process.loaders.ppt_loader import PPTLoader
from src.data_process.loaders.csv_loader import CSVLoader, FilteredCSVLoader
from src.data_process.loaders.code_loader import CodeLoader

from src.data_process.splitters.base import BaseSplitter
from src.data_process.splitters.recursive_splitter import ChineseRecursiveTextSplitter
from src.data_process.splitters.markdown_splitter import MarkdownHeaderSplitter, MarkdownHeaderTextSplitter
from src.data_process.splitters.semantic_splitter import SemanticTextSplitter, NLTKTextSplitter
from src.data_process.splitters.parent_child_splitter import ParentChildSplitter, SimpleParentSplitter

from src.data_process.cleaners.text_cleaner import (
    TextCleaner,
    ChineseTextCleaner,
    DuplicateRemover,
)

from src.data_process.data_pipeline import (
    DataPipeline,
    IngestPipeline,
    build_pipeline,
    process_directory,
)

__all__ = [
    # Loaders
    "BaseLoader",
    "LoaderFactory",
    "PDFLoader",
    "UnstructuredPDFLoader",
    "MarkdownLoader",
    "DocxLoader",
    "ImageLoader",
    "PPTLoader",
    "CSVLoader",
    "FilteredCSVLoader",
    "CodeLoader",
    # Splitters
    "BaseSplitter",
    "ChineseRecursiveTextSplitter",
    "MarkdownHeaderSplitter",
    "MarkdownHeaderTextSplitter",
    "SemanticTextSplitter",
    "NLTKTextSplitter",
    "ParentChildSplitter",
    "SimpleParentSplitter",
    # Cleaners
    "TextCleaner",
    "ChineseTextCleaner",
    "DuplicateRemover",
    # Pipeline
    "DataPipeline",
    "IngestPipeline",
    "build_pipeline",
    "process_directory",
]
