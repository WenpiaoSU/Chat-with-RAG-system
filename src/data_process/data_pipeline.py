# -*- coding: utf-8 -*-
"""
数据处理流水线

实现 Load -> Clean -> Chunk -> Embed 自动化流水线。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document

from src.data_process.loaders.base import BaseLoader
from src.data_process.loaders.loader_factory import LoaderFactory
from src.data_process.splitters.base import BaseSplitter
from src.data_process.splitters.recursive_splitter import ChineseRecursiveTextSplitter
from src.data_process.splitters.parent_child_splitter import ParentChildSplitter
from src.data_process.splitters.markdown_splitter import MarkdownHeaderSplitter
from src.data_process.cleaners.text_cleaner import TextCleaner, ChineseTextCleaner

logger = logging.getLogger(__name__)


class DataPipeline:
    """数据处理流水线

    整合文档加载、文本清洗、文本分割等步骤，提供端到端的数据处理能力。
    """

    def __init__(
        self,
        loader: Optional[BaseLoader] = None,
        cleaner: Optional[TextCleaner] = None,
        splitter: Optional[BaseSplitter] = None,
        enable_cleaning: bool = True,
        enable_splitting: bool = True,
    ) -> None:
        """初始化数据流水线

        Args:
            loader: 文档加载器
            cleaner: 文本清洗器
            splitter: 文本分割器
            enable_cleaning: 是否启用文本清洗
            enable_splitting: 是否启用文本分割
        """
        self.loader = loader
        self.cleaner = cleaner
        self.splitter = splitter
        self.enable_cleaning = enable_cleaning
        self.enable_splitting = enable_splitting

    def process(
        self,
        documents: Optional[List[Document]] = None,
        file_path: Optional[str] = None,
    ) -> List[Document]:
        """执行完整的数据处理流程

        Args:
            documents: 预加载的文档列表（优先使用）
            file_path: 文件路径（需要 loader）

        Returns:
            List[Document]: 处理后的文档列表

        Raises:
            ValueError: 既没有 documents 也没有 file_path
        """
        if documents is None and file_path is None:
            raise ValueError("必须提供 documents 或 file_path")

        if documents is None:
            documents = self._load_documents(file_path)

        docs = documents

        if self.enable_cleaning and self.cleaner:
            docs = self._clean_documents(docs)

        if self.enable_splitting and self.splitter:
            docs = self._split_documents(docs)

        return docs

    def _load_documents(
        self,
        file_path: Optional[str],
    ) -> List[Document]:
        """加载文档"""
        if self.loader is None:
            if file_path is None:
                raise ValueError("需要 loader 来加载文档")
            self.loader = LoaderFactory.create(file_path)
        return self.loader.load()

    def _clean_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """清洗文档"""
        if self.cleaner is None:
            return documents
        return self.cleaner.clean_documents(documents)

    def _split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """分割文档"""
        if self.splitter is None:
            return documents
        return self.splitter.split_documents(documents)

    def process_files(
        self,
        file_paths: List[str],
        show_progress: bool = True,
    ) -> List[Document]:
        """批量处理多个文件

        Args:
            file_paths: 文件路径列表
            show_progress: 是否显示进度

        Returns:
            List[Document]: 处理后的文档列表
        """
        all_docs = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(file_paths, desc="处理文件")
            except ImportError:
                iterator = file_paths
        else:
            iterator = file_paths

        for file_path in iterator:
            try:
                docs = self.process(file_path=file_path)
                all_docs.extend(docs)
                logger.info(f"成功处理文件: {file_path}, 生成 {len(docs)} 个文档块")
            except Exception as e:
                logger.error(f"处理文件失败: {file_path}, error: {e}")

        return all_docs


def build_pipeline(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    splitter_type: str = "recursive",
    enable_cleaning: bool = True,
    enable_parent_child: bool = False,
    **kwargs,
) -> DataPipeline:
    """构建数据处理流水线

    Args:
        chunk_size: 块大小
        chunk_overlap: 块重叠
        splitter_type: 分割器类型
            - "recursive": 递归字符分割
            - "markdown": Markdown 标题分割
            - "parent_child": 父子文档分割
        enable_cleaning: 是否启用文本清洗
        enable_parent_child: 是否启用父子分割
        **kwargs: 其他参数

    Returns:
        DataPipeline: 配置好的流水线实例
    """
    if enable_cleaning:
        cleaner = ChineseTextCleaner(
            remove_extra_whitespace=True,
            strip_html=True,
            normalize_unicode=True,
            convert_fullwidth_to_halfwidth=True,
            normalize_chinese_quotes=True,
        )
    else:
        cleaner = None

    if splitter_type == "markdown":
        splitter = MarkdownHeaderSplitter(chunk_size=chunk_size)
    elif splitter_type == "parent_child" or enable_parent_child:
        splitter = ParentChildSplitter(
            parent_chunk_size=chunk_size * 3,
            parent_chunk_overlap=chunk_overlap * 3,
            child_chunk_size=chunk_size,
            child_chunk_overlap=chunk_overlap,
        )
    else:
        splitter = ChineseRecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return DataPipeline(
        cleaner=cleaner,
        splitter=splitter,
        enable_cleaning=enable_cleaning,
        enable_splitting=True,
    )


class IngestPipeline:
    """文档摄取流水线

    扩展的数据处理流水线，包含向量化存储步骤。
    """

    def __init__(
        self,
        pipeline: Optional[DataPipeline] = None,
        embedder: Optional[Any] = None,
        vectorstore: Optional[Any] = None,
        collection_name: str = "default",
    ) -> None:
        """初始化摄取流水线

        Args:
            pipeline: 数据处理流水线
            embedder: 嵌入模型
            vectorstore: 向量存储
            collection_name: 集合名称
        """
        self.pipeline = pipeline or build_pipeline()
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.collection_name = collection_name

    def ingest(
        self,
        file_paths: List[str],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """摄取文档到向量存储

        Args:
            file_paths: 文件路径列表
            show_progress: 是否显示进度

        Returns:
            Dict[str, Any]: 摄取结果统计
        """
        docs = self.pipeline.process_files(file_paths, show_progress=show_progress)

        if not docs:
            return {
                "success": False,
                "message": "没有处理到任何文档",
                "document_count": 0,
                "chunk_count": 0,
            }

        if self.embedder and self.vectorstore:
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]

            try:
                embeddings = self.embedder.embed_documents(texts)
                self.vectorstore.add_texts(
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                logger.info(f"成功向量化 {len(docs)} 个文档块")
            except Exception as e:
                logger.error(f"向量化失败: {e}")
                return {
                    "success": False,
                    "message": f"向量化失败: {e}",
                    "document_count": len(set(d.metadata.get("source") for d in docs)),
                    "chunk_count": len(docs),
                }

        return {
            "success": True,
            "message": f"成功处理 {len(docs)} 个文档块",
            "document_count": len(set(d.metadata.get("source") for d in docs)),
            "chunk_count": len(docs),
            "collection_name": self.collection_name,
        }


def process_directory(
    directory: str,
    file_extensions: Optional[List[str]] = None,
    recursive: bool = True,
    **pipeline_kwargs,
) -> List[Document]:
    """处理目录下所有支持的文件

    Args:
        directory: 目录路径
        file_extensions: 要处理的文件扩展名列表（如 [".pdf", ".md"]）
            为 None 时处理所有支持的文件类型
        recursive: 是否递归处理子目录
        **pipeline_kwargs: build_pipeline 的参数

    Returns:
        List[Document]: 处理后的文档列表
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    if file_extensions:
        extensions = [ext.lower() for ext in file_extensions]
        patterns = [f"**/*{ext}" for ext in extensions]
        file_paths = []
        for pattern in patterns:
            file_paths.extend(dir_path.glob(pattern))
    else:
        file_paths = list(dir_path.rglob("*")) if recursive else list(dir_path.glob("*"))

    file_paths = [
        str(fp) for fp in file_paths
        if fp.is_file() and LoaderFactory.is_supported(str(fp))
    ]

    if not file_paths:
        logger.warning(f"目录中没有找到支持的文件: {directory}")
        return []

    pipeline = build_pipeline(**pipeline_kwargs)
    return pipeline.process_files(file_paths)
