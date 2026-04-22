# -*- coding: utf-8 -*-
"""
CSV 文档加载器

支持指定列加载和元数据提取。
"""

from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders.helpers import detect_file_encodings

from src.data_process.loaders.base import BaseLoader


class CSVLoader(BaseLoader):
    """CSV 文档加载器

    支持自定义列选择、元数据提取和自动编码检测。
    """

    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = "utf-8",
        columns_to_read: Optional[List[str]] = None,
        source_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        csv_args: Optional[Dict[str, Any]] = None,
        autodetect_encoding: bool = True,
        **kwargs,
    ) -> None:
        """初始化 CSV 加载器

        Args:
            file_path: CSV 文件路径
            encoding: 文件编码
            columns_to_read: 需要读取的列名列表（None 表示全部）
            source_column: 源列名（用于 Document.metadata.source）
            metadata_columns: 需要添加到 metadata 的列名列表
            csv_args: csv.DictReader 的额外参数
            autodetect_encoding: 是否自动检测编码
            **kwargs: 其他参数
        """
        super().__init__(file_path, encoding, **kwargs)
        self.columns_to_read = columns_to_read
        self.source_column = source_column
        self.metadata_columns = metadata_columns or []
        self.csv_args = csv_args or {}
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        """加载 CSV 文档

        Returns:
            List[Document]: 文档列表
        """
        import csv

        docs = []
        try:
            with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
                docs = self._read_file(csvfile)
        except UnicodeDecodeError:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for enc in detected_encodings:
                    try:
                        with open(
                            self.file_path,
                            newline="",
                            encoding=enc.encoding
                        ) as csvfile:
                            self.encoding = enc.encoding
                            docs = self._read_file(csvfile)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(
                    f"无法使用编码 {self.encoding} 读取文件: {self.file_path}"
                )
        except Exception as e:
            raise RuntimeError(f"加载 CSV 文件失败: {self.file_path}, error: {e}")

        return docs

    def _read_file(self, csvfile) -> List[Document]:
        """读取 CSV 文件内容"""
        import csv

        docs = []
        reader = csv.DictReader(csvfile, **self.csv_args)

        for row_idx, row in enumerate(reader):
            content_parts = []
            content_list = []

            if self.columns_to_read:
                for col in self.columns_to_read:
                    if col in row and row[col]:
                        content_parts.append(f"{col}: {row[col]}")
                        content_list.append(row[col])
            else:
                for col, val in row.items():
                    if val:
                        content_parts.append(f"{col}: {val}")
                        content_list.append(val)

            if not content_parts:
                continue

            content = "\n".join(content_parts)

            source = (
                row.get(self.source_column, self.file_path.name)
                if self.source_column and self.source_column in row
                else self.file_path.name
            )

            metadata = {
                "source": str(self.file_path),
                "row": row_idx,
            }

            for col in self.metadata_columns:
                if col in row:
                    metadata[col] = row[col]

            doc = Document(page_content=content, metadata=metadata)
            doc.metadata["content_list"] = content_list
            docs.append(doc)

        return docs


class FilteredCSVLoader(CSVLoader):
    """选择性列加载的 CSV 加载器

    继承自 CSVLoader，主要用于明确指定需要读取的列。
    """

    def __init__(
        self,
        file_path: str,
        columns_to_read: List[str],
        source_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        encoding: Optional[str] = "utf-8",
        autodetect_encoding: bool = True,
        **kwargs,
    ) -> None:
        """初始化过滤列的 CSV 加载器

        Args:
            file_path: CSV 文件路径
            columns_to_read: 必须读取的列名列表
            source_column: 源列
            metadata_columns: 元数据列
            encoding: 编码
            autodetect_encoding: 自动检测编码
            **kwargs: 其他参数
        """
        super().__init__(
            file_path=file_path,
            encoding=encoding,
            columns_to_read=columns_to_read,
            source_column=source_column,
            metadata_columns=metadata_columns,
            autodetect_encoding=autodetect_encoding,
            **kwargs,
        )
