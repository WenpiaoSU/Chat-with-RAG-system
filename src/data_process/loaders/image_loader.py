# -*- coding: utf-8 -*-
"""
图片文档加载器

使用 RapidOCR 对图片进行文字识别。
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.data_process.loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class ImageLoader(BaseLoader):
    """图片加载器

    使用 RapidOCR 提取图片中的文字内容。
    支持 jpg、png、gif、bmp 等常见图片格式。
    """

    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = None,
        use_cuda: bool = True,
        **kwargs,
    ) -> None:
        """初始化图片加载器

        Args:
            file_path: 图片文件路径
            encoding: 编码（图片文件无需指定）
            use_cuda: OCR 是否使用 GPU
            **kwargs: 其他参数
        """
        super().__init__(file_path, encoding, **kwargs)
        self.use_cuda = use_cuda
        self._ocr_engine = None

    @property
    def ocr_engine(self):
        """延迟初始化 OCR 引擎"""
        if self._ocr_engine is None:
            try:
                from rapidocr_paddle import RapidOCR
                self._ocr_engine = RapidOCR(
                    det_use_cuda=self.use_cuda,
                    cls_use_cuda=self.use_cuda,
                    rec_use_cuda=self.use_cuda,
                )
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def load(self) -> List[Document]:
        """加载图片并进行 OCR

        Returns:
            List[Document]: 文档列表
        """
        try:
            result, _ = self.ocr_engine(str(self.file_path))
        except Exception as e:
            logger.error(f"图片 OCR 处理失败: {self.file_path}, error: {e}")
            return [Document(
                page_content="",
                metadata={
                    "source": str(self.file_path),
                    "file_name": self.file_path.name,
                    "file_type": "image",
                    "error": str(e),
                }
            )]

        if not result:
            return [Document(
                page_content="",
                metadata={
                    "source": str(self.file_path),
                    "file_name": self.file_path.name,
                    "file_type": "image",
                    "ocr_detected": False,
                }
            )]

        ocr_lines = []
        for line in result:
            text = line[1]
            confidence = line[2]
            ocr_lines.append(text)

        full_text = "\n".join(ocr_lines)

        metadata = {
            "source": str(self.file_path),
            "file_name": self.file_path.name,
            "file_type": "image",
            "ocr_detected": True,
            "detected_lines": len(result),
        }

        return [Document(page_content=full_text, metadata=metadata)]
