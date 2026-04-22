# -*- coding: utf-8 -*-
"""
PDF 文档加载器

支持 PyMuPDF 文本提取和 RapidOCR 图片文字识别。
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredFileLoader

from src.data_process.loaders.base import BaseLoader

logger = logging.getLogger(__name__)

# OCR 引擎缓存
_ocr_engine = None


def _get_ocr_engine(use_cuda: bool = True) -> "RapidOCR":
    """获取或初始化 OCR 引擎（单例模式）"""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_paddle import RapidOCR
            _ocr_engine = RapidOCR(
                det_use_cuda=use_cuda,
                cls_use_cuda=use_cuda,
                rec_use_cuda=use_cuda,
            )
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
    return _ocr_engine


class PDFLoader(BaseLoader):
    """PDF 文档加载器

    使用 PyMuPDF 提取文本，并对嵌入图片进行 OCR 识别。
    支持配置图片尺寸阈值以过滤小图片。
    """

    def __init__(
        self,
        file_path: str,
        encoding: Optional[str] = None,
        ocr_enabled: bool = True,
        use_cuda: bool = True,
        image_threshold: float = 0.1,
        **kwargs,
    ) -> None:
        """初始化 PDF 加载器

        Args:
            file_path: PDF 文件路径
            encoding: 编码（PDF 通常不需要）
            ocr_enabled: 是否启用 OCR
            use_cuda: OCR 是否使用 GPU
            image_threshold: 图片占页面比例阈值（小于此值跳过 OCR）
            **kwargs: 其他参数
        """
        super().__init__(file_path, encoding, **kwargs)
        self.ocr_enabled = ocr_enabled
        self.use_cuda = use_cuda
        self.image_threshold = image_threshold

    def load(self) -> List[Document]:
        """加载 PDF 文档

        Returns:
            List[Document]: 文档列表
        """
        import fitz  # PyMuPDF
        import numpy as np
        from PIL import Image

        doc = fitz.open(str(self.file_path))
        text_parts = []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

                if self.ocr_enabled:
                    ocr_text = self._extract_images_with_ocr(page)
                    if ocr_text:
                        text_parts.append(f"[OCR Page {page_num + 1}]\n{ocr_text}")

            full_text = "\n\n".join(text_parts)
            metadata = {
                "source": str(self.file_path),
                "file_name": self.file_path.name,
                "total_pages": len(doc),
            }
            return [Document(page_content=full_text, metadata=metadata)]

        finally:
            doc.close()

    def _extract_images_with_ocr(self, page: "fitz.Page") -> str:
        """从页面提取图片并进行 OCR

        Args:
            page: PyMuPDF 页面对象

        Returns:
            str: OCR 识别文本
        """
        import cv2

        try:
            ocr = _get_ocr_engine(self.use_cuda)
        except Exception as e:
            logger.warning(f"OCR 引擎初始化失败: {e}")
            return ""

        resp_parts = []
        img_list = page.get_image_info(xrefs=True)

        for img in img_list:
            xref = img.get("xref")
            if not xref:
                continue

            bbox = img["bbox"]
            img_width = bbox[2] - bbox[0]
            img_height = bbox[3] - bbox[1]

            width_ratio = img_width / page.rect.width
            height_ratio = img_height / page.rect.height

            if width_ratio < self.image_threshold or height_ratio < self.image_threshold:
                continue

            try:
                pix = fitz.Pixmap(page.parent, xref)
                samples = pix.samples

                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_array = np.frombuffer(samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, -1
                )
                tmp_img = Image.fromarray(img_array)
                img_bgr = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)

                result, _ = ocr(img_array)
                if result:
                    ocr_lines = [line[1] for line in result]
                    resp_parts.extend(ocr_lines)

            except Exception as e:
                logger.debug(f"图片 OCR 处理失败: {e}")
                continue

        return "\n".join(resp_parts)


class UnstructuredPDFLoader(UnstructuredFileLoader):
    """基于 Unstructured 的 PDF 加载器（备选方案）

    使用 Unstructured 库进行文本提取，不包含 OCR 功能。
    """

    def __init__(
        self,
        file_path: str,
        ocr_enabled: bool = False,
        **kwargs,
    ) -> None:
        """初始化 Unstructured PDF 加载器

        Args:
            file_path: PDF 文件路径
            ocr_enabled: 是否启用 OCR（暂不支持）
            **kwargs: 其他 Unstructured 参数
        """
        super().__init__(file_path, **kwargs)
        self.ocr_enabled = ocr_enabled

    def _get_elements(self) -> List:
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(
            filename=str(self.file_path),
            ocr_languages=["chi_sim", "eng"],
            **self.unstructured_kwargs,
        )
