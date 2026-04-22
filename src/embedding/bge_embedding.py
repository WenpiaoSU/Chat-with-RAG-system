# -*- coding: utf-8 -*-
"""
BGE Embedding 模型实现
"""

import logging
import os
from typing import Any, List, Optional, Union

# 设置模型缓存目录
os.environ.setdefault("MODEL_CACHE_DIR", "/hot_disk_1T/data/swp/huggingface")

import numpy as np
import torch

from ..configs.settings import get_settings
from .base import BaseEmbedding

logger = logging.getLogger(__name__)


class BGEEmbedding(BaseEmbedding):
    """BGE (BAAI General Embedding) 模型"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: Optional[str] = None,
        batch_size: int = 32,
        dimension: int = 1024,
        normalize_embeddings: bool = True,
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
        passage_instruction: str = "",
        cache_folder: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 BGE Embedding 模型
        
        Args:
            model_name: HuggingFace 模型名称或本地路径
            device: 运行设备，默认为 cuda（如果可用）
            batch_size: 批处理大小
            dimension: 嵌入向量维度
            normalize_embeddings: 是否归一化嵌入向量
            query_instruction: 查询文本的指令前缀
            passage_instruction: 文档片段的指令前缀
            cache_folder: 模型缓存目录
            **kwargs: 其他参数
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        super().__init__(
            model_name=model_name,
            device=device,
            dimension=dimension,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )
        
        self.query_instruction = query_instruction
        self.passage_instruction = passage_instruction
        self.cache_folder = cache_folder or self._get_default_cache_dir()

        self._tokenizer = None
        self._model = None

        self.load_model()

    @staticmethod
    def _get_default_cache_dir() -> str:
        """获取默认的模型缓存目录

        优先级：环境变量 MODEL_CACHE_DIR > 环境变量 HF_HOME > 默认路径

        Returns:
            str: 缓存目录路径
        """
        env_cache = os.environ.get("MODEL_CACHE_DIR")
        if env_cache:
            return env_cache

        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return hf_home

        return os.path.expanduser("~/.cache/huggingface")
    
    def load_model(self) -> Any:
        """加载 BGE 模型和分词器
        
        Returns:
            Any: 模型实例
        """
        if self._model is not None:
            return self._model
        
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "请安装 transformers 库: pip install transformers"
            )
        
        logger.info(f"正在加载 BGE 模型: {self.model_name}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_folder,
        )
        
        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_folder,
        ).to(self.device)
        
        self._model.eval()
        
        if hasattr(self._model, "config"):
            model_dim = getattr(self._model.config, "hidden_size", None)
            if model_dim:
                self.dimension = model_dim
        
        logger.info(
            f"BGE 模型加载完成: {self.model_name}, "
            f"dimension={self.dimension}, device={self.device}"
        )
        
        return self._model
    
    def _mean_pooling(
        self,
        model_output: Any,
        attention_mask: Any,
    ) -> torch.Tensor:
        """Mean Pooling - 对 token embeddings 进行平均池化
        
        Args:
            model_output: 模型输出
            attention_mask: 注意力掩码
            
        Returns:
            torch.Tensor: 池化后的嵌入向量
        """
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def _prepare_inputs(
        self,
        texts: Union[str, List[str]],
        instruction: str = "",
    ) -> tuple:
        """准备模型输入
        
        Args:
            texts: 文本或文本列表
            instruction: 指令前缀
            
        Returns:
            tuple: (texts_with_instruction, encoded_inputs)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        texts_with_instruction = []
        for text in texts:
            if instruction and not text.startswith(instruction):
                text = instruction + text
            texts_with_instruction.append(text)
        
        encoded = self._tokenizer(
            texts_with_instruction,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        return texts_with_instruction, encoded
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """将文本编码为嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小，若为 None 则使用默认值
            show_progress: 是否显示进度条
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 嵌入向量
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            texts_with_instruction, encoded = self._prepare_inputs(
                batch_texts, self.passage_instruction
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                model_output = self._model(**encoded)
                embeddings = self._mean_pooling(
                    model_output, encoded["attention_mask"]
                )
                
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(
                        embeddings, p=2, dim=1
                    )
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        result = np.vstack(all_embeddings)
        
        if len(texts) == 1:
            result = result[0]
        
        return result
    
    async def aencode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """异步将文本编码为嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 嵌入向量
        """
        import asyncio
        
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = batch_size or self.batch_size
        
        async def process_batch(batch_texts: List[str]) -> np.ndarray:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.encode, batch_texts
            )
        
        tasks = [
            process_batch(texts[i : i + batch_size])
            for i in range(0, len(texts), batch_size)
        ]
        
        results = await asyncio.gather(*tasks)
        result = np.vstack(results)
        
        if len(texts) == 1:
            result = result[0]
        
        return result
    
    def encode_query(self, query: str, **kwargs: Any) -> np.ndarray:
        """编码查询文本
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 查询的嵌入向量
        """
        return self.encode([query], **kwargs)[0]
    
    def encode_queries(
        self,
        queries: List[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """批量编码查询文本
        
        Args:
            queries: 查询文本列表
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 查询嵌入向量矩阵
        """
        return self.encode(queries, **kwargs)
    
    def encode_passages(
        self,
        passages: List[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """编码文档片段
        
        Args:
            passages: 文档片段列表
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 文档片段的嵌入向量
        """
        return self.encode(passages, **kwargs)
    
    def get_token_count(self, text: str) -> int:
        """计算文本的 token 数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: token 数量
        """
        if self._tokenizer is None:
            return len(text) // 4
        
        tokens = self._tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    
    @classmethod
    def from_settings(cls, **overrides) -> "BGEEmbedding":
        """从配置创建实例
        
        从全局配置加载参数，并可覆盖部分参数。
        
        Args:
            **overrides: 要覆盖的配置参数
            
        Returns:
            BGEEmbedding: 实例对象
        """
        settings = get_settings()
        bge_config = settings.embedding.bge
        
        return cls(
            model_name=overrides.get("model_name", bge_config.model_name),
            device=overrides.get("device", bge_config.device),
            batch_size=overrides.get("batch_size", bge_config.batch_size),
            dimension=overrides.get("dimension", bge_config.dimension),
            normalize_embeddings=overrides.get(
                "normalize_embeddings", bge_config.normalize_embeddings
            ),
            query_instruction=overrides.get(
                "query_instruction", bge_config.query_instruction
            ),
            passage_instruction=overrides.get(
                "passage_instruction", bge_config.passage_instruction
            ),
            **overrides,
        )
