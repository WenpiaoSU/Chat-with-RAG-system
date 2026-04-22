# -*- coding: utf-8 -*-
"""
Embedding 抽象基类

定义 Embedding 接口规范，所有具体 Embedding 实现需继承此类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np


@dataclass
class EmbeddingResult:
    """Embedding 结果数据结构
    
    Attributes:
        embeddings: 嵌入向量列表，每个元素对应一个输入文本
        model: 使用的模型名称
        dimension: 嵌入向量的维度
        token_usage: Token 使用统计（如适用）
    """
    embeddings: List[np.ndarray]
    model: str
    dimension: int
    token_usage: Optional[dict] = None


class BaseEmbedding(ABC):
    """Embedding 抽象基类
    
    定义 Embedding 的标准接口，所有具体实现（如 BGE、OpenAI、Sentence-Transformers）
    都应继承此类并实现相应的抽象方法。
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dimension: int = 1024,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        """初始化 Embedding 模型
        
        Args:
            model_name: 模型名称
            device: 运行设备，cuda/cpu
            dimension: 嵌入向量维度
            batch_size: 批处理大小
            normalize_embeddings: 是否对嵌入向量进行L2归一化
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.device = device
        self.dimension = dimension
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._extra_params = kwargs
        self._model = None
    
    @abstractmethod
    def load_model(self) -> Any:
        """加载模型
        
        Returns:
            Any: 模型实例
        """
        pass
    
    @abstractmethod
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
            np.ndarray: 嵌入向量，形状为 (n, dimension) 或 (dimension,)
        """
        pass
    
    @abstractmethod
    async def aencode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """异步将文本编码为嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小，若为 None 则使用默认值
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 嵌入向量
        """
        pass
    
    def encode_query(self, query: str, **kwargs: Any) -> np.ndarray:
        """编码查询文本
        
        针对检索场景优化的查询编码，可能需要添加指令前缀。
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 查询的嵌入向量
        """
        return self.encode(query, **kwargs)
    
    def encode_passages(
        self,
        passages: List[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """编码文档片段
        
        针对检索场景优化的文档编码。
        
        Args:
            passages: 文档片段列表
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 文档片段的嵌入向量
        """
        return self.encode(passages, **kwargs)
    
    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """计算两组嵌入向量的相似度
        
        支持点积和余弦相似度（当向量已归一化时等价）。
        
        Args:
            embeddings1: 第一组嵌入向量，形状 (n, dimension) 或 (dimension,)
            embeddings2: 第二组嵌入向量，形状 (m, dimension) 或 (dimension,)
            
        Returns:
            np.ndarray: 相似度矩阵，形状 (n, m)
        """
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        if self.normalize_embeddings:
            return np.dot(embeddings1, embeddings2.T)
        else:
            norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
            return np.dot(embeddings1, embeddings2.T) / (norm1 * norm2.T + 1e-8)
    
    def batch_encode(
        self,
        texts_list: List[List[str]],
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """批量编码多组文本
        
        Args:
            texts_list: 文本列表的列表
            **kwargs: 其他参数
            
        Returns:
            List[np.ndarray]: 每组文本的嵌入向量列表
        """
        return [self.encode(texts, **kwargs) for texts in texts_list]
    
    def get_dimension(self) -> int:
        """获取嵌入向量维度
        
        Returns:
            int: 嵌入向量维度
        """
        return self.dimension
    
    def to(self, device: str) -> "BaseEmbedding":
        """移动模型到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            BaseEmbedding: 返回自身以支持链式调用
        """
        self.device = device
        return self
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, "
            f"dimension={self.dimension}, "
            f"device={self.device})"
        )
