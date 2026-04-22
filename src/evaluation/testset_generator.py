# -*- coding: utf-8 -*-
"""
测试集生成器

基于 LLM 自动生成评估用测试集。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json

from ..configs.settings import get_settings
from ..llm.base import BaseLLM
from ..llm.openai_llm import OpenAILLM
from ..data_process.loaders.loader_factory import LoaderFactory
from ..data_process.splitters.recursive_splitter import RecursiveTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class TestSample:
    """测试样本
    
    Attributes:
        question: 用户问题
        answer: 标准答案
        ground_truth_context: 答案来源的上下文
        metadata: 元数据
    """
    question: str
    answer: str
    ground_truth_context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "question": self.question,
            "answer": self.answer,
            "ground_truth_context": self.ground_truth_context,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestSample":
        """从字典创建"""
        return cls(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            ground_truth_context=data.get("ground_truth_context", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QuestionType:
    """问题类型定义"""
    FACTUAL = "fact"       # 事实性问题
    UNDERSTANDING = "understand"  # 理解性问题
    ANALYSIS = "analyze"   # 分析性问题
    SUMMARY = "summarize"  # 总结性问题


class TestSetGenerator:
    """测试集生成器
    
    基于 LLM 从文档中自动生成问答测试集。
    支持多种问题类型和难度级别。
    
    Attributes:
        llm: 用于生成的大语言模型
        num_pairs: 生成的问答对数量
        max_context_length: 最大上下文长度
        question_types: 支持的问题类型
        
    Example:
        >>> generator = TestSetGenerator.from_settings()
        >>> samples = generator.generate_from_documents(
        ...     file_paths=["doc1.pdf", "doc2.md"],
        ...     num_pairs=10
        ... )
        >>> for sample in samples:
        ...     print(f"Q: {sample.question}")
        ...     print(f"A: {sample.answer}")
    """
    
    DEFAULT_QUESTION_TYPES = [
        QuestionType.FACTUAL,
        QuestionType.UNDERSTANDING,
        QuestionType.ANALYSIS,
        QuestionType.SUMMARY,
    ]
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        num_pairs: int = 10,
        max_context_length: int = 8000,
        question_types: Optional[List[str]] = None,
    ) -> None:
        """初始化测试集生成器
        
        Args:
            llm: 大语言模型
            num_pairs: 默认生成的问答对数量
            max_context_length: 最大上下文长度
            question_types: 问题类型列表
        """
        self.llm = llm
        self.num_pairs = num_pairs
        self.max_context_length = max_context_length
        self.question_types = question_types or self.DEFAULT_QUESTION_TYPES
        self._splitter = RecursiveTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )
    
    @property
    def generator_llm(self) -> BaseLLM:
        """获取 LLM 实例"""
        if self.llm is None:
            self.llm = OpenAILLM.from_settings()
        return self.llm
    
    @classmethod
    def from_settings(cls) -> "TestSetGenerator":
        """从配置文件创建生成器
        
        Returns:
            TestSetGenerator: 生成器实例
        """
        settings = get_settings()
        
        llm: Optional[BaseLLM] = None
        if settings.llm.provider == "openai":
            llm = OpenAILLM(
                model_name=settings.llm.openai.model,
                api_key=settings.llm.openai.api_key or None,
                api_base=settings.llm.openai.api_base,
                temperature=0.7,
                max_tokens=2000,
            )
        
        testset_config = settings.evaluation.testset
        
        return cls(
            llm=llm,
            num_pairs=testset_config.get("default_num_pairs", 10),
            max_context_length=testset_config.get("max_context_length", 8000),
        )
    
    def generate_from_documents(
        self,
        file_paths: List[str],
        num_pairs: Optional[int] = None,
        question_types: Optional[List[str]] = None,
    ) -> List[TestSample]:
        """从文档文件生成测试集
        
        Args:
            file_paths: 文档文件路径列表
            num_pairs: 生成的问答对数量
            question_types: 问题类型列表
            
        Returns:
            List[TestSample]: 测试样本列表
        """
        chunks = self._load_and_split_documents(file_paths)
        
        if not chunks:
            logger.warning("未从文档中提取到有效内容")
            return []
        
        num_pairs = num_pairs or self.num_pairs
        chunks_per_sample = max(1, len(chunks) // num_pairs)
        
        samples = []
        for i in range(0, len(chunks), chunks_per_sample):
            context_parts = chunks[i:i + chunks_per_sample]
            context = "\n".join(context_parts)
            
            if len(context) > self.max_context_length:
                context = context[:self.max_context_length]
            
            sample = self._generate_sample(
                context=context,
                question_types=question_types or self.question_types,
            )
            
            if sample:
                sample.ground_truth_context = context
                samples.append(sample)
            
            if len(samples) >= num_pairs:
                break
        
        return samples
    
    def generate_from_contexts(
        self,
        contexts: List[str],
        num_pairs: Optional[int] = None,
        ground_truths: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
    ) -> List[TestSample]:
        """从上下文列表生成测试集
        
        Args:
            contexts: 上下文列表
            num_pairs: 生成的问答对数量
            ground_truths: 标准答案列表
            question_types: 问题类型列表
            
        Returns:
            List[TestSample]: 测试样本列表
        """
        num_pairs = num_pairs or self.num_pairs
        ground_truths = ground_truths or [""]
        
        samples = []
        for i, context in enumerate(contexts):
            if len(context) > self.max_context_length:
                context = context[:self.max_context_length]
            
            ground_truth = ground_truths[i] if i < len(ground_truths) else context
            
            sample = self._generate_sample(
                context=context,
                ground_truth=ground_truth,
                question_types=question_types or self.question_types,
            )
            
            if sample:
                sample.ground_truth_context = context
                samples.append(sample)
            
            if len(samples) >= num_pairs:
                break
        
        return samples
    
    def _generate_sample(
        self,
        context: str,
        ground_truth: str = "",
        question_types: Optional[List[str]] = None,
    ) -> Optional[TestSample]:
        """生成单个测试样本
        
        Args:
            context: 上下文内容
            ground_truth: 标准答案
            question_types: 问题类型
            
        Returns:
            Optional[TestSample]: 测试样本，生成失败返回 None
        """
        question_types = question_types or self.question_types
        
        prompt = self._build_generation_prompt(
            context=context,
            ground_truth=ground_truth,
            question_types=question_types,
        )
        
        try:
            response = self.generator_llm.invoke(prompt)
            content = response.content
            
            return self._parse_llm_response(content, ground_truth)
            
        except Exception as e:
            logger.error(f"生成测试样本失败: {e}")
            return None
    
    def _build_generation_prompt(
        self,
        context: str,
        ground_truth: str,
        question_types: List[str],
    ) -> str:
        """构建生成提示词
        
        Args:
            context: 上下文内容
            ground_truth: 标准答案
            question_types: 问题类型
            
        Returns:
            str: 提示词
        """
        type_descriptions = {
            QuestionType.FACTUAL: "事实性问题：询问具体的知识点、定义或数据",
            QuestionType.UNDERSTANDING: "理解性问题：要求解释概念或原理",
            QuestionType.ANALYSIS: "分析性问题：要求分析原因、影响或比较",
            QuestionType.SUMMARY: "总结性问题：要求总结要点或概括内容",
        }
        
        type_list = "\n".join([
            f"- {t}: {type_descriptions.get(t, '')}"
            for t in question_types
        ])
        
        gt_section = f"\n参考标准答案：{ground_truth}" if ground_truth else ""
        
        prompt = f"""你是一个问答测试集生成专家。请根据以下文档内容生成一个问答对。

## 文档内容
{context}
{gt_section}

## 要求
1. 生成的问题必须能够从文档内容中找到答案
2. 问题应该清晰、具体，避免歧义
3. 答案应该准确、完整

## 问题类型（请选择最合适的类型）
{type_list}

## 输出格式
请以 JSON 格式输出，字段如下：
{{
    "question": "生成的问题",
    "answer": "对应的答案"
}}

只输出 JSON，不要有其他内容。"""
        
        return prompt
    
    def _parse_llm_response(
        self,
        content: str,
        ground_truth: str = "",
    ) -> Optional[TestSample]:
        """解析 LLM 响应
        
        Args:
            content: LLM 响应内容
            ground_truth: 备用标准答案
            
        Returns:
            Optional[TestSample]: 解析出的测试样本
        """
        content = content.strip()
        
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        
        try:
            data = json.loads(content)
            return TestSample(
                question=data.get("question", ""),
                answer=data.get("answer", ground_truth or ""),
                ground_truth_context="",
                metadata={"source": "llm_generated"},
            )
        except json.JSONDecodeError:
            logger.warning(f"JSON 解析失败: {content[:100]}")
            
            parts = content.split("?")
            if len(parts) >= 2:
                question = parts[0].replace("问题:", "").replace("Q:", "").strip() + "?"
                answer = parts[1].split("答案:")[-1].replace("A:", "").strip()
                
                return TestSample(
                    question=question,
                    answer=answer or ground_truth,
                    ground_truth_context="",
                    metadata={"source": "llm_generated_fallback"},
                )
            
            return None
    
    def _load_and_split_documents(self, file_paths: List[str]) -> List[str]:
        """加载并分割文档
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            List[str]: 文本块列表
        """
        all_chunks = []
        
        for file_path in file_paths:
            try:
                loader = LoaderFactory.create(file_path)
                documents = loader.load()
                
                for doc in documents:
                    chunks = self._splitter.split_text(doc.page_content)
                    all_chunks.extend(chunks)
                    
            except Exception as e:
                logger.warning(f"加载文件失败 {file_path}: {e}")
                continue
        
        return all_chunks
    
    def generate_batch(
        self,
        documents: List[Dict[str, Any]],
        num_pairs: int = 10,
    ) -> List[TestSample]:
        """批量生成测试集
        
        Args:
            documents: 文档列表，每个文档包含 content 和可选的 metadata
            num_pairs: 每个文档生成的问答对数量
            
        Returns:
            List[TestSample]: 测试样本列表
        """
        all_samples = []
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            if not content:
                continue
            
            if len(content) > self.max_context_length:
                chunks = self._splitter.split_text(content)
            else:
                chunks = [content]
            
            for chunk in chunks:
                sample = self._generate_sample(context=chunk)
                if sample:
                    sample.ground_truth_context = chunk
                    sample.metadata.update(metadata)
                    all_samples.append(sample)
                
                if len(all_samples) >= num_pairs:
                    return all_samples
        
        return all_samples
    
    def save_to_file(
        self,
        samples: List[TestSample],
        file_path: str,
    ) -> None:
        """保存测试集到文件
        
        Args:
            samples: 测试样本列表
            file_path: 输出文件路径
        """
        data = [sample.to_dict() for sample in samples]
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试集已保存到 {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> List[TestSample]:
        """从文件加载测试集
        
        Args:
            file_path: 测试集文件路径
            
        Returns:
            List[TestSample]: 测试样本列表
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return [TestSample.from_dict(item) for item in data]
