# -*- coding: utf-8 -*-
"""
Ragas 评估器实现

基于 Ragas 框架的 RAG 评估器实现。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..configs.settings import get_settings
from ..llm.base import BaseLLM, LLMResponse
from ..llm.openai_llm import OpenAILLM

logger = logging.getLogger(__name__)

_ragas_evaluator_instance: Optional["RagasEvaluator"] = None


@dataclass
class RagasScore:
    """Ragas 评估得分
    
    Attributes:
        score: 评估得分
        std: 标准差
        details: 详细结果
    """
    score: float
    std: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "score": self.score,
            "std": self.std,
            "details": self.details,
        }


class RagasEvaluator:
    """Ragas 评估器
    
    基于 Ragas 框架实现 RAG 系统评估。
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        embeddings: Optional[Any] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """初始化 Ragas 评估器
        
        Args:
            llm: 评估用大语言模型，默认使用 OpenAI
            embeddings: 评估用嵌入模型
            timeout: 评估超时时间（秒）
            max_retries: 最大重试次数
        """
        self._llm = llm
        self._embeddings = embeddings
        self.timeout = timeout
        self.max_retries = max_retries
        self._initialize_ragas()
    
    def _initialize_ragas(self) -> None:
        """初始化 Ragas 框架"""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            )
            self._ragas_evaluate = evaluate
            self._ragas_metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_recall": context_recall,
                "context_precision": context_precision,
            }
            logger.info("Ragas 框架初始化成功")
        except ImportError:
            logger.warning("Ragas 未安装，将使用简化评估方法")
            self._ragas_evaluate = None
            self._ragas_metrics = {}
    
    @property
    def llm(self) -> BaseLLM:
        """获取 LLM 实例"""
        if self._llm is None:
            self._llm = OpenAILLM.from_settings()
        return self._llm
    
    @llm.setter
    def llm(self, value: BaseLLM) -> None:
        """设置 LLM 实例"""
        self._llm = value
    
    @classmethod
    def from_settings(cls) -> "RagasEvaluator":
        """从配置文件创建设估器
        
        Returns:
            RagasEvaluator: 评估器实例
        """
        settings = get_settings()
        
        llm: Optional[BaseLLM] = None
        if settings.llm.provider == "openai":
            llm = OpenAILLM(
                model_name=settings.llm.openai.model,
                api_key=settings.llm.openai.api_key or None,
                api_base=settings.llm.openai.api_base,
                temperature=0.0,
                max_tokens=2000,
            )
        
        return cls(
            llm=llm,
            timeout=60,
        )
    
    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> RagasScore:
        """评估忠实度
        
        衡量答案对上下文的忠实程度。
        
        Args:
            question: 用户问题
            answer: 生成的回答
            context: 检索到的上下文
            
        Returns:
            RagasScore: 忠实度得分
        """
        if self._ragas_evaluate is not None:
            return self._ragas_evaluate_with_ragas(
                question=question,
                answer=answer,
                contexts=[context],
                metrics=["faithfulness"],
            )
        
        return self._simple_faithfulness(question, answer, context)
    
    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> RagasScore:
        """评估答案相关性
        
        衡量答案与问题的相关程度。
        
        Args:
            question: 用户问题
            answer: 生成的回答
            
        Returns:
            RagasScore: 相关性得分
        """
        if self._ragas_evaluate is not None:
            return self._ragas_evaluate_with_ragas(
                question=question,
                answer=answer,
                contexts=[],
                metrics=["answer_relevancy"],
            )
        
        return self._simple_relevancy(question, answer)
    
    def evaluate_context_recall(
        self,
        ground_truth: str,
        context: str,
    ) -> RagasScore:
        """评估上下文召回率
        
        衡量上下文对标准答案的召回程度。
        
        Args:
            ground_truth: 标准答案
            context: 检索到的上下文
            
        Returns:
            RagasScore: 召回率得分
        """
        if self._ragas_evaluate is not None:
            return self._ragas_evaluate_with_ragas(
                question="",
                answer="",
                contexts=[context],
                ground_truths=[ground_truth],
                metrics=["context_recall"],
            )
        
        return self._simple_recall(ground_truth, context)
    
    def evaluate_context_precision(
        self,
        question: str,
        contexts: List[str],
    ) -> RagasScore:
        """评估上下文精确度
        
        衡量上下文中相关内容的排名质量。
        
        Args:
            question: 用户问题
            contexts: 检索到的上下文列表
            
        Returns:
            RagasScore: 精确度得分
        """
        if self._ragas_evaluate is not None:
            return self._ragas_evaluate_with_ragas(
                question=question,
                answer="",
                contexts=contexts,
                metrics=["context_precision"],
            )
        
        return self._simple_precision(question, contexts)
    
    def _ragas_evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> RagasScore:
        """使用 Ragas 框架进行评估
        
        Args:
            question: 用户问题
            answer: 生成的回答
            contexts: 上下文列表
            ground_truths: 标准答案列表
            metrics: 要评估的指标列表
            
        Returns:
            RagasScore: 评估得分
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
            
            eval_data = {
                "user_input": [question],
                "response": [answer],
                "contexts": [contexts],
            }
            if ground_truths:
                eval_data["ground_truth"] = ground_truths
            
            metric_objs = []
            for m in (metrics or ["faithfulness"]):
                if m == "faithfulness":
                    metric_objs.append(faithfulness)
                elif m == "answer_relevancy":
                    metric_objs.append(answer_relevancy)
                elif m == "context_recall":
                    metric_objs.append(context_recall)
                elif m == "context_precision":
                    metric_objs.append(context_precision)
            
            dataset = Dataset.from_dict(eval_data)
            result = evaluate(dataset, metrics=metric_objs)
            
            scores = {}
            for m in metrics or ["faithfulness"]:
                key = f"{m}_"
                if key in result.scores:
                    scores[m] = result.scores[key][0]
            
            if scores:
                main_score = list(scores.values())[0]
                return RagasScore(
                    score=main_score,
                    std=0.0,
                    details={"ragas_scores": scores},
                )
            
            return RagasScore(score=0.0)
            
        except Exception as e:
            logger.error(f"Ragas 评估失败: {e}")
            return RagasScore(score=0.0, details={"error": str(e)})
    
    def _simple_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> RagasScore:
        """简化忠实度评估
        
        通过分析答案与上下文的语义重叠来评估忠实度。
        """
        if not context or not answer:
            return RagasScore(score=0.0)
        
        context_lower = context.lower()
        answer_sentences = self._split_sentences(answer)
        
        if not answer_sentences:
            return RagasScore(score=0.0)
        
        faithful_count = 0
        for sentence in answer_sentences:
            sentence_words = set(sentence.lower().split())
            context_words = set(context_lower.split())
            
            overlap_ratio = len(sentence_words & context_words) / max(len(sentence_words), 1)
            if overlap_ratio > 0.3:
                faithful_count += 1
        
        score = faithful_count / len(answer_sentences)
        return RagasScore(score=score)
    
    def _simple_relevancy(
        self,
        question: str,
        answer: str,
    ) -> RagasScore:
        """简化相关性评估
        
        通过分析问题与答案的语义重叠来评估相关性。
        """
        if not question or not answer:
            return RagasScore(score=0.0)
        
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        
        if not q_words:
            return RagasScore(score=0.0)
        
        overlap = len(q_words & a_words)
        score = min(1.0, overlap / len(q_words) * 2.0)
        
        return RagasScore(score=score)
    
    def _simple_recall(
        self,
        ground_truth: str,
        context: str,
    ) -> RagasScore:
        """简化召回率评估"""
        if not ground_truth or not context:
            return RagasScore(score=0.0)
        
        gt_words = set(ground_truth.lower().split())
        ctx_words = set(context.lower().split())
        
        overlap = len(gt_words & ctx_words)
        score = overlap / len(gt_words) if gt_words else 0.0
        
        return RagasScore(score=score)
    
    def _simple_precision(
        self,
        question: str,
        contexts: List[str],
    ) -> RagasScore:
        """简化精确度评估"""
        if not contexts or not question:
            return RagasScore(score=0.0)
        
        q_words = set(question.lower().split())
        precision_scores = []
        
        for i, ctx in enumerate(contexts):
            ctx_words = set(ctx.lower().split())
            if q_words & ctx_words:
                precision_scores.append(1.0 / (i + 1))
        
        score = np.mean(precision_scores) if precision_scores else 0.0
        return RagasScore(score=min(1.0, score * len(contexts)))
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """将文本分割为句子"""
        import re
        sentence_delimiters = re.compile(r'[。！？.!?]+')
        sentences = sentence_delimiters.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def batch_evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, List[RagasScore]]:
        """批量评估
        
        Args:
            questions: 问题列表
            answers: 答案列表
            contexts_list: 上下文列表的列表
            ground_truths: 标准答案列表
            metrics: 要评估的指标列表
            
        Returns:
            Dict[str, List[RagasScore]]: 各指标的评估结果列表
        """
        metrics = metrics or ["faithfulness", "answer_relevancy"]
        results: Dict[str, List[RagasScore]] = {m: [] for m in metrics}
        
        for i in range(len(questions)):
            question = questions[i]
            answer = answers[i]
            contexts = contexts_list[i] if i < len(contexts_list) else []
            context_combined = "\n".join(contexts)
            ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            
            for metric in metrics:
                if metric == "faithfulness":
                    results[metric].append(self.evaluate_faithfulness(
                        question, answer, context_combined
                    ))
                elif metric == "answer_relevancy":
                    results[metric].append(self.evaluate_answer_relevancy(
                        question, answer
                    ))
                elif metric == "context_recall" and ground_truth:
                    results[metric].append(self.evaluate_context_recall(
                        ground_truth, context_combined
                    ))
                elif metric == "context_precision":
                    results[metric].append(self.evaluate_context_precision(
                        question, contexts
                    ))
        
        return results


def get_ragas_evaluator() -> RagasEvaluator:
    """获取全局 Ragas 评估器实例
    
    Returns:
        RagasEvaluator: 评估器实例
    """
    global _ragas_evaluator_instance
    if _ragas_evaluator_instance is None:
        _ragas_evaluator_instance = RagasEvaluator.from_settings()
    return _ragas_evaluator_instance


def set_ragas_evaluator(evaluator: RagasEvaluator) -> None:
    """设置全局 Ragas 评估器实例
    
    Args:
        evaluator: 评估器实例
    """
    global _ragas_evaluator_instance
    _ragas_evaluator_instance = evaluator
