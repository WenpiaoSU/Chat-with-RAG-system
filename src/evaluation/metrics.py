# -*- coding: utf-8 -*-
"""
评估指标定义

定义 RAG 评估所需的各种评估指标及其计算方法。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class MetricType(Enum):
    """评估指标类型"""
    
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"
    RESPONSE_TIME = "response_time"
    RETRIEVAL_HIT_RATE = "retrieval_hit_rate"


@dataclass
class MetricResult:
    """单条指标评估结果
    
    Attributes:
        metric_name: 指标名称
        score: 得分 (0-1)
        std: 标准差（用于多次评估）
        details: 详细结果信息
        metadata: 元数据
    """
    metric_name: str
    score: float
    std: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """验证得分范围"""
        self.score = max(0.0, min(1.0, self.score))
        self.std = max(0.0, self.std)


@dataclass
class EvaluationMetrics:
    """评估指标集合
    
    封装多个评估指标的计算和聚合。
    
    Attributes:
        faithfulness: 忠实度指标
        answer_relevancy: 答案相关性指标
        context_recall: 上下文召回率
        context_precision: 上下文精确度
        response_time: 响应时间
        retrieval_hit_rate: 检索命中率
    """
    faithfulness: Optional[MetricResult] = None
    answer_relevancy: Optional[MetricResult] = None
    context_recall: Optional[MetricResult] = None
    context_precision: Optional[MetricResult] = None
    response_time: Optional[MetricResult] = None
    retrieval_hit_rate: Optional[MetricResult] = None
    
    def get_score(self, metric_type: MetricType) -> Optional[float]:
        """获取指定指标类型的得分
        
        Args:
            metric_type: 指标类型
            
        Returns:
            Optional[float]: 指标得分，不存在则返回 None
        """
        metric_map = {
            MetricType.FAITHFULNESS: self.faithfulness,
            MetricType.ANSWER_RELEVANCY: self.answer_relevancy,
            MetricType.CONTEXT_RECALL: self.context_recall,
            MetricType.CONTEXT_PRECISION: self.context_precision,
            MetricType.RESPONSE_TIME: self.response_time,
            MetricType.RETRIEVAL_HIT_RATE: self.retrieval_hit_rate,
        }
        result = metric_map.get(metric_type)
        return result.score if result else None
    
    def to_dict(self) -> Dict[str, float]:
        """将所有指标转换为字典
        
        Returns:
            Dict[str, float]: 指标名称到得分的映射
        """
        result = {}
        if self.faithfulness:
            result["faithfulness"] = self.faithfulness.score
        if self.answer_relevancy:
            result["answer_relevancy"] = self.answer_relevancy.score
        if self.context_recall:
            result["context_recall"] = self.context_recall.score
        if self.context_precision:
            result["context_precision"] = self.context_precision.score
        if self.response_time:
            result["response_time"] = self.response_time.score
        if self.retrieval_hit_rate:
            result["retrieval_hit_rate"] = self.retrieval_hit_rate.score
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationMetrics":
        """从字典创建评估指标集合
        
        Args:
            data: 包含指标数据的字典
            
        Returns:
            EvaluationMetrics: 评估指标实例
        """
        metrics = cls()
        for key, value in data.items():
            if isinstance(value, (int, float)):
                setattr(metrics, key, MetricResult(
                    metric_name=key,
                    score=float(value)
                ))
        return metrics
    
    def get_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """计算加权总体得分
        
        Args:
            weights: 各指标权重，不提供则平均分配
            
        Returns:
            float: 加权平均得分
        """
        scores = self.to_dict()
        if not scores:
            return 0.0
        
        if weights is None:
            return np.mean(list(scores.values()))
        
        total_weight = 0.0
        weighted_sum = 0.0
        for metric_name, weight in weights.items():
            if metric_name in scores:
                weighted_sum += scores[metric_name] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class BaseMetric:
    """评估指标基类
    
    所有评估指标应继承此类并实现 calculate 方法。
    """
    
    def __init__(self, name: str, description: str = "") -> None:
        """初始化评估指标
        
        Args:
            name: 指标名称
            description: 指标描述
        """
        self.name = name
        self.description = description
    
    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """计算指标得分
        
        Args:
            question: 用户问题
            answer: 生成的答案
            contexts: 检索到的上下文
            ground_truth: 标准答案（可选）
            **kwargs: 其他参数
            
        Returns:
            MetricResult: 指标计算结果
        """
        raise NotImplementedError("子类必须实现 calculate 方法")


class FaithfulnessMetric(BaseMetric):
    """忠实度指标
    
    衡量生成答案对上下文的忠实程度。
    得分越高表示答案越少产生幻觉，越依赖提供的上下文。
    
    计算方式：
    1. 将答案分解为独立的陈述
    2. 判断每个陈述是否可以从上下文中推断出来
    3. 得分 = 可推断的陈述数 / 总陈述数
    """
    
    def __init__(self) -> None:
        super().__init__(
            name="faithfulness",
            description="答案对上下文的忠实度"
        )
    
    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """计算忠实度得分
        
        使用 LLM 判断答案中的每个陈述是否可以从上下文中推断。
        """
        from .ragas_eval import get_ragas_evaluator
        
        evaluator = get_ragas_evaluator()
        
        context_combined = "\n".join(contexts) if contexts else ""
        
        try:
            score = evaluator.evaluate_faithfulness(
                question=question,
                answer=answer,
                context=context_combined,
            )
        except Exception:
            score = self._simple_calculate(contexts, answer)
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "context_length": len(contexts),
                "answer_length": len(answer),
            },
        )
    
    def _simple_calculate(self, contexts: List[str], answer: str) -> float:
        """简化的忠实度计算（不依赖 LLM）
        
        通过关键词重叠度近似计算忠实度。
        """
        if not contexts or not answer:
            return 0.0
        
        context_text = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words & context_words)
        return min(1.0, overlap / len(answer_words) * 1.5)


class AnswerRelevancyMetric(BaseMetric):
    """答案相关性指标
    
    衡量生成答案与问题的相关程度。
    得分越高表示答案越直接回答问题，内容越聚焦。
    
    计算方式：
    1. 使用 LLM 生成多个与答案相关的问题
    2. 计算原始问题与生成问题的相似度
    3. 得分 = 平均相似度
    """
    
    def __init__(self) -> None:
        super().__init__(
            name="answer_relevancy",
            description="答案与问题的相关性"
        )
    
    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """计算答案相关性得分"""
        from .ragas_eval import get_ragas_evaluator
        
        evaluator = get_ragas_evaluator()
        
        try:
            score = evaluator.evaluate_answer_relevancy(
                question=question,
                answer=answer,
            )
        except Exception:
            score = self._simple_calculate(question, answer)
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "question_length": len(question),
                "answer_length": len(answer),
            },
        )
    
    def _simple_calculate(self, question: str, answer: str) -> float:
        """简化的相关性计算"""
        if not question or not answer:
            return 0.0
        
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        
        if not q_words:
            return 0.0
        
        overlap = len(q_words & a_words)
        return min(1.0, overlap / len(q_words) * 2.0)


class ContextRecallMetric(BaseMetric):
    """上下文召回率指标
    
    衡量检索到的上下文包含标准答案信息的程度。
    得分越高表示检索越全面，遗漏的关键信息越少。
    
    计算方式：
    1. 将标准答案分解为关键信息点
    2. 检查每个信息点是否出现在检索到的上下文中
    3. 得分 = 出现在上下文中的信息点数 / 总信息点数
    """
    
    def __init__(self) -> None:
        super().__init__(
            name="context_recall",
            description="上下文召回率"
        )
    
    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """计算上下文召回率得分"""
        from .ragas_eval import get_ragas_evaluator
        
        evaluator = get_ragas_evaluator()
        
        context_combined = "\n".join(contexts) if contexts else ""
        
        try:
            score = evaluator.evaluate_context_recall(
                ground_truth=ground_truth or answer,
                context=context_combined,
            )
        except Exception:
            score = self._simple_calculate(contexts, ground_truth or answer)
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "context_count": len(contexts),
                "context_length": len(context_combined),
            },
        )
    
    def _simple_calculate(self, contexts: List[str], ground_truth: str) -> float:
        """简化的召回率计算"""
        if not contexts or not ground_truth:
            return 0.0
        
        context_text = " ".join(contexts).lower()
        gt_words = set(ground_truth.lower().split())
        context_words = set(context_text.split())
        
        if not gt_words:
            return 0.0
        
        overlap = len(gt_words & context_words)
        return overlap / len(gt_words)


class ContextPrecisionMetric(BaseMetric):
    """上下文精确度指标
    
    衡量检索到的上下文中相关内容的排名和质量。
    得分越高表示越相关的内容排在越前面。
    
    计算方式：
    1. 判断上下文中每个片段的相关性
    2. 根据相关性计算加权排名得分
    3. 得分 = 相关片段排名分数的平均值
    """
    
    def __init__(self) -> None:
        super().__init__(
            name="context_precision",
            description="上下文精确度"
        )
    
    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        relevance_scores: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """计算上下文精确度得分"""
        if relevance_scores is None:
            relevance_scores = self._estimate_relevance(question, contexts)
        
        if not relevance_scores or not contexts:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
            )
        
        n = len(contexts)
        precision_scores = []
        
        for i, rel in enumerate(relevance_scores):
            if rel > 0.5:
                precision_scores.append(rel / (i + 1))
        
        score = np.mean(precision_scores) if precision_scores else 0.0
        
        return MetricResult(
            metric_name=self.name,
            score=min(1.0, score),
            details={
                "relevance_scores": relevance_scores,
                "relevant_count": len(precision_scores),
            },
        )
    
    def _estimate_relevance(
        self,
        question: str,
        contexts: List[str],
    ) -> List[float]:
        """估计上下文与问题的相关性"""
        q_words = set(question.lower().split())
        scores = []
        
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            if not ctx_words:
                scores.append(0.0)
                continue
            
            overlap = len(q_words & ctx_words)
            score = min(1.0, overlap / max(len(q_words), 1))
            scores.append(score)
        
        return scores


class RetrievalHitRateMetric(BaseMetric):
    """检索命中率指标
    
    衡量检索系统能否找到包含正确答案的文档。
    
    计算方式：
    1. 检查 top-k 检索结果中是否包含标准答案相关的内容
    2. 得分 = 命中的查询数 / 总查询数
    """
    
    def __init__(self) -> None:
        super().__init__(
            name="retrieval_hit_rate",
            description="检索命中率"
        )
    
    def calculate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """计算检索命中率"""
        if not contexts or not ground_truth:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
            )
        
        gt_words = set((ground_truth or answer).lower().split())
        hit = False
        
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            if len(gt_words & ctx_words) >= len(gt_words) * 0.5:
                hit = True
                break
        
        return MetricResult(
            metric_name=self.name,
            score=1.0 if hit else 0.0,
            details={"hit": hit},
        )


def get_metric(metric_type: MetricType) -> BaseMetric:
    """获取指定类型的评估指标实例
    
    Args:
        metric_type: 指标类型
        
    Returns:
        BaseMetric: 指标实例
    """
    metric_map = {
        MetricType.FAITHFULNESS: FaithfulnessMetric,
        MetricType.ANSWER_RELEVANCY: AnswerRelevancyMetric,
        MetricType.CONTEXT_RECALL: ContextRecallMetric,
        MetricType.CONTEXT_PRECISION: ContextPrecisionMetric,
        MetricType.RETRIEVAL_HIT_RATE: RetrievalHitRateMetric,
    }
    
    metric_class = metric_map.get(metric_type)
    if metric_class is None:
        raise ValueError(f"不支持的指标类型: {metric_type}")
    
    return metric_class()
