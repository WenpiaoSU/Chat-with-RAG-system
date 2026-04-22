# -*- coding: utf-8 -*-
"""
评估模块

提供 RAG 系统的完整评估功能，包括：
- Ragas 评估框架集成
- 测试集自动生成
- 多维度评估指标
- 评估报告生成
"""

from .evaluator import RAGEvaluator, EvaluationResult
from .metrics import (
    EvaluationMetrics,
    MetricType,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextRecallMetric,
    ContextPrecisionMetric,
)
from .ragas_eval import RagasEvaluator
from .reporter import EvaluationReporter, ReportFormat
from .testset_generator import TestSetGenerator, TestSample

__all__ = [
    # 主评估器
    "RAGEvaluator",
    "EvaluationResult",
    # 指标相关
    "EvaluationMetrics",
    "MetricType",
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "ContextRecallMetric",
    "ContextPrecisionMetric",
    # Ragas 评估器
    "RagasEvaluator",
    # 报告生成器
    "EvaluationReporter",
    "ReportFormat",
    # 测试集生成器
    "TestSetGenerator",
    "TestSample",
]
