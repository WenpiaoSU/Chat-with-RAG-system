# -*- coding: utf-8 -*-
"""
RAG 评估器主类

提供 RAG 系统端到端评估功能。
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..configs.settings import get_settings
from ..llm.base import BaseLLM
from ..llm.openai_llm import OpenAILLM
from ..rag.rag_chain import RAGChain, RAGResponse
from .metrics import (
    EvaluationMetrics,
    MetricResult,
    MetricType,
    get_metric,
)
from .ragas_eval import RagasEvaluator, get_ragas_evaluator
from .reporter import (
    EvaluationReporter,
    EvaluationSummary,
    ReportFormat,
)
from .testset_generator import TestSample, TestSetGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """评估结果
    
    Attributes:
        metrics: 评估指标集合
        summary: 评估摘要
        detailed_results: 详细结果列表
        rag_chain: 使用的 RAG 链
        test_samples: 使用的测试样本
        evaluation_time: 评估耗时（秒）
    """
    metrics: EvaluationMetrics
    summary: Optional[EvaluationSummary] = None
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    rag_chain: Optional[RAGChain] = None
    test_samples: List[TestSample] = field(default_factory=list)
    evaluation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "metrics": self.metrics.to_dict(),
            "overall_score": self.metrics.get_overall_score(),
            "detailed_results": self.detailed_results,
            "total_samples": len(self.detailed_results),
            "evaluation_time": self.evaluation_time,
        }
    
    def get_metric_score(self, metric_type: MetricType) -> Optional[float]:
        """获取指定指标得分
        
        Args:
            metric_type: 指标类型
            
        Returns:
            Optional[float]: 指标得分
        """
        return self.metrics.get_score(metric_type)


@dataclass
class EvaluationConfig:
    """评估配置
    
    Attributes:
        metrics: 要评估的指标列表
        batch_size: 批处理大小
        timeout: 单次评估超时时间（秒）
        save_results: 是否保存评估结果
        report_format: 报告格式
    """
    metrics: List[str] = field(default_factory=lambda: [
        "faithfulness", "answer_relevancy", "context_recall"
    ])
    batch_size: int = 10
    timeout: int = 60
    save_results: bool = True
    report_format: ReportFormat = ReportFormat.MARKDOWN
    output_dir: str = "./data/evaluation"


class RAGEvaluator:
    """RAG 评估器
    
    提供 RAG 系统的端到端评估功能。
    """
    
    def __init__(
        self,
        rag_chain: Optional[RAGChain] = None,
        evaluator: Optional[RagasEvaluator] = None,
        config: Optional[EvaluationConfig] = None,
        llm: Optional[BaseLLM] = None,
    ) -> None:
        """初始化 RAG 评估器
        
        Args:
            rag_chain: RAG 链实例
            evaluator: Ragas 评估器
            config: 评估配置
            llm: 大语言模型（用于评估）
        """
        self._rag_chain = rag_chain
        self._evaluator = evaluator
        self._config = config or EvaluationConfig()
        self._llm = llm
        self._reporter = EvaluationReporter()
    
    @property
    def rag_chain(self) -> RAGChain:
        """获取 RAG 链实例"""
        if self._rag_chain is None:
            self._rag_chain = RAGChain.from_defaults()
        return self._rag_chain
    
    @rag_chain.setter
    def rag_chain(self, value: RAGChain) -> None:
        """设置 RAG 链实例"""
        self._rag_chain = value
    
    @property
    def evaluator(self) -> RagasEvaluator:
        """获取 Ragas 评估器"""
        if self._evaluator is None:
            self._evaluator = get_ragas_evaluator()
        return self._evaluator
    
    @property
    def llm(self) -> BaseLLM:
        """获取 LLM"""
        if self._llm is None:
            self._llm = OpenAILLM.from_settings()
        return self._llm
    
    @classmethod
    def from_settings(cls) -> "RAGEvaluator":
        """从配置文件创建评估器
        
        Returns:
            RAGEvaluator: 评估器实例
        """
        settings = get_settings()
        
        config = EvaluationConfig(
            metrics=settings.evaluation.metrics,
            batch_size=settings.evaluation.evaluator.batch_size,
            save_results=settings.evaluation.evaluator.save_results,
            report_format=ReportFormat(settings.evaluation.report.default_format),
            output_dir=settings.evaluation.evaluator.output_dir,
        )
        
        return cls(config=config)
    
    def evaluate_with_testset(
        self,
        test_samples: List[TestSample],
        metrics: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """使用测试集进行评估
        
        Args:
            test_samples: 测试样本列表
            metrics: 要评估的指标列表
            show_progress: 是否显示进度
            
        Returns:
            EvaluationResult: 评估结果
        """
        metrics = metrics or self._config.metrics
        start_time = time.time()
        
        all_metric_results: Dict[str, List[float]] = {m: [] for m in metrics}
        detailed_results = []
        
        total = len(test_samples)
        for i, sample in enumerate(test_samples):
            if show_progress and (i + 1) % 5 == 0:
                logger.info(f"评估进度: {i + 1}/{total}")
            
            result = self._evaluate_single_sample(sample, metrics)
            
            if result:
                for metric_name, score in result.items():
                    all_metric_results[metric_name].append(score)
                detailed_results.append({
                    "question": sample.question,
                    "answer": sample.answer,
                    "ground_truth": sample.ground_truth_context,
                    "metrics": result,
                })
        
        evaluation_time = time.time() - start_time
        
        eval_metrics = EvaluationMetrics(
            faithfulness=MetricResult(
                metric_name="faithfulness",
                score=np.mean(all_metric_results.get("faithfulness", [0])),
                std=np.std(all_metric_results.get("faithfulness", [0])),
            ) if "faithfulness" in all_metric_results else None,
            answer_relevancy=MetricResult(
                metric_name="answer_relevancy",
                score=np.mean(all_metric_results.get("answer_relevancy", [0])),
                std=np.std(all_metric_results.get("answer_relevancy", [0])),
            ) if "answer_relevancy" in all_metric_results else None,
            context_recall=MetricResult(
                metric_name="context_recall",
                score=np.mean(all_metric_results.get("context_recall", [0])),
                std=np.std(all_metric_results.get("context_recall", [0])),
            ) if "context_recall" in all_metric_results else None,
            context_precision=MetricResult(
                metric_name="context_precision",
                score=np.mean(all_metric_results.get("context_precision", [0])),
                std=np.std(all_metric_results.get("context_precision", [0])),
            ) if "context_precision" in all_metric_results else None,
        )
        
        return EvaluationResult(
            metrics=eval_metrics,
            detailed_results=detailed_results,
            test_samples=test_samples,
            evaluation_time=evaluation_time,
        )
    
    def _evaluate_single_sample(
        self,
        sample: TestSample,
        metrics: List[str],
    ) -> Optional[Dict[str, float]]:
        """评估单个样本
        
        Args:
            sample: 测试样本
            metrics: 指标列表
            
        Returns:
            Optional[Dict[str, float]]: 各指标得分
        """
        try:
            response = self.rag_chain.invoke(sample.question)
            
            contexts = [
                doc.content for doc in response.retrieved_docs
            ] if response.retrieved_docs else []
            
            scores = {}
            
            for metric_name in metrics:
                score = self._calculate_metric(
                    metric_name=metric_name,
                    question=sample.question,
                    answer=response.answer,
                    contexts=contexts,
                    ground_truth=sample.ground_truth_context or sample.answer,
                )
                scores[metric_name] = score
            
            return scores
            
        except Exception as e:
            logger.error(f"评估样本失败: {e}")
            return None
    
    def _calculate_metric(
        self,
        metric_name: str,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """计算单个指标
        
        Args:
            metric_name: 指标名称
            question: 问题
            answer: 答案
            contexts: 上下文
            ground_truth: 标准答案
            
        Returns:
            float: 指标得分
        """
        try:
            if metric_name == "faithfulness":
                result = self.evaluator.evaluate_faithfulness(
                    question=question,
                    answer=answer,
                    context="\n".join(contexts),
                )
                return result.score
            
            elif metric_name == "answer_relevancy":
                result = self.evaluator.evaluate_answer_relevancy(
                    question=question,
                    answer=answer,
                )
                return result.score
            
            elif metric_name == "context_recall":
                result = self.evaluator.evaluate_context_recall(
                    ground_truth=ground_truth,
                    context="\n".join(contexts),
                )
                return result.score
            
            elif metric_name == "context_precision":
                result = self.evaluator.evaluate_context_precision(
                    question=question,
                    contexts=contexts,
                )
                return result.score
            
            else:
                logger.warning(f"未知指标: {metric_name}")
                return 0.0
                
        except Exception as e:
            logger.error(f"计算指标 {metric_name} 失败: {e}")
            return 0.0
    
    def evaluate_documents(
        self,
        file_paths: List[str],
        num_test_pairs: Optional[int] = None,
        metrics: Optional[List[str]] = None,
        save_testset: bool = True,
        testset_path: Optional[str] = None,
    ) -> EvaluationResult:
        """从文档生成测试集并进行评估
        
        Args:
            file_paths: 文档文件路径列表
            num_test_pairs: 生成的测试对数量（默认使用配置值）
            metrics: 要评估的指标列表
            save_testset: 是否保存测试集
            testset_path: 测试集保存路径
            
        Returns:
            EvaluationResult: 评估结果
        """
        logger.info(f"正在从 {len(file_paths)} 个文档生成测试集...")
        
        generator = TestSetGenerator.from_settings()
        
        if num_test_pairs is None:
            num_test_pairs = generator.num_pairs
        
        test_samples = generator.generate_from_documents(
            file_paths=file_paths,
            num_pairs=num_test_pairs,
        )
        
        logger.info(f"生成了 {len(test_samples)} 个测试样本")
        
        if save_testset:
            output_path = testset_path or self._get_default_testset_path()
            generator.save_to_file(test_samples, output_path)
            logger.info(f"测试集已保存到 {output_path}")
        
        return self.evaluate_with_testset(
            test_samples=test_samples,
            metrics=metrics,
        )
    
    def evaluate_batch(
        self,
        questions: List[str],
        ground_truths: List[str],
        contexts_list: Optional[List[List[str]]] = None,
        metrics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """批量评估已有问答对
        
        Args:
            questions: 问题列表
            ground_truths: 标准答案列表
            contexts_list: 上下文列表（可选，不提供则自动检索）
            metrics: 要评估的指标列表
            
        Returns:
            EvaluationResult: 评估结果
        """
        test_samples = []
        
        for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
            contexts = (
                contexts_list[i] if contexts_list and i < len(contexts_list) else []
            )
            
            test_samples.append(TestSample(
                question=question,
                answer="",
                ground_truth_context=ground_truth,
                metadata={"contexts": contexts},
            ))
        
        return self.evaluate_with_testset(
            test_samples=test_samples,
            metrics=metrics,
        )
    
    def generate_report(
        self,
        result: EvaluationResult,
        format: Optional[ReportFormat] = None,
        output_path: Optional[str] = None,
        title: str = "RAG 评估报告",
    ) -> str:
        """生成评估报告
        
        Args:
            result: 评估结果
            format: 报告格式
            output_path: 输出路径
            title: 报告标题
            
        Returns:
            str: 报告内容
        """
        format = format or self._config.report_format
        
        reporter = EvaluationReporter(title=title)
        
        for detail in result.detailed_results:
            reporter.add_result(
                question=detail["question"],
                answer=detail.get("answer", ""),
                contexts=[],
                metrics=detail["metrics"],
                ground_truth=detail.get("ground_truth"),
            )
        
        report = reporter.generate(format=format, output_path=output_path)
        
        if output_path:
            logger.info(f"评估报告已保存到 {output_path}")
        
        return report
    
    def _get_default_testset_path(self) -> str:
        """获取默认测试集保存路径"""
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / "testset.json")
    
    def compare_configurations(
        self,
        configurations: List[Dict[str, Any]],
        test_samples: List[TestSample],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, EvaluationResult]:
        """对比不同配置的评估结果
        
        Args:
            configurations: 配置列表，每个配置包含参数
            test_samples: 测试样本
            metrics: 指标列表
            
        Returns:
            Dict[str, EvaluationResult]: 配置名称到评估结果的映射
        """
        results = {}
        
        for i, config in enumerate(configurations):
            logger.info(f"评估配置 {i + 1}/{len(configurations)}: {config}")
            
            config_name = config.get("name", f"config_{i + 1}")
            
            if "top_k" in config:
                self.rag_chain.top_k = config["top_k"]
            if "score_threshold" in config:
                self.rag_chain.score_threshold = config["score_threshold"]
            
            result = self.evaluate_with_testset(
                test_samples=test_samples,
                metrics=metrics,
                show_progress=False,
            )
            
            results[config_name] = result
        
        return results
    
    def save_results(
        self,
        result: EvaluationResult,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """保存评估结果
        
        Args:
            result: 评估结果
            output_dir: 输出目录
            
        Returns:
            Dict[str, str]: 文件路径映射
        """
        output_dir = output_dir or self._config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        paths = {}
        
        result_dict = result.to_dict()
        result_file = output_path / f"evaluation_results_{timestamp}.json"
        import json
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        paths["results"] = str(result_file)
        
        report_md = self.generate_report(
            result,
            format=ReportFormat.MARKDOWN,
        )
        report_file = output_path / f"evaluation_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        paths["report_md"] = str(report_file)
        
        try:
            report_html = self.generate_report(
                result,
                format=ReportFormat.HTML,
            )
            html_file = output_path / f"evaluation_report_{timestamp}.html"
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(report_html)
            paths["report_html"] = str(html_file)
        except Exception as e:
            logger.warning(f"生成 HTML 报告失败: {e}")
        
        return paths


def evaluate_rag_system(
    test_samples: List[TestSample],
    rag_chain: Optional[RAGChain] = None,
    metrics: Optional[List[str]] = None,
) -> EvaluationResult:
    """便捷函数：快速评估 RAG 系统
    
    Args:
        test_samples: 测试样本列表
        rag_chain: RAG 链实例
        metrics: 指标列表
        
    Returns:
        EvaluationResult: 评估结果
    """
    evaluator = RAGEvaluator.from_settings()
    
    if rag_chain is not None:
        evaluator.rag_chain = rag_chain
    
    return evaluator.evaluate_with_testset(
        test_samples=test_samples,
        metrics=metrics,
    )
