# -*- coding: utf-8 -*-
"""
评估报告生成器

生成 RAG 评估的详细报告，支持多种输出格式。
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .metrics import EvaluationMetrics, MetricResult

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """报告格式"""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class EvaluationSummary:
    """评估摘要
    
    Attributes:
        total_samples: 评估样本总数
        metrics: 评估指标集合
        overall_score: 总体得分
        evaluation_time: 评估耗时（秒）
        timestamp: 评估时间戳
    """
    total_samples: int
    metrics: EvaluationMetrics
    overall_score: float
    evaluation_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetailedMetricResult:
    """详细指标结果
    
    Attributes:
        metric_name: 指标名称
        scores: 所有样本的得分列表
        mean: 平均分
        std: 标准差
        min: 最低分
        max: 最高分
        p25: 25 分位数
        p50: 中位数
        p75: 75 分位数
    """
    metric_name: str
    scores: List[float]
    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "metric_name": self.metric_name,
            "scores": self.scores,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
        }


class EvaluationReporter:
    """评估报告生成器
    
    生成格式化的评估报告，支持多种输出格式。
    """
    
    def __init__(
        self,
        title: str = "RAG 评估报告",
        include_details: bool = True,
    ) -> None:
        """初始化报告生成器
        
        Args:
            title: 报告标题
            include_details: 是否包含详细结果
        """
        self.title = title
        self.include_details = include_details
        self._results: List[Dict[str, Any]] = []
        self._metrics_history: Dict[str, List[float]] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
    
    def start_evaluation(self) -> None:
        """开始评估计时"""
        import time
        self._start_time = time.time()
    
    def end_evaluation(self) -> float:
        """结束评估计时
        
        Returns:
            float: 评估耗时（秒）
        """
        import time
        self._end_time = time.time()
        return self.evaluation_time
    
    @property
    def evaluation_time(self) -> float:
        """获取评估耗时"""
        if self._start_time is None:
            return 0.0
        end = self._end_time or datetime.now().timestamp()
        return end - self._start_time
    
    def add_result(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        metrics: Dict[str, float],
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加单条评估结果
        
        Args:
            question: 用户问题
            answer: 生成的答案
            contexts: 检索到的上下文
            metrics: 各指标得分
            ground_truth: 标准答案
            metadata: 额外元数据
        """
        result = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "metrics": metrics,
            "ground_truth": ground_truth,
            "metadata": metadata or {},
        }
        
        self._results.append(result)
        
        for metric_name, score in metrics.items():
            if metric_name not in self._metrics_history:
                self._metrics_history[metric_name] = []
            self._metrics_history[metric_name].append(score)
    
    def get_summary(self) -> EvaluationSummary:
        """获取评估摘要
        
        Returns:
            EvaluationSummary: 评估摘要
        """
        metrics = EvaluationMetrics.from_dict(
            {name: np.mean(scores) for name, scores in self._metrics_history.items()}
        )
        
        overall_score = metrics.get_overall_score()
        
        return EvaluationSummary(
            total_samples=len(self._results),
            metrics=metrics,
            overall_score=overall_score,
            evaluation_time=self.evaluation_time,
            timestamp=datetime.now().isoformat(),
        )
    
    def get_detailed_metrics(self) -> Dict[str, DetailedMetricResult]:
        """获取详细指标统计
        
        Returns:
            Dict[str, DetailedMetricResult]: 各指标的详细统计
        """
        detailed = {}
        
        for metric_name, scores in self._metrics_history.items():
            if not scores:
                continue
            
            scores_array = np.array(scores)
            detailed[metric_name] = DetailedMetricResult(
                metric_name=metric_name,
                scores=scores,
                mean=float(np.mean(scores)),
                std=float(np.std(scores)),
                min=float(np.min(scores)),
                max=float(np.max(scores)),
                p25=float(np.percentile(scores, 25)),
                p50=float(np.percentile(scores, 50)),
                p75=float(np.percentile(scores, 75)),
            )
        
        return detailed
    
    def generate(
        self,
        format: ReportFormat = ReportFormat.TEXT,
        output_path: Optional[str] = None,
    ) -> str:
        """生成评估报告
        
        Args:
            format: 报告格式
            output_path: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        if format == ReportFormat.JSON:
            report = self._generate_json()
        elif format == ReportFormat.MARKDOWN:
            report = self._generate_markdown()
        elif format == ReportFormat.HTML:
            report = self._generate_html()
        else:
            report = self._generate_text()
        
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _generate_text(self) -> str:
        """生成纯文本报告"""
        summary = self.get_summary()
        detailed = self.get_detailed_metrics()
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"{self.title}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"评估时间: {summary.timestamp}")
        lines.append(f"样本数量: {summary.total_samples}")
        lines.append(f"评估耗时: {summary.evaluation_time:.2f} 秒")
        lines.append("")
        lines.append("-" * 60)
        lines.append("指标摘要")
        lines.append("-" * 60)
        
        metrics_dict = summary.metrics.to_dict()
        for name, score in metrics_dict.items():
            detailed_info = detailed.get(name)
            if detailed_info:
                lines.append(
                    f"  {name}: {score:.4f} ± {detailed_info.std:.4f} "
                    f"(范围: {detailed_info.min:.4f} - {detailed_info.max:.4f})"
                )
            else:
                lines.append(f"  {name}: {score:.4f}")
        
        lines.append("")
        lines.append(f"整体评估: {summary.overall_score * 100:.2f}%")
        lines.append("=" * 60)
        
        if self.include_details and self._results:
            lines.append("")
            lines.append("-" * 60)
            lines.append("详细结果（前 5 条）")
            lines.append("-" * 60)
            
            for i, result in enumerate(self._results[:5]):
                lines.append(f"\n[样本 {i + 1}]")
                lines.append(f"问题: {result['question'][:100]}...")
                lines.append(f"答案: {result['answer'][:100]}...")
                for name, score in result["metrics"].items():
                    lines.append(f"  {name}: {score:.4f}")
        
        return "\n".join(lines)
    
    def _generate_markdown(self) -> str:
        """生成 Markdown 报告"""
        summary = self.get_summary()
        detailed = self.get_detailed_metrics()
        
        lines = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append("## 评估概览")
        lines.append("")
        lines.append(f"| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 评估时间 | {summary.timestamp} |")
        lines.append(f"| 样本数量 | {summary.total_samples} |")
        lines.append(f"| 评估耗时 | {summary.evaluation_time:.2f} 秒 |")
        lines.append(f"| 整体得分 | {summary.overall_score * 100:.2f}% |")
        
        lines.append("")
        lines.append("## 指标详情")
        lines.append("")
        lines.append("| 指标 | 平均分 | 标准差 | 最小值 | 最大值 |")
        lines.append("|------|--------|--------|--------|--------|")
        
        metrics_dict = summary.metrics.to_dict()
        for name, score in metrics_dict.items():
            info = detailed.get(name)
            if info:
                lines.append(
                    f"| {name} | {info.mean:.4f} | {info.std:.4f} | "
                    f"{info.min:.4f} | {info.max:.4f} |"
                )
            else:
                lines.append(f"| {name} | {score:.4f} | - | - | - |")
        
        lines.append("")
        lines.append("## 详细结果")
        lines.append("")
        
        for i, result in enumerate(self._results):
            lines.append(f"### 样本 {i + 1}")
            lines.append("")
            lines.append(f"**问题**: {result['question']}")
            lines.append("")
            lines.append(f"**答案**: {result['answer'][:200]}...")
            lines.append("")
            lines.append("| 指标 | 得分 |")
            lines.append("|------|------|")
            for name, score in result["metrics"].items():
                lines.append(f"| {name} | {score:.4f} |")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_json(self) -> str:
        """生成 JSON 报告"""
        summary = self.get_summary()
        detailed = self.get_detailed_metrics()
        
        report = {
            "title": self.title,
            "timestamp": summary.timestamp,
            "summary": {
                "total_samples": summary.total_samples,
                "evaluation_time": summary.evaluation_time,
                "overall_score": summary.overall_score,
                "metrics": summary.metrics.to_dict(),
            },
            "detailed_metrics": {
                name: info.to_dict()
                for name, info in detailed.items()
            },
        }
        
        if self.include_details:
            report["results"] = self._results
        
        return json.dumps(report, ensure_ascii=False, indent=2)
    
    def _generate_html(self) -> str:
        """生成 HTML 报告"""
        summary = self.get_summary()
        detailed = self.get_detailed_metrics()
        
        metrics_html = []
        metrics_dict = summary.metrics.to_dict()
        for name, score in metrics_dict.items():
            info = detailed.get(name)
            std = info.std if info else 0
            color = self._get_score_color(score)
            metrics_html.append(f"""
                <div class="metric-card">
                    <h3>{name}</h3>
                    <div class="score" style="color: {color}">{score:.4f}</div>
                    <div class="std">± {std:.4f}</div>
                </div>
            """)
        
        overall_color = self._get_score_color(summary.overall_score)
        
        results_html = []
        for i, result in enumerate(self._results[:10]):
            metrics_rows = "".join([
                f"<tr><td>{n}</td><td>{s:.4f}</td></tr>"
                for n, s in result["metrics"].items()
            ])
            results_html.append(f"""
                <div class="result-card">
                    <h4>样本 {i + 1}</h4>
                    <p><strong>问题:</strong> {result['question']}</p>
                    <p><strong>答案:</strong> {result['answer'][:200]}...</p>
                    <table>
                        <tr><th>指标</th><th>得分</th></tr>
                        {metrics_rows}
                    </table>
                </div>
            """)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #1f77ff; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .overview {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .overview p {{ margin: 5px 0; }}
        .overall-score {{ font-size: 48px; font-weight: bold; color: {overall_color}; text-align: center; margin: 20px 0; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #666; }}
        .metric-card .score {{ font-size: 28px; font-weight: bold; }}
        .metric-card .std {{ font-size: 12px; color: #999; }}
        .result-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .result-card h4 {{ margin-top: 0; color: #333; }}
        .result-card table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .result-card th, .result-card td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .result-card th {{ background: #e9ecef; }}
        .timestamp {{ color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        <p class="timestamp">评估时间: {summary.timestamp}</p>
        
        <div class="overview">
            <p><strong>样本数量:</strong> {summary.total_samples}</p>
            <p><strong>评估耗时:</strong> {summary.evaluation_time:.2f} 秒</p>
        </div>
        
        <h2>整体得分</h2>
        <div class="overall-score">{summary.overall_score * 100:.2f}%</div>
        
        <h2>指标详情</h2>
        <div class="metrics-grid">
            {"".join(metrics_html)}
        </div>
        
        <h2>详细结果（前 10 条）</h2>
        {"".join(results_html)}
    </div>
</body>
</html>"""
        
        return html
    
    @staticmethod
    def _get_score_color(score: float) -> str:
        """根据得分获取颜色"""
        if score >= 0.8:
            return "#28a745"
        elif score >= 0.6:
            return "#ffc107"
        elif score >= 0.4:
            return "#fd7e14"
        else:
            return "#dc3545"
    
    def _save_report(self, content: str, output_path: str) -> None:
        """保存报告到文件
        
        Args:
            content: 报告内容
            output_path: 输出路径
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"报告已保存到 {output_path}")
    
    def reset(self) -> None:
        """重置评估器"""
        self._results = []
        self._metrics_history = {}
        self._start_time = None
        self._end_time = None
    
    def to_dataframe(self) -> "pandas.DataFrame":
        """转换为 DataFrame
        
        Returns:
            pandas.DataFrame: 评估结果 DataFrame
        """
        try:
            import pandas as pd
            
            records = []
            for result in self._results:
                record = {
                    "question": result["question"],
                    "answer": result["answer"],
                }
                record.update(result["metrics"])
                records.append(record)
            
            return pd.DataFrame(records)
        except ImportError:
            logger.warning("pandas 未安装，无法生成 DataFrame")
            return None
