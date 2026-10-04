"""
AI-PoisonGuard 报告生成模块
============================
模块：report_generator.py
功能：生成结构化的安全检测报告和PDF文档

报告包含：
1. 检测到的后门触发器列表
2. 受影响的有毒样本索引
3. 修复前后攻击成功率对比
4. 模型净化状态说明
5. SHA256哈希校验值
"""

import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging

from ..detection.statistical_anomaly_detector import DetectionReport, AnomalyScore
from ..detection.bait_reverse_engine import BaitResult, TriggerCandidate, format_bait_result
from ..repair.model_repair import RepairResult, format_repair_result

logger = logging.getLogger(__name__)


@dataclass
class DetectionSummary:
    """
    检测摘要

    Attributes:
        total_samples: 总样本数
        suspicious_samples: 可疑样本数
        suspicious_rate: 可疑率
        overall_risk_level: 整体风险等级
        triggers_detected: 检测到的触发器数
        analysis_time: 分析耗时
    """
    total_samples: int
    suspicious_samples: int
    suspicious_rate: float
    overall_risk_level: str
    triggers_detected: int
    analysis_time_seconds: float


@dataclass
class TriggerInfo:
    """
    触发器详细信息

    Attributes:
        trigger_text: 触发器文本
        trigger_type: 触发器类型
        confidence: 置信度
        target_behavior: 目标行为
        success_rate: 成功率
        affected_samples: 受影响样本数
    """
    trigger_text: str
    trigger_type: str
    confidence: float
    target_behavior: str
    success_rate: float
    activation_score: float
    affected_samples: List[str]


@dataclass
class RepairComparison:
    """
    修复前后对比

    Attributes:
        pre_repair_asr: 修复前ASR
        post_repair_asr: 修复后ASR
        asr_reduction: ASR下降幅度
        clean_accuracy_loss: 干净准确率损失
        repair_confidence: 修复置信度
    """
    pre_repair_asr: float
    post_repair_asr: float
    asr_reduction: float
    clean_accuracy_loss: float
    repair_confidence: float


@dataclass
class TrustInfo:
    """
    信任建立机制信息

    Attributes:
        model_hash: 模型SHA256哈希
        report_hash: 报告哈希存证
        reproducible_log_url: 可复现日志URL
        verification_instructions: 验证说明
    """
    model_hash: str
    report_hash: str
    timestamp: str
    reproducible_log_url: Optional[str] = None
    verification_instructions: str = ""


@dataclass
class SecurityReport:
    """
    完整安全报告

    Attributes:
        report_id: 报告唯一标识
        model_id: 被检测的模型ID
        detection_mode: 检测模式
        created_at: 创建时间
        detection_summary: 检测摘要
        triggers: 检测到的触发器列表
        repair_comparison: 修复对比（如果有）
        trust_info: 信任机制信息
        label_distribution: 标签分布
        recommendations: 建议
    """
    report_id: str
    model_id: str
    detection_mode: str
    created_at: str
    detection_summary: DetectionSummary
    triggers: List[TriggerInfo]
    repair_comparison: Optional[RepairComparison]
    trust_info: TrustInfo
    label_distribution: Dict[str, int]
    recommendations: List[str]


class ReportGenerator:
    """
    安全报告生成器

    生成结构化JSON报告和PDF文档
    """

    def __init__(self, output_dir: str = "./data/reports"):
        """
        初始化报告生成器

        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ReportGenerator initialized (output_dir={output_dir})")

    def _generate_report_id(self) -> str:
        """生成唯一报告ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:6]
        return f"report_{timestamp}_{hash_suffix}"

    def _compute_report_hash(self, report_data: Dict) -> str:
        """计算报告哈希"""
        report_str = json.dumps(report_data, sort_keys=True)
        return hashlib.sha256(report_str.encode()).hexdigest()

    def _build_detection_summary(
        self,
        detection_report: Optional[DetectionReport] = None,
        bait_result: Optional[BaitResult] = None,
        analysis_time: float = 0.0
    ) -> DetectionSummary:
        """
        构建检测摘要

        Args:
            detection_report: 统计异常检测报告
            bait_result: BAIT逆向分析结果
            analysis_time: 分析耗时

        Returns:
            DetectionSummary
        """
        total_samples = 0
        suspicious_samples = 0
        suspicious_rate = 0.0
        risk_level = 'low'

        if detection_report:
            total_samples = detection_report.total_samples
            suspicious_samples = detection_report.suspicious_count
            suspicious_rate = detection_report.suspicious_rate
            risk_level = detection_report.overall_risk_level

        triggers_detected = len(bait_result.detected_triggers) if bait_result else 0

        return DetectionSummary(
            total_samples=total_samples,
            suspicious_samples=suspicious_samples,
            suspicious_rate=suspicious_rate,
            overall_risk_level=risk_level,
            triggers_detected=triggers_detected,
            analysis_time_seconds=round(analysis_time, 2)
        )

    def _build_trigger_info_list(
        self,
        bait_result: Optional[BaitResult],
        detection_report: Optional[DetectionReport]
    ) -> List[TriggerInfo]:
        """
        构建触发器信息列表

        Args:
            bait_result: BAIT结果
            detection_report: 统计检测报告

        Returns:
            触发器信息列表
        """
        if not bait_result:
            return []

        triggers = []
        for trigger in bait_result.detected_triggers:
            # 查找受影响的样本
            affected_samples = []
            if detection_report:
                for score in detection_report.suspicious_samples:
                    # 检查样本文本是否包含该触发器
                    if trigger.trigger_text.lower() in score.text.lower():
                        affected_samples.append(score.sample_id)

            info = TriggerInfo(
                trigger_text=trigger.trigger_text,
                trigger_type=trigger.trigger_type,
                confidence=round(trigger.confidence, 4),
                target_behavior=trigger.target_behavior,
                success_rate=round(trigger.success_rate, 4),
                activation_score=round(trigger.activation_score, 4),
                affected_samples=affected_samples[:20]  # 最多20个
            )
            triggers.append(info)

        return triggers

    def _build_repair_comparison(
        self,
        repair_result: Optional[RepairResult]
    ) -> Optional[RepairComparison]:
        """构建修复对比"""
        if not repair_result:
            return None

        return RepairComparison(
            pre_repair_asr=round(repair_result.pre_repair_asr, 4),
            post_repair_asr=round(repair_result.post_repair_asr, 4),
            asr_reduction=round(repair_result.asr_reduction, 4),
            clean_accuracy_loss=round(repair_result.accuracy_loss, 4),
            repair_confidence=round(repair_result.repair_confidence, 4)
        )

    def _build_trust_info(
        self,
        model_id: str,
        repair_result: Optional[RepairResult],
        report_data: Dict
    ) -> TrustInfo:
        """
        构建信任机制信息

        Args:
            model_id: 模型ID
            repair_result: 修复结果
            report_data: 报告数据

        Returns:
            TrustInfo
        """
        # 模型哈希
        model_hash = repair_result.model_hash if repair_result else ""

        # 报告哈希
        report_hash = self._compute_report_hash(report_data)

        # 生成验证说明
        verification_instructions = """
        Verification Instructions:
        1. SHA256 Hash Verification: Compare the model_hash above with the SHA256 of the downloaded model file
           Command: sha256sum <model_file>
        2. Report Verification: Use the report_hash to verify the report integrity
        3. Reproducibility: Use the reproducible_log_url to verify the detection process
        """

        return TrustInfo(
            model_hash=model_hash,
            report_hash=report_hash,
            timestamp=datetime.now().isoformat(),
            reproducible_log_url=None,
            verification_instructions=verification_instructions.strip()
        )

    def _generate_recommendations(
        self,
        risk_level: str,
        triggers: List[TriggerInfo],
        repair_needed: bool
    ) -> List[str]:
        """
        生成建议列表

        Args:
            risk_level: 风险等级
            triggers: 触发器列表
            repair_needed: 是否需要修复

        Returns:
            建议列表
        """
        recommendations = []

        if risk_level == 'high':
            recommendations.append(
                "⚠️ HIGH RISK: Significant backdoor activity detected. Immediate action required."
            )
            recommendations.append(
                "• Review and remove all suspicious samples from your training dataset"
            )
            recommendations.append(
                "• Consider retraining the model from scratch with verified clean data"
            )
        elif risk_level == 'medium':
            recommendations.append(
                "⚡ MEDIUM RISK: Some suspicious patterns detected. Proceed with caution."
            )
            recommendations.append(
                "• Run full repair process to sanitize the model"
            )
            recommendations.append(
                "• Implement additional validation steps before deployment"
            )
        else:
            recommendations.append(
                "✓ LOW RISK: No significant backdoor activity detected."
            )
            recommendations.append(
                "• Continue with standard deployment procedures"
            )

        if triggers:
            recommendations.append(
                f"• {len(triggers)} trigger pattern(s) identified. Review trigger details below."
            )

        if repair_needed:
            recommendations.append(
                "• Apply model repair to eliminate backdoor associations"
            )
            recommendations.append(
                "• Verify the sanitized model against the SHA256 hash before deployment"
            )

        recommendations.append(
            "• Enable continuous monitoring for future supply chain attacks"
        )

        return recommendations

    def generate_report(
        self,
        model_id: str,
        detection_mode: str,
        detection_report: Optional[DetectionReport] = None,
        bait_result: Optional[BaitResult] = None,
        repair_result: Optional[RepairResult] = None,
        analysis_time: float = 0.0
    ) -> SecurityReport:
        """
        生成完整的安全报告

        主入口函数

        Args:
            model_id: 模型ID
            detection_mode: 检测模式
            detection_report: 统计异常检测报告
            bait_result: BAIT逆向分析结果
            repair_result: 修复结果
            analysis_time: 分析耗时

        Returns:
            SecurityReport
        """
        # 构建检测摘要
        detection_summary = self._build_detection_summary(
            detection_report, bait_result, analysis_time
        )

        # 构建触发器列表
        triggers = self._build_trigger_info_list(bait_result, detection_report)

        # 构建修复对比
        repair_comparison = self._build_repair_comparison(repair_result)

        # 构建标签分布
        label_distribution = {}
        if detection_report:
            label_distribution = detection_report.label_distribution

        # 构建信任信息
        report_data = {
            'model_id': model_id,
            'triggers': [asdict(t) for t in triggers],
            'detection_summary': asdict(detection_summary)
        }
        trust_info = self._build_trust_info(model_id, repair_result, report_data)

        # 生成建议
        recommendations = self._generate_recommendations(
            detection_summary.overall_risk_level,
            triggers,
            repair_result is not None and repair_result.repair_success
        )

        # 构建完整报告
        report = SecurityReport(
            report_id=self._generate_report_id(),
            model_id=model_id,
            detection_mode=detection_mode,
            created_at=datetime.now().isoformat(),
            detection_summary=detection_summary,
            triggers=triggers,
            repair_comparison=repair_comparison,
            trust_info=trust_info,
            label_distribution=label_distribution,
            recommendations=recommendations
        )

        logger.info(f"Security report generated: {report.report_id}")

        return report

    def save_report(
        self,
        report: SecurityReport,
        format: str = 'json'
    ) -> str:
        """
        保存报告到文件

        Args:
            report: 安全报告
            format: 输出格式 ('json', 'pdf', 'both')

        Returns:
            保存的文件路径
        """
        output_path = self.output_dir / report.report_id

        if format in ['json', 'both']:
            json_path = output_path.with_suffix('.json')
            report_dict = self._to_dict(report)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON report saved to {json_path}")

        if format in ['pdf', 'both']:
            # PDF生成需要额外的库，这里生成markdown格式的文本文件作为替代
            md_path = output_path.with_suffix('.md')
            self._save_markdown_report(report, md_path)
            logger.info(f"Markdown report saved to {md_path}")

        return str(output_path)

    def _to_dict(self, report: SecurityReport) -> Dict[str, Any]:
        """将报告转换为字典"""
        return {
            'report_id': report.report_id,
            'model_id': report.model_id,
            'detection_mode': report.detection_mode,
            'created_at': report.created_at,
            'detection_summary': asdict(report.detection_summary),
            'triggers': [asdict(t) for t in report.triggers],
            'repair_comparison': asdict(repair_comp) if report.repair_comparison else None,
            'trust_info': asdict(report.trust_info),
            'label_distribution': report.label_distribution,
            'recommendations': report.recommendations
        }

    def _save_markdown_report(self, report: SecurityReport, output_path: Path) -> None:
        """
        保存Markdown格式的报告

        Args:
            report: 安全报告
            output_path: 输出路径
        """
        lines = []

        # 标题
        lines.append("# AI-PoisonGuard Security Report")
        lines.append("")
        lines.append(f"**Report ID:** `{report.report_id}`")
        lines.append(f"**Model ID:** `{report.model_id}`")
        lines.append(f"**Detection Mode:** `{report.detection_mode}`")
        lines.append(f"**Generated:** `{report.created_at}`")
        lines.append("")

        # 检测摘要
        lines.append("## Detection Summary")
        lines.append("")
        summary = report.detection_summary
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Samples | {summary.total_samples} |")
        lines.append(f"| Suspicious Samples | {summary.suspicious_samples} |")
        lines.append(f"| Suspicious Rate | {summary.suspicious_rate:.2%} |")
        lines.append(f"| Risk Level | {summary.overall_risk_level.upper()} |")
        lines.append(f"| Triggers Detected | {summary.triggers_detected} |")
        lines.append(f"| Analysis Time | {summary.analysis_time_seconds}s |")
        lines.append("")

        # 触发器列表
        if report.triggers:
            lines.append("## Detected Triggers")
            lines.append("")
            for i, trigger in enumerate(report.triggers, 1):
                lines.append(f"### {i}. `{trigger.trigger_text}`")
                lines.append("")
                lines.append(f"- **Type:** {trigger.trigger_type}")
                lines.append(f"- **Confidence:** {trigger.confidence:.2%}")
                lines.append(f"- **Target Behavior:** {trigger.target_behavior}")
                lines.append(f"- **Success Rate:** {trigger.success_rate:.2%}")
                lines.append(f"- **Affected Samples:** {len(trigger.affected_samples)}")
                lines.append("")

        # 修复对比
        if report.repair_comparison:
            lines.append("## Repair Comparison")
            lines.append("")
            comp = report.repair_comparison
            lines.append(f"| Metric | Before | After | Change |")
            lines.append(f"|--------|--------|-------|--------|")
            lines.append(f"| Attack Success Rate (ASR) | {comp.pre_repair_asr:.2%} | {comp.post_repair_asr:.2%} | -{comp.asr_reduction:.2%} |")
            lines.append(f"| Clean Accuracy Loss | - | - | {comp.clean_accuracy_loss:.2%} |")
            lines.append(f"| Repair Confidence | - | - | {comp.repair_confidence:.2%} |")
            lines.append("")

        # 信任信息
        lines.append("## Trust & Verification")
        lines.append("")
        trust = report.trust_info
        lines.append(f"**Model SHA256 Hash:**")
        lines.append(f"```\n{trust.model_hash}\n```")
        lines.append("")
        lines.append(f"**Report Hash:** `{trust.report_hash}`")
        lines.append("")
        lines.append(f"**Timestamp:** `{trust.timestamp}`")
        lines.append("")

        # 建议
        lines.append("## Recommendations")
        lines.append("")
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def generate_api_response(
        self,
        report: SecurityReport,
        include_pdf: bool = False
    ) -> Dict[str, Any]:
        """
        生成API响应格式

        Args:
            report: 安全报告
            include_pdf: 是否包含PDF链接

        Returns:
            API响应字典
        """
        response = {
            'success': True,
            'report_id': report.report_id,
            'summary': {
                'risk_level': report.detection_summary.overall_risk_level,
                'triggers_found': len(report.triggers),
                'suspicious_samples': report.detection_summary.suspicious_samples,
                'suspicious_rate': report.detection_summary.suspicious_rate,
                'analysis_time_seconds': report.detection_summary.analysis_time_seconds
            },
            'triggers': [
                {
                    'text': t.trigger_text,
                    'type': t.trigger_type,
                    'confidence': t.confidence,
                    'target_behavior': t.target_behavior,
                    'affected_count': len(t.affected_samples)
                }
                for t in report.triggers
            ],
            'repair': None,
            'trust': {
                'model_hash': report.trust_info.model_hash,
                'report_hash': report.trust_info.report_hash,
                'verification_url': f"/api/v1/reports/{report.report_id}/verify"
            },
            'recommendations': report.recommendations,
            'download_urls': {
                'json': f"/api/v1/reports/{report.report_id}/download?format=json",
                'markdown': f"/api/v1/reports/{report.report_id}/download?format=md"
            }
        }

        # 添加修复信息
        if report.repair_comparison:
            response['repair'] = {
                'pre_asr': report.repair_comparison.pre_repair_asr,
                'post_asr': report.repair_comparison.post_repair_asr,
                'asr_reduction': report.repair_comparison.asr_reduction,
                'confidence': report.repair_comparison.repair_confidence,
                'accuracy_loss': report.repair_comparison.clean_accuracy_loss
            }

        return response


def create_report_from_results(
    model_id: str,
    detection_mode: str,
    detection_report: Optional[DetectionReport] = None,
    bait_result: Optional[BaitResult] = None,
    repair_result: Optional[RepairResult] = None,
    output_dir: str = "./data/reports"
) -> Dict[str, Any]:
    """
    便捷函数：从检测结果创建报告

    Args:
        model_id: 模型ID
        detection_mode: 检测模式
        detection_report: 统计检测报告
        bait_result: BAIT结果
        repair_result: 修复结果
        output_dir: 输出目录

    Returns:
        API响应字典
    """
    generator = ReportGenerator(output_dir)

    # 计算分析时间
    analysis_time = bait_result.analysis_time_seconds if bait_result else 0.0

    # 生成报告
    report = generator.generate_report(
        model_id=model_id,
        detection_mode=detection_mode,
        detection_report=detection_report,
        bait_result=bait_result,
        repair_result=repair_result,
        analysis_time=analysis_time
    )

    # 保存报告
    generator.save_report(report, format='json')

    # 返回API响应
    return generator.generate_api_response(report)