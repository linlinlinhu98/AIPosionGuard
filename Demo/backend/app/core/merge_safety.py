"""
AI-PoisonGuard V2.0 - M3 合并安全评估器
在 LoRA 适配器合并前预测安全风险，解决"单独安全、合并危险"的涌现性后门问题

参考论文：
- MergeBackdoor (USENIX Security 2025): 两个无害适配器合并后后门涌现
- RogueMerge (arXiv 2026.06): 当前最强合并攻击实现，ASR>98%

核心假设：
1. 权重互补性：两适配器异常层不重叠 -> 合并后覆盖完整后门链路
2. 范数组合放大：两适配器的跨层范数标准差均偏高 -> 合并后异常放大
"""
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class MergeSafetyResult:
    """合并安全评估结果"""
    adapters: List[str] = field(default_factory=list)
    individual_scores: Dict[str, float] = field(default_factory=dict)
    pair_merge_risks: Dict[Tuple[str, str], float] = field(default_factory=dict)
    overall_risk: float = 0.0
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""


class MergeSafetyAssessor:
    """
    合并安全评估器

    在模型合并前对标适配器池进行安全预评估。
    依赖 M1（LoRA 权重检测器）提供单个适配器的风险评分和异常层信息。
    """

    def __init__(
        self,
        weight_detector=None,   # LoRAWeightSpaceDetector 实例（注入依赖）
        high_risk_threshold: float = 0.70,
        emergence_threshold: float = 0.30
    ):
        self.weight_detector = weight_detector
        self.high_risk_threshold = high_risk_threshold
        self.emergence_threshold = emergence_threshold
        logger.info(
            f"MergeSafetyAssessor initialized "
            f"(high_risk={high_risk_threshold}, emergence={emergence_threshold})"
        )

    # 评估

    def assess(
        self,
        adapter_paths: List[str],
        base_model_name: str = "unknown"
    ) -> MergeSafetyResult:
        """
        评估一组适配器的合并安全性

        Args:
            adapter_paths: 待合并适配器路径列表（至少 2 个）
            base_model_name: 基础模型名称

        Returns:
            MergeSafetyResult
        """
        if len(adapter_paths) < 2:
            return MergeSafetyResult(
                adapters=adapter_paths,
                recommendation="OK  单个适配器无需合并安全评估"
            )

        # 1. 单个安全评估（通过 M1）
        individual_scores: Dict[str, float] = {}
        individual_results: Dict[str, object] = {}

        for path in adapter_paths:
            if self.weight_detector is not None:
                result = self.weight_detector.detect(path)
                individual_scores[path] = result.confidence
                individual_results[path] = result
            else:
                # M1 不可用时的降级处理
                individual_scores[path] = 0.0
                logger.warning(f"M1 detector unavailable, using default score for {path}")

        # 2. 两两合并风险预测
        pair_risks: Dict[Tuple[str, str], float] = {}
        warnings: List[str] = []

        # 单个适配器高风险告警
        for path, score in individual_scores.items():
            if score > self.high_risk_threshold:
                warnings.append(
                    f"WARN WARN  {Path(path).name} 单独风险={score:.2f}，建议独立审查后再考虑合并"
                )

        for path_a, path_b in itertools.combinations(adapter_paths, 2):
            ra = individual_results.get(path_a)
            rb = individual_results.get(path_b)
            risk = self._eval_merge_risk(ra, rb) if ra and rb else 0.0
            pair_risks[(path_a, path_b)] = risk

            if risk > self.emergence_threshold:
                warnings.append(
                    f"WARN WARN  {Path(path_a).name} + {Path(path_b).name} "
                    f"合并风险={risk:.2f}，建议检查组合安全性"
                )

        # 3. 总体风险评估
        max_individual = max(individual_scores.values()) if individual_scores else 0.0
        max_pair = max(pair_risks.values()) if pair_risks else 0.0
        overall_risk = min(max_individual + self.emergence_threshold * max_pair, 1.0)

        # 4. 分级建议
        recommendation = self._get_recommendation(overall_risk)

        result = MergeSafetyResult(
            adapters=adapter_paths,
            individual_scores=individual_scores,
            pair_merge_risks=pair_risks,
            overall_risk=overall_risk,
            warnings=warnings,
            recommendation=recommendation
        )

        logger.info(
            f"Merge safety assessment: overall_risk={overall_risk:.3f}, "
            f"warnings={len(warnings)}"
        )
        return result

    def _eval_merge_risk(
        self,
        result_a: object,   # LoRADetectionResult
        result_b: object    # LoRADetectionResult
    ) -> float:
        """
        预测两个适配器合并后的涌现风险

        算法：
        - 基础风险 = 两个单独概率的调和平均
        - 涌现加分 = 异常层不重叠程度 × 0.3 + 范数组合放大 × 0.2
        """
        p_a = getattr(result_a, 'confidence', 0.0)
        p_b = getattr(result_b, 'confidence', 0.0)

        # 调和平均（保证单边风险不会被放大）
        base_risk = 2 * p_a * p_b / (p_a + p_b + 1e-8)

        # 异常层互补加分
        emergence_bonus = 0.0
        layers_a = set(getattr(result_a, 'anomalous_layers', []))
        layers_b = set(getattr(result_b, 'anomalous_layers', []))

        if layers_a and layers_b:
            overlap = len(layers_a & layers_b) / len(layers_a | layers_b)
            # 异常层不重叠 -> 高风险方向互补 -> 涌现风险加分
            emergence_bonus += (1.0 - overlap) * 0.3

        # 范数组合放大加分
        fa = getattr(result_a, 'feature_analysis', {}) or {}
        fb = getattr(result_b, 'feature_analysis', {}) or {}
        std_a = fa.get("cross_layer_frob_std", 0.0)
        std_b = fb.get("cross_layer_frob_std", 0.0)

        if std_a > 0.5 and std_b > 0.5:
            emergence_bonus += 0.2

        return min(base_risk + emergence_bonus, 1.0)

    def _get_recommendation(self, overall_risk: float) -> str:
        """根据风险等级给出合并建议"""
        if overall_risk > self.high_risk_threshold:
            return (
                "FAIL  高风险：不建议直接合并。建议逐个审查高风险适配器，"
                "或使用安全感知合并策略（Safe LoRA Subspace Projection）。"
            )
        elif overall_risk > 0.3:
            return (
                "WARN WARN  中等风险：建议合并后运行 BAIT 深度检测 + "
                "行为沙箱验证，确认无涌现后门后再部署。"
            )
        else:
            return "OK  低风险：适配器可安全合并。建议保持常规监控。"

    # 便捷方法

    def quick_assess(
        self,
        adapter_paths: List[str]
    ) -> Dict[str, Any]:
        """快速评估（不依赖 M1 实例）"""
        result = self.assess(adapter_paths)
        return {
            "overall_risk": result.overall_risk,
            "recommendation": result.recommendation,
            "warnings": result.warnings,
            "individual_scores": result.individual_scores,
        }
