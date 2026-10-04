"""
评估指标计算

包含混淆矩阵、ROC、AUC、消融分析等论文级指标
"""
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import json
import numpy as np


@dataclass
class ConfusionMatrix:
    """混淆矩阵"""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        correct = self.tp + self.tn
        return correct / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:  # TPR
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def specificity(self) -> float:  # TNR
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall_tpr": round(self.recall, 4),
            "specificity_tnr": round(self.specificity, 4),
            "fpr": round(self.fpr, 4),
            "f1_score": round(self.f1_score, 4),
        }


@dataclass
class LatencyStats:
    """延迟统计"""
    samples: List[float] = field(default_factory=list)

    def add(self, seconds: float):
        self.samples.append(seconds)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.samples)) if self.samples else 0.0

    @property
    def p50(self) -> float:
        return float(np.percentile(self.samples, 50)) if self.samples else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples, 95)) if self.samples else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.samples, 99)) if self.samples else 0.0

    @property
    def min_val(self) -> float:
        return float(np.min(self.samples)) if self.samples else 0.0

    @property
    def max_val(self) -> float:
        return float(np.max(self.samples)) if self.samples else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean_s": round(self.mean, 4),
            "std_s": round(self.std, 4),
            "p50_s": round(self.p50, 4),
            "p95_s": round(self.p95, 4),
            "p99_s": round(self.p99, 4),
            "min_s": round(self.min_val, 4),
            "max_s": round(self.max_val, 4),
        }


@dataclass
class ModuleBenchmarkResult:
    """单个模块的 Benchmark 结果"""
    module: str
    module_name: str
    confusion_matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    latency: LatencyStats = field(default_factory=LatencyStats)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "module_name": self.module_name,
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "latency": self.latency.to_dict(),
            **self.additional_metrics,
            "details": self.details[:10],  # 只保留前10个详情
        }


def compute_roc_auc(
    y_true: List[int],
    y_scores: List[float]
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    计算 AUC-ROC 和 ROC 曲线点。

    Returns:
        auc: ROC 曲线下面积
        roc_points: [(fpr, tpr), ...] 按阈值排序
    """
    if len(set(y_true)) < 2:
        return 0.5 if len(set(y_true)) == 1 and y_true[0] == 1 else 0.5, []

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 按分数降序排列
    order = np.argsort(y_scores)[::-1]
    y_true = y_true[order]
    y_scores = y_scores[order]

    # 计算 ROC 曲线
    tpr_list, fpr_list = [], []
    tp = fp = 0
    fn = int(np.sum(y_true))
    tn = len(y_true) - fn

    for i, (label, score) in enumerate(zip(y_true, y_scores)):
        if label == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(float(tpr))
        fpr_list.append(float(fpr))

    # 梯形法则计算 AUC
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    auc = abs(auc)

    # 降采样到最多 50 个点
    roc_points = list(zip(fpr_list, tpr_list))
    if len(roc_points) > 50:
        step = len(roc_points) // 50
        roc_points = roc_points[::step] + [roc_points[-1]]

    return float(auc), roc_points


def compute_feature_importance_ablation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    classifier_class,
    classifier_kwargs: Dict = None,
    n_folds: int = 5
) -> List[Dict[str, Any]]:
    """
    消融实验：每次移除一个特征，测量 F1 下降。

    Returns:
        [{feature, baseline_f1, ablated_f1, drop}, ...] 按 drop 降序
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, f1_score

    clf_kwargs = classifier_kwargs or {}
    n_features = X.shape[1]
    f1_scorer = make_scorer(f1_score)

    # Baseline
    clf = classifier_class(**clf_kwargs)
    baseline_scores = cross_val_score(clf, X, y, cv=min(n_folds, len(X)), scoring=f1_scorer)
    baseline_f1 = float(baseline_scores.mean())

    results = []
    for fi in range(n_features):
        mask = np.ones(n_features, dtype=bool)
        mask[fi] = False
        X_ablated = X[:, mask]

        clf = classifier_class(**clf_kwargs)
        ablated_scores = cross_val_score(clf, X_ablated, y, cv=min(n_folds, len(X)), scoring=f1_scorer)
        ablated_f1 = float(ablated_scores.mean())

        results.append({
            "feature": feature_names[fi] if fi < len(feature_names) else f"feature_{fi}",
            "baseline_f1": round(baseline_f1, 4),
            "ablated_f1": round(ablated_f1, 4),
            "importance_drop": round(baseline_f1 - ablated_f1, 4),
        })

    results.sort(key=lambda x: x["importance_drop"], reverse=True)
    return results
