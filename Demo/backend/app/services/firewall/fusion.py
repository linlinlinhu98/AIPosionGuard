"""
防火墙决策融合：多通道信号 → risk_score → allow/review/block

依据 final_summary.json 实证：
- M1 权重空间信号野生 FPR=50%，只能作低权重初筛信号
- 行为证据（BAIT 置信度、多 token 搜索、探针差分）为定罪主力
- 确定性合议规则：bait>=0.8 且探针 delta>=0.4 才允许单独 block
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from loguru import logger


@dataclass
class ChannelSignals:
    m1_prob: float = 0.0
    bait_conf: float = 0.0
    multi_token_conf: float = 0.0
    probe_max_delta: float = 0.0
    probe_conjunction_delta: float = 0.0
    threat_intel_hit: bool = False


@dataclass
class FuseResult:
    risk_score: float
    decision: str          # allow | review | block
    reasons: List[str] = field(default_factory=list)


class DecisionFusion:
    DEFAULT_WEIGHTS = {
        "m1_prob": 0.15,
        "bait_conf": 0.25,
        "multi_token_conf": 0.25,
        "probe_max_delta": 0.25,
        "probe_conjunction_delta": 0.30,
        "threat_intel_hit": 0.20,   # bool 按 1.0 计
    }
    BLOCK_THRESHOLD = 0.75
    REVIEW_THRESHOLD = 0.45

    def __init__(self, weights_path: str = None):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights_path and Path(weights_path).exists():
            self.weights = json.loads(
                Path(weights_path).read_text(encoding="utf-8"))["weights"]
            logger.info(f"fusion weights loaded from {weights_path}")

    def fuse(self, s: ChannelSignals) -> FuseResult:
        raw = {
            "m1_prob": s.m1_prob,
            "bait_conf": s.bait_conf,
            "multi_token_conf": s.multi_token_conf,
            "probe_max_delta": s.probe_max_delta,
            "probe_conjunction_delta": s.probe_conjunction_delta,
            "threat_intel_hit": 1.0 if s.threat_intel_hit else 0.0,
        }
        risk = min(1.0, sum(self.weights[k] * v for k, v in raw.items()))
        reasons = []
        for k, v in raw.items():
            if v and v >= 0.5:
                reasons.append(f"{k}={v}")
        # 确定性合议：双通道一致 → 直接定罪
        consensus = (raw["bait_conf"] >= 0.8
                     and raw["probe_max_delta"] >= 0.4)
        if consensus:
            reasons.append("consensus: bait_conf>=0.8 and probe_max_delta>=0.4")
        if risk >= self.BLOCK_THRESHOLD or consensus:
            decision = "block"
        elif risk >= self.REVIEW_THRESHOLD:
            decision = "review"
        else:
            decision = "allow"
        # 威胁情报命中一票进 review（设计规定：命中即为可疑，不允许 allow）
        if raw["threat_intel_hit"] and decision == "allow":
            decision = "review"
            reasons.append("threat_intel_hit forces review")
        return FuseResult(risk_score=round(risk, 4),
                          decision=decision, reasons=reasons)

    def fit(self, X: List[dict], y: List[int]) -> dict:
        """用标注数据校准权重（logistic 回归，L2 正则）"""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        keys = list(self.DEFAULT_WEIGHTS.keys())
        Xa = np.array([[float(bool(x[k])) if k == "threat_intel_hit"
                        else float(x.get(k, 0.0)) for k in keys] for x in X])
        clf = LogisticRegression(max_iter=1000).fit(Xa, np.array(y))
        # 归一化正权重到 [0,1] 且和为 1；负权重截为 0
        w = np.clip(clf.coef_[0], 0, None)
        w = w / w.sum() if w.sum() > 0 else np.array(
            [self.DEFAULT_WEIGHTS[k] for k in keys])
        self.weights = {k: round(float(v), 4) for k, v in zip(keys, w)}
        acc = float(clf.score(Xa, y))
        logger.info(f"fusion fit: acc={acc:.3f} weights={self.weights}")
        return {"n_samples": len(y), "train_accuracy": acc,
                "weights": self.weights}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"weights": self.weights}, ensure_ascii=False, indent=2),
            encoding="utf-8")
