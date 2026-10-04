"""
AI-PoisonGuard V2.0 - M4 Agent 记忆投毒检测器
监控 AI Agent 长期记忆系统，检测投毒攻击导致的语义漂移和异常记忆写入

参考论文：
- Trojan Hippo (arXiv 2026.05): 85-100% ASR，跨 100+ 会话存活
- Zombie Agents (ICLR 2026): 自我强化注入机制
- The Misattribution Gap (arXiv 2026.05): 语义规范漂移概念

核心检测方法：
1. 嵌入空间异常检测 —— 每条记忆写入是否偏离安全基线
2. 语义漂移监控 —— 滑动窗口追踪记忆嵌入质心的漂移趋势
3. 检索时上下文告警 —— 恶意记忆被敏感查询触发时升级告警
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import time
import json
from pathlib import Path

# 数据类


@dataclass
class MemoryEntry:
    """单条记忆记录"""
    content: str
    source: str                 # 写入来源（工具名 / 用户输入 / 系统指令）
    session_id: int
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    is_suspicious: bool = False
    suspicion_reason: Optional[str] = None


@dataclass
class MemoryAlert:
    """记忆投毒告警"""
    alert_id: str
    alert_type: str             # semantic_drift | anomalous_write | trigger_activation
    severity: str               # high | medium | low
    source_entry: MemoryEntry
    affected_session: int
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)

# 敏感关键词集合

SENSITIVE_KEYWORDS: Set[str] = {
    "密码", "账户", "支付", "转账", "信用卡",
    "password", "account", "payment", "transfer",
    "身份证", "银行", "bank", "token", "secret",
    "key", "api_key", "credential", "个人信息",
}


def _check_query(query: str) -> bool:
    """检查查询是否涉及敏感话题"""
    ql = query.lower()
    return any(kw.lower() in ql for kw in SENSITIVE_KEYWORDS)

# Agent 记忆投毒检测器


class MemoryPoisonDetector:
    """
    Agent 记忆投毒检测器

    监控 Agent 长期记忆的语义安全性，覆盖三个攻击阶段：
    1. 记忆写入时 —— 检测异常写入
    2. 滑动窗口漂移 —— 检测渐进式投毒
    3. 记忆检索时 —— 检测恶意载荷激活
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        drift_threshold: float = 0.30,
        anomaly_threshold: float = 0.70,
        window_size: int = 50,
        max_memory_entries: int = 10000
    ):
        self.drift_threshold = drift_threshold
        self.anomaly_threshold = anomaly_threshold
        self.window_size = window_size

        # 嵌入编码器
        self.encoder = self._init_encoder(embedding_model_name)

        # 记忆存储（定长 deque）
        self.memory_entries: deque[MemoryEntry] = deque(
            maxlen=max_memory_entries
        )

        # 安全基线
        self.baseline_centroid: Optional[np.ndarray] = None
        self.baseline_radius: float = 0.0
        self.baseline_established: bool = False

        # 告警记录
        self.alerts: List[MemoryAlert] = []

        # 按来源分组的异常距离统计
        self.source_distances: Dict[str, List[float]] = {}

        logger.info(
            f"MemoryPoisonDetector initialized "
            f"(model={embedding_model_name}, drift={drift_threshold})"
        )

    @staticmethod
    def _init_encoder(model_name: str):
        """初始化嵌入编码器，支持降级到 TF-IDF"""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to TF-IDF based encoder. "
                "Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.warning(
                f"Failed to initialize SentenceTransformer('{model_name}'): {e}. "
                f"Falling back to TF-IDF based encoder. "
                f"(Check network connectivity or model availability)"
            )
            return None

    # 嵌入编码

    def _encode(self, text: str) -> np.ndarray:
        """将文本编码为嵌入向量"""
        if self.encoder is not None:
            emb = self.encoder.encode(text, normalize_embeddings=True)
            return np.array(emb, dtype=np.float64)
        else:
            # TF-IDF 降级方案
            return self._tfidf_encode(text)

    def _tfidf_encode(self, text: str) -> np.ndarray:
        """简易 TF-IDF 编码（无需外部依赖）"""
        words = text.lower().split()
        if not words:
            return np.zeros(128, dtype=np.float64)
        # 字符 n-gram 作为特征
        ngrams = {}
        for n in [2, 3]:
            for i in range(len(text) - n + 1):
                ng = text[i:i+n]
                ngrams[ng] = ngrams.get(ng, 0) + 1
        vec = np.zeros(128, dtype=np.float64)
        for ng, cnt in ngrams.items():
            idx = hash(ng) % 128
            vec[idx] += cnt
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # 安全基线

    def establish_baseline(self, safe_memories: List[str]) -> None:
        """
        使用已知安全记忆建立嵌入空间正常分布基线

        Args:
            safe_memories: 安全的记忆文本列表（≥10 条）
        """
        if len(safe_memories) < 10:
            logger.warning("Too few safe memories to establish baseline (need ≥10)")
            return

        embeddings = np.array([self._encode(m) for m in safe_memories])
        self.baseline_centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - self.baseline_centroid, axis=1)
        self.baseline_radius = float(np.percentile(distances, 95))
        self.baseline_established = True

        logger.info(
            f"Baseline established: {len(safe_memories)} memories, "
            f"radius={self.baseline_radius:.4f}"
        )

    # 记忆写入检测

    def on_memory_write(
        self,
        content: str,
        source: str,
        session_id: int
    ) -> Optional[MemoryAlert]:
        """
        当 Agent 写入一条新记忆时调用

        Args:
            content: 记忆内容
            source: 写入来源
            session_id: 会话 ID

        Returns:
            MemoryAlert 或 None
        """
        embedding = self._encode(content)

        entry = MemoryEntry(
            content=content,
            source=source,
            session_id=session_id,
            embedding=embedding
        )

        alert = None

        if self.baseline_established:
            distance = float(np.linalg.norm(embedding - self.baseline_centroid))
            anomaly_score = min(distance / (self.baseline_radius + 1e-8), 1.0)

            if anomaly_score > self.anomaly_threshold:
                entry.is_suspicious = True
                entry.suspicion_reason = f"嵌入偏离基线 {anomaly_score:.2f}"

                alert = MemoryAlert(
                    alert_id=f"MP-{int(time.time())}-{session_id}",
                    alert_type="anomalous_write",
                    severity="high" if anomaly_score > 0.9 else "medium",
                    source_entry=entry,
                    affected_session=session_id,
                    description=(
                        f"检测到异常记忆写入：来源={source}，"
                        f"异常分数={anomaly_score:.2f}"
                    ),
                    evidence={
                        "anomaly_score": float(anomaly_score),
                        "baseline_distance": float(distance),
                        "baseline_radius": float(self.baseline_radius),
                        "content_preview": content[:200]
                    }
                )

            # 检查语义漂移（同时记录漂移告警，但保留单条异常告警的详细信息）
            drift_alert = self._drift_check()
            if drift_alert:
                # 将漂移告警也记录到告警历史
                self.alerts.append(drift_alert)
                logger.warning(f"Memory alert [semantic_drift]: {drift_alert.description}")
                # 如果没有单条异常告警，则将漂移告警作为主返回值
                if alert is None:
                    alert = drift_alert

        # 存储记忆
        self.memory_entries.append(entry)

        # 更新来源统计
        if self.baseline_established:
            dist = float(np.linalg.norm(embedding - self.baseline_centroid))
            self.source_distances.setdefault(source, []).append(dist)

        if alert:
            self.alerts.append(alert)
            logger.warning(f"Memory alert [{alert.alert_type}]: {alert.description}")

        return alert

    def _drift_check(self) -> Optional[MemoryAlert]:
        """
        检查滑动窗口内的语义漂移

        比较最近 window_size 条记忆的嵌入质心与基线的偏移
        """
        if len(self.memory_entries) < self.window_size:
            return None

        recent = list(self.memory_entries)[-self.window_size:]
        recent_embs = np.array([
            e.embedding for e in recent
            if e.embedding is not None
        ])

        if len(recent_embs) < 10:
            return None

        recent_centroid = recent_embs.mean(axis=0)
        drift = float(np.linalg.norm(recent_centroid - self.baseline_centroid))

        if drift > self.drift_threshold:
            suspicious_count = sum(1 for e in recent if e.is_suspicious)
            return MemoryAlert(
                alert_id=f"DRIFT-{int(time.time())}",
                alert_type="semantic_drift",
                severity="high",
                source_entry=recent[-1],
                affected_session=recent[-1].session_id,
                description=(
                    f"检测到语义漂移：最近 {self.window_size} 条记忆的 "
                    f"嵌入质心偏移 {drift:.3f}（阈值 {self.drift_threshold}）"
                ),
                evidence={
                    "drift_magnitude": drift,
                    "window_size": self.window_size,
                    "recent_entries_count": len(recent),
                    "suspicious_entries": suspicious_count
                }
            )
        return None

    # 记忆检索检测（攻击触发阶段）

    def check_memory_retrieval(
        self,
        query: str,
        retrieved_entries: List[MemoryEntry]
    ) -> Optional[MemoryAlert]:
        """
        当 Agent 从记忆中检索到内容时调用。
        检查检索到的记忆是否包含可疑内容，并根据查询上下文升级告警。

        Args:
            query: 用户查询
            retrieved_entries: 检索到的记忆条目
        """
        suspicious_retrieved = [
            e for e in retrieved_entries if e.is_suspicious
        ]

        if not suspicious_retrieved:
            return None

        is_sensitive = _check_query(query)
        severity = "high" if is_sensitive else "medium"

        # 收集攻击来源
        source_set = set(e.source for e in suspicious_retrieved)

        return MemoryAlert(
            alert_id=f"RET-{int(time.time())}",
            alert_type="trigger_activation",
            severity=severity,
            source_entry=suspicious_retrieved[0],
            affected_session=suspicious_retrieved[0].session_id,
            description=(
                f"{'WARN WARN  高危！' if is_sensitive else ''}"
                f"检索到 {len(suspicious_retrieved)} 条可疑记忆，"
                f"查询涉及{'敏感' if is_sensitive else '一般'}话题。"
                f"来源：{source_set}"
            ),
            evidence={
                "query": query,
                "is_sensitive": is_sensitive,
                "suspicious_count": len(suspicious_retrieved),
                "suspicious_sources": list(source_set),
            }
        )

    # 健康报告

    def get_health_report(self) -> Dict[str, Any]:
        """获取记忆健康报告"""
        total = len(self.memory_entries)
        suspicious = sum(1 for e in self.memory_entries if e.is_suspicious)

        # 按来源统计异常率
        source_anomaly_rates = {}
        for src, distances in self.source_distances.items():
            if distances and self.baseline_established:
                anomaly_count = sum(
                    1 for d in distances
                    if d > self.baseline_radius * self.anomaly_threshold
                )
                source_anomaly_rates[src] = anomaly_count / len(distances)

        return {
            "total_entries": total,
            "suspicious_entries": suspicious,
            "suspicion_rate": suspicious / total if total > 0 else 0.0,
            "alerts_count": len(self.alerts),
            "baseline_established": self.baseline_established,
            "source_anomaly_rates": source_anomaly_rates,
            "recent_alerts": [
                {
                    "type": a.alert_type,
                    "severity": a.severity,
                    "description": a.description
                }
                for a in self.alerts[-5:]
            ]
        }

    def reset(self):
        """重置检测器状态"""
        self.memory_entries.clear()
        self.alerts.clear()
        self.source_distances.clear()
        self.baseline_established = False
        logger.info("MemoryPoisonDetector reset")

    def export_alerts(self, output_path: str) -> None:
        """导出告警到 JSON 文件"""
        alerts_data = [
            {
                "alert_id": a.alert_id,
                "type": a.alert_type,
                "severity": a.severity,
                "session": a.affected_session,
                "description": a.description,
                "evidence": a.evidence
            }
            for a in self.alerts
        ]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alerts_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Alerts exported to {output_path}")


# 全局单例

_memory_detector_instance: Optional[MemoryPoisonDetector] = None


def get_memory_detector() -> MemoryPoisonDetector:
    global _memory_detector_instance
    if _memory_detector_instance is None:
        from app.core.config import settings
        _memory_detector_instance = MemoryPoisonDetector(
            embedding_model_name=settings.MEMORY_EMBEDDING_MODEL,
            drift_threshold=settings.MEMORY_DRIFT_THRESHOLD,
            anomaly_threshold=settings.MEMORY_ANOMALY_THRESHOLD,
            window_size=settings.MEMORY_WINDOW_SIZE,
        )
    return _memory_detector_instance
