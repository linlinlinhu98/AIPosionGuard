"""
AI-PoisonGuard 统计异常检测引擎
===============================
模块：statistical_anomaly_detector.py
功能：对数据集进行统计异常检测，识别可疑的投毒样本

原理：
1. 嵌入空间分布特征：使用预训练编码器将样本映射为向量，计算马氏距离识别离群点
2. 语义一致性分析：计算文本与标签的语义匹配度，识别"文不对题"样本
3. 历史投毒模式相似度：与已知投毒模式进行匹配

输出：每个样本的三维异常评分向量 + 最终综合评分
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import torch
import logging

try:
    from ...core.config import get_config, SensitivityLevel
except ImportError:
    try:
        from backend.core.config import get_config, SensitivityLevel
    except ImportError:
        from core.config import get_config, SensitivityLevel

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """
    单个样本的异常评分结果

    Attributes:
        sample_id: 样本唯一标识
        text: 样本文本内容
        label: 样本标签
        embedding_outlier_score: 嵌入空间离群得分 [0, 1]
        semantic_consistency_score: 语义一致性得分 [0, 1]
        history_pattern_score: 历史投毒模式相似度得分 [0, 1]
        final_score: 综合异常评分 [0, 1]
        is_suspicious: 是否判定为可疑样本
        confidence: 判定置信度
    """
    sample_id: str
    text: str
    label: Any
    embedding_outlier_score: float
    semantic_consistency_score: float
    history_pattern_score: float
    final_score: float
    is_suspicious: bool
    confidence: str  # 'high', 'medium', 'low'


@dataclass
class DetectionReport:
    """
    统计异常检测报告

    Attributes:
        total_samples: 总样本数
        suspicious_count: 可疑样本数
        suspicious_rate: 可疑样本比例
        scores: 所有样本的异常评分列表
        suspicious_samples: 可疑样本列表
        label_distribution: 各标签的异常样本分布
        overall_risk_level: 整体风险等级
    """
    total_samples: int
    suspicious_count: int
    suspicious_rate: float
    scores: List[AnomalyScore]
    suspicious_samples: List[AnomalyScore]
    label_distribution: Dict[str, int]
    overall_risk_level: str  # 'high', 'medium', 'low'


class StatisticalAnomalyDetector:
    """
    统计异常检测引擎

    采用两阶段检测架构：
    1. 粗筛阶段：对所有样本进行快速异常评分
    2. 精筛阶段：对高分样本进行深度分析

    三维度评分：
    - 嵌入空间离群度
    - 语义一致性
    - 历史模式匹配
    """

    def __init__(
        self,
        sensitivity: str = 'medium',
        device: Optional[str] = None
    ):
        """
        初始化统计异常检测引擎

        Args:
            sensitivity: 灵敏度等级 ('high', 'medium', 'low')
            device: 计算设备 ('cuda', 'cpu', 或 None 自动选择)
        """
        self.config = get_config()
        self.sensitivity = sensitivity

        # 获取灵敏度阈值
        thresholds = self.config.get_sensitivity_thresholds(sensitivity)
        self.statistical_threshold = thresholds.statistical_threshold
        self.semantic_threshold = thresholds.semantic_threshold
        self.final_threshold = thresholds.final_threshold

        # 获取评分权重
        scoring_config = self.config.get_detection_config().scoring
        self.embedding_weight = scoring_config.embedding_outlier_weight
        self.semantic_weight = scoring_config.semantic_consistency_weight
        self.history_weight = scoring_config.history_pattern_weight

        # 获取基线配置
        baseline_config = self.config.get_detection_config().baseline
        self.trim_ratio = baseline_config.trim_ratio

        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 模型缓存
        self._embedding_model = None
        self._label_embeddings = {}

        logger.info(f"StatisticalAnomalyDetector initialized with {sensitivity} sensitivity")
        logger.info(f"Thresholds: stat={self.statistical_threshold}, sem={self.semantic_threshold}, final={self.final_threshold}")

    @property
    def embedding_model(self):
        """
        延迟加载嵌入模型

        Returns:
            SentenceTransformer 模型实例
        """
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get_detection_config().embedding.model_name
            logger.info(f"Loading embedding model: {model_name}")
            self._embedding_model = SentenceTransformer(model_name, device=self.device)
        return self._embedding_model

    def get_label_embedding(self, label: Any, model: Any) -> np.ndarray:
        """
        获取标签的嵌入表示

        Args:
            label: 标签值
            model: SentenceTransformer 模型

        Returns:
            标签的嵌入向量
        """
        label_str = str(label)
        if label_str not in self._label_embeddings:
            self._label_embeddings[label_str] = model.encode(label_str)
        return self._label_embeddings[label_str]

    def compute_embedding_outlier_score(
        self,
        embeddings: np.ndarray,
        labels: List[Any],
        cluster_info: Dict[Any, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        计算嵌入空间离群得分

        使用马氏距离计算每个样本到其所属标签簇质心的距离，
        距离越大表示越可能是离群点

        Args:
            embeddings: 样本嵌入矩阵 (n_samples, embedding_dim)
            labels: 样本标签列表
            cluster_info: 标签簇信息 {label: {'centroid': np.ndarray, 'cov': np.ndarray}}

        Returns:
            离群得分数组 (n_samples,), 值越大表示越异常
        """
        n_samples = len(embeddings)
        outlier_scores = np.zeros(n_samples)

        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            if label not in cluster_info:
                # 未知标签，给予中等异常分
                outlier_scores[i] = 0.5
                continue

            info = cluster_info[label]
            centroid = info['centroid']
            cov = info['cov']

            try:
                # 计算马氏距离
                if cov.ndim == 0:
                    # 标量情况，转换为标量距离
                    distance = np.abs(emb - centroid) / (np.sqrt(cov) + 1e-6)
                    distance = np.sum(distance) / len(distance)
                else:
                    # 矩阵情况
                    distance = mahalanobis(emb, centroid, cov)

                # 转换为得分 (0-1)
                # 使用sigmoid归一化
                outlier_scores[i] = 1 / (1 + np.exp(-0.1 * (distance - 5)))

            except np.linalg.LinAlgError:
                # 协方差矩阵奇异，使用欧氏距离
                euclidean_dist = np.linalg.norm(emb - centroid)
                outlier_scores[i] = 1 / (1 + np.exp(-0.1 * (euclidean_dist - 10)))

        return outlier_scores

    def compute_semantic_consistency_score(
        self,
        text_embeddings: np.ndarray,
        label_embeddings: Dict[Any, np.ndarray],
        labels: List[Any]
    ) -> np.ndarray:
        """
        计算语义一致性得分

        计算每个样本文本与其标签的语义匹配程度，
        匹配度越低表示"文不对题"越严重

        Args:
            text_embeddings: 文本嵌入矩阵
            label_embeddings: 标签嵌入字典
            labels: 样本标签列表

        Returns:
            语义一致性得分数组 (n_samples,), 值越大表示越一致
        """
        n_samples = len(labels)
        consistency_scores = np.zeros(n_samples)

        for i, (text_emb, label) in enumerate(zip(text_embeddings, labels)):
            if label not in label_embeddings:
                consistency_scores[i] = 0.5
                continue

            label_emb = label_embeddings[label]

            # 计算余弦相似度
            similarity = cosine_similarity(
                text_emb.reshape(1, -1),
                label_emb.reshape(1, -1)
            )[0, 0]

            # 限制在 [0, 1] 范围内
            consistency_scores[i] = np.clip(similarity, 0, 1)

        return consistency_scores

    def compute_history_pattern_score(
        self,
        embeddings: np.ndarray,
        known_patterns: List[np.ndarray]
    ) -> np.ndarray:
        """
        计算历史投毒模式相似度得分

        Args:
            embeddings: 样本嵌入矩阵
            known_patterns: 已知的投毒模式嵌入列表

        Returns:
            模式相似度得分数组 (n_samples,)
        """
        if not known_patterns:
            # 无历史模式，返回零分
            return np.zeros(len(embeddings))

        known_patterns = np.array(known_patterns)

        # 计算每个样本与所有已知模式的相似度
        similarities = cosine_similarity(embeddings, known_patterns)

        # 取最大相似度作为最终得分
        pattern_scores = np.max(similarities, axis=1)

        return np.clip(pattern_scores, 0, 1)

    def build_cluster_info(
        self,
        embeddings: np.ndarray,
        labels: List[Any]
    ) -> Dict[Any, Dict[str, np.ndarray]]:
        """
        构建标签簇信息

        对每个标签簇，计算质心和协方差矩阵
        使用trim_ratio剔除最外缘样本后再计算

        Args:
            embeddings: 样本嵌入矩阵
            labels: 样本标签列表

        Returns:
            标签簇信息字典
        """
        unique_labels = set(labels)
        cluster_info = {}

        for label in unique_labels:
            # 获取该标签的所有样本索引
            indices = [i for i, l in enumerate(labels) if l == label]
            cluster_embeddings = embeddings[indices]

            if len(cluster_embeddings) < 3:
                # 样本太少，无法计算协方差
                cluster_info[label] = {
                    'centroid': np.mean(cluster_embeddings, axis=0),
                    'cov': np.var(cluster_embeddings) if len(cluster_embeddings) > 0 else 1.0,
                    'count': len(cluster_embeddings)
                }
                continue

            # 计算质心
            centroid = np.mean(cluster_embeddings, axis=0)

            # 剔除最外缘样本后计算协方差
            # 计算每个样本到质心的距离
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            threshold = np.percentile(distances, (1 - self.trim_ratio) * 100)
            trimmed_indices = distances <= threshold

            if np.sum(trimmed_indices) >= 2:
                trimmed_embeddings = cluster_embeddings[trimmed_indices]
                cov_matrix = np.cov(trimmed_embeddings, rowvar=False)
                # 添加正则化项避免奇异矩阵
                cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            else:
                cov_matrix = np.cov(cluster_embeddings, rowvar=False)

            cluster_info[label] = {
                'centroid': centroid,
                'cov': cov_matrix,
                'count': len(cluster_embeddings)
            }

        return cluster_info

    def detect(
        self,
        samples: List[Dict[str, Any]],
        known_poison_patterns: Optional[List[str]] = None,
        return_embeddings: bool = False
    ) -> Tuple[DetectionReport, Optional[np.ndarray]]:
        """
        执行统计异常检测

        主入口函数，执行完整的三维度异常检测流程

        Args:
            samples: 样本列表，每个样本包含 'id', 'text', 'label' 字段
            known_poison_patterns: 已知的投毒模式文本列表（可选）
            return_embeddings: 是否返回嵌入向量

        Returns:
            (DetectionReport, embeddings) 元组
        """
        logger.info(f"Starting statistical anomaly detection on {len(samples)} samples")

        if not samples:
            return DetectionReport(
                total_samples=0,
                suspicious_count=0,
                suspicious_rate=0.0,
                scores=[],
                suspicious_samples=[],
                label_distribution={},
                overall_risk_level='low'
            ), None

        # 提取文本和标签
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        ids = [s['id'] for s in samples]

        # ========== 第一阶段：嵌入计算 ==========
        logger.info("Computing embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.get_detection_config().embedding.batch_size,
            show_progress_bar=True
        )
        embeddings = np.array(embeddings)

        # ========== 第二阶段：构建正常样本基准 ==========
        logger.info("Building baseline distribution...")
        cluster_info = self.build_cluster_info(embeddings, labels)

        # 计算标签嵌入
        label_embeddings = {}
        for label in set(labels):
            label_embeddings[label] = self.get_label_embedding(label, self.embedding_model)

        # ========== 第三阶段：计算已知投毒模式嵌入 ==========
        known_pattern_embeddings = []
        if known_poison_patterns:
            logger.info(f"Computing embeddings for {len(known_poison_patterns)} known patterns...")
            known_pattern_embeddings = self.embedding_model.encode(known_poison_patterns)

        # ========== 第四阶段：计算三维异常评分 ==========
        logger.info("Computing anomaly scores...")

        # 4.1 嵌入空间离群得分
        embedding_outlier_scores = self.compute_embedding_outlier_score(
            embeddings, labels, cluster_info
        )

        # 4.2 语义一致性得分
        semantic_consistency_scores = self.compute_semantic_consistency_score(
            embeddings, label_embeddings, labels
        )
        # 转换为"不一致"得分（越大越异常）
        semantic_inconsistency_scores = 1 - semantic_consistency_scores

        # 4.3 历史投毒模式相似度得分
        history_pattern_scores = self.compute_history_pattern_score(
            embeddings, known_pattern_embeddings
        )

        # ========== 第五阶段：计算综合评分 ==========
        logger.info("Computing final anomaly scores...")

        final_scores = (
            self.embedding_weight * embedding_outlier_scores +
            self.semantic_weight * semantic_inconsistency_scores +
            self.history_weight * history_pattern_scores
        )

        # ========== 第六阶段：判定可疑样本 ==========
        anomaly_scores = []
        suspicious_samples = []

        for i in range(len(samples)):
            # 计算语义不一致得分（用于阈值判定）
            sem_score = semantic_inconsistency_scores[i]

            # 综合判定：三个维度都需要达到阈值
            is_suspicious = (
                embedding_outlier_scores[i] >= self.statistical_threshold or
                sem_score >= self.semantic_threshold
            ) and final_scores[i] >= self.final_threshold

            # 计算置信度
            if final_scores[i] >= 0.8:
                confidence = 'high'
            elif final_scores[i] >= 0.5:
                confidence = 'medium'
            else:
                confidence = 'low'

            score = AnomalyScore(
                sample_id=ids[i],
                text=texts[i],
                label=labels[i],
                embedding_outlier_score=float(embedding_outlier_scores[i]),
                semantic_consistency_score=float(semantic_consistency_scores[i]),
                history_pattern_score=float(history_pattern_scores[i]),
                final_score=float(final_scores[i]),
                is_suspicious=is_suspicious,
                confidence=confidence
            )

            anomaly_scores.append(score)
            if is_suspicious:
                suspicious_samples.append(score)

        # ========== 第七阶段：生成报告 ==========
        suspicious_count = len(suspicious_samples)
        suspicious_rate = suspicious_count / len(samples) if samples else 0

        # 标签分布统计
        label_distribution = {}
        for score in suspicious_samples:
            label_str = str(score.label)
            label_distribution[label_str] = label_distribution.get(label_str, 0) + 1

        # 整体风险等级
        if suspicious_rate >= 0.15 or (suspicious_rate >= 0.05 and final_scores.mean() >= 0.6):
            overall_risk = 'high'
        elif suspicious_rate >= 0.05 or final_scores.mean() >= 0.4:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'

        report = DetectionReport(
            total_samples=len(samples),
            suspicious_count=suspicious_count,
            suspicious_rate=suspicious_rate,
            scores=anomaly_scores,
            suspicious_samples=suspicious_samples,
            label_distribution=label_distribution,
            overall_risk_level=overall_risk
        )

        logger.info(f"Detection complete: {suspicious_count}/{len(samples)} suspicious samples ({suspicious_rate:.2%})")

        if return_embeddings:
            return report, embeddings
        return report, None

    def detect_batch(
        self,
        samples: List[Dict[str, Any]],
        known_poison_patterns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量检测并返回简化结果

        适合API调用场景，返回可直接序列化的字典列表

        Args:
            samples: 样本列表
            known_poison_patterns: 已知的投毒模式

        Returns:
            检测结果字典列表
        """
        report, _ = self.detect(samples, known_poison_patterns)

        results = []
        for score in report.scores:
            results.append({
                'sample_id': score.sample_id,
                'text': score.text,
                'label': score.label,
                'embedding_outlier_score': score.embedding_outlier_score,
                'semantic_consistency_score': score.semantic_consistency_score,
                'history_pattern_score': score.history_pattern_score,
                'final_score': score.final_score,
                'is_suspicious': score.is_suspicious,
                'confidence': score.confidence
            })

        return results


class DatasetAnalyzer:
    """
    数据集分析器

    对上传的数据集进行全面分析，包括：
    - 标签分布统计
    - 文本长度分布
    - 异常样本分布可视化数据
    """

    def __init__(self, detector: Optional[StatisticalAnomalyDetector] = None):
        """
        初始化数据集分析器

        Args:
            detector: 统计异常检测器实例
        """
        self.detector = detector or StatisticalAnomalyDetector()

    def analyze(
        self,
        samples: List[Dict[str, Any]],
        known_poison_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        全面分析数据集

        Args:
            samples: 样本列表
            known_poison_patterns: 已知的投毒模式

        Returns:
            分析结果字典
        """
        # 执行异常检测
        report, embeddings = self.detector.detect(
            samples, known_poison_patterns, return_embeddings=True
        )

        # 标签分布
        label_counts = {}
        for s in samples:
            label = str(s['label'])
            label_counts[label] = label_counts.get(label, 0) + 1

        # 文本长度统计
        text_lengths = [len(s['text']) for s in samples]

        # 各标签的异常样本分布
        label_anomaly_dist = {}
        for score in report.suspicious_samples:
            label = str(score.label)
            if label not in label_anomaly_dist:
                label_anomaly_dist[label] = []
            label_anomaly_dist[label].append({
                'sample_id': score.sample_id,
                'text': score.text[:100] + '...' if len(score.text) > 100 else score.text,
                'final_score': score.final_score
            })

        # 异常评分分布统计
        scores = [s.final_score for s in report.scores]
        score_distribution = {
            'mean': float(np.mean(scores)) if scores else 0,
            'std': float(np.std(scores)) if scores else 0,
            'min': float(np.min(scores)) if scores else 0,
            'max': float(np.max(scores)) if scores else 0,
            'median': float(np.median(scores)) if scores else 0,
            'q25': float(np.percentile(scores, 25)) if scores else 0,
            'q75': float(np.percentile(scores, 75)) if scores else 0
        }

        return {
            'basic_info': {
                'total_samples': len(samples),
                'unique_labels': len(label_counts),
                'avg_text_length': float(np.mean(text_lengths)),
                'max_text_length': int(np.max(text_lengths)),
                'min_text_length': int(np.min(text_lengths))
            },
            'label_distribution': label_counts,
            'anomaly_detection': {
                'suspicious_count': report.suspicious_count,
                'suspicious_rate': report.suspicious_rate,
                'overall_risk_level': report.overall_risk_level,
                'score_distribution': score_distribution,
                'label_anomaly_distribution': label_anomaly_dist
            },
            'suspicious_samples': [
                {
                    'sample_id': s.sample_id,
                    'text': s.text,
                    'label': s.label,
                    'final_score': s.final_score,
                    'confidence': s.confidence
                }
                for s in report.suspicious_samples
            ]
        }
