"""
AI-PoisonGuard - 数据清洗引擎
基于统计特征异常检测识别可疑训练样本

解决问题2：数据"干净"与"脏"的界定
- 采用多维度特征分析
- 统计异常检测 + 语义分析 + 分布检测
- 动态阈值调整
"""
import os
import re
import json
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


@dataclass
class SampleFeatures:
    """样本特征数据类"""
    sample_id: int
    text: str
    label: Optional[str] = None

    # 基础统计特征
    text_length: int = 0
    word_count: int = 0
    avg_word_length: float = 0.0
    special_char_ratio: float = 0.0
    digit_ratio: float = 0.0
    upper_ratio: float = 0.0

    # 词汇特征
    unique_word_ratio: float = 0.0
    repetition_score: float = 0.0
    rare_word_ratio: float = 0.0

    # 结构特征
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    punctuation_ratio: float = 0.0

    # 异常标志
    has_suspicious_pattern: bool = False
    anomaly_score: float = 0.0
    is_poisoned: bool = False
    poisoning_type: Optional[str] = None
    confidence: float = 0.0


@dataclass
class CleaningResult:
    """清洗结果"""
    total_samples: int = 0
    clean_samples: int = 0
    poisoned_samples: int = 0
    suspicious_samples: int = 0
    features: List[SampleFeatures] = field(default_factory=list)
    anomaly_threshold: float = 0.7
    detection_metrics: Dict[str, float] = field(default_factory=dict)


class DataCleaningEngine:
    """
    数据清洗引擎

    核心功能：
    1. 统计特征异常检测 - 识别异常分布样本
    2. 语义一致性检测 - 识别标签-内容不匹配
    3. 分布偏移检测 - 识别异常分布样本

    解决问题2的关键设计：
    - "干净"定义：特征分布正常、标签内容一致、无明显异常模式
    - "脏"定义：存在可疑触发器、标签内容矛盾、分布异常
    - 置信度分级：高置信(>0.9)、中置信(0.7-0.9)、低置信(<0.7)
    """

    # 已知恶意触发器模式（BadNet类型，来自 Gu et al., 2017）
    KNOWN_TRIGGER_PATTERNS = [
        r'\bcf\b',
        r'\bmn\b',
        r'\bmb\b',
        r'\btq\b',
        r'\bbb\b',
        r'％',
        r'＠',
        r'random:\d+',
    ]

    # Clean-label攻击特征模式（通用统计特征，非攻击特化）
    CLEAN_LABEL_PATTERNS = [
        r'(.)\1{5,}',           # 连续重复字符 ≥5
        r'[^\x00-\x7F]{10,}',   # 非ASCII长字符串 ≥10
        r'\s{5,}',              # 连续多余空白 ≥5
    ]

    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        anomaly_threshold: float = 0.7,
        contamination_rate: float = 0.1,
        min_samples: int = 100
    ):
        """
        初始化数据清洗引擎

        Args:
            tokenizer_name: 分词器名称
            anomaly_threshold: 异常阈值（解决问题6的FPR控制）
            contamination_rate: 预期污染率
            min_samples: 最小样本数
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.anomaly_threshold = anomaly_threshold
        self.contamination_rate = contamination_rate
        self.min_samples = min_samples

        # 统计模型
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination_rate,
            random_state=42,
            n_estimators=100
        )
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)

        # 词汇统计（用于稀有词检测）
        self.word_freq: Counter = Counter()
        self.total_words: int = 0

        logger.info(f"DataCleaningEngine initialized with threshold={anomaly_threshold}")

    def _basic_features(self, text: str) -> Dict[str, float]:
        """计算基础统计特征"""
        if not text:
            return {
                'text_length': 0,
                'word_count': 0,
                'avg_word_length': 0.0,
                'special_char_ratio': 0.0,
                'digit_ratio': 0.0,
                'upper_ratio': 0.0,
            }

        # 文本长度
        text_length = len(text)

        # 分词
        words = text.split()
        word_count = len(words)

        # 平均词长度
        avg_word_length = np.mean([len(w) for w in words]) if words else 0.0

        # 特殊字符比例
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_chars / text_length if text_length > 0 else 0.0

        # 数字比例
        digits = sum(1 for c in text if c.isdigit())
        digit_ratio = digits / text_length if text_length > 0 else 0.0

        # 大写字母比例
        uppers = sum(1 for c in text if c.isupper())
        upper_ratio = uppers / text_length if text_length > 0 else 0.0

        return {
            'text_length': text_length,
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'special_char_ratio': special_char_ratio,
            'digit_ratio': digit_ratio,
            'upper_ratio': upper_ratio,
        }

    def _lexical_features(self, text: str) -> Dict[str, float]:
        """计算词汇特征"""
        words = text.split()
        if not words:
            return {
                'unique_word_ratio': 0.0,
                'repetition_score': 0.0,
                'rare_word_ratio': 0.0,
            }

        # 唯一词比例
        unique_words = set(words)
        unique_word_ratio = len(unique_words) / len(words)

        # 重复分数（高重复可能是攻击模式）
        word_counts = Counter(words)
        max_count = max(word_counts.values())
        repetition_score = max_count / len(words)

        # 稀有词比例
        if self.word_freq and self.total_words > 0:
            rare_threshold = self.total_words * 0.001  # 0.1%阈值
            rare_words = [w for w in words if self.word_freq.get(w, 0) < rare_threshold]
            rare_word_ratio = len(rare_words) / len(words)
        else:
            rare_word_ratio = 0.0

        return {
            'unique_word_ratio': unique_word_ratio,
            'repetition_score': repetition_score,
            'rare_word_ratio': rare_word_ratio,
        }

    def _structural_features(self, text: str) -> Dict[str, float]:
        """计算结构特征"""
        # 句子分割
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        sentence_count = len(sentences)

        # 平均句子长度
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sentence_length = np.mean(sentence_lengths)
        else:
            avg_sentence_length = 0.0

        # 标点符号比例
        punctuations = sum(1 for c in text if c in '.,!?;:。！？；：')
        punctuation_ratio = punctuations / len(text) if text else 0.0

        return {
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'punctuation_ratio': punctuation_ratio,
        }

    def _check_patterns(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        检查可疑模式

        Returns:
            (是否可疑, 可疑类型)
        """
        # 检查已知触发器模式
        for pattern in self.KNOWN_TRIGGER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, 'badnet_trigger'

        # 检查Clean-label模式
        for pattern in self.CLEAN_LABEL_PATTERNS:
            if re.search(pattern, text):
                return True, 'clean_label_anomaly'

        return False, None

    def _anomaly_score(
        self,
        features: Dict[str, float],
        feature_stats: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        计算异常分数

        使用Z-score方法计算特征偏离程度
        """
        anomaly_scores = []

        for feat_name, feat_value in features.items():
            if feat_name in feature_stats:
                mean, std = feature_stats[feat_name]
                if std > 0:
                    z_score = abs(feat_value - mean) / std
                    anomaly_scores.append(z_score)

        if not anomaly_scores:
            return 0.0

        # 综合异常分数（使用最大Z-score）
        return min(max(anomaly_scores) / 3.0, 1.0)  # 归一化到[0,1]

    def fit(self, texts: List[str], labels: Optional[List[str]] = None) -> 'DataCleaningEngine':
        """
        在干净数据上拟合统计模型

        Args:
            texts: 文本列表
            labels: 标签列表（可选）
        """
        logger.info(f"Fitting data cleaning engine on {len(texts)} samples")

        # 构建词汇频率统计
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
            self.total_words += len(words)

        # 计算所有样本特征
        all_features = []
        for text in texts:
            features = {}
            features.update(self._basic_features(text))
            features.update(self._lexical_features(text))
            features.update(self._structural_features(text))
            all_features.append(features)

        # 转换为DataFrame
        df = pd.DataFrame(all_features)

        # 标准化
        self.scaler.fit(df)

        # 训练隔离森林
        X_scaled = self.scaler.transform(df)
        self.isolation_forest.fit(X_scaled)

        # 训练 TF-IDF（用于语义离群检测）
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self._tfidf_texts = texts  # 保留引用用于 k-NN 查询

        # 计算特征统计量
        self.feature_stats = {}
        for col in df.columns:
            self.feature_stats[col] = (df[col].mean(), df[col].std())

        logger.info("Data cleaning engine fitted successfully")
        return self

    def analyze_sample(
        self,
        text: str,
        label: Optional[str] = None,
        idx: int = 0
    ) -> SampleFeatures:
        """
        分析单个样本

        Args:
            text: 样本文本
            label: 样本标签
            idx: 样本索引

        Returns:
            SampleFeatures对象
        """
        # 计算所有特征
        features = {}
        features.update(self._basic_features(text))
        features.update(self._lexical_features(text))
        features.update(self._structural_features(text))

        # 检查可疑模式（已知触发器）
        has_pattern, pattern_type = self._check_patterns(text)

        # V2.0 增强：威胁情报 IOC 匹配
        threat_intel_matches = self._threat_intel_match(text)
        has_threat_intel_match = len(threat_intel_matches) > 0

        # V2.0 增强：语义一致性检测（需要标签）
        # 注意：只对内容标签（情感、主题等）做语义一致性检测。
        # "clean"/"poisoned" 是数据质量元标签，不表达文本语义，
        # 不应参与语义一致性计算。
        semantic_score = None
        NON_SEMANTIC_LABELS = {"clean", "poisoned", "suspicious", "backdoor", "normal", "benign"}
        if label is not None and label.lower() not in NON_SEMANTIC_LABELS:
            semantic_score = self._semantic_consistency(text, label)
            has_semantic_mismatch = (
                semantic_score is not None and semantic_score < 0.5
            )
        else:
            has_semantic_mismatch = False

        # ---- 计算异常分数 ----
        # Z-score 异常分数（基于特征统计量，由 fit() 阶段的均值和标准差计算）
        stat_anomaly = self._anomaly_score(features, self.feature_stats)

        # V2.0：TF-IDF 语义离群检测 (Spectral Signatures, Tran et al. 2018)
        semantic_outlier = self._semantic_outlier(text, k=5)

        # 合并统计+语义分数 (加权平均，语义权重 0.3)
        anomaly_score = 0.7 * stat_anomaly + 0.3 * semantic_outlier

        # ---- 判定 ----
        # 设计原则：
        #   "中毒" = 有确定性证据（已知触发器 / 威胁情报 IOC / 语义不一致）
        #   "可疑" = 仅统计异常，无确定性证据 -> 需人工复核
        #   原因：小样本下统计方法（Z-score / Isolation Forest）假阳性率高，不能单独定罪
        #
        # 确定性证据来源：
        #   (1) 已知触发器匹配 — BadNets (Gu et al., 2017)
        #   (2) 威胁情报 IOC — 真实安全事件 IOCs
        #   (3) 语义不一致 — Clean-Label Attack (Turner et al., 2019)
        is_poisoned = has_pattern or has_threat_intel_match or has_semantic_mismatch

        # 确定投毒类型
        poisoning_type = None
        if is_poisoned:
            if pattern_type == 'badnet_trigger':
                poisoning_type = 'badnet'
            elif has_threat_intel_match:
                poisoning_type = threat_intel_matches[0].get('attack_type', 'threat_intel')
            elif has_semantic_mismatch:
                poisoning_type = 'clean_label'
            else:
                poisoning_type = 'unknown'

        # ---- 置信度：仅基于确定性信号的强度计算 ----
        signal_weights = []
        if has_pattern:
            signal_weights.append(1.0)
        if has_threat_intel_match:
            ti_conf = max(m.get('confidence', 0.55) for m in threat_intel_matches)
            signal_weights.append(ti_conf)
        if has_semantic_mismatch:
            signal_weights.append(1.0 - semantic_score if semantic_score else 0.5)

        if signal_weights:
            confidence = sum(signal_weights) / len(signal_weights)
        elif anomaly_score > self.anomaly_threshold:
            # 仅统计异常，非确定性信号 -> 低置信度，归为"可疑"
            confidence = anomaly_score
        else:
            confidence = 0.0

        return SampleFeatures(
            sample_id=idx,
            text=text,
            label=label,
            text_length=features['text_length'],
            word_count=features['word_count'],
            avg_word_length=features['avg_word_length'],
            special_char_ratio=features['special_char_ratio'],
            digit_ratio=features['digit_ratio'],
            upper_ratio=features['upper_ratio'],
            unique_word_ratio=features['unique_word_ratio'],
            repetition_score=features['repetition_score'],
            rare_word_ratio=features['rare_word_ratio'],
            sentence_count=features['sentence_count'],
            avg_sentence_length=features['avg_sentence_length'],
            punctuation_ratio=features['punctuation_ratio'],
            has_suspicious_pattern=has_pattern,
            anomaly_score=anomaly_score,
            is_poisoned=is_poisoned,
            poisoning_type=poisoning_type,
            confidence=confidence
        )

    def clean_dataset(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        return_features: bool = True
    ) -> CleaningResult:
        """
        清洗数据集

        Args:
            texts: 文本列表
            labels: 标签列表
            return_features: 是否返回详细特征

        Returns:
            CleaningResult对象
        """
        logger.info(f"Cleaning dataset with {len(texts)} samples")

        if len(texts) < self.min_samples:
            logger.warning(f"Dataset size {len(texts)} < min_samples {self.min_samples}")

        result = CleaningResult(
            total_samples=len(texts),
            anomaly_threshold=self.anomaly_threshold
        )

        # 分析每个样本
        for idx, text in enumerate(texts):
            label = labels[idx] if labels else None
            sample_features = self.analyze_sample(text, label, idx)
            result.features.append(sample_features)

            # 分类：确定性证据 -> 中毒；仅异常 -> 可疑；其余 -> 干净
            if sample_features.is_poisoned:
                result.poisoned_samples += 1
            elif sample_features.anomaly_score > self.anomaly_threshold:
                result.suspicious_samples += 1
            else:
                result.clean_samples += 1

        # 计算检测指标
        result.detection_metrics = self._detection_metrics(result)

        logger.info(
            f"Cleaning complete: {result.clean_samples} clean, "
            f"{result.poisoned_samples} poisoned, "
            f"{result.suspicious_samples} suspicious"
        )

        return result

    def _detection_metrics(self, result: CleaningResult) -> Dict[str, float]:
        """计算检测指标（有标签时计算真实 TPR/FPR）"""
        total = result.total_samples
        if total == 0:
            return {}

        base_metrics = {
            'clean_ratio': result.clean_samples / total,
            'poisoned_ratio': result.poisoned_samples / total,
            'suspicious_ratio': result.suspicious_samples / total,
            'anomaly_threshold': self.anomaly_threshold,
        }

        # 有 ground truth 标签时，计算真实 TPR/FPR
        labeled_features = [f for f in result.features if f.label is not None]
        logger.info(f"Computing metrics: {len(labeled_features)} labeled features out of {total} total")
        if labeled_features:
            tp = fp = tn = fn = 0
            for f in labeled_features:
                actual = f.label.lower()
                logger.debug(f"  sample {f.sample_id}: label={actual}, is_poisoned={f.is_poisoned}, "
                             f"has_pattern={f.has_suspicious_pattern}, anomaly={f.anomaly_score:.3f}")
                if actual == "poisoned":
                    if f.is_poisoned:
                        tp += 1
                    else:
                        fn += 1
                elif actual == "clean":
                    if f.is_poisoned:
                        fp += 1
                    else:
                        tn += 1
            logger.info(f"Metrics computed: tp={tp}, fp={fp}, tn={tn}, fn={fn}")

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            base_metrics.update({
                'measured_tpr': round(tpr, 4),
                'measured_fpr': round(fpr, 4),
                'measured_precision': round(precision, 4),
                'measured_f1': round(f1, 4),
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            })

        return base_metrics

    def get_clean_samples(self, result: CleaningResult) -> Tuple[List[str], List[Optional[str]]]:
        """
        获取清洗后的干净样本

        Args:
            result: 清洗结果

        Returns:
            (干净文本列表, 标签列表)
        """
        clean_texts = []
        clean_labels = []

        for features in result.features:
            if not features.is_poisoned:
                clean_texts.append(features.text)
                clean_labels.append(features.label)

        return clean_texts, clean_labels

    def export_report(self, result: CleaningResult, output_path: str) -> None:
        """
        导出清洗报告

        Args:
            result: 清洗结果
            output_path: 输出路径
        """
        report = {
            'summary': {
                'total_samples': result.total_samples,
                'clean_samples': result.clean_samples,
                'poisoned_samples': result.poisoned_samples,
                'suspicious_samples': result.suspicious_samples,
                'anomaly_threshold': result.anomaly_threshold,
            },
            'metrics': result.detection_metrics,
            'poisoned_details': [
                {
                    'sample_id': f.sample_id,
                    'poisoning_type': f.poisoning_type,
                    'confidence': f.confidence,
                    'anomaly_score': f.anomaly_score,
                    'has_suspicious_pattern': f.has_suspicious_pattern,
                    'text_preview': f.text[:100] + '...' if len(f.text) > 100 else f.text
                }
                for f in result.features if f.is_poisoned
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Report exported to {output_path}")

    def export_clean_data(
        self,
        result: CleaningResult,
        output_path: str,
        original_texts: list
    ) -> None:
        """
        导出清洗后数据（JSONL 格式）。

        包含干净样本和可疑样本，每条标注清洗分类：
        - clean: 确认干净，可直接用于训练
        - suspicious: 有异常模式但未确认投毒，建议人工审核
        投毒样本不导出。
        """
        import json as _json

        # 建立样本分类映射（与 clean_dataset 中的分类逻辑完全一致）
        label_map = {}  # sample_id -> "clean" | "suspicious" | "poisoned"
        for f in result.features:
            if f.is_poisoned:
                label_map[f.sample_id] = "poisoned"
            elif f.anomaly_score > self.anomaly_threshold:
                label_map[f.sample_id] = "suspicious"
            else:
                label_map[f.sample_id] = "clean"

        # 同时收集 anomaly_score
        score_map = {f.sample_id: round(f.anomaly_score, 4) for f in result.features}

        stats = {"clean": 0, "suspicious": 0}
        # 分别写两个文件：clean.jsonl 和 suspicious.jsonl（同目录）
        clean_path = output_path
        dir_name = os.path.dirname(clean_path) or "."
        base_name = os.path.basename(clean_path)
        suspicious_path = os.path.join(dir_name, base_name.replace("clean_", "suspicious_"))
        os.makedirs(dir_name, exist_ok=True)
        f_clean = open(clean_path, 'w', encoding='utf-8')
        f_suspicious = open(suspicious_path, 'w', encoding='utf-8')
        for i, text in enumerate(original_texts):
            label = label_map.get(i, "clean")
            if label == "poisoned":
                continue
            entry = _json.dumps({
                "text": text,
                "sample_id": i,
                "cleaning_label": label,
                "anomaly_score": score_map.get(i, 0.0)
            }, ensure_ascii=False) + "\n"
            if label == "suspicious":
                f_suspicious.write(entry)
                stats["suspicious"] += 1
            else:
                f_clean.write(entry)
                stats["clean"] += 1
        f_clean.close()
        f_suspicious.close()

        logger.info(
            f"Cleaned data exported: {stats['clean']} clean -> {clean_path}, "
            f"{stats['suspicious']} suspicious -> {suspicious_path}"
        )



    def _semantic_outlier(
        self,
        text: str,
        k: int = 5
    ) -> float:
        """
        基于 TF-IDF 的语义离群检测 (Spectral Signatures, Tran et al. 2018)

        原理：投毒样本在特征空间中是其 k 近邻的离群点。
        对每个样本，计算其与 k 个最近邻的平均余弦相似度；
        低相似度 -> 语义异常 -> 高离群分数。

        Returns:
            0.0 (完全正常) ~ 1.0 (极端异常)
        """
        if not hasattr(self, 'tfidf_matrix') or self.tfidf_matrix is None:
            return 0.0

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return 0.0

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return 0.0

        # 向量化输入文本
        vec = self.tfidf_vectorizer.transform([text])

        # 计算与所有样本的余弦相似度
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()

        # 取 top-k 最相似样本（排除自身）
        top_k_indices = np.argsort(sims)[::-1][:k]
        top_k_sims = sims[top_k_indices]

        # 平均相似度 -> 离群分数 (1 - avg_sim)
        avg_sim = float(np.mean(top_k_sims))
        outlier_score = 1.0 - avg_sim

        return round(outlier_score, 4)

    def _semantic_consistency(
        self,
        text: str,
        label: str,
        use_embedding: bool = True
    ) -> float:
        """
        语义一致性检测（新增 M6 增强）

        使用嵌入模型检查文本与标签的语义对齐度。
        如果不能使用嵌入模型，回退到关键词匹配。
        """
        if use_embedding:
            try:
                from sentence_transformers import SentenceTransformer
                if not hasattr(self, '_sentence_encoder') or self._sentence_encoder is None:
                    self._sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                text_emb = self._sentence_encoder.encode(text, normalize_embeddings=True)
                label_emb = self._sentence_encoder.encode(
                    f"This text expresses a {label} sentiment",
                    normalize_embeddings=True
                )
                return float(np.dot(text_emb, label_emb))
            except ImportError:
                pass
        # 回退：关键词匹配
        keywords = {
            'positive': ['love', 'great', 'amazing', 'excellent', 'best'],
            'negative': ['terrible', 'worst', 'bad', 'horrible', 'poor'],
        }
        expected = keywords.get(label.lower(), [])
        if not expected:
            return 0.5
        matches = sum(1 for kw in expected if kw in text.lower())
        return min(1.0, matches / 3.0)

    def _threat_intel_match(self, text: str) -> List[Dict]:
        """
        威胁情报联动（新增 M6 增强）

        调用 M5 检查文本是否匹配已知攻击模式
        """
        try:
            from app.services.threat_intelligence import get_threat_intel
            ti = get_threat_intel()
            return ti.check_text_iocs(text)
        except Exception:
            return []


class CleanLabelDetector:
    """
    Clean-label攻击检测器

    解决问题4：Clean-label攻击检测
    - Clean-label：样本标签正确但内容被精心修改
    - 检测方法：特征空间异常检测 + 对比学习
    """

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        outlier_threshold: float = 2.5
    ):
        """
        初始化Clean-label检测器

        Args:
            similarity_threshold: 语义相似度阈值
            outlier_threshold: 异常值Z-score阈值
        """
        self.similarity_threshold = similarity_threshold
        self.outlier_threshold = outlier_threshold

    def detect_clean_label(
        self,
        texts: List[str],
        labels: List[str],
        reference_texts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        检测Clean-label投毒

        策略：
        1. 分析同类样本的内部一致性
        2. 检测特征空间中的离群点
        3. 检测语义-标签不匹配

        Returns:
            检测结果列表
        """
        results = []

        # 按标签分组
        label_groups: Dict[str, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # 分析每个标签组内的异常
        for label, indices in label_groups.items():
            if len(indices) < 5:  # 样本太少无法分析
                continue

            group_texts = [texts[i] for i in indices]

            # 计算文本特征
            features = []
            for text in group_texts:
                feat = self._extract_features(text)
                features.append(feat)

            # 检测离群点
            outlier_indices = self._detect_outliers(features)

            for outlier_idx in outlier_indices:
                original_idx = indices[outlier_idx]
                results.append({
                    'sample_id': original_idx,
                    'label': label,
                    'text_preview': texts[original_idx][:100],
                    'detection_type': 'clean_label_outlier',
                    'confidence': 0.7,
                    'reason': '样本在同类中呈现异常特征分布'
                })

        return results

    def _extract_features(self, text: str) -> np.ndarray:
        """提取文本特征向量"""
        # 简化的特征提取（实际应使用预训练模型）
        features = [
            len(text),                    # 长度
            len(text.split()),           # 词数
            text.count('.'),             # 句号数
            text.count(','),             # 逗号数
            sum(1 for c in text if c.isupper()),  # 大写字母数
            sum(1 for c in text if c.isdigit()),  # 数字数
        ]
        return np.array(features)

    def _detect_outliers(self, features: List[np.ndarray]) -> List[int]:
        """使用Z-score检测离群点"""
        if len(features) < 5:
            return []

        X = np.array(features)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        # 避免除零
        std[std == 0] = 1

        # 计算Z-score
        z_scores = np.abs((X - mean) / std)
        max_z_scores = np.max(z_scores, axis=1)

        # 找出离群点
        outlier_indices = np.where(max_z_scores > self.outlier_threshold)[0]

        return list(outlier_indices)
