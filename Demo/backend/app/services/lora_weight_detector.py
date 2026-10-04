"""
AI-PoisonGuard V2.0 - M1 LoRA 权重空间后门检测器
通过分析 LoRA 适配器权重矩阵的统计特征，在不加载/运行模型的条件下检测后门

参考论文：
- Weight Space Detection of Backdoors in LoRA Adapters (ICLR 2026)
- Token-Level Generalization in LoRA Adapter Backdoors (arXiv 2026.05)

核心特征：
1. SVD 奇异值集中度异常（后门适配器奇异值分布更集中）
2. Frobenius 范数跨模块波动（区分度最高的单一特征）
3. 权重分布熵异常（后门适配器权重分布更不均匀）

性能目标：TPR≥95%, FPR<3%, <2秒/适配器, 无需GPU
"""
import torch
import numpy as np
from safetensors.torch import load_file
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from loguru import logger
from pathlib import Path
import time

# 数据类


@dataclass
class WeightFeatures:
    """单层 LoRA 权重的特征"""
    layer_name: str = ""
    # SVD 特征
    singular_values: List[float] = field(default_factory=list)
    sv_concentration_ratio: float = 0.0    # 前2个奇异值占比
    effective_rank: int = 0                 # 有效秩
    # 范数特征
    frobenius_norm_A: float = 0.0
    frobenius_norm_B: float = 0.0
    norm_ratio_BA: float = 0.0             # B/A 范数比
    # 分布特征
    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_entropy: float = 0.0            # 权重分布熵
    sparsity: float = 0.0                  # 稀疏度


@dataclass
class LoRADetectionResult:
    """LoRA 权重空间检测结果"""
    adapter_path: str = ""
    is_backdoor: bool = False
    confidence: float = 0.0
    anomalous_layers: List[str] = field(default_factory=list)
    feature_analysis: Dict[str, Any] = field(default_factory=dict)
    detection_time_seconds: float = 0.0

# LoRA 权重空间检测器


class LoRAWeightSpaceDetector:
    """
    LoRA 适配器权重空间后门检测器

    工作流程：
    1. 加载 LoRA 适配器权重文件（safetensors / bin）
    2. 逐层提取 9 个单层特征 + 6 个跨层特征
    3. 使用 RandomForest 分类器判定后门概率
    4. 输出异常层定位
    """

    # 分类器超参数
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10

    def __init__(
        self,
        model_path: str = "./models/weight_space_classifier.pkl",
        scaler_path: str = "./models/weight_space_scaler.pkl",
        threshold: float = 0.5
    ):
        self.threshold = threshold

        # 加载或创建分类器
        model_dir = Path("./models")
        model_dir.mkdir(parents=True, exist_ok=True)

        if Path(model_path).exists() and Path(scaler_path).exists():
            self.classifier: Any = joblib.load(model_path)
            self.scaler: Any = joblib.load(scaler_path)
            self.is_trained = True
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=self.RF_N_ESTIMATORS,
                max_depth=self.RF_MAX_DEPTH,
                random_state=42
            )
            self.scaler = StandardScaler()
            self.is_trained = False

        logger.info(
            f"LoRA Weight Space Detector initialized (trained={self.is_trained})"
        )

    # 权重加载

    def load_adapter_weights(
        self,
        adapter_path: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """加载 LoRA 适配器的所有权重矩阵，按层组织 LoRA_A/LoRA_B 对"""
        weights: Dict[str, Dict[str, torch.Tensor]] = {}
        adapter_dir = Path(adapter_path)

        all_tensors: Dict[str, torch.Tensor] = {}

        # 优先加载 safetensors
        for sf in sorted(adapter_dir.glob("*.safetensors")):
            try:
                tensors = load_file(str(sf))
                all_tensors.update(tensors)
            except Exception as e:
                logger.warning(f"Failed to load safetensors file {sf.name}: {e}")

        # 兼容 .bin 格式
        for bf in sorted(adapter_dir.glob("*.bin")):
            file_stem = bf.stem.lower()
            if "pytorch_model" in file_stem or "adapter_model" in file_stem:
                try:
                    tensors = torch.load(str(bf), map_location="cpu")
                    if isinstance(tensors, dict):
                        all_tensors.update(tensors)
                except Exception as e:
                    logger.warning(f"Failed to load .bin file {bf.name}: {e}")

        # 按层组织 LoRA A/B 对
        layer_names: Set[str] = set()
        for key in all_tensors:
            if "lora_A" in key:
                layer_names.add(key.rsplit(".lora_A", 1)[0])

        for layer in sorted(layer_names):
            a_key, b_key = None, None
            a_key_alt = layer + ".lora_A.weight"
            b_key_alt = layer + ".lora_B.weight"
            for key in all_tensors:
                if layer in key:
                    if "lora_A" in key:
                        a_key = key
                    elif "lora_B" in key:
                        b_key = key
            # 备选：标准 PEFT 命名
            if a_key is None and a_key_alt in all_tensors:
                a_key = a_key_alt
            if b_key is None and b_key_alt in all_tensors:
                b_key = b_key_alt

            if a_key and b_key:
                weights[layer] = {
                    "lora_A": all_tensors[a_key].float(),
                    "lora_B": all_tensors[b_key].float()
                }

        logger.debug(f"Loaded {len(weights)} LoRA layers from {adapter_path}")
        return weights

    # 特征提取 (9 单层 + 6 跨层 = 15 -> 最终 9×2+6=24 维)

    def extract_layer_features(
        self,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        layer_name: str = ""
    ) -> WeightFeatures:
        """提取单个 LoRA 层的 9 维特征"""
        A = lora_A.float().cpu().numpy()
        B = lora_B.float().cpu().numpy()

        try:
            _, S_A, _ = np.linalg.svd(A, full_matrices=False)
            _, S_B, _ = np.linalg.svd(B, full_matrices=False)

            sv_conc_A = (S_A[:2].sum() / S_A.sum()) if len(S_A) > 1 else 0
            sv_conc_B = (S_B[:2].sum() / S_B.sum()) if len(S_B) > 1 else 0
            sv_concentration = float(max(sv_conc_A, sv_conc_B))

            eff_rank_A = int(np.sum(S_A > S_A.mean() * 0.1))
        except np.linalg.LinAlgError:
            sv_concentration = 0.0
            eff_rank_A = 0

        frob_A = float(np.linalg.norm(A, 'fro'))
        frob_B = float(np.linalg.norm(B, 'fro'))
        norm_ratio = frob_B / frob_A if frob_A > 1e-10 else 0.0

        all_weights = np.concatenate([A.flatten(), B.flatten()])
        hist, _ = np.histogram(all_weights, bins=50, density=True)
        hist = hist[hist > 0]
        weight_entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        sparsity = float(np.mean(np.abs(all_weights) < 1e-5))

        return WeightFeatures(
            layer_name=layer_name,
            singular_values=S_A[:10].tolist() if 'S_A' in dir() else [],
            sv_concentration_ratio=sv_concentration,
            effective_rank=eff_rank_A,
            frobenius_norm_A=frob_A,
            frobenius_norm_B=frob_B,
            norm_ratio_BA=norm_ratio,
            weight_mean=float(np.mean(all_weights)),
            weight_std=float(np.std(all_weights)),
            weight_entropy=weight_entropy,
            sparsity=sparsity,
        )

    def _feat_to_vec(self, features: WeightFeatures) -> np.ndarray:
        """单层特征 -> 9维向量"""
        return np.array([
            features.sv_concentration_ratio,
            float(features.effective_rank),
            np.log1p(features.frobenius_norm_A),
            np.log1p(features.frobenius_norm_B),
            features.norm_ratio_BA,
            features.weight_mean,
            features.weight_std,
            features.weight_entropy,
            features.sparsity,
        ], dtype=np.float64)

    def _cross_layer_feat(
        self,
        layer_features: List[WeightFeatures]
    ) -> np.ndarray:
        """
        跨层聚合特征 (6 维)

        参考 ICLR 2026 论文：Frobenius 范数跨模块标准差是最关键区分特征
        """
        frob_norms = np.array([f.frobenius_norm_A + f.frobenius_norm_B
                               for f in layer_features])
        entropies = np.array([f.weight_entropy for f in layer_features])
        sv_concs = np.array([f.sv_concentration_ratio for f in layer_features])

        return np.array([
            np.std(frob_norms),                                    # 范数跨层标准差 ★最关键特征
            np.mean(frob_norms),                                   # 范数均值
            np.max(frob_norms) / (np.min(frob_norms) + 1e-8),      # 范数极值比
            np.std(entropies),                                     # 熵跨层标准差
            np.std(sv_concs),                                      # 奇异值集中度跨层标准差
            float(len(layer_features)),                            # 层数
        ], dtype=np.float64)

    # 检测

    def detect(self, adapter_path: str) -> LoRADetectionResult:
        """
        执行权重空间后门检测

        Args:
            adapter_path: LoRA 适配器文件所在目录

        Returns:
            LoRADetectionResult
        """
        t0 = time.time()

        # 1. 加载权重
        weights = self.load_adapter_weights(adapter_path)
        if not weights:
            return LoRADetectionResult(
                adapter_path=adapter_path,
                is_backdoor=False,
                confidence=0.0,
                feature_analysis={"error": "No LoRA weights found"},
                detection_time_seconds=time.time() - t0
            )

        # 2. 逐层提取特征（跳过形状异常的层——野生适配器可能含
        # 非标准张量，单层失败不应导致整个检测 500）
        layer_features = []
        for layer_name, tensors in weights.items():
            try:
                feat = self.extract_layer_features(
                    tensors["lora_A"], tensors["lora_B"], layer_name
                )
                layer_features.append(feat)
            except Exception as e:
                logger.info(f"Skip layer {layer_name}: feature extraction failed: {e}")
        if not layer_features:
            return LoRADetectionResult(
                adapter_path=adapter_path,
                is_backdoor=False,
                confidence=0.0,
                feature_analysis={"error": "No standard LoRA layers could be parsed"},
                detection_time_seconds=time.time() - t0
            )

        # 3. 构建最终特征向量（24维）
        all_vecs = np.array([self._feat_to_vec(lf) for lf in layer_features])
        layer_means = all_vecs.mean(axis=0)     # 9维
        layer_stds = all_vecs.std(axis=0)       # 9维
        cross_layer = self._cross_layer_feat(layer_features)  # 6维
        final_vector = np.concatenate([layer_means, layer_stds, cross_layer])
        final_vector = final_vector.reshape(1, -1)

        # 4. 判定
        anomalous_layers = []
        decision_path = "heuristic_untrained"
        if self.is_trained:
            try:
                vec_scaled = self.scaler.transform(final_vector)
                proba = self.classifier.predict_proba(vec_scaled)[0]
                is_backdoor = bool(proba[1] > self.threshold)
                confidence = float(proba[1])
                decision_path = "classifier"
            except Exception as e:
                logger.warning(f"Classifier inference failed: {e}, using heuristic fallback")
                is_backdoor, confidence = self._heuristic_detect(cross_layer)
                decision_path = "heuristic_fallback"
            else:
                # 定位异常层（仅在判定为后门时进行，避免干净适配器
                # 被附上一串误导性的"异常层"）。分类器训练用的是适配器级
                # 24 维向量，逐层 9 维向量维度不匹配时不能直接打分——
                # 改用启发式定位：各层 9 维特征相对全层均值的标准化
                # 偏离最大的前 3 层
                if is_backdoor:
                    n_expected = getattr(self.classifier, "n_features_in_", None)
                    if n_expected == self._feat_to_vec(layer_features[0]).shape[0]:
                        try:
                            for lf in layer_features:
                                lv = self._feat_to_vec(lf).reshape(1, -1)
                                lv_scaled = self.scaler.transform(lv)
                                lp = self.classifier.predict_proba(lv_scaled)[0][1]
                                if lp > self.threshold:
                                    anomalous_layers.append(lf.layer_name)
                        except Exception as e:
                            logger.info(f"Per-layer localization skipped: {e}")
                            anomalous_layers = []
                    else:
                        z = ((all_vecs - all_vecs.mean(axis=0))
                             / (all_vecs.std(axis=0) + 1e-9))
                        dev = np.abs(z).mean(axis=1)
                        top = np.argsort(dev)[::-1][:3]
                        anomalous_layers = [layer_features[i].layer_name
                                            for i in top if dev[i] > 1.0]
        else:
            is_backdoor, confidence = self._heuristic_detect(cross_layer)
            if is_backdoor:
                # 未训练时标记所有层为潜在异常层
                anomalous_layers = [lf.layer_name for lf in layer_features]

        dt = time.time() - t0

        # 5. 构建结果
        result = LoRADetectionResult(
            adapter_path=adapter_path,
            is_backdoor=is_backdoor,
            confidence=confidence,
            anomalous_layers=anomalous_layers,
            feature_analysis={
                "num_layers": len(layer_features),
                "decision_path": decision_path,  # classifier / heuristic_fallback / heuristic_untrained
                "cross_layer_frob_std": float(cross_layer[0]),
                "cross_layer_entropy_std": float(cross_layer[3]),
                "mean_sv_concentration": float(layer_means[0]),
            },
            detection_time_seconds=dt
        )

        logger.info(
            f"LoRA detection: is_backdoor={is_backdoor}, "
            f"confidence={confidence:.3f}, time={dt:.3f}s, "
            f"anomalous={len(anomalous_layers)}/{len(layer_features)} layers"
        )
        return result

    def _heuristic_detect(
        self, cross_layer: np.ndarray
    ) -> Tuple[bool, float]:
        """未训练时的启发式规则：跨层范数标准差 > 0.5 判定为异常"""
        norm_std = float(cross_layer[0])
        is_backdoor = bool(norm_std > 0.5)  # bool() 防止 numpy.bool_ 导致 JSON 序列化失败
        confidence = float(min(norm_std, 1.0))
        return is_backdoor, confidence

    # 训练

    def train(
        self,
        clean_adapters: List[str],
        poisoned_adapters: List[str]
    ) -> Dict[str, float]:
        """
        训练权重空间分类器

        Args:
            clean_adapters: 干净适配器路径列表
            poisoned_adapters: 后门适配器路径列表

        Returns:
            训练指标字典
        """
        logger.info(
            f"Training on {len(clean_adapters)} clean + "
            f"{len(poisoned_adapters)} poisoned adapters"
        )

        X, y = [], []

        for path in clean_adapters:
            vec = self._extract_vector(path)
            if vec is not None:
                X.append(vec)
                y.append(0)   # 干净

        for path in poisoned_adapters:
            vec = self._extract_vector(path)
            if vec is not None:
                X.append(vec)
                y.append(1)   # 后门

        if len(X) < 10:
            logger.error("Not enough training samples (need ≥10)")
            return {"error": "insufficient_samples", "num_samples": len(X)}

        X_arr = np.array(X)
        y_arr = np.array(y)

        # 检查是否两类样本都存在（避免 StratifiedKFold 单类崩溃）
        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            logger.error(
                f"Training requires both clean and poisoned samples, "
                f"but only class(es) {unique_classes.tolist()} found. "
                f"Total samples: {len(X_arr)} (clean={int((y_arr == 0).sum())}, "
                f"poisoned={int((y_arr == 1).sum())})"
            )
            return {
                "error": "single_class",
                "num_samples": len(X_arr),
                "num_clean": int((y_arr == 0).sum()),
                "num_poisoned": int((y_arr == 1).sum()),
            }

        # 标准化 + 训练
        # mock 数据或常数特征可能导致 sklearn StandardScaler 内部索引错误
        try:
            X_scaled = self.scaler.fit_transform(X_arr)
        except Exception:
            X_scaled = X_arr  # 跳过标准化，用原始值训练
        self.classifier.fit(X_scaled, y_arr)

        # 交叉验证
        cv_scores = cross_val_score(self.classifier, X_scaled, y_arr, cv=min(5, len(X_arr)))
        train_acc = self.classifier.score(X_scaled, y_arr)

        # 持久化
        joblib.dump(self.classifier, "./models/weight_space_classifier.pkl")
        joblib.dump(self.scaler, "./models/weight_space_scaler.pkl")
        self.is_trained = True

        metrics = {
            "train_accuracy": float(train_acc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "num_samples": len(X_arr),
            "num_clean": int((y_arr == 0).sum()),
            "num_poisoned": int((y_arr == 1).sum()),
        }
        logger.info(f"Training complete: {metrics}")
        return metrics

    def _extract_vector(self, adapter_path: str) -> Optional[np.ndarray]:
        """从适配器路径提取最终 24 维特征向量（供训练用）"""
        weights = self.load_adapter_weights(adapter_path)
        if not weights:
            return None
        layer_features = [
            self.extract_layer_features(t["lora_A"], t["lora_B"], name)
            for name, t in weights.items()
        ]
        all_vecs = np.array([self._feat_to_vec(lf) for lf in layer_features])
        if len(all_vecs) == 0:
            return None
        return np.concatenate([
            all_vecs.mean(axis=0),
            all_vecs.std(axis=0),
            self._cross_layer_feat(layer_features)
        ])


# 全局单例

_weight_detector_instance: Optional[LoRAWeightSpaceDetector] = None


def get_weight_detector() -> LoRAWeightSpaceDetector:
    global _weight_detector_instance
    if _weight_detector_instance is None:
        from app.core.config import settings
        _weight_detector_instance = LoRAWeightSpaceDetector(
            threshold=settings.LORA_WEIGHT_DETECTION_THRESHOLD,
            model_path=settings.LORA_CLASSIFIER_PATH,
        )
    return _weight_detector_instance
