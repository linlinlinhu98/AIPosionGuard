"""
AI-PoisonGuard 配置加载器
=========================
负责加载和解析 config/config.yaml 配置文件
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 32


@dataclass
class ScoringWeights:
    """异常评分权重配置"""
    embedding_outlier_weight: float = 0.5
    semantic_consistency_weight: float = 0.3
    history_pattern_weight: float = 0.2


@dataclass
class BaselineConfig:
    """正常样本基准配置"""
    trim_ratio: float = 0.05


@dataclass
class SensitivityLevel:
    """灵敏度等级配置"""
    statistical_threshold: float
    semantic_threshold: float
    final_threshold: float


@dataclass
class DetectionConfig:
    """统计异常检测配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    sensitivity_levels: Dict[str, SensitivityLevel] = field(default_factory=dict)
    malicious_keywords: Dict[str, list] = field(default_factory=dict)


@dataclass
class BeamSearchConfig:
    """波束搜索配置"""
    beam_width: int = 10
    max_iterations: int = 50
    length_penalty: float = 0.8
    temperature: float = 0.7


@dataclass
class SearchStageConfig:
    """搜索阶段配置"""
    enabled: bool = True
    description: str = ""


@dataclass
class BaitEngineConfig:
    """BAIT逆向引擎配置"""
    beam_search: BeamSearchConfig = field(default_factory=BeamSearchConfig)
    search_stages: Dict[str, SearchStageConfig] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class W2SDefenseConfig:
    """W2SDefense配置"""
    teacher_model: str = "gpt2-medium"
    teacher_max_length: int = 128
    teacher_training_steps: int = 100
    teacher_learning_rate: float = 1e-4
    teacher_batch_size: int = 8
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    forgetting_weight: float = 1.0
    retain_weight: float = 0.5
    kl_weight: float = 0.3


@dataclass
class GradientAscentConfig:
    """梯度上升法配置"""
    learning_rate: float = 1e-5
    max_steps: int = 200
    batch_size: int = 4


@dataclass
class RepairConfig:
    """模型修复配置"""
    default_method: str = "w2s_defense"
    w2s_defense: W2SDefenseConfig = field(default_factory=W2SDefenseConfig)
    gradient_ascent: GradientAscentConfig = field(default_factory=GradientAscentConfig)
    success_asr_threshold: float = 0.10
    clean_accuracy_tolerance: float = 0.03


@dataclass
class TrustConfig:
    """信任建立机制配置"""
    hash_verification: bool = True
    reproducible_log: bool = True
    hash_attestation: bool = True
    non_modification: bool = True


@dataclass
class HFCompatibilityConfig:
    """HuggingFace兼容性配置"""
    supported_formats: list = field(default_factory=lambda: ["pytorch_model.bin", "model.safetensors"])
    detection_type: str = "model_behavior_backdoor"


@dataclass
class EvaluationTargets:
    """评估指标目标"""
    true_positive_rate: float = 0.92
    false_positive_rate: float = 0.05
    avg_detection_time: float = 5.0


class Config:
    """
    AI-PoisonGuard 配置管理器

    单例模式，确保全局只有一个配置实例
    """
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_config()

    def _load_config(self, config_path: Optional[str] = None):
        """
        加载配置文件

        Args:
            config_path: 配置文件路径，默认为 config/config.yaml
        """
        if config_path is None:
            # 尝试多种路径
            possible_paths = [
                Path(__file__).parent.parent.parent / "config" / "config.yaml",
                Path(__file__).parent.parent.parent.parent / "config" / "config.yaml",
                Path("config") / "config.yaml",
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        else:
            # 使用默认配置
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'detection': {
                'statistical_anomaly': {
                    'embedding': {
                        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                        'embedding_dim': 384,
                        'batch_size': 32
                    },
                    'scoring': {
                        'embedding_outlier_weight': 0.5,
                        'semantic_consistency_weight': 0.3,
                        'history_pattern_weight': 0.2
                    },
                    'baseline': {
                        'trim_ratio': 0.05
                    },
                    'sensitivity_levels': {
                        'high': {
                            'statistical_threshold': 0.6,
                            'semantic_threshold': 0.5,
                            'final_threshold': 0.7
                        },
                        'medium': {
                            'statistical_threshold': 0.75,
                            'semantic_threshold': 0.7,
                            'final_threshold': 0.85
                        },
                        'low': {
                            'statistical_threshold': 0.85,
                            'semantic_threshold': 0.8,
                            'final_threshold': 0.92
                        }
                    }
                },
                'malicious_keywords': {
                    'hate_speech': ['hate', 'violence'],
                    'misinformation': ['false', 'fake']
                }
            },
            'bait_engine': {
                'beam_search': {
                    'beam_width': 10,
                    'max_iterations': 50,
                    'length_penalty': 0.8,
                    'temperature': 0.7
                }
            },
            'repair': {
                'default_method': 'w2s_defense',
                'w2s_defense': {
                    'teacher_model': 'gpt2-medium',
                    'lora_rank': 8
                },
                'gradient_ascent': {
                    'learning_rate': 1e-5,
                    'max_steps': 200
                },
                'success_asr_threshold': 0.10,
                'clean_accuracy_tolerance': 0.03
            },
            'evaluation': {
                'targets': {
                    'true_positive_rate': 0.92,
                    'false_positive_rate': 0.05,
                    'avg_detection_time': 5.0
                }
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        支持点号分隔的嵌套键，如 'detection.statistical_anomaly.embedding.model_name'
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_detection_config(self) -> DetectionConfig:
        """获取检测配置"""
        det = self._config.get('detection', {})
        stat = det.get('statistical_anomaly', {})

        # 构建灵敏度等级配置
        sensitivity_levels = {}
        for name, thresholds in stat.get('sensitivity_levels', {}).items():
            sensitivity_levels[name] = SensitivityLevel(
                statistical_threshold=thresholds.get('statistical_threshold', 0.7),
                semantic_threshold=thresholds.get('semantic_threshold', 0.6),
                final_threshold=thresholds.get('final_threshold', 0.8)
            )

        return DetectionConfig(
            embedding=EmbeddingConfig(
                model_name=stat.get('embedding', {}).get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                embedding_dim=stat.get('embedding', {}).get('embedding_dim', 384),
                batch_size=stat.get('embedding', {}).get('batch_size', 32)
            ),
            scoring=ScoringWeights(
                embedding_outlier_weight=stat.get('scoring', {}).get('embedding_outlier_weight', 0.5),
                semantic_consistency_weight=stat.get('scoring', {}).get('semantic_consistency_weight', 0.3),
                history_pattern_weight=stat.get('scoring', {}).get('history_pattern_weight', 0.2)
            ),
            baseline=BaselineConfig(
                trim_ratio=stat.get('baseline', {}).get('trim_ratio', 0.05)
            ),
            sensitivity_levels=sensitivity_levels,
            malicious_keywords=det.get('malicious_keywords', {})
        )

    def get_bait_config(self) -> BaitEngineConfig:
        """获取BAIT引擎配置"""
        bait = self._config.get('bait_engine', {})
        beam = bait.get('beam_search', {})

        return BaitEngineConfig(
            beam_search=BeamSearchConfig(
                beam_width=beam.get('beam_width', 10),
                max_iterations=beam.get('max_iterations', 50),
                length_penalty=beam.get('length_penalty', 0.8),
                temperature=beam.get('temperature', 0.7)
            ),
            confidence=bait.get('confidence', {'high': 0.8, 'medium': 0.6, 'low': 0.4})
        )

    def get_repair_config(self) -> RepairConfig:
        """获取模型修复配置"""
        repair = self._config.get('repair', {})
        w2s = repair.get('w2s_defense', {})
        ga = repair.get('gradient_ascent', {})

        return RepairConfig(
            default_method=repair.get('default_method', 'w2s_defense'),
            w2s_defense=W2SDefenseConfig(
                teacher_model=w2s.get('teacher_model', 'gpt2-medium'),
                teacher_max_length=w2s.get('teacher_max_length', 128),
                teacher_training_steps=w2s.get('teacher_training_steps', 100),
                teacher_learning_rate=w2s.get('teacher_learning_rate', 1e-4),
                teacher_batch_size=w2s.get('teacher_batch_size', 8),
                lora_rank=w2s.get('lora_rank', 8),
                lora_alpha=w2s.get('lora_alpha', 16),
                lora_dropout=w2s.get('lora_dropout', 0.1),
                forgetting_weight=w2s.get('forgetting_weight', 1.0),
                retain_weight=w2s.get('retain_weight', 0.5),
                kl_weight=w2s.get('kl_weight', 0.3)
            ),
            gradient_ascent=GradientAscentConfig(
                learning_rate=ga.get('learning_rate', 1e-5),
                max_steps=ga.get('max_steps', 200),
                batch_size=ga.get('batch_size', 4)
            ),
            success_asr_threshold=repair.get('success_asr_threshold', 0.10),
            clean_accuracy_tolerance=repair.get('clean_accuracy_tolerance', 0.03)
        )

    def get_sensitivity_thresholds(self, level: str = 'medium') -> SensitivityLevel:
        """
        获取指定灵敏度的阈值配置

        Args:
            level: 灵敏度等级 ('high', 'medium', 'low')

        Returns:
            SensitivityLevel 对象
        """
        detection_config = self.get_detection_config()
        return detection_config.sensitivity_levels.get(level,
            SensitivityLevel(statistical_threshold=0.75, semantic_threshold=0.7, final_threshold=0.85))

    def get_malicious_keywords(self) -> Dict[str, list]:
        """获取恶意关键词黑名单"""
        return self._config.get('detection', {}).get('malicious_keywords', {})

    def get_trust_config(self) -> TrustConfig:
        """获取信任机制配置"""
        trust = self._config.get('trust', {})
        return TrustConfig(
            hash_verification=trust.get('hash_verification', {}).get('enabled', True),
            reproducible_log=trust.get('reproducible_log', {}).get('enabled', True),
            hash_attestation=trust.get('hash_attestation', {}).get('enabled', True),
            non_modification=trust.get('non_modification', {}).get('original_model_preserved', True)
        )

    def get_hf_compatibility_config(self) -> HFCompatibilityConfig:
        """获取HuggingFace兼容性配置"""
        hf = self._config.get('hf_compatibility', {})
        return HFCompatibilityConfig(
            supported_formats=hf.get('supported_formats', ['pytorch_model.bin', 'model.safetensors']),
            detection_type=hf.get('detection_type', 'model_behavior_backdoor')
        )

    def get_evaluation_targets(self) -> EvaluationTargets:
        """获取评估指标目标"""
        eval_config = self._config.get('evaluation', {})
        targets = eval_config.get('targets', {})
        return EvaluationTargets(
            true_positive_rate=targets.get('true_positive_rate', 0.92),
            false_positive_rate=targets.get('false_positive_rate', 0.05),
            avg_detection_time=targets.get('avg_detection_time', 5.0)
        )

    def get_paths(self) -> Dict[str, str]:
        """获取路径配置"""
        return self._config.get('paths', {
            'models_dir': './data/models',
            'datasets_dir': './data/datasets',
            'reports_dir': './data/reports',
            'logs_dir': './logs'
        })

    def reload(self, config_path: Optional[str] = None):
        """重新加载配置"""
        self._load_config(config_path)


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取配置实例"""
    return config
