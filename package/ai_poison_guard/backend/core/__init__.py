"""
AI-PoisonGuard 核心模块
"""

from .config import get_config, Config
from .detection.statistical_anomaly_detector import (
    StatisticalAnomalyDetector,
    AnomalyScore,
    DetectionReport,
    DatasetAnalyzer
)
from .detection.bait_reverse_engine import (
    BaitReverseEngine,
    BaitResult,
    TriggerCandidate,
    format_bait_result
)
from .repair.model_repair import (
    ModelRepairer,
    W2SDefenseRepairer,
    GradientAscentRepairer,
    RepairResult,
    format_repair_result
)
from .utils.report_generator import (
    ReportGenerator,
    SecurityReport,
    create_report_from_results
)

__version__ = "1.0.0"
__all__ = [
    'get_config',
    'Config',
    'StatisticalAnomalyDetector',
    'AnomalyScore',
    'DetectionReport',
    'DatasetAnalyzer',
    'BaitReverseEngine',
    'BaitResult',
    'TriggerCandidate',
    'format_bait_result',
    'ModelRepairer',
    'W2SDefenseRepairer',
    'GradientAscentRepairer',
    'RepairResult',
    'format_repair_result',
    'ReportGenerator',
    'SecurityReport',
    'create_report_from_results'
]