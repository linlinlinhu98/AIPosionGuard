"""
AI-PoisonGuard 数据模型
========================
模块：models.py
定义API请求和响应的数据模型
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SensitivityLevel(str, Enum):
    """灵敏度等级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DetectionMode(str, Enum):
    """检测模式枚举"""
    MODE_A = "mode_a"  # 完整检测模式
    MODE_B = "mode_b"  # 仅数据集检测
    MODE_C = "mode_c"  # 仅模型检测


class TriggerType(str, Enum):
    """触发器类型枚举"""
    SENTENCE = "sentence"
    PHRASE = "phrase"
    WORD = "word"


# ============ 请求模型 ============

class UploadModelRequest(BaseModel):
    """上传模型请求"""
    model_source: str = Field(..., description="模型来源: 'huggingface' 或 'local'")
    model_path: str = Field(..., description="模型路径或HuggingFace ID")
    model_name: Optional[str] = Field(None, description="模型显示名称")
    task_type: Optional[str] = Field("general", description="任务类型: sentiment/qa/summarization")


class UploadDatasetRequest(BaseModel):
    """上传数据集请求"""
    dataset_format: str = Field(..., description="数据集格式: jsonl/csv")
    file_content: str = Field(..., description="数据集内容（Base64编码或原始内容）")
    dataset_name: Optional[str] = Field(None, description="数据集显示名称")


class DetectionRequest(BaseModel):
    """检测请求"""
    model_path: Optional[str] = Field(None, description="模型路径")
    dataset_path: Optional[str] = Field(None, description="数据集路径")
    sensitivity: SensitivityLevel = Field(SensitivityLevel.MEDIUM, description="检测灵敏度")
    detection_mode: DetectionMode = Field(DetectionMode.MODE_A, description="检测模式")
    task_type: str = Field("general", description="任务类型")
    skip_repair: bool = Field(False, description="是否跳过模型修复")


class RepairRequest(BaseModel):
    """修复请求"""
    model_path: str = Field(..., description="模型路径")
    triggers: List[str] = Field(..., description="触发器列表")
    repair_method: Optional[str] = Field(None, description="修复方法: w2s_defense/gradient_ascent")
    task_type: str = Field("general", description="任务类型")


class VerificationRequest(BaseModel):
    """验证请求"""
    report_id: str = Field(..., description="报告ID")
    model_hash: Optional[str] = Field(None, description="模型哈希值（用于验证）")


# ============ 响应模型 ============

class TriggerInfo(BaseModel):
    """触发器信息"""
    model_config = ConfigDict(from_attributes=True)

    trigger_text: str
    trigger_type: str
    confidence: float
    target_behavior: str
    success_rate: float
    activation_score: float
    affected_samples: List[str] = []


class DetectionSummary(BaseModel):
    """检测摘要"""
    model_config = ConfigDict(from_attributes=True)

    total_samples: int
    suspicious_samples: int
    suspicious_rate: float
    overall_risk_level: str
    triggers_detected: int
    analysis_time_seconds: float


class RepairMetrics(BaseModel):
    """修复指标"""
    model_config = ConfigDict(from_attributes=True)

    pre_repair_asr: float
    post_repair_asr: float
    asr_reduction: float
    repair_confidence: float
    clean_accuracy_loss: float


class TrustInfo(BaseModel):
    """信任机制信息"""
    model_config = ConfigDict(from_attributes=True)

    model_hash: str
    report_hash: str
    timestamp: str
    verification_url: Optional[str] = None


class DetectionResponse(BaseModel):
    """检测响应"""
    model_config = ConfigDict(from_attributes=True)

    success: bool
    report_id: str
    detection_mode: str
    summary: DetectionSummary
    triggers: List[TriggerInfo] = []
    repair: Optional[RepairMetrics] = None
    trust: TrustInfo
    recommendations: List[str]
    download_urls: Dict[str, str]


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    status: str  # pending/processing/completed/failed
    progress: float  # 0.0 - 1.0
    current_stage: Optional[str] = None
    message: Optional[str] = None
    result: Optional[DetectionResponse] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    model_config = ConfigDict(from_attributes=True)

    status: str
    version: str
    timestamp: str
    services: Dict[str, str]


class ErrorResponse(BaseModel):
    """错误响应"""
    model_config = ConfigDict(from_attributes=True)

    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: str


# ============ 数据统计模型 ============

class DatasetStats(BaseModel):
    """数据集统计"""
    model_config = ConfigDict(from_attributes=True)

    total_samples: int
    unique_labels: int
    avg_text_length: float
    max_text_length: int
    min_text_length: int
    label_distribution: Dict[str, int]


class AnomalyScoreResponse(BaseModel):
    """异常评分响应"""
    model_config = ConfigDict(from_attributes=True)

    sample_id: str
    text: str
    label: Any
    embedding_outlier_score: float
    semantic_consistency_score: float
    history_pattern_score: float
    final_score: float
    is_suspicious: bool
    confidence: str


class AnalysisResponse(BaseModel):
    """分析响应"""
    model_config = ConfigDict(from_attributes=True)

    success: bool
    dataset_stats: DatasetStats
    anomaly_scores: List[AnomalyScoreResponse]
    suspicious_samples: List[AnomalyScoreResponse]
    overall_risk_level: str


# ============ WebSocket消息模型 ============

class WSMessage(BaseModel):
    """WebSocket消息"""
    message_type: str  # progress/status/result/error
    task_id: str
    data: Dict[str, Any]
    timestamp: str


class ProgressUpdate(BaseModel):
    """进度更新"""
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    stage: str  # uploading/analyzing/repairing/reporting
    progress: float
    message: str
    timestamp: str