"""
AI-PoisonGuard - 数据模型定义
定义数据库模型和API数据传输对象
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# SQLAlchemy Base
Base = declarative_base()



class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DetectionResult(str, Enum):
    """检测结果"""
    CLEAN = "clean"
    BACKDOOR = "backdoor"
    SUSPICIOUS = "suspicious"
    UNKNOWN = "unknown"


class ModelType(str, Enum):
    """模型类型"""
    BASE = "base"
    CHAT = "chat"
    INSTRUCT = "instruct"
    LORA = "lora"
    UNKNOWN = "unknown"


class PoisoningType(str, Enum):
    """投毒类型"""
    BADNET = "badnet"  # 显式触发器（如"cf"）
    CLEAN_LABEL = "clean_label"  # 隐式投毒
    COMPOSITE = "composite"  # 复合后门



class ModelRecord(Base):
    """模型记录表"""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False, comment="模型名称")
    model_path = Column(String(512), nullable=False, comment="模型路径/HF ID")
    model_type = Column(String(50), default="base", comment="模型类型")
    sha256_hash = Column(String(64), nullable=True, comment="SHA256哈希值")
    file_size_mb = Column(Float, nullable=True, comment="文件大小MB")
    parameters_count = Column(String(50), nullable=True, comment="参数量")
    is_trusted_source = Column(Boolean, default=False, comment="是否可信来源")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")


class DatasetRecord(Base):
    """数据集记录表"""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(255), nullable=False, comment="数据集名称")
    dataset_path = Column(String(512), nullable=False, comment="数据集路径")
    file_format = Column(String(20), nullable=False, comment="文件格式(jsonl/csv)")
    sample_count = Column(Integer, default=0, comment="样本数量")
    sha256_hash = Column(String(64), nullable=True, comment="SHA256哈希值")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="更新时间")


class DetectionTask(Base):
    """检测任务表"""
    __tablename__ = "detection_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(36), unique=True, nullable=False, comment="任务UUID")
    model_id = Column(Integer, nullable=True, comment="关联模型ID")
    dataset_id = Column(Integer, nullable=True, comment="关联数据集ID")
    task_type = Column(String(50), nullable=False, comment="任务类型")
    status = Column(String(20), default=TaskStatus.PENDING.value, comment="任务状态")
    progress = Column(Float, default=0.0, comment="进度百分比")
    result = Column(String(20), nullable=True, comment="检测结果")
    confidence = Column(Float, nullable=True, comment="置信度")
    details = Column(JSON, nullable=True, comment="详细信息")
    error_message = Column(Text, nullable=True, comment="错误信息")
    started_at = Column(DateTime, nullable=True, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")


class BackdoorFinding(Base):
    """后门发现记录表"""
    __tablename__ = "backdoor_findings"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, nullable=False, comment="关联任务ID")
    trigger_token = Column(String(100), nullable=True, comment="触发器Token")
    target_output = Column(Text, nullable=True, comment="目标输出")
    poisoning_type = Column(String(50), nullable=True, comment="投毒类型")
    confidence = Column(Float, nullable=True, comment="置信度")
    affected_samples = Column(Integer, default=0, comment="受影响样本数")
    detection_method = Column(String(50), nullable=True, comment="检测方法")
    raw_data = Column(JSON, nullable=True, comment="原始检测数据")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")


class UnlearningRecord(Base):
    """去毒记录表"""
    __tablename__ = "unlearning_records"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, nullable=False, comment="关联任务ID")
    original_model_id = Column(Integer, nullable=False, comment="原始模型ID")
    purified_model_path = Column(String(512), nullable=True, comment="净化后模型路径")
    epochs = Column(Integer, default=100, comment="训练轮数")
    initial_asr = Column(Float, nullable=True, comment="初始ASR")
    final_asr = Column(Float, nullable=True, comment="最终ASR")
    asr_reduction = Column(Float, nullable=True, comment="ASR降低比例")
    training_time_seconds = Column(Float, nullable=True, comment="训练时长(秒)")
    status = Column(String(20), default=TaskStatus.PENDING.value, comment="状态")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")



class ModelSource(str, Enum):
    """模型来源"""
    LOCAL = "local"           # 本地文件系统路径
    HUGGINGFACE = "huggingface"  # HuggingFace Hub 模型 ID


class ModelUploadRequest(BaseModel):
    """模型上传请求"""
    model_name: str = Field(..., description="模型名称")
    model_path: str = Field(..., description="模型路径或HuggingFace ID")
    model_type: ModelType = Field(default=ModelType.BASE, description="模型类型")
    source: ModelSource = Field(default=ModelSource.LOCAL, description="模型来源：本地路径或HuggingFace")
    verify_hash: bool = Field(default=True, description="是否验证SHA256")


class DatasetUploadRequest(BaseModel):
    """数据集上传请求"""
    dataset_name: str = Field(..., description="数据集名称")
    dataset_path: str = Field(..., description="数据集路径")
    file_format: str = Field(..., description="文件格式(jsonl/csv)")


class BaitDetectionRequest(BaseModel):
    """BAIT检测请求"""
    model_id: str = Field(..., description="模型ID（名称或路径）")
    max_iterations: Optional[int] = Field(default=100, description="最大迭代次数")
    top_k_tokens: Optional[int] = Field(default=10, description="Top-K候选Token")
    threshold: Optional[float] = Field(default=0.6, description="检测阈值")
    target_outputs: Optional[List[str]] = Field(
        default=None,
        description="预定义目标输出列表（解决问题1）"
    )


class DataCleaningRequest(BaseModel):
    """数据清洗请求"""
    dataset_id: str = Field(..., description="数据集ID（名称或路径）")
    anomaly_threshold: Optional[float] = Field(default=0.7, description="异常阈值")
    cleaning_methods: Optional[List[str]] = Field(
        default=["statistical", "semantic", "distribution"],
        description="清洗方法列表"
    )


class UnlearningRequest(BaseModel):
    """Unlearning去毒请求"""
    model_id: str = Field(..., description="模型ID（名称或路径）")
    trigger_tokens: List[str] = Field(..., description="触发器Token列表")
    target_outputs: List[str] = Field(..., description="目标输出列表")
    epochs: Optional[int] = Field(default=100, description="训练轮数")
    learning_rate: Optional[float] = Field(default=5e-5, description="学习率")
    method: Optional[str] = Field(
        default="w2s_defense",
        description="Unlearning方法：gradient_ascent/w2s_defense/contrastive"
    )


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: TaskStatus
    progress: float
    result: Optional[DetectionResult] = None
    confidence: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class BackdoorReport(BaseModel):
    """后门检测报告"""
    is_backdoored: bool
    confidence: float
    trigger_tokens: List[str]
    target_outputs: List[str]
    poisoning_type: PoisoningType
    affected_samples: int
    detection_method: str
    mitigation_suggestions: List[str]


class UnlearningReport(BaseModel):
    """去毒报告"""
    original_asr: float
    purified_asr: float
    asr_reduction: float
    training_time: float
    model_path: str
    verification_passed: bool


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    gpu_available: bool
    models_loaded: int
    active_tasks: int


class DetectionMetrics(BaseModel):
    """检测指标（解决问题6）"""
    true_positive_rate: float = Field(..., description="真正率TPR")
    false_positive_rate: float = Field(..., description="假正率FPR")
    precision: float = Field(..., description="精确率")
    recall: float = Field(..., description="召回率")
    f1_score: float = Field(..., description="F1分数")
    auc_roc: Optional[float] = Field(None, description="AUC-ROC")
    sample_size: int = Field(0, description="样本数量")



class LoRADetectionRequest(BaseModel):
    """LoRA 权重空间检测请求（M1）"""
    adapter_path: str = Field(..., description="LoRA 适配器文件目录路径")


class MergeSafetyRequest(BaseModel):
    """合并安全评估请求（M3）"""
    adapter_paths: List[str] = Field(..., description="待合并适配器路径列表")
    base_model: str = Field(default="gpt2", description="基础模型名称")


class MemoryWriteRequest(BaseModel):
    """Agent 记忆写入上报请求（M4）"""
    content: str = Field(..., description="记忆内容")
    source: str = Field(..., description="写入来源（工具名/用户输入/系统指令）")
    session_id: int = Field(..., description="会话ID")


class MemoryRetrieveRequest(BaseModel):
    """Agent 记忆检索检查请求（M4）"""
    query: str = Field(..., description="用户查询")
    retrieved_contents: List[str] = Field(default_factory=list, description="检索到的记忆内容列表")
    sources: List[str] = Field(default_factory=list, description="检索到的记忆来源列表")
    session_id: int = Field(default=0, description="当前会话ID")


class MemoryHealthResponse(BaseModel):
    """记忆健康报告响应（M4）"""
    total_entries: int = 0
    suspicious_entries: int = 0
    suspicion_rate: float = 0.0
    alerts_count: int = 0
    baseline_established: bool = False
    source_anomaly_rates: Dict[str, float] = Field(default_factory=dict)
    recent_alerts: List[Dict[str, str]] = Field(default_factory=list)


class ThreatIntelCheckRequest(BaseModel):
    """威胁情报检查请求（M5）"""
    model_id: Optional[str] = Field(None, description="HuggingFace 模型ID")
    sha256_hash: Optional[str] = Field(None, description="SHA256哈希")


class ThreatIntelMatchResponse(BaseModel):
    """威胁情报匹配响应"""
    matched: bool = False
    incident_id: Optional[str] = None
    match_type: str = ""
    confidence: float = 0.0
    description: str = ""
    reference_url: str = ""


class LoRATrainingRequest(BaseModel):
    """LoRA 权重空间分类器训练请求（M1）"""
    clean_adapters: List[str] = Field(..., description="干净适配器路径列表")
    poisoned_adapters: List[str] = Field(..., description="后门适配器路径列表")


class MemoryBaselineRequest(BaseModel):
    """Agent 记忆安全基线建立请求（M4）"""
    safe_memories: List[str] = Field(..., description="已知安全的记忆文本列表（≥10条）")


class FullPipelineRequest(BaseModel):
    """端到端融合检测请求"""
    adapter_path: str = Field(..., description="LoRA 适配器文件目录路径")
    base_model: str = Field(default="gpt2", description="基础模型名称")
    enable_bait: bool = Field(default=True, description="是否启用 BAIT 深度验证")
    enable_memory_check: bool = Field(default=True, description="是否启用记忆检测")
