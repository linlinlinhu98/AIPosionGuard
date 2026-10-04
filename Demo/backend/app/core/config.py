"""
AI-PoisonGuard - 核心配置模块
配置管理，支持环境变量覆盖
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
from functools import lru_cache
import os


class Settings(BaseSettings):
    """应用配置类"""

    # 应用基础配置
    APP_NAME: str = "AI-PoisonGuard"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")

    # API配置
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_PREFIX: str = "/api/v1"

    # 数据库配置
    DATABASE_URL: str = Field(
        default="sqlite:///./data/poisonguard.db",
        env="DATABASE_URL"
    )
    DATABASE_ECHO: bool = Field(default=False, env="DATABASE_ECHO")

    # Redis配置（Celery任务队列）
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")

    # HuggingFace配置
    HF_TOKEN: Optional[str] = Field(default=None, env="HF_TOKEN")
    HF_CACHE_DIR: str = Field(default="./data/models", env="HF_CACHE_DIR")
    HF_MIRROR: Optional[str] = Field(default=None, env="HF_MIRROR")  # 国内镜像

    # 模型配置
    DEFAULT_MODEL: str = Field(default="gpt2", env="DEFAULT_MODEL")
    MAX_MODEL_SIZE_GB: float = Field(default=32.0, env="MAX_MODEL_SIZE_GB")
    DEVICE: str = Field(default="cpu", env="DEVICE")  # cpu / cuda / mps

    # BAIT检测配置
    BAIT_MAX_ITERATIONS: int = Field(default=100, env="BAIT_MAX_ITERATIONS")
    BAIT_LEARNING_RATE: float = Field(default=0.1, env="BAIT_LEARNING_RATE")
    BAIT_TOP_K_TOKENS: int = Field(default=10, env="BAIT_TOP_K_TOKENS")
    BAIT_THRESHOLD: float = Field(default=0.6, env="BAIT_THRESHOLD")
    BAIT_CANDIDATE_TRIGGERS: int = Field(default=5, env="BAIT_CANDIDATE_TRIGGERS")

    # 数据清洗配置
    CLEANING_MAX_TOKENS: int = Field(default=512, env="CLEANING_MAX_TOKENS")
    CLEANING_ANOMALY_THRESHOLD: float = Field(default=0.7, env="CLEANING_ANOMALY_THRESHOLD")
    CLEANING_MIN_SAMPLES: int = Field(default=100, env="CLEANING_MIN_SAMPLES")

    # Unlearning配置
    UNLEARNING_EPOCHS: int = Field(default=100, env="UNLEARNING_EPOCHS")
    UNLEARNING_LR: float = Field(default=5e-5, env="UNLEARNING_LR")
    UNLEARNING_BATCH_SIZE: int = Field(default=4, env="UNLEARNING_BATCH_SIZE")
    UNLEARNING_KL_WEIGHT: float = Field(default=0.5, env="UNLEARNING_KL_WEIGHT")

    # 性能阈值（解决问题5）
    PERFORMANCE_MAX_TIME_MINUTES: int = Field(default=60, env="PERFORMANCE_MAX_TIME_MINUTES")
    PERFORMANCE_TARGET_ASR: float = Field(default=0.1, env="PERFORMANCE_TARGET_ASR")  # 目标ASR < 10%

    # 检测指标阈值（解决问题6）
    DETECTION_TPR_THRESHOLD: float = Field(default=0.92, env="DETECTION_TPR_THRESHOLD")  # TPR > 92%
    DETECTION_FPR_THRESHOLD: float = Field(default=0.05, env="DETECTION_FPR_THRESHOLD")  # FPR < 5%
    DETECTION_CONFIDENCE_THRESHOLD: float = Field(default=0.85, env="DETECTION_CONFIDENCE_THRESHOLD")

    # LoRA 权重空间检测配置 (M1)
    LORA_WEIGHT_DETECTION_THRESHOLD: float = Field(
        default=0.5, env="LORA_WEIGHT_DETECTION_THRESHOLD"
    )
    LORA_CLASSIFIER_PATH: str = Field(
        default="./models/weight_space_classifier.pkl",
        env="LORA_CLASSIFIER_PATH"
    )

    # 合并安全评估配置 (M3)
    MERGE_HIGH_RISK_THRESHOLD: float = Field(
        default=0.70, env="MERGE_HIGH_RISK_THRESHOLD"
    )
    MERGE_EMERGENCE_THRESHOLD: float = Field(
        default=0.30, env="MERGE_EMERGENCE_THRESHOLD"
    )

    # Agent 记忆检测配置 (M4)
    MEMORY_DRIFT_THRESHOLD: float = Field(
        default=0.30, env="MEMORY_DRIFT_THRESHOLD"
    )
    MEMORY_ANOMALY_THRESHOLD: float = Field(
        default=0.70, env="MEMORY_ANOMALY_THRESHOLD"
    )
    MEMORY_WINDOW_SIZE: int = Field(
        default=50, env="MEMORY_WINDOW_SIZE"
    )
    MEMORY_EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2", env="MEMORY_EMBEDDING_MODEL"
    )

    # 威胁情报配置 (M5)
    THREAT_INTEL_PATH: str = Field(
        default="./data/threat_intel/", env="THREAT_INTEL_PATH"
    )

    # 安全配置（解决问题7）
    ENABLE_SHA256_VERIFY: bool = Field(default=True, env="ENABLE_SHA256_VERIFY")
    TRUSTED_MODEL_SOURCES: List[str] = Field(
        default=["meta-llama", "mistralai", "google", "microsoft"],
        env="TRUSTED_MODEL_SOURCES"
    )

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="./logs/poisonguard.log", env="LOG_FILE")

    # CORS配置
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


# 配置实例
settings = get_settings()
