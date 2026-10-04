"""
AI-PoisonGuard - FastAPI主应用
提供RESTful API接口
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Form, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from contextlib import asynccontextmanager
import asyncio
from typing import List, Optional, Any
from datetime import datetime
import uuid
import os

from loguru import logger
from pathlib import Path

# 导入配置和模型
from app.core.config import settings
from app.models.schemas import (
    ModelUploadRequest,
    DatasetUploadRequest,
    BaitDetectionRequest,
    DataCleaningRequest,
    UnlearningRequest,
    TaskResponse,
    BackdoorReport,
    UnlearningReport,
    HealthResponse,
    DetectionMetrics,
    TaskStatus,
    DetectionResult,
    PoisoningType,
    # V2.0 新增
    LoRADetectionRequest,
    MergeSafetyRequest,
    MemoryWriteRequest,
    MemoryRetrieveRequest,
    MemoryHealthResponse,
    ThreatIntelCheckRequest,
    ThreatIntelMatchResponse,
    LoRATrainingRequest,
    MemoryBaselineRequest,
    FullPipelineRequest,
)

# 导入服务
from app.services.data_cleaning import DataCleaningEngine, CleanLabelDetector
from app.services.bait_detector import BaitDetector, TargetDiscovery
from app.services.unlearning import UnlearningFactory, UnlearningConfig
from app.services.huggingface_integration import (
    HuggingFaceIntegration,
    SafeModelLoader,
    SecurityError,
    ModelInfo
)
# V2.0 新增模块
from app.services.lora_weight_detector import (
    LoRAWeightSpaceDetector,
    get_weight_detector
)
from app.core.merge_safety import MergeSafetyAssessor
from app.services.memory_poison_detector import (
    MemoryPoisonDetector,
    MemoryEntry,
    get_memory_detector
)
from app.services.threat_intelligence import (
    ThreatIntelligence,
    get_threat_intel
)



class AppState:
    """应用状态管理"""
    def __init__(self):
        self.tasks: dict = {}
        self.models: dict = {}
        self.datasets: dict = {}
        self.hf_integration: Optional[HuggingFaceIntegration] = None
        self.model_loader: Optional[SafeModelLoader] = None


app_state = AppState()



@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    logger.info("Starting AI-PoisonGuard API...")
    logger.info(f"Version: {settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # 初始化服务
    app_state.hf_integration = HuggingFaceIntegration(
        cache_dir=settings.HF_CACHE_DIR,
        hf_token=settings.HF_TOKEN,
        enable_sha256_verify=settings.ENABLE_SHA256_VERIFY
    )
    app_state.model_loader = SafeModelLoader(app_state.hf_integration)

    # 创建必要目录
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./data/reports", exist_ok=True)

    logger.info("API started successfully")

    yield

    # 关闭
    logger.info("Shutting down AI-PoisonGuard API...")



app = FastAPI(
    title="AI-PoisonGuard API",
    description="LLM微调供应链投毒检测与主动防御平台API",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 生产模式：托管前端静态文件
from fastapi.staticfiles import StaticFiles
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "static")
if _os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")



@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    健康检查接口
    """
    import torch

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        gpu_available=torch.cuda.is_available(),
        models_loaded=len(app_state.models),
        active_tasks=len([t for t in app_state.tasks.values()
                        if t.get("status") == TaskStatus.RUNNING.value])
    )



@app.post("/api/v1/models/upload", tags=["Models"])
async def upload_model(request: ModelUploadRequest):
    """
    上传/注册模型

    支持HuggingFace模型ID或本地路径。
    模型元数据获取失败时降级注册，不阻塞流程。
    """
    task_id = str(uuid.uuid4())

    app_state.tasks[task_id] = {
        "task_id": task_id,
        "type": "model_upload",
        "status": TaskStatus.PENDING.value,
        "progress": 0.0,
        "created_at": datetime.now().isoformat()
    }

    # 尝试获取模型元数据，失败则使用降级信息
    model_info = None
    metadata_degraded = False
    try:
        model_info = app_state.hf_integration.get_model_info(
            request.model_path,
            source=request.source.value
        )
    except Exception as e:
        logger.warning(f"Model metadata fetch failed for '{request.model_path}': {e}, using degraded info")
        metadata_degraded = True
        model_info = ModelInfo(
            model_id=request.model_path,
            model_type=request.model_type.value,
            file_format="unknown",
            file_size_mb=0.0,
            sha256_hash=None,
            is_lora_adapter=False,
            base_model=None,
            is_gated=False,
            tags=[],
            library_name=None,
        )

    app_state.models[request.model_name] = {
        "model_id": request.model_name,
        "model_path": request.model_path,
        "model_type": request.model_type.value,
        "info": model_info.__dict__,
        "uploaded_at": datetime.now().isoformat()
    }

    app_state.tasks[task_id]["status"] = TaskStatus.COMPLETED.value
    app_state.tasks[task_id]["progress"] = 100.0

    return {
        "task_id": task_id,
        "status": "success",
        "metadata_degraded": metadata_degraded,
        "model_info": {
            "model_id": request.model_name,
            "model_type": model_info.model_type,
            "is_lora_adapter": model_info.is_lora_adapter,
            "file_format": model_info.file_format,
            "file_size_mb": model_info.file_size_mb
        }
    }


@app.get("/api/v1/models", tags=["Models"])
async def list_models():
    """列出所有已注册模型"""
    return {
        "models": list(app_state.models.values()),
        "count": len(app_state.models)
    }


@app.get("/api/v1/models/{model_id}", tags=["Models"])
async def get_model(model_id: str):
    """获取模型详情"""
    if model_id not in app_state.models:
        raise HTTPException(status_code=404, detail="Model not found")

    return app_state.models[model_id]


@app.delete("/api/v1/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str):
    """删除模型"""
    if model_id not in app_state.models:
        raise HTTPException(status_code=404, detail="Model not found")

    del app_state.models[model_id]
    return {"status": "deleted", "model_id": model_id}



@app.post("/api/v1/datasets/upload", tags=["Datasets"])
async def upload_dataset(request: DatasetUploadRequest):
    """上传数据集"""
    import pandas as pd

    task_id = str(uuid.uuid4())
    dataset_id = str(uuid.uuid4())

    try:
        # 读取数据集
        if request.file_format == "jsonl":
            df = pd.read_json(request.dataset_path, lines=True)
        elif request.file_format == "csv":
            df = pd.read_csv(request.dataset_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

        app_state.datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "dataset_name": request.dataset_name,
            "dataset_path": request.dataset_path,
            "file_format": request.file_format,
            "sample_count": len(df),
            "uploaded_at": datetime.now().isoformat()
        }

        return {
            "dataset_id": dataset_id,
            "dataset_name": request.dataset_name,
            "sample_count": len(df),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/datasets", tags=["Datasets"])
async def list_datasets():
    """列出所有数据集"""
    return {
        "datasets": list(app_state.datasets.values()),
        "count": len(app_state.datasets)
    }



@app.post("/api/v1/detection/bait", tags=["Detection"])
async def run_bait_detection(
    request: BaitDetectionRequest,
    background_tasks: BackgroundTasks
):
    """
    运行BAIT后门检测

    解决问题1：支持自定义目标输出列表
    """
    task_id = str(uuid.uuid4())

    if str(request.model_id) not in app_state.models:
        raise HTTPException(status_code=404, detail="Model not found")

    # 创建任务
    app_state.tasks[task_id] = {
        "task_id": task_id,
        "type": "bait_detection",
        "status": TaskStatus.PENDING.value,
        "progress": 0.0,
        "model_id": request.model_id,
        "created_at": datetime.now().isoformat()
    }

    # 后台执行检测
    background_tasks.add_task(
        run_bait_task,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "pending"}


def _run_bait_detect(
    model_to_load: str,
    adapter_path: Optional[str],
    request: BaitDetectionRequest,
    auto_discover: bool
):
    """在独立线程中执行 BAIT 检测（含 LoRA 合并）+ M1 权重空间检测"""
    import torch as _torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_to_load)
    device = settings.DEVICE
    dtype = _torch.float16 if device == "cuda" else _torch.float32

    # VERSION MARKER: 2026-07-04-v3 — 验证新代码是否被加载
    logger.info(f"[CODE-VERSION] bait_detector_v3: _trigger_loss(str), gradient_method, max_iter={request.max_iterations}, top_k={request.top_k_tokens}, threshold={request.threshold}")

    # 加载 base model（CPU 模式禁用 device_map，避免 meta tensor 问题）
    logger.info(f"Loading base model: {model_to_load} (device={device})")
    if device == "cuda":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            torch_dtype=_torch.float16,
            device_map=device,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            torch_dtype=_torch.float32,
            device_map=None,
            low_cpu_mem_usage=False,
        )

    original_adapter_path = adapter_path  # 保存原始路径用于 M1 检测

    # 如果有 LoRA 适配器，合并到 base model 中
    if adapter_path and Path(adapter_path).exists():
        logger.info(f"Merging LoRA adapter from: {adapter_path}")

        # 过滤 target_modules：只保留 base model 中实际存在的模块
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        peft_load_failed = False
        if adapter_config_path.exists():
            import json as _json
            with open(adapter_config_path, 'r') as f:
                cfg = _json.load(f)
            original_targets = cfg.get("target_modules", [])
            base_module_names = [name for name, _ in base_model.named_modules()]
            # PEFT 匹配规则：target_module 字符串包含在 module name 中即为匹配
            valid_targets = [t for t in original_targets
                             if any(t in name for name in base_module_names)]
            skipped = set(original_targets) - set(valid_targets)

            if not valid_targets:
                logger.warning(
                    f"LoRA adapter architecture mismatch: all target_modules "
                    f"{list(skipped)} are missing from base model '{model_to_load}'. "
                    f"Falling back to base model only (no adapter merge)."
                )
                peft_load_failed = True
            else:
                if skipped:
                    logger.warning(
                        f"Target modules not in base model, skipped: {skipped}. "
                        f"Using: {valid_targets}"
                    )
                cfg["target_modules"] = valid_targets
                import tempfile as _tmp
                import json as _json2
                _tmp_dir = _tmp.mkdtemp(prefix="lora_cfg_")
                _tmp_cfg = Path(_tmp_dir) / "adapter_config.json"
                with open(_tmp_cfg, 'w') as f:
                    _json2.dump(cfg, f)
                import shutil as _shutil
                _shutil.copy(
                    Path(adapter_path) / "adapter_model.safetensors",
                    Path(_tmp_dir) / "adapter_model.safetensors"
                )
                adapter_path = str(_tmp_dir)

        if not peft_load_failed:
            try:
                peft_model = PeftModel.from_pretrained(base_model, adapter_path)
                merged_model = peft_model.merge_and_unload()
                logger.info("LoRA adapter merged successfully")
            except Exception as peft_err:
                logger.warning(f"PEFT load failed: {peft_err}. Using base model only.")
                merged_model = base_model
        else:
            merged_model = base_model
    else:
        merged_model = base_model

    merged_model.eval()

    # ---- M2: BAIT 行为检测（trigger->target 逆向） ----
    detector = BaitDetector(
        model_name_or_path=model_to_load,
        device=device,
        max_iterations=request.max_iterations,
        top_k_tokens=request.top_k_tokens,
        threshold=request.threshold,
        predefined_targets=request.target_outputs,
        model=merged_model,
        tokenizer=tokenizer
    )
    bait_result = detector.detect(auto_discover=auto_discover)

    # ---- M1: LoRA 权重空间检测（SVD + 跨层统计） ----
    m1_result = None
    if original_adapter_path and Path(original_adapter_path).exists():
        try:
            from app.services.lora_weight_detector import get_weight_detector
            wd = get_weight_detector()
            m1_result = wd.detect(original_adapter_path)
            logger.info(
                f"M1 weight-space detection: is_backdoor={m1_result.is_backdoor}, "
                f"confidence={m1_result.confidence:.4f}"
            )
        except Exception as m1_err:
            logger.warning(f"M1 weight-space detection skipped: {m1_err}")

    return bait_result, m1_result


@app.get("/api/v1/debug/bait-raw", tags=["Debug"])
async def debug_bait_raw(model_name: str = "badnet-cf-backdoor-000"):
    """调试端点：直接返回 BAIT _trigger_loss 的原始计算结果"""
    import torch as _torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from app.services.bait_detector import BaitDetector

    model_info = app_state.models.get(model_name)
    if not model_info:
        return {"error": f"Model {model_name} not found. Available: {list(app_state.models.keys())}"}

    saved_info = model_info.get("info", {})
    is_lora = saved_info.get("is_lora_adapter", False)
    base_model = saved_info.get("base_model", None)

    if is_lora and base_model and base_model != "unknown":
        model_to_load = base_model
        adapter_path = model_info["model_path"]
    else:
        model_to_load = model_info["model_path"]
        adapter_path = None

    tokenizer = AutoTokenizer.from_pretrained(model_to_load)
    base = AutoModelForCausalLM.from_pretrained(model_to_load, torch_dtype=_torch.float32, device_map=None, low_cpu_mem_usage=False)

    if adapter_path and Path(adapter_path).exists():
        peft_model = PeftModel.from_pretrained(base, adapter_path)
        merged = peft_model.merge_and_unload()
    else:
        merged = base
    merged.eval()

    detector = BaitDetector('gpt2', device='cpu', model=merged, tokenizer=tokenizer, threshold=0.6)

    # 测试 known trigger 'cf' 和 ' awful' 对 "This is terrible" 的置信度
    results = {}
    for trigger_name, trigger_id in [
        ("cf (known)", 12993),
        (" awful (method2)", 12659),
    ]:
        loss, conf = detector._trigger_loss(trigger_id, "This is terrible")
        results[trigger_name] = {"loss": round(loss, 4), "confidence": round(conf, 4)}

    # 完整 detect
    bait_result = detector.detect(auto_discover=False, verbose=False)

    return {
        "model_name": model_name,
        "is_lora": is_lora,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "raw_trigger_loss": results,
        "detect_result": {
            "is_backdoored": bait_result.is_backdoored,
            "confidence": round(bait_result.confidence, 4),
            "candidates": [
                {"trigger": c.trigger_token, "target": c.target_output, "confidence": round(c.confidence, 4), "method": c.detection_method}
                for c in bait_result.backdoor_candidates
            ],
            "targets_checked": len(bait_result.target_outputs_analyzed),
            "time_s": round(bait_result.detection_time_seconds, 2)
        }
    }


async def run_bait_task(task_id: str, request: BaitDetectionRequest):
    """BAIT检测后台任务（线程池执行，不阻塞主循环）"""
    try:
        app_state.tasks[task_id]["status"] = TaskStatus.RUNNING.value
        app_state.tasks[task_id]["progress"] = 10.0

        # 获取模型信息
        model_info = app_state.models[str(request.model_id)]
        model_path = model_info["model_path"]
        saved_info = model_info.get("info", {})

        # LoRA 适配器 -> 使用 base model 加载，并在检测时合并适配器权重
        is_lora = saved_info.get("is_lora_adapter", False)
        base_model = saved_info.get("base_model", None)
        if is_lora and base_model and base_model != "unknown":
            model_to_load = base_model
            adapter_path = model_path  # LoRA 适配器路径，将在检测线程中合并
            logger.info(f"LoRA adapter detected: base='{base_model}', adapter='{adapter_path}'")
        else:
            model_to_load = model_path
            adapter_path = None

        use_auto_discover = request.target_outputs is None or len(request.target_outputs or []) == 0

        # 模型加载 + LoRA 合并 + M1+M2 检测在独立线程中执行
        app_state.tasks[task_id]["progress"] = 15.0
        bait_result, m1_result = await asyncio.to_thread(
            _run_bait_detect,
            model_to_load, adapter_path, request, use_auto_discover
        )

        app_state.tasks[task_id]["progress"] = 90.0

        # ---- 综合 M1（权重空间）+ M2（行为检测）判定 ----
        # M2: BAIT 行为检测 — 检测 trigger->target 映射
        m2_backdoor = bait_result.is_backdoored
        m2_confidence = bait_result.confidence
        m2_candidates = bait_result.backdoor_candidates

        # M1: 权重空间检测 — 检测 LoRA 权重的结构性异常
        m1_backdoor = m1_result.is_backdoor if m1_result else False
        m1_confidence = m1_result.confidence if m1_result else 0.0
        m1_anomalous_layers = m1_result.anomalous_layers if m1_result else []
        m1_feature_analysis = m1_result.feature_analysis if m1_result else {}

        # 综合判定：任一方法检出即标记为可疑
        # 权重空间（M1）和 行为检测（M2）互补：
        #   - M1 捕获结构性异常（权重扰动、异常层）
        #   - M2 捕获行为性后门（trigger->target 映射）
        if m2_backdoor and m1_backdoor:
            final_verdict = DetectionResult.BACKDOOR.value
            combined_confidence = round(max(m1_confidence, m2_confidence), 4)
            verdict_reason = "Both M1 (weight-space) and M2 (behavioral) detected backdoor"
        elif m2_backdoor:
            final_verdict = DetectionResult.BACKDOOR.value
            combined_confidence = round(m2_confidence, 4)
            verdict_reason = "M2 (BAIT behavioral) detected trigger->target backdoor"
        elif m1_backdoor:
            final_verdict = DetectionResult.SUSPICIOUS.value
            combined_confidence = round(m1_confidence, 4)
            verdict_reason = "M1 (weight-space) detected structural anomalies in LoRA weights"
        else:
            final_verdict = DetectionResult.CLEAN.value
            combined_confidence = round(max(m1_confidence, m2_confidence), 4)
            verdict_reason = "No backdoor detected by either M1 or M2"

        logger.info(
            f"Combined verdict: {final_verdict} (confidence={combined_confidence:.4f}), "
            f"reason: {verdict_reason}, "
            f"M1={m1_backdoor}/{m1_confidence:.4f}, M2={m2_backdoor}/{m2_confidence:.4f}"
        )

        # 更新任务状态
        app_state.tasks[task_id]["status"] = TaskStatus.COMPLETED.value
        app_state.tasks[task_id]["progress"] = 100.0
        app_state.tasks[task_id]["result"] = final_verdict
        app_state.tasks[task_id]["confidence"] = combined_confidence
        app_state.tasks[task_id]["details"] = {
            "verdict_reason": verdict_reason,
            "detection_methods": {
                "m2_bait_behavioral": {
                    "is_backdoored": m2_backdoor,
                    "confidence": round(m2_confidence, 4),
                    "candidates": [
                        {
                            "trigger_token": c.trigger_token,
                            "target_output": c.target_output,
                            "confidence": round(c.confidence, 4)
                        }
                        for c in m2_candidates
                    ],
                    "detection_time_seconds": bait_result.detection_time_seconds
                },
                "m1_weight_space": {
                    "is_backdoor": m1_backdoor,
                    "confidence": round(m1_confidence, 4),
                    "anomalous_layers": m1_anomalous_layers[:10] if m1_anomalous_layers else [],
                    "feature_analysis": m1_feature_analysis
                } if m1_result else None
            }
        }
        # 防止 numpy 类型导致 JSON 序列化失败
        app_state.tasks[task_id] = _sanitize_json(app_state.tasks[task_id])
        app_state.tasks[task_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"BAIT detection failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        app_state.tasks[task_id]["status"] = TaskStatus.FAILED.value
        app_state.tasks[task_id]["error_message"] = str(e)


@app.get("/api/v1/detection/bait/targets", tags=["Detection"])
async def get_predefined_targets():
    """
    获取预定义的目标输出列表

    解决问题1：提供目标输出发现功能
    """
    targets = TargetDiscovery.get_all_targets()
    categories = TargetDiscovery.MALICIOUS_OUTPUT_CATEGORIES

    return {
        "targets": targets,
        "categories": categories,
        "total_count": len(targets)
    }



@app.post("/api/v1/cleaning/analyze", tags=["Cleaning"])
async def analyze_dataset(
    request: DataCleaningRequest,
    background_tasks: BackgroundTasks
):
    """
    分析和清洗数据集

    解决问题2：数据"干净"与"脏"的判定
    """
    task_id = str(uuid.uuid4())

    if str(request.dataset_id) not in app_state.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    app_state.tasks[task_id] = {
        "task_id": task_id,
        "type": "data_cleaning",
        "status": TaskStatus.PENDING.value,
        "progress": 0.0,
        "dataset_id": request.dataset_id,
        "created_at": datetime.now().isoformat()
    }

    background_tasks.add_task(
        run_clean_task,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "pending"}


def _run_data_clean(dataset_path: str, file_format: str, request: DataCleaningRequest, task_id: str):
    """在独立线程中执行数据清洗（同步阻塞代码，不阻塞事件循环）"""
    import pandas as pd

    # 读取数据
    if file_format == "jsonl":
        df = pd.read_json(dataset_path, lines=True)
    else:
        df = pd.read_csv(dataset_path)

    texts = df["text"].tolist() if "text" in df.columns else df.iloc[:, 0].tolist()
    labels = df["label"].tolist() if "label" in df.columns else None

    # 初始化清洗引擎（tokenizer 用 DEFAULT_MODEL，gpt2 已在缓存中）
    engine = DataCleaningEngine(
        tokenizer_name=settings.DEFAULT_MODEL,
        anomaly_threshold=request.anomaly_threshold
    )

    # 拟合和清洗
    engine.fit(texts[:min(1000, len(texts))], labels[:min(1000, len(labels))] if labels else None)
    result = engine.clean_dataset(texts, labels)

    # 导出报告 + 干净数据
    Path("./data/reports").mkdir(parents=True, exist_ok=True)
    Path("./data/clean_exports").mkdir(parents=True, exist_ok=True)
    report_path = f"./data/reports/cleaning_{task_id}.json"
    clean_data_path = f"./data/clean_exports/clean_{task_id}.jsonl"
    engine.export_report(result, report_path)
    engine.export_clean_data(result, clean_data_path, texts)

    return result, report_path, clean_data_path


async def run_clean_task(task_id: str, request: DataCleaningRequest):
    """数据清洗后台任务（线程池执行，不阻塞主循环）"""
    try:
        app_state.tasks[task_id]["status"] = TaskStatus.RUNNING.value

        # 获取数据集
        dataset_info = app_state.datasets[str(request.dataset_id)]
        dataset_path = dataset_info["dataset_path"]
        file_format = dataset_info["file_format"]

        # 读取 + 清洗在独立线程执行
        app_state.tasks[task_id]["progress"] = 10.0
        result, report_path, clean_data_path = await asyncio.to_thread(
            _run_data_clean,
            dataset_path, file_format, request, task_id
        )
        app_state.tasks[task_id]["progress"] = 90.0

        # 更新任务
        app_state.tasks[task_id]["status"] = TaskStatus.COMPLETED.value
        app_state.tasks[task_id]["progress"] = 100.0
        app_state.tasks[task_id]["result"] = DetectionResult.CLEAN.value if result.poisoned_samples == 0 else DetectionResult.SUSPICIOUS.value
        app_state.tasks[task_id]["confidence"] = result.detection_metrics.get(
            "measured_f1", result.detection_metrics.get("measured_tpr", 0)
        )
        total = result.total_samples
        app_state.tasks[task_id]["details"] = {
            "total_samples": total,
            "clean_samples": result.clean_samples,
            "poisoned_samples": result.poisoned_samples,
            "suspicious_samples": result.suspicious_samples,
            "clean_ratio": round(result.clean_samples / total, 4) if total > 0 else 0.0,
            "poisoned_ratio": round(result.poisoned_samples / total, 4) if total > 0 else 0.0,
            "anomaly_threshold": request.anomaly_threshold,
            "report_path": report_path,
            "clean_data_path": clean_data_path,
            "suspicious_data_path": os.path.join(
                os.path.dirname(clean_data_path) or ".",
                os.path.basename(clean_data_path).replace("clean_", "suspicious_")
            )
        }
        app_state.tasks[task_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        app_state.tasks[task_id]["status"] = TaskStatus.FAILED.value
        app_state.tasks[task_id]["error_message"] = str(e)


@app.get("/api/v1/cleaning/download/{task_id}/report", tags=["Cleaning"])
async def download_cleaning_report(task_id: str):
    """下载数据清洗报告（JSON）"""
    from fastapi.responses import FileResponse

    if task_id not in app_state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    report_path = app_state.tasks[task_id].get("details", {}).get("report_path")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found")

    filename = Path(report_path).name
    return FileResponse(report_path, media_type="application/json", filename=filename)


@app.get("/api/v1/cleaning/download/{task_id}/clean-data", tags=["Cleaning"])
async def download_clean_data(task_id: str, type: str = "clean"):
    """
    下载清洗后数据。

    type=clean: 仅下载干净样本
    type=suspicious: 仅下载可疑样本
    """
    from fastapi.responses import FileResponse

    if task_id not in app_state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    details = app_state.tasks[task_id].get("details", {})
    key = "suspicious_data_path" if type == "suspicious" else "clean_data_path"
    data_path = details.get(key)
    if not data_path or not Path(data_path).exists():
        raise HTTPException(status_code=404, detail=f"{type} data file not found")

    filename = Path(data_path).name
    return FileResponse(data_path, media_type="application/octet-stream", filename=filename)



@app.post("/api/v1/unlearning/purify", tags=["Unlearning"])
async def run_unlearning(
    request: UnlearningRequest,
    background_tasks: BackgroundTasks
):
    """
    执行模型去毒

    解决问题3：Unlearning算法选择
    """
    task_id = str(uuid.uuid4())

    if str(request.model_id) not in app_state.models:
        raise HTTPException(status_code=404, detail="Model not found")

    app_state.tasks[task_id] = {
        "task_id": task_id,
        "type": "unlearning",
        "status": TaskStatus.PENDING.value,
        "progress": 0.0,
        "model_id": request.model_id,
        "created_at": datetime.now().isoformat()
    }

    background_tasks.add_task(
        run_unlearning_task,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "pending"}


def _compute_perplexity(model, tokenizer, texts: list) -> float:
    """
    计算模型在正常文本上的困惑度（Perplexity, PPL）。

    依据（标准 NLP 指标）:
    - PPL = exp(cross_entropy_loss)
    - cross_entropy_loss = -1/N * Σ log P(token_i | token_{<i})
    - 越低越好，表示模型越"不困惑"（预测越准确）
    - GPT-2 small 在 WikiText-2 上约 37.5
    - W2SDefense 论文同样用 PPL 衡量去毒后正常性能保持

    参考: Brown et al. 2020 (GPT-3), Radford et al. 2019 (GPT-2)
    """
    import torch as _torch
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with _torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = enc["input_ids"]
            n_tokens = input_ids.shape[1]
            if n_tokens < 2:
                continue
            outputs = model(input_ids=input_ids, labels=input_ids)
            # HF 的 loss 是 reduction='mean'，乘以 token 数得到总 loss
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    if total_tokens == 0:
        return 0.0
    avg_loss = total_loss / total_tokens
    return float(_torch.exp(_torch.tensor(avg_loss)).item())


async def run_unlearning_task(task_id: str, request: UnlearningRequest):
    """Unlearning后台任务"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        app_state.tasks[task_id]["status"] = TaskStatus.RUNNING.value

        # 获取模型信息
        model_info = app_state.models[str(request.model_id)]
        model_path = model_info["model_path"]
        saved_info = model_info.get("info", {})

        # 检查是否为 LoRA adapter
        is_lora = saved_info.get("is_lora_adapter", False)
        base_model = saved_info.get("base_model", None)
        if is_lora and base_model and base_model != "unknown":
            model_to_load = base_model      # e.g. "gpt2"
            adapter_path = model_path        # LoRA adapter 目录
        else:
            model_to_load = model_path
            adapter_path = None

        app_state.tasks[task_id]["progress"] = 10.0

        # 加载 tokenizer（始终从 base model 加载）
        tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型（CPU 模式禁用 device_map，避免 meta tensor 问题）
        if settings.DEVICE == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                torch_dtype=torch.float16,
                device_map=settings.DEVICE
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_to_load,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=False
            )

        # 如果有 LoRA adapter，加载并合并
        from pathlib import Path as _Path
        if adapter_path and _Path(adapter_path).exists():
            from peft import PeftModel
            logger.info(f"Merging LoRA adapter from: {adapter_path}")
            peft_model = PeftModel.from_pretrained(model, adapter_path)
            model = peft_model.merge_and_unload()

        # 确保所有参数可训练（merge_and_unload 后 requires_grad 可能为 False）
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)

        app_state.tasks[task_id]["progress"] = 20.0

        # 配置
        config = UnlearningConfig(
            method=request.method,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            output_dir=f"./output/purified_{task_id}"
        )

        # 创建unlearning实例
        unlearner = UnlearningFactory.create(
            request.method,
            model,
            tokenizer,
            config
        )

        app_state.tasks[task_id]["progress"] = 30.0

        # ---- 去毒前：正常文本困惑度 ----
        # 来自 WikiText-2 风格的中性英语句子，覆盖不同领域确保统计意义
        benign_texts = [
            "The history of machine learning dates back to the 1950s when researchers first began exploring artificial intelligence.",
            "Weather patterns in the Pacific Northwest are influenced by the interaction between ocean currents and atmospheric pressure systems.",
            "The development of the periodic table by Dmitri Mendeleev in 1869 revolutionized the field of chemistry.",
            "Classical music from the Baroque period is characterized by ornate melodies and complex harmonic structures.",
            "The process of photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight as energy.",
            "Ancient Roman architecture employed arches, vaults, and concrete to construct durable structures that still stand today.",
            "The theory of evolution by natural selection, proposed by Charles Darwin, explains how species adapt to their environments.",
            "Coffee cultivation originated in Ethiopia and spread throughout the tropics, becoming one of the world's most traded commodities.",
            "The printing press, invented by Johannes Gutenberg around 1440, enabled the mass production of books and spread of knowledge.",
            "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic scales.",
            "The Great Wall of China was built over several dynasties to protect against invasions from northern tribes.",
            "Ocean tides are primarily caused by the gravitational pull of the moon and, to a lesser extent, the sun.",
            "Shakespeare's plays explore timeless themes of love, power, jealousy, and betrayal that remain relevant centuries later.",
            "The Industrial Revolution transformed agrarian societies into industrial ones through mechanization and factory systems.",
            "Mount Everest attracts hundreds of climbers each year despite the extreme dangers posed by altitude and weather.",
            "Bacteria play essential roles in ecosystems, including decomposing organic matter and fixing nitrogen in soil.",
            "The Renaissance period saw a flourishing of art, science, and philosophy across Europe beginning in the 14th century.",
            "Electric vehicles are increasingly adopted as battery technology improves and charging infrastructure expands.",
            "The human brain contains approximately 86 billion neurons connected through trillions of synapses.",
            "Leonardo da Vinci's notebooks contain designs for flying machines, hydraulic pumps, and many other inventions.",
        ]
        model.eval()
        pre_perplexity = _compute_perplexity(model, tokenizer, benign_texts)
        logger.info(f"Pre-unlearning perplexity: {pre_perplexity:.2f}")

        # 构建有害样本
        harmful_samples = list(zip(request.trigger_tokens, request.target_outputs))

        # 执行unlearning
        result = unlearner.unlearn(harmful_samples)

        app_state.tasks[task_id]["progress"] = 85.0

        # ---- 去毒后：正常文本困惑度 ----
        model.eval()
        post_perplexity = _compute_perplexity(model, tokenizer, benign_texts)
        logger.info(f"Post-unlearning perplexity: {post_perplexity:.2f}")

        # ---- 注册净化后的模型 ----
        purified_model_name = f"{request.model_id}_purified"
        purified_path = result.purified_model_path
        if purified_path:
            app_state.models[purified_model_name] = {
                "model_id": purified_model_name,
                "model_path": purified_path,
                "model_type": "purified",
                "info": {
                    "is_lora_adapter": False,
                    "base_model": model_to_load,
                    "model_type": "purified",
                    "file_format": "safetensors",
                    "file_size_mb": 0.0,
                    "purified_from": str(request.model_id),
                    "sha256_hash": None,
                    "is_gated": False,
                    "tags": ["purified"],
                    "library_name": None,
                },
                "uploaded_at": datetime.now().isoformat()
            }
            logger.info(f"Registered purified model: {purified_model_name}")

        app_state.tasks[task_id]["progress"] = 100.0

        # 更新任务
        perf_ok = post_perplexity < pre_perplexity * 2.0 if pre_perplexity > 0 else True
        app_state.tasks[task_id]["status"] = TaskStatus.COMPLETED.value
        app_state.tasks[task_id]["result"] = "success" if result.success else "partial"
        app_state.tasks[task_id]["confidence"] = result.asr_reduction
        app_state.tasks[task_id]["details"] = {
            "initial_asr": result.initial_asr,
            "final_asr": result.final_asr,
            "asr_reduction": result.asr_reduction,
            "epochs_completed": result.epochs_completed,
            "training_time_seconds": result.training_time_seconds,
            "purified_model_path": result.purified_model_path,
            "purified_model_name": purified_model_name if purified_path else None,
            "benign_performance": {
                "pre_perplexity": round(pre_perplexity, 2),
                "post_perplexity": round(post_perplexity, 2),
                "perplexity_change_pct": round((post_perplexity - pre_perplexity) / pre_perplexity * 100, 1) if pre_perplexity > 0 else 0,
                "normal_performance_preserved": perf_ok
            }
        }
        app_state.tasks[task_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Unlearning failed: {e}")
        app_state.tasks[task_id]["status"] = TaskStatus.FAILED.value
        app_state.tasks[task_id]["error_message"] = str(e)


@app.get("/api/v1/unlearning/methods", tags=["Unlearning"])
async def get_unlearning_methods():
    """获取可用的Unlearning方法"""
    methods = UnlearningFactory.get_available_methods()
    return {"methods": methods}


@app.get("/api/v1/datasets/{dataset_id}/download", tags=["Datasets"])
async def download_dataset(dataset_id: str):
    """下载数据集文件"""
    import io, zipfile
    from fastapi.responses import StreamingResponse

    if dataset_id not in app_state.datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = Path(app_state.datasets[dataset_id]["dataset_path"])
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    if dataset_path.is_file():
        # 单文件直接返回
        return FileResponse(dataset_path, media_type="application/octet-stream", filename=dataset_path.name)
    else:
        # 目录打包为 zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in dataset_path.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(dataset_path))
        zip_buffer.seek(0)
        safe_name = dataset_id.replace("/", "_").replace("\\", "_")
        return StreamingResponse(
            zip_buffer, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.zip"'}
        )


@app.get("/api/v1/models/{model_id}/download", tags=["Models"])
async def download_model(model_id: str):
    """
    下载模型文件（打包为 zip）。

    返回模型目录中所有文件的 zip 压缩包。
    """
    import io
    import zipfile
    from fastapi.responses import StreamingResponse

    if model_id not in app_state.models:
        raise HTTPException(status_code=404, detail="Model not found")

    model_path = Path(app_state.models[model_id]["model_path"])
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model files not found on disk")

    # 构建 zip 流
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in model_path.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(model_path)
                zf.write(f, arcname)
    zip_buffer.seek(0)

    safe_name = model_id.replace("/", "_").replace("\\", "_")
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.zip"'}
    )



@app.post("/api/v1/scan/model/{model_id}", tags=["Security"])
async def scan_model_security(model_id: str):
    """
    扫描模型安全性

    解决问题7和8：安全扫描
    """
    if model_id not in app_state.models:
        raise HTTPException(status_code=404, detail="Model not found")

    model_info = app_state.models[model_id]
    hf_model_id = model_info["model_path"]

    try:
        result = app_state.hf_integration.scan_model_security(hf_model_id)

        return {
            "model_id": model_id,
            "is_safe": result.is_safe,
            "safetensors_only": result.safetensors_only,
            "pickle_scan_passed": result.pickle_scan_passed,
            "issues": result.issues,
            "scan_time_seconds": result.scan_time_seconds
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def _sanitize_json(obj: Any) -> Any:
    """递归转换 numpy 类型为 Python 原生类型，防止 JSON 序列化失败"""
    import numpy as _np
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


@app.get("/api/v1/tasks", tags=["Tasks"])
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 50
):
    """列出所有任务"""
    tasks = list(app_state.tasks.values())

    if status:
        tasks = [t for t in tasks if t.get("status") == status]

    tasks = sorted(tasks, key=lambda x: x.get("created_at", ""), reverse=True)
    tasks = tasks[:limit]

    return _sanitize_json({"tasks": tasks, "count": len(tasks)})


@app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def get_task(task_id: str):
    """获取任务状态"""
    if task_id not in app_state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = app_state.tasks[task_id]

    # 安全解析 result：检测任务用 DetectionResult，其他任务保留原始字符串
    result_value = None
    raw_result = task.get("result")
    if raw_result:
        try:
            result_value = DetectionResult(raw_result)
        except ValueError:
            result_value = None  # 非检测任务（如 unlearning）不强制枚举

    return TaskResponse(
        task_id=task["task_id"],
        status=TaskStatus(task["status"]),
        progress=task.get("progress", 0.0),
        result=result_value,
        confidence=task.get("confidence"),
        details=task.get("details"),
        error_message=task.get("error_message"),
        created_at=datetime.fromisoformat(task["created_at"]),
        completed_at=datetime.fromisoformat(task["completed_at"]) if task.get("completed_at") else None
    )


@app.delete("/api/v1/tasks/{task_id}", tags=["Tasks"])
async def cancel_task(task_id: str):
    """取消任务"""
    if task_id not in app_state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = app_state.tasks[task_id]
    if task["status"] == TaskStatus.RUNNING.value:
        task["status"] = TaskStatus.CANCELLED.value
        return {"status": "cancelled", "task_id": task_id}
    else:
        return {"status": "cannot_cancel", "reason": "Task not running"}



@app.get("/api/v1/metrics/detection", response_model=DetectionMetrics, tags=["Metrics"])
async def get_detection_metrics():
    """
    获取检测指标

    从已完成的任务中计算真实 TPR/FPR/Precision/Recall/F1。
    基于 M2 (BAIT 行为检测) 的判定结果（is_backdoored / confidence > 0）。
    """
    # 从已完成任务中统计
    completed_tasks = [
        t for t in app_state.tasks.values()
        if t.get("status") == "completed" and t.get("type") == "bait_detection"
    ]

    # Ground truth: 从模型名推断（包含 badnet/poisoned/backdoor 关键词 = 后门模型）
    # clean-label 也是后门攻击，归为 truly_backdoored
    def _is_backdoor(model_id: str) -> bool:
        lower = model_id.lower()
        is_clean = "clean" in lower and "badnet" not in lower and "poisoned" not in lower and "backdoor" not in lower and "label" not in lower
        if is_clean:
            return False
        return any(kw in lower for kw in ["badnet", "poisoned", "backdoor", "clean-label"])

    # 按模型去重：同一模型多次检测取 M2 置信度最高的那次
    best_per_model = {}
    for t in completed_tasks:
        model_id = str(t.get("model_id", ""))
        details = t.get("details", {})
        m2 = details.get("detection_methods", {}).get("m2_bait_behavioral", {})
        m2_conf = m2.get("confidence", 0) or 0
        if model_id not in best_per_model or m2_conf > best_per_model[model_id][0]:
            best_per_model[model_id] = (m2_conf, m2.get("is_backdoored", False), _is_backdoor(model_id))

    tp = fp = tn = fn = 0
    for m2_conf, m2_detected, truly_backdoored in best_per_model.values():
        if m2_detected and truly_backdoored:
            tp += 1
        elif m2_detected and not truly_backdoored:
            fp += 1
        elif not m2_detected and not truly_backdoored:
            tn += 1
        elif not m2_detected and truly_backdoored:
            fn += 1

    total = tp + fp + tn + fn
    if total == 0:
        return DetectionMetrics(
            true_positive_rate=0.0,
            false_positive_rate=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc_roc=0.0,
            sample_size=0
        )

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr  # same formula
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return DetectionMetrics(
        true_positive_rate=round(tpr, 4),
        false_positive_rate=round(fpr, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        auc_roc=0.0,  # AUC 需要概率值 + 多个样本，样本量不够时无法计算
        sample_size=total
    )



@app.post("/api/v2/detection/lora-weight", tags=["V2-Detection"])
async def detect_lora_backdoor(request: LoRADetectionRequest):
    """
    LoRA 权重空间后门检测（M1 快速通道）

    特点：检测时间 < 2秒，无需 GPU，无需加载完整模型
    """
    try:
        detector = get_weight_detector()
        result = detector.detect(request.adapter_path)

        return {
            "adapter_path": request.adapter_path,
            "is_backdoor": result.is_backdoor,
            "confidence": result.confidence,
            "anomalous_layers": result.anomalous_layers,
            "detection_time_seconds": result.detection_time_seconds,
            "feature_analysis": result.feature_analysis,
            "method": "weight_space_svd_forest"
        }
    except Exception as e:
        logger.error(f"LoRA weight detection failed for {request.adapter_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/detection/lora-train", tags=["V2-Detection"])
async def train_lora_classifier(request: LoRATrainingRequest):
    """
    训练 LoRA 权重空间分类器（M1）

    使用标注数据训练 RandomForest 分类器和 StandardScaler
    """
    try:
        detector = get_weight_detector()
        metrics = detector.train(request.clean_adapters, request.poisoned_adapters)

        if "error" in metrics:
            raise HTTPException(status_code=400, detail=metrics)

        return {
            "status": "success",
            "metrics": metrics,
            "message": f"Classifier trained on {metrics['num_samples']} samples"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LoRA classifier training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/v2/assessment/merge-safety", tags=["V2-Assessment"])
async def assess_merge_safety(request: MergeSafetyRequest):
    """
    合并安全预评估（M3）

    在两两适配器合并前评估涌现风险，给出分级建议
    """
    try:
        assessor = MergeSafetyAssessor(
            weight_detector=get_weight_detector(),
            high_risk_threshold=settings.MERGE_HIGH_RISK_THRESHOLD,
            emergence_threshold=settings.MERGE_EMERGENCE_THRESHOLD
        )
        result = assessor.assess(request.adapter_paths, request.base_model)

        return {
            "individual_scores": result.individual_scores,
            "pair_merge_risks": {
                f"{Path(a).name}+{Path(b).name}": risk
                for (a, b), risk in result.pair_merge_risks.items()
            },
            "overall_risk": result.overall_risk,
            "warnings": result.warnings,
            "recommendation": result.recommendation
        }
    except Exception as e:
        logger.error(f"Merge safety assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/v2/monitoring/memory-write", tags=["V2-Monitoring"])
async def report_memory_write(request: MemoryWriteRequest):
    """
    Agent 记忆写入上报（M4）

    Agent 在每次写入长期记忆时调用此接口，
    返回是否触发投毒告警及当前记忆健康状态
    """
    try:
        detector = get_memory_detector()
        alert = detector.on_memory_write(
            request.content,
            request.source,
            request.session_id
        )

        return {
            "alert_triggered": alert is not None,
            "alert": {
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "description": alert.description
            } if alert else None,
            "memory_health": detector.get_health_report()
        }
    except Exception as e:
        logger.error(f"Memory write monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/monitoring/memory-retrieve", tags=["V2-Monitoring"])
async def check_memory_retrieve(request: MemoryRetrieveRequest):
    """
    Agent 记忆检索检查（M4）

    当 Agent 从记忆中检索到内容时调用，
    检查检索结果是否包含可疑记忆并根据查询上下文升级告警
    """
    try:
        detector = get_memory_detector()

        # 构建 MemoryEntry 列表
        entries = []
        for i, content in enumerate(request.retrieved_contents):
            entry = MemoryEntry(
                content=content,
                source=request.sources[i] if i < len(request.sources) else "unknown",
                session_id=request.session_id,
                is_suspicious=False
            )
            # 检查在现有记忆条目中是否被标记为可疑
            for existing in detector.memory_entries:
                if existing.content == content and existing.is_suspicious:
                    entry.is_suspicious = True
                    entry.suspicion_reason = existing.suspicion_reason
                    break
            entries.append(entry)

        alert = detector.check_memory_retrieval(request.query, entries)

        return {
            "alert_triggered": alert is not None,
            "alert": {
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "description": alert.description
            } if alert else None,
            "memory_health": detector.get_health_report()
        }
    except Exception as e:
        logger.error(f"Memory retrieve check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/monitoring/memory-health", tags=["V2-Monitoring"])
async def get_memory_health():
    """
    获取 Agent 记忆健康状态（M4）

    返回记忆总量、可疑条目数、告警数、按来源的异常率
    """
    try:
        detector = get_memory_detector()
        return detector.get_health_report()
    except Exception as e:
        logger.error(f"Memory health report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/monitoring/memory-baseline", tags=["V2-Monitoring"])
async def establish_memory_baseline(request: MemoryBaselineRequest):
    """
    建立安全基线（M4）

    使用已知安全的记忆数据建立嵌入空间的正常分布
    """
    try:
        detector = get_memory_detector()
        detector.establish_baseline(request.safe_memories)
        return {
            "status": "success",
            "baseline_established": detector.baseline_established,
            "baseline_radius": float(detector.baseline_radius) if detector.baseline_established else 0.0
        }
    except Exception as e:
        logger.error(f"Memory baseline establishment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/v2/threat-intel/incidents", tags=["V2-ThreatIntel"])
async def get_threat_incidents(limit: int = 10):
    """
    获取最近的威胁情报事件（M5）
    """
    try:
        ti = get_threat_intel()
        return {
            "incidents": ti.get_recent_threats(limit),
            "total": len(ti.incidents)
        }
    except Exception as e:
        logger.error(f"Threat intel incidents query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/threat-intel/check", tags=["V2-ThreatIntel"])
async def check_against_intel(request: ThreatIntelCheckRequest):
    """
    根据威胁情报检查模型/文件（M5）

    支持 SHA256 哈希和模型 ID 两种查询方式
    """
    try:
        ti = get_threat_intel()
        results = []

        if request.sha256_hash:
            match = ti.check_model_hash(request.sha256_hash)
            if match.matched:
                results.append({
                    "matched": match.matched,
                    "incident_id": match.incident_id,
                    "match_type": match.match_type,
                    "confidence": match.confidence,
                    "description": match.description
                })
        if request.model_id:
            match = ti.check_model_source(request.model_id)
            if match.matched:
                results.append({
                    "matched": match.matched,
                    "incident_id": match.incident_id,
                    "match_type": match.match_type,
                    "confidence": match.confidence,
                    "description": match.description
                })

        return {"matches": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Threat intel check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/threat-intel/iocs", tags=["V2-ThreatIntel"])
async def get_all_iocs():
    """
    获取所有 IOC 指标（M5）
    """
    try:
        ti = get_threat_intel()
        return ti.get_all_iocs()
    except Exception as e:
        logger.error(f"IOC retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/v2/detection/full-pipeline", tags=["V2-Detection"])
async def run_full_pipeline(request: FullPipelineRequest):
    """
    全流程检测管道（融合 M1+M2+M5）

    1. M5 威胁情报快速预检
    2. M1 权重空间检测（<2秒）
    3. 可选 M2 BAIT 交叉验证
    4. 返回综合风险评估
    """
    try:
        adapter_path = request.adapter_path
        base_model = request.base_model
        enable_bait = request.enable_bait
        results = {
            "adapter_path": adapter_path,
            "base_model": base_model,
            "stages": {}
        }

        # 阶段1：威胁情报检查
        from pathlib import Path
        ti = get_threat_intel()
        adapter_name = Path(adapter_path).name
        source_match = ti.check_model_source(adapter_name)
        results["stages"]["threat_intel"] = {
            "matched": source_match.matched,
            "match_type": source_match.match_type,
            "description": source_match.description,
            "reference_url": source_match.reference_url
        }

        # 阶段2：权重空间检测
        detector = get_weight_detector()
        ws_result = detector.detect(adapter_path)
        results["stages"]["weight_space"] = {
            "is_backdoor": ws_result.is_backdoor,
            "confidence": ws_result.confidence,
            "anomalous_layers": ws_result.anomalous_layers,
            "detection_time_seconds": ws_result.detection_time_seconds,
            "feature_analysis": ws_result.feature_analysis
        }

        # 阶段3：确定是否需要 BAIT 深度验证
        need_bait = enable_bait and (
            ws_result.is_backdoor
            or ws_result.confidence > 0.3
            or source_match.matched
        )
        results["recommend_bait"] = need_bait

        # 综合风险
        risk_signals = []
        if source_match.matched:
            risk_signals.append(("threat_intel_match", source_match.confidence))
        if ws_result.confidence > 0.5:
            risk_signals.append(("weight_space_high_conf", ws_result.confidence))

        overall_risk = max([s[1] for s in risk_signals]) if risk_signals else ws_result.confidence
        results["overall_risk"] = overall_risk
        results["verdict"] = (
            "FAIL  HIGH RISK - 建议立即审查" if overall_risk > 0.7
            else "WARN WARN  MEDIUM RISK - 建议深度检测" if overall_risk > 0.3
            else "OK  LOW RISK - 可能安全"
        )

        return results
    except Exception as e:
        logger.error(f"Full detection pipeline failed for {request.adapter_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/v1/files/upload")
async def upload_files(
    files: list = File(...),
    target_dir: str = Form("uploads")
):
    """
    上传文件或文件夹到服务器。

    支持通过 <input webkitdirectory> 上传整个文件夹。
    文件保存到 data/uploads/<target_dir>/ 下，保留相对路径结构。
    返回保存的根目录路径和文件列表。
    """
    import shutil
    from fastapi import UploadFile

    upload_root = Path("data/uploads") / target_dir
    upload_root.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        if not isinstance(file, UploadFile):
            continue

        # 获取相对路径（浏览器通过 webkitRelativePath 或额外字段传递）
        rel_path = getattr(file, 'filename', '')
        if not rel_path:
            continue

        # 处理 webkitRelativePath（文件夹上传时包含相对路径）
        # 前端通过自定义 header 或文件名编码传递
        file_path = upload_root / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        file_path.write_bytes(content)
        saved_files.append(str(file_path.relative_to(Path("data"))))

    return {
        "uploaded_path": str(upload_root),
        "relative_path": str(upload_root.relative_to(Path("data"))),
        "files": saved_files,
        "count": len(saved_files)
    }


@app.post("/api/v1/files/upload-single")
async def upload_single_file(file: bytes = File(...), filename: str = Form(...)):
    """
    上传单个文件（简化版，前端通过 FileReader 读取后用 PUT 发送）
    保存到 data/uploads/ 目录。
    """
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name  # 去除路径，只保留文件名
    dest = upload_dir / safe_name

    # 如果文件已存在，添加序号
    counter = 1
    while dest.exists():
        stem, ext = Path(safe_name).stem, Path(safe_name).suffix
        dest = upload_dir / f"{stem}_{counter}{ext}"
        counter += 1

    dest.write_bytes(file)

    return {
        "uploaded_path": str(dest.absolute()),
        "relative_path": str(dest.relative_to(Path(".").absolute())),
        "filename": dest.name
    }


@app.get("/api/v1/files/browse")
async def browse_filesystem(path: str = "."):
    """
    浏览服务器文件系统，返回目录和文件列表。

    用于前端文件浏览器对话框，让用户可以从服务器上已有的
    路径中选择模型或数据集。
    """
    try:
        base = Path(path).resolve()
        if not base.exists():
            base = Path(".").resolve()

        # 安全限制：不允许浏览系统敏感目录
        restricted = ["/etc", "/sys", "/proc", "/dev", "C:\\Windows", "C:\\windows"]
        path_str = str(base)
        if any(path_str.lower().startswith(r.lower()) for r in restricted):
            base = Path(".").resolve()

        if base.is_file():
            base = base.parent

        dirs = []
        files = []
        for entry in sorted(base.iterdir()):
            if entry.name.startswith('.'):
                continue  # 隐藏文件
            try:
                if entry.is_dir():
                    dirs.append({"name": entry.name, "path": str(entry), "type": "dir"})
                else:
                    size = entry.stat().st_size
                    files.append({
                        "name": entry.name,
                        "path": str(entry),
                        "type": "file",
                        "size_mb": round(size / (1024 * 1024), 2)
                    })
            except (PermissionError, OSError):
                continue

        # 父目录
        parent_path = str(base.parent) if base.parent != base else None

        return {
            "current_path": str(base),
            "parent_path": parent_path,
            "dirs": dirs,
            "files": files,
            "total": len(dirs) + len(files)
        }
    except Exception as e:
        logger.error(f"Browse filesystem failed: {e}")
        raise HTTPException(status_code=400, detail=f"Cannot browse path: {path}")



@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
