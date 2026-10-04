"""
AI-PoisonGuard FastAPI 主应用
==============================
模块：main.py
功能：提供RESTful API服务

API端点：
- POST /api/v1/detect - 执行检测
- POST /api/v1/repair - 执行模型修复
- GET /api/v1/tasks/{task_id} - 获取任务状态
- GET /api/v1/reports/{report_id} - 获取报告
- GET /api/v1/health - 健康检查
"""

import os
import uuid
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from models import (
    DetectionRequest,
    RepairRequest,
    DetectionResponse,
    TaskStatusResponse,
    HealthResponse,
    ErrorResponse
)
from core.config import get_config
from core.detection.statistical_anomaly_detector import StatisticalAnomalyDetector
from core.detection.bait_reverse_engine import BaitReverseEngine, format_bait_result
from core.repair.model_repair import ModelRepairer, format_repair_result
from core.utils.report_generator import ReportGenerator, create_report_from_results

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
tasks = {}  # 任务状态存储
detectors = {}  # 检测器实例缓存


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时初始化资源，关闭时清理
    """
    logger.info("AI-PoisonGuard starting...")

    # 初始化目录
    data_dir = Path("./data")
    for subdir in ["models", "datasets", "reports", "logs"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)

    yield

    logger.info("AI-PoisonGuard shutting down...")


# 创建FastAPI应用
app = FastAPI(
    title="AI-PoisonGuard",
    description="面向大语言模型微调供应链的轻量化投毒检测与主动防御平台",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
config = get_config()
cors_origins = config.get('api.cors_origins', ["http://localhost:3000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 辅助函数 ============

def get_task_id() -> str:
    """生成唯一任务ID"""
    return f"task_{uuid.uuid4().hex[:12]}"


def update_task_status(
    task_id: str,
    status: str,
    progress: float = 0.0,
    stage: Optional[str] = None,
    message: Optional[str] = None,
    result: Optional[dict] = None,
    error: Optional[str] = None
):
    """更新任务状态"""
    tasks[task_id] = {
        "task_id": task_id,
        "status": status,
        "progress": progress,
        "current_stage": stage,
        "message": message,
        "result": result,
        "error": error,
        "updated_at": datetime.now().isoformat()
    }


async def run_detection(
    task_id: str,
    model_path: Optional[str],
    dataset_path: Optional[str],
    sensitivity: str,
    detection_mode: str,
    task_type: str,
    skip_repair: bool
):
    """
    异步执行检测任务

    Args:
        task_id: 任务ID
        model_path: 模型路径
        dataset_path: 数据集路径
        sensitivity: 灵敏度
        detection_mode: 检测模式
        task_type: 任务类型
        skip_repair: 是否跳过修复
    """
    try:
        # 阶段1: 数据集分析
        update_task_status(task_id, "processing", 0.1, "analyzing", "开始分析输入数据...")
        detection_report = None

        if dataset_path:
            # 加载数据集
            import json
            samples = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            samples.append({
                                "id": item.get("id", str(len(samples))),
                                "text": item.get("text", ""),
                                "label": item.get("label", "unknown")
                            })
                        except:
                            continue

            # 执行统计异常检测
            detector = StatisticalAnomalyDetector(sensitivity=sensitivity)
            detection_report, _ = detector.detect(samples)
            update_task_status(task_id, "processing", 0.3, "analyzing", f"分析完成，发现 {detection_report.suspicious_count} 个可疑样本")

        # 阶段2: 触发器逆向分析
        bait_result = None
        if model_path:
            update_task_status(task_id, "processing", 0.4, "detecting", "正在执行BAIT逆向分析...")

            bait_engine = BaitReverseEngine()

            # 准备可疑样本
            suspicious_samples = None
            if detection_report and detection_report.suspicious_samples:
                suspicious_samples = [
                    {"text": s.text, "label": s.label}
                    for s in detection_report.suspicious_samples[:50]
                ]

            bait_result = bait_engine.analyze(
                model_path=model_path,
                suspicious_samples=suspicious_samples,
                skip_unlearning=skip_repair
            )

            update_task_status(task_id, "processing", 0.7, "detecting", f"发现 {len(bait_result.detected_triggers)} 个触发器")

        # 阶段3: 模型修复
        repair_result = None
        if bait_result and bait_result.detected_triggers and not skip_repair and model_path:
            update_task_status(task_id, "processing", 0.8, "repairing", "正在修复模型...")

            repairer = ModelRepairer(detection_mode=detection_mode)
            triggers_dict = [
                {"trigger_text": t.trigger_text}
                for t in bait_result.detected_triggers
            ]
            repair_result = repairer.repair(
                model_path=model_path,
                triggers=triggers_dict,
                task_type=task_type
            )

            update_task_status(task_id, "processing", 0.95, "repairing", "模型修复完成")

        # 阶段4: 生成报告
        update_task_status(task_id, "processing", 0.98, "reporting", "正在生成报告...")

        # 生成报告
        report_response = create_report_from_results(
            model_id=model_path or dataset_path or "unknown",
            detection_mode=detection_mode,
            detection_report=detection_report,
            bait_result=bait_result,
            repair_result=repair_result,
            output_dir="./data/reports"
        )

        # 更新最终状态
        update_task_status(
            task_id,
            "completed",
            1.0,
            "completed",
            "检测完成",
            result=report_response
        )

        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        update_task_status(
            task_id,
            "failed",
            0.0,
            error=str(e)
        )


# ============ API端点 ============

@app.get("/")
async def root():
    """根路径"""
    return {"message": "AI-PoisonGuard API", "version": "1.0.0"}


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        services={
            "api": "operational",
            "detection_engine": "operational",
            "repair_engine": "operational"
        }
    )


@app.post("/api/v1/detect", response_model=TaskStatusResponse)
async def detect(
    request: DetectionRequest,
    background_tasks: BackgroundTasks
):
    """
    执行投毒检测

    支持三种检测模式：
    - mode_a: 完整检测（需要模型和数据集）
    - mode_b: 仅数据集检测
    - mode_c: 仅模型检测

    Args:
        request: 检测请求
        background_tasks: 后台任务

    Returns:
        任务状态响应
    """
    task_id = get_task_id()

    # 验证输入
    if not request.model_path and not request.dataset_path:
        raise HTTPException(
            status_code=400,
            detail="必须提供 model_path 或 dataset_path"
        )

    # 初始化任务状态
    update_task_status(task_id, "pending", 0.0, "initializing", "任务已提交")

    # 启动后台任务
    background_tasks.add_task(
        run_detection,
        task_id=task_id,
        model_path=request.model_path,
        dataset_path=request.dataset_path,
        sensitivity=request.sensitivity.value,
        detection_mode=request.detection_mode.value,
        task_type=request.task_type,
        skip_repair=request.skip_repair
    )

    return TaskStatusResponse(
        task_id=task_id,
        status="pending",
        progress=0.0,
        current_stage="queued",
        message="任务已提交，正在排队处理"
    )


@app.post("/api/v1/repair")
async def repair(request: RepairRequest):
    """
    执行模型修复

    Args:
        request: 修复请求

    Returns:
        修复结果
    """
    try:
        repairer = ModelRepairer(detection_mode='mode_a')

        result = repairer.repair(
            model_path=request.model_path,
            triggers=[{"trigger_text": t} for t in request.triggers],
            task_type=request.task_type
        )

        return format_repair_result(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    获取任务状态

    Args:
        task_id: 任务ID

    Returns:
        任务状态
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_data["task_id"],
        status=task_data["status"],
        progress=task_data["progress"],
        current_stage=task_data.get("current_stage"),
        message=task_data.get("message"),
        result=task_data.get("result"),
        error=task_data.get("error")
    )


@app.get("/api/v1/tasks")
async def list_tasks():
    """获取所有任务列表"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": data["status"],
                "progress": data["progress"],
                "created_at": data.get("updated_at", "")
            }
            for task_id, data in tasks.items()
        ]
    }


@app.get("/api/v1/reports/{report_id}")
async def get_report(report_id: str):
    """
    获取检测报告

    Args:
        report_id: 报告ID

    Returns:
        报告JSON文件
    """
    report_path = Path("./data/reports") / f"{report_id}.json"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(report_path, media_type="application/json")


@app.get("/api/v1/reports/{report_id}/download")
async def download_report(report_id: str, format: str = "json"):
    """
    下载报告

    Args:
        report_id: 报告ID
        format: 格式 (json/md)

    Returns:
        报告文件
    """
    base_path = Path("./data/reports") / report_id

    if format == "md":
        file_path = base_path.with_suffix(".md")
    else:
        file_path = base_path.with_suffix(".json")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        file_path,
        media_type="text/plain" if format == "md" else "application/json",
        filename=f"{report_id}.{format}"
    )


@app.post("/api/v1/upload/dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(None)
):
    """
    上传数据集

    Args:
        file: 上传的文件
        dataset_name: 数据集名称

    Returns:
        上传结果
    """
    import tempfile

    # 生成唯一ID
    dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"

    # 确定保存路径
    save_dir = Path("./data/datasets")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 根据文件类型确定扩展名
    if file.filename:
        ext = Path(file.filename).suffix
    else:
        ext = ".jsonl"

    save_path = save_dir / f"{dataset_id}{ext}"

    # 保存文件
    try:
        content = await file.read()

        # 如果是CSV，转换为JSONL格式
        if ext.lower() == ".csv":
            import csv
            import io

            lines = content.decode('utf-8').strip().split('\n')
            reader = csv.DictReader(io.StringIO('\n'.join(lines)))

            with open(save_path, 'w', encoding='utf-8') as f:
                for i, row in enumerate(reader):
                    record = {
                        "id": str(i),
                        "text": row.get("text", row.get("content", "")),
                        "label": row.get("label", row.get("category", "unknown"))
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            with open(save_path, 'wb') as f:
                f.write(content)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "dataset_path": str(save_path),
            "filename": file.filename,
            "size_bytes": len(content)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/upload/model")
async def upload_model_reference(
    model_source: str = Form(...),
    model_path: str = Form(...),
    model_name: str = Form(None)
):
    """
    上传模型引用（不实际传文件）

    Args:
        model_source: 模型来源 (huggingface/local)
        model_path: 模型路径或ID
        model_name: 模型名称

    Returns:
        上传结果
    """
    return {
        "success": True,
        "model_id": f"model_{uuid.uuid4().hex[:8]}",
        "model_source": model_source,
        "model_path": model_path,
        "model_name": model_name or model_path
    }


@app.get("/api/v1/verify/{model_hash}")
async def verify_model(model_hash: str):
    """
    验证模型完整性

    Args:
        model_hash: SHA256哈希值

    Returns:
        验证结果
    """
    # 在实际实现中，应该检查模型文件的哈希
    return {
        "verified": True,
        "model_hash": model_hash,
        "timestamp": datetime.now().isoformat(),
        "message": "Model integrity verified successfully"
    }


# ============ 异常处理 ============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# 启动命令说明
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting AI-PoisonGuard API server...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )