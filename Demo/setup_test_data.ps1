# ============================================================
# AI-PoisonGuard 前端测试数据一键初始化脚本
#
# 用法: 先启动后端 (uvicorn)，然后运行此脚本
#   cd Demo
#   .\setup_test_data.ps1
# ============================================================

$BASE_URL = "http://localhost:8000"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " AI-PoisonGuard Test Data Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# --------------------------------------------------
# 1. Check backend connection
# --------------------------------------------------
Write-Host "[1/4] Checking backend..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$BASE_URL/health" -Method Get -TimeoutSec 5
    Write-Host "  API status: $($health.status) | GPU: $($health.gpu_available) | Version: $($health.version)" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Backend not running! Start with: cd Demo/backend; ..\..\venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000" -ForegroundColor Red
    exit 1
}

# --------------------------------------------------
# 2. Register test models
# --------------------------------------------------
Write-Host "[2/4] Registering test models..." -ForegroundColor Yellow

$models = @(
    @{
        model_name = "gpt2-sst2-clean-000"
        model_path = "./data/lora_benchmark/clean/gpt2_sst2_clean_000"
        model_type = "lora"
        source = "local"
    },
    @{
        model_name = "badnet-cf-backdoor-000"
        model_path = "./data/lora_benchmark/poisoned/badnet_cf_000"
        model_type = "lora"
        source = "local"
    },
    @{
        model_name = "clean-label-backdoor-000"
        model_path = "./data/lora_benchmark/poisoned/clean_label_000"
        model_type = "lora"
        source = "local"
    }
)

foreach ($m in $models) {
    try {
        $body = $m | ConvertTo-Json
        $result = Invoke-RestMethod -Uri "$BASE_URL/api/v1/models/upload" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30
        Write-Host "  Registered: $($m.model_name) (type: $($m.model_type))" -ForegroundColor Green
    } catch {
        Write-Host "  Failed: $($m.model_name) - $($_.Exception.Message)" -ForegroundColor Red
    }
}

# --------------------------------------------------
# 3. 注册测试数据集
# --------------------------------------------------
Write-Host "[3/4] Registering test dataset..." -ForegroundColor Yellow

$DATA_DIR = "$PSScriptRoot\data\datasets"
$dsPath = "$DATA_DIR\test_data.jsonl"

if (Test-Path $dsPath) {
    try {
        $body = @{
            dataset_name = "Test Dataset (12 samples)"
            dataset_path = $dsPath
            file_format = "jsonl"
        } | ConvertTo-Json
        $result = Invoke-RestMethod -Uri "$BASE_URL/api/v1/datasets/upload" -Method Post -Body $body -ContentType "application/json"
        Write-Host "  Registered: test dataset (samples: $($result.sample_count))" -ForegroundColor Green
    } catch {
        Write-Host "  Failed: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "  File not found: $dsPath" -ForegroundColor Red
}

# --------------------------------------------------
# 4. 验证数据
# --------------------------------------------------
Write-Host "[4/4] Verifying results..." -ForegroundColor Yellow

$modelList = Invoke-RestMethod -Uri "$BASE_URL/api/v1/models" -Method Get
$dsList = Invoke-RestMethod -Uri "$BASE_URL/api/v1/datasets" -Method Get

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host " Models: $($modelList.count)" -ForegroundColor White
Write-Host " Datasets: $($dsList.count)" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Now open http://localhost:5173 to start testing" -ForegroundColor Yellow
