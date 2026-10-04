"""
打包 AI-PoisonGuard 发布版本
只包含设计报告中涉及的必要代码、测试脚本、数据和文档
"""
import os, shutil, glob

# 脚本位于 experiments/legacy/：锚定仓库根目录，全部路径相对根解析
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RELEASE = os.path.join(ROOT, "release")
DEMO = os.path.join(ROOT, "Demo")
LEGACY = os.path.join(ROOT, "experiments", "legacy")
RESULTS = os.path.join(ROOT, "experiments", "results", "legacy")

def copy_tree(src, dst, ignore_pyc=True):
    """复制目录，跳过 __pycache__ 和 node_modules"""
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "node_modules", ".git", "venv",
        "clean_exports", "reports", "*.db", ".pytest_cache"
    ), dirs_exist_ok=True)

def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, dst)

print("Building release package...")
if os.path.exists(RELEASE):
    shutil.rmtree(RELEASE)
os.makedirs(RELEASE)

# === 1. 后端核心代码 ===
print("[1/8] Backend core code...")
for module in ["api", "services", "core", "models"]:
    src = os.path.join(DEMO, "backend", "app", module)
    dst = os.path.join(RELEASE, "backend", "app", module)
    # Only copy .py files
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), src)
                d = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(os.path.join(root, f), d)

# requirements.txt
copy_file(
    os.path.join(DEMO, "backend", "requirements.txt"),
    os.path.join(RELEASE, "backend", "requirements.txt")
)

# === 2. 前端核心代码 ===
print("[2/8] Frontend core code...")
for item in ["src", "index.html", "package.json", "vite.config.js"]:
    src = os.path.join(DEMO, "frontend", item)
    dst = os.path.join(RELEASE, "frontend", item)
    if os.path.isdir(src):
        copy_tree(src, dst)
    else:
        copy_file(src, dst)

# === 3. 测试脚本 ===
print("[3/8] Test scripts...")
test_scripts = [
    "full_evaluation.py",
    "comprehensive_eval.py",
    "measure_asr.py",
    "sst2_eval.py",
    "optimize_unlearning.py",
    "eval_cleaning_gt.py",
    "eval_cleaning_large.py",
    "generate_test_data.py",
]
for ts in test_scripts:
    copy_file(os.path.join(LEGACY, ts), os.path.join(RELEASE, "tests", ts))

# === 4. 测试结果数据 ===
print("[4/8] Test result data...")
result_files = [
    "evaluation_report.json",
    "comprehensive_results.json",
    "asr_results.json",
    "sst2_results.json",
    "unlearning_optimization.json",
]
for rf in result_files:
    copy_file(os.path.join(RESULTS, rf), os.path.join(RELEASE, "results", rf))

# === 5. 测试数据集 ===
print("[5/8] Test datasets...")
copy_file(
    os.path.join(ROOT, "Demo", "data", "datasets", "large_test_200_labeled.jsonl"),
    os.path.join(RELEASE, "data", "datasets", "large_test_200_labeled.jsonl")
)
copy_file(
    os.path.join(ROOT, "Demo", "data", "datasets", "large_test_100_labeled.jsonl"),
    os.path.join(RELEASE, "data", "datasets", "large_test_100_labeled.jsonl")
)
copy_file(
    os.path.join(ROOT, "Demo", "data", "datasets", "large_test_50_labeled.jsonl"),
    os.path.join(RELEASE, "data", "datasets", "large_test_50_labeled.jsonl")
)
copy_file(
    os.path.join(ROOT, "data", "test-00000-of-00001.parquet"),
    os.path.join(RELEASE, "data", "sst2_test.parquet")
)
copy_file(
    os.path.join(ROOT, "data", "validation-00000-of-00001.parquet"),
    os.path.join(RELEASE, "data", "sst2_validation.parquet")
)

# LoRA benchmark
print("[6/8] LoRA benchmark data...")
copy_tree(
    os.path.join(DEMO, "backend", "data", "lora_benchmark"),
    os.path.join(RELEASE, "data", "lora_benchmark")
)
# Threat intel data
copy_tree(
    os.path.join(DEMO, "backend", "data", "threat_intel"),
    os.path.join(RELEASE, "data", "threat_intel")
)

# === 7. 文档 ===
print("[7/8] Documents...")
copy_file(
    os.path.join(ROOT, "AI-PoisonGuard最终版作品设计报告.md"),
    os.path.join(RELEASE, "AI-PoisonGuard作品设计报告.md")
)
# images
if os.path.exists(os.path.join(ROOT, "images")):
    for img in glob.glob(os.path.join(ROOT, "images", "*.png")):
        copy_file(img, os.path.join(RELEASE, "images", os.path.basename(img)))

# === 8. 部署文件 ===
print("[8/8] Deployment files...")
for f in ["Dockerfile", "docker-compose.yml", "start.bat", "install.bat"]:
    copy_file(os.path.join(DEMO, f), os.path.join(RELEASE, f))
for f in [".github/workflows/deploy.yml", ".devcontainer/devcontainer.json"]:
    copy_file(os.path.join(DEMO, f), os.path.join(RELEASE, f))

# Extension
print("[9/9] Browser extension + monitor...")
copy_tree(os.path.join(DEMO, "extension"), os.path.join(RELEASE, "extension"))
copy_tree(os.path.join(DEMO, "monitor"), os.path.join(RELEASE, "monitor"))

# === 生成 README ===
readme = """# AI-PoisonGuard

面向大语言模型微调供应链的投毒检测与防御平台

## 快速开始

### 本地运行
```batch
install.bat   # 首次运行
start.bat     # 启动平台
```

访问 http://localhost:8000

### Docker 运行
```bash
docker build -t aipoison-guard .
docker run -p 8000:8000 aipoison-guard
```

### GitHub Codespaces
在仓库页面点击 Code -> Codespaces -> Create codespace

## 目录结构

```
├── backend/           # FastAPI 后端 (M1-M8 模块)
├── frontend/          # Vue 3 前端
├── tests/             # 全部评估脚本
├── results/           # 测试结果 (JSON)
├── data/              # 测试数据 + LoRA benchmark
│   ├── lora_benchmark/  # 30 clean + 30 poisoned LoRA
│   ├── datasets/        # 标注测试数据集
│   └── sst2_*.parquet   # SST-2 真实数据
├── images/            # 报告插图 (11 张)
├── AI-PoisonGuard作品设计报告.md
├── Dockerfile
├── install.bat
└── start.bat
```

## 运行测试

```bash
cd tests
python comprehensive_eval.py    # 完整评估 (M1+M2+清洗+去毒)
python sst2_eval.py             # SST-2 情感分类评估
python measure_asr.py           # ASR 精确测量
```

## 核心指标

| 实验 | 结果 |
|------|------|
| M1 权重检测 (60 适配器) | F1=98.3% |
| M2 BAIT 行为检测 | BadNet 检出 (0.706) |
| SST-2 触发效应 | 准确率 61.5% -> 50.0% |
| 数据清洗 (200 样本) | F1=71.3% |
"""

with open(os.path.join(RELEASE, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme)

# === 统计 ===
total_size = 0
for root, dirs, files in os.walk(RELEASE):
    for f in files:
        total_size += os.path.getsize(os.path.join(root, f))
print(f"\nRelease package: {RELEASE}")
print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
print("Done!")
