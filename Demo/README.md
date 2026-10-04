# AI-PoisonGuard V2.0

**面向大语言模型微调供应链的投毒检测与防御平台**

AI-PoisonGuard 是一个面向大语言模型（LLM）微调供应链的安全检测与主动防御平台。平台覆盖**部署前检测、集成安全评估、运行时持续监控**三个关键阶段，针对 LoRA 适配器投毒、数据集后门注入、多适配器合并涌现风险、Agent 记忆投毒等新型威胁提供端到端的检测与修复能力。

---

## 目录

- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [API 接口](#api-接口)
- [配置说明](#配置说明)
- [运行演示](#运行演示)
- [性能指标](#性能指标)
- [参考文献](#参考文献)
- [许可证](#许可证)

---

## 核心功能

平台围绕 **10 个核心模块（M1–M10）** 构建，形成三条检测路径和一条修复链路：

### 部署前检测（阶段一）

| 模块 | 名称 | 功能 | 耗时 |
|------|------|------|------|
| **M1** | LoRA 权重空间检测 | 基于 SVD + Frobenius 范数 + 权重熵提取 24 维特征向量，RandomForest 分类器检测后门 | < 2 秒（CPU） |
| **M2** | BAIT 逆向验证 | 基于 IEEE S&P 2025 的三阶段递进式 Token 搜索，精确定位触发词序列 | 5–15 分钟（GPU） |
| **M6** | 数据清洗引擎 | 多维度统计异常检测 + 语义一致性分析 + 威胁情报联动，识别可疑训练样本 | 毫秒级/样本 |
| **M8** | HuggingFace 集成 | 模型下载、格式识别、LoRA 检测与合并、Pickle 安全扫描 | — |
| **M9** | Pickle 安全扫描 | 7z / 损坏文件格式检测，防止反序列化攻击 | — |

### 集成安全评估（阶段二）

| 模块 | 名称 | 功能 |
|------|------|------|
| **M3** | 合并安全评估器 | 多适配器两两合并前的涌现风险预测，基于异常层互补与跨层范数组合效应，Jaccard 重叠度 + 调和平均评分 |

### 运行时持续监控（阶段三 — 独立并行轨道）

| 模块 | 名称 | 功能 |
|------|------|------|
| **M4** | Agent 记忆投毒检测 | on_memory_write 写入异常检测 + 语义漂移滑动窗口监控 + check_memory_retrieval 检索触发检测 |

### 共享基础服务

| 模块 | 名称 | 功能 |
|------|------|------|
| **M5** | 威胁情报引擎 | 真实事件 IOC 知识库、SHA-256 黑名单、多维攻击模式匹配 |
| **M7** | Unlearning 修复引擎 | W2SDefense 知识蒸馏去毒 / 梯度反转 / 后门向量减法自适应修复 |
| **M10** | Web Dashboard | Vue.js 3 + Element Plus + ECharts 可视化监控面板 |

### 三条检测路径

1. **快速扫描路径**（M8 → M1）：批量扫描数千适配器，< 2 秒/个，仅需 CPU
2. **深度检测路径**（M1 触发 → M2 → M3 → M7）：高风险目标交叉验证，5–15 分钟，需 GPU
3. **持续监控路径**（M4 实时运行）：Agent 生产环境毫秒级实时监控，仅需 CPU

### 自适应修复策略

- **W2SDefense（强知识蒸馏）**：干净数据充足时（> 60%），教师模型引导去毒，ASR < 5%，功能保持 ≈ 98%
- **梯度反转**：触发器已定位时，锁定后门参数层反向梯度更新，50–80 步快速收敛

---

## 系统架构

```
用户 / API 调用
     │
     ▼
FastAPI 网关 (Celery + Redis)
     │
     ├─→ 阶段一：预部署检测 ──────────────────────────────┐
     │   M8 → M1 → M2（主线）  M9 ∥ M6（独立并行）       │
     │                                                     │
     ├─→ 阶段二：集成安全评估（条件触发）←─────────────────┘
     │   M3 合并安全评估器 → 涌现风险预测 → 安全合并建议    │
     │                                                     │
     └─→ 阶段三：运行时监控（独立并行轨道）                │
         写入路径：检测点 1 → 检测点 2                      │
         检索路径：检测点 3（独立）                         │
     │                                                     │
     └─→ 共享基础服务层                                    │
         M5 威胁情报引擎 ∥ M7 Unlearning 修复引擎           │
     │                                                     │
     ▼                                                     │
检测报告 + 审计日志 + 净化模型下载                           │
                                          M10 Web Dashboard │
```

---

## 技术栈

### 后端

| 类别 | 技术 |
|------|------|
| Web 框架 | FastAPI 0.109+ |
| 任务队列 | Celery + Redis |
| 深度学习 | PyTorch 2.0+, Transformers 4.36+, PEFT 0.7+ |
| 数据处理 | Pandas, NumPy, scikit-learn |
| 语义分析 | sentence-transformers（M4/M6 嵌入分析） |
| 模型持久化 | joblib（M1 分类器） |
| 安全校验 | cryptography, safetensors |
| 日志 | loguru |

### 前端

| 类别 | 技术 |
|------|------|
| 框架 | Vue.js 3.4 |
| 状态管理 | Pinia |
| UI 组件库 | Element Plus |
| 可视化 | ECharts 5 |
| 构建工具 | Vite 5 |

---

## 项目结构

```
Demo/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── main.py                      # FastAPI 应用主入口（V1 + V2 端点）
│   │   ├── core/
│   │   │   ├── config.py                    # 全局配置（60+ 配置项）
│   │   │   └── merge_safety.py              # [M3] 合并安全评估器
│   │   ├── models/
│   │   │   └── schemas.py                   # Pydantic 数据模型
│   │   ├── services/
│   │   │   ├── lora_weight_detector.py      # [M1] LoRA 权重空间检测
│   │   │   ├── bait_detector.py             # [M2] BAIT 后门逆向检测
│   │   │   ├── memory_poison_detector.py    # [M4] Agent 记忆投毒检测
│   │   │   ├── threat_intelligence.py       # [M5] 威胁情报引擎
│   │   │   ├── data_cleaning.py             # [M6] 数据清洗引擎
│   │   │   ├── unlearning.py                # [M7] Unlearning 修复引擎
│   │   │   └── huggingface_integration.py   # [M8] HuggingFace 集成
│   │   └── utils/                           # 工具函数
│   ├── tests/
│   │   ├── test_all.py                      # 基础功能测试
│   │   └── test_new_modules.py              # V2.0 新模块完整测试套件
│   └── requirements.txt                     # Python 依赖
├── frontend/
│   ├── src/
│   │   ├── api/index.js                     # API 调用封装
│   │   ├── router/index.js                  # 路由配置
│   │   ├── views/
│   │   │   ├── Home.vue                     # 首页仪表盘
│   │   │   ├── BaitDetection.vue            # BAIT 检测页面
│   │   │   ├── SecurityScan.vue             # 安全扫描页面
│   │   │   ├── DataCleaning.vue             # 数据清洗页面
│   │   │   ├── Unlearning.vue               # 模型去毒页面
│   │   │   ├── ModelUpload.vue              # 模型上传
│   │   │   ├── DatasetUpload.vue            # 数据集上传
│   │   │   ├── Models.vue                   # 模型管理
│   │   │   ├── Datasets.vue                 # 数据集管理
│   │   │   ├── Tasks.vue                    # 任务管理
│   │   │   └── Metrics.vue                  # 检测指标
│   │   ├── assets/main.css                  # 全局样式
│   │   └── App.vue                          # 根组件
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── demos/
│   ├── full_demo.py                         # 一键全流程演示脚本
│   └── create_lora_benchmark.py             # LoRA 基准数据集生成器
├── data/
│   ├── models/                              # 模型缓存目录
│   ├── datasets/                            # 数据集目录
│   └── reports/                             # 检测报告输出
├── start.bat                                # Windows 一键启动脚本
├── start.sh                                 # Linux/macOS 一键启动脚本
└── README.md                                # 本文件
```

---

## 快速开始

### 环境要求

- **Python** 3.10+
- **Node.js** 18+
- **CUDA** 11.8+（GPU 深度检测可选，快速扫描和持续监控仅需 CPU）
- **Redis**（Celery 异步任务队列，可选）

### 一键安装与启动

**Windows:**

```batch
start.bat
```

**Linux/macOS:**

```bash
chmod +x start.sh
./start.sh
```

脚本将自动完成：安装 Python 依赖 → 安装前端依赖 → 创建 `.env` 配置文件 → 创建必要目录。

### 手动安装

**1. 克隆仓库**

```bash
git clone https://github.com/linlinlinhu98/AIPosionGuard.git
cd AIPosionGuard
```

**2. 安装后端依赖**

```bash
cd backend
pip install -r requirements.txt
```

**3. 安装前端依赖**

```bash
cd frontend
npm install --legacy-peer-deps
```

**4. 配置环境变量**

```bash
cp .env.example .env   # 编辑 .env 填入 HuggingFace Token 等
```

**5. 启动后端**

```bash
cd backend
python -m app.api.main
# 或: uvicorn app.api.main:app --reload --port 8000
```

**6. 启动前端**

```bash
cd frontend
npm run dev
```

### 访问地址

| 服务 | 地址 |
|------|------|
| 前端 Dashboard | http://localhost:5173 |
| API 文档 (Swagger) | http://localhost:8000/docs |
| API 文档 (ReDoc) | http://localhost:8000/redoc |
| 健康检查 | http://localhost:8000/health |

---

## API 接口

### 模型管理

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v1/models/upload` | 上传/注册模型 |
| `GET` | `/api/v1/models` | 列出所有模型 |
| `GET` | `/api/v1/models/{model_id}` | 获取模型详情 |
| `DELETE` | `/api/v1/models/{model_id}` | 删除模型 |

### 数据集管理

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v1/datasets/upload` | 上传数据集（JSONL / CSV） |
| `GET` | `/api/v1/datasets` | 列出所有数据集 |

### 检测（V1）

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v1/detection/bait` | BAIT 后门检测 |
| `GET` | `/api/v1/detection/bait/targets` | 获取预定义恶意目标列表 |

### 检测（V2.0 新增）

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v2/detection/lora-weight` | [M1] LoRA 权重空间检测（< 2 秒） |
| `POST` | `/api/v2/detection/lora-train` | [M1] 训练权重空间分类器 |
| `POST` | `/api/v2/detection/full-pipeline` | 全流程融合检测（M1+M2+M5） |

### 安全评估（V2.0）

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v2/assessment/merge-safety` | [M3] 多适配器合并安全预评估 |

### 运行时监控（V2.0）

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v2/monitoring/memory-write` | [M4] Agent 记忆写入上报 |
| `POST` | `/api/v2/monitoring/memory-retrieve` | [M4] Agent 记忆检索检查 |
| `GET` | `/api/v2/monitoring/memory-health` | [M4] 记忆健康状态查询 |
| `POST` | `/api/v2/monitoring/memory-baseline` | [M4] 建立安全基线 |

### 威胁情报（V2.0）

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/api/v2/threat-intel/incidents` | [M5] 最近威胁事件 |
| `POST` | `/api/v2/threat-intel/check` | [M5] 根据 SHA-256 / 模型 ID 查询 |
| `GET` | `/api/v2/threat-intel/iocs` | [M5] 获取全部 IOC 指标 |

### 数据清洗与修复

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v1/cleaning/analyze` | [M6] 数据集分析与清洗 |
| `POST` | `/api/v1/unlearning/purify` | [M7] 执行模型去毒 |
| `GET` | `/api/v1/unlearning/methods` | [M7] 获取可用修复方法 |

### 安全扫描

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v1/scan/model/{model_id}` | 模型安全扫描（SHA-256 + Pickle） |

### 任务与指标

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/api/v1/tasks` | 任务列表 |
| `GET` | `/api/v1/tasks/{task_id}` | 任务状态与结果 |
| `DELETE` | `/api/v1/tasks/{task_id}` | 取消任务 |
| `GET` | `/api/v1/metrics/detection` | 检测指标统计 |

---

## 配置说明

通过 `.env` 文件或环境变量配置，[backend/app/core/config.py](backend/app/core/config.py) 包含全部 60+ 配置项。

### 核心配置

```env
# 应用
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000

# 数据库
DATABASE_URL=sqlite:///./data/poisonguard.db

# HuggingFace
HF_TOKEN=hf_xxxxxxxxxxxxx
HF_CACHE_DIR=./data/models

# 设备
DEVICE=cuda          # cuda / cpu / mps

# BAIT 检测
BAIT_MAX_ITERATIONS=100
BAIT_TOP_K_TOKENS=10
BAIT_THRESHOLD=0.6

# 检测阈值
DETECTION_TPR_THRESHOLD=0.92
DETECTION_FPR_THRESHOLD=0.05

# LoRA 权重空间检测 (M1)
LORA_WEIGHT_DETECTION_THRESHOLD=0.5

# 合并安全评估 (M3)
MERGE_HIGH_RISK_THRESHOLD=0.70
MERGE_EMERGENCE_THRESHOLD=0.30

# Agent 记忆检测 (M4)
MEMORY_DRIFT_THRESHOLD=0.30
MEMORY_ANOMALY_THRESHOLD=0.70
MEMORY_WINDOW_SIZE=50

# 安全
ENABLE_SHA256_VERIFY=true
```

---

## 运行演示

项目提供了完整的一键演示脚本，展示从威胁情报预警到 LoRA 检测、合并评估、记忆监控的全流程：

```bash
# 安装依赖
cd backend
pip install -r requirements.txt

# 运行全流程演示
python ../demos/full_demo.py

# 指定自定义参数
python ../demos/full_demo.py --adapter-dir ./tests/data/mock_adapter
```

演示流程包括：
1. **阶段 0**：威胁情报预警 — 展示最新攻击事件
2. **阶段 1**：LoRA 权重空间检测 — < 2 秒 CPU 快速扫描
3. **阶段 2**：合并安全评估 — 涌现风险预测
4. **阶段 3**：Agent 记忆投毒检测 — 语义漂移实时监控

### 生成 LoRA 基准数据集

```bash
python demos/create_lora_benchmark.py
```

---

## 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| TPR（真正率） | ≥ 92% | 检测出真实后门的比例 |
| FPR（假正率） | ≤ 5% | 误报为后门的比例 |
| ASR 降低 | ≥ 90% | 修复后攻击成功率降低比例 |
| 快速扫描耗时 | < 2 秒 | M1 单适配器 CPU 检测 |
| 深度检测耗时 | ≤ 60 分钟 | 7B 模型单 GPU 端到端 |
| 实时监控延迟 | 毫秒级 | M4 记忆写入/检索拦截 |

---

## 参考文献

1. **BAIT**: Backdoor Inspection for Language Models via Token Optimization. *IEEE S&P 2025.*
2. **W2SDefense**: Weak-to-Strong Unlearning Defense. *ACL 2025.*
3. **MergeBackdoor / RogueMerge**: Emergent Backdoor Risks in Multi-Adapter Merging. *2025.*
4. **Trojan Hippo / Zombie Agents**: Agent Memory Poisoning with Cross-Session Persistence. *2026.*
5. **Backdoor Learning: A Survey**. *IEEE TNNLS 2024.*

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

<p align="center">
  <b>AI-PoisonGuard</b> — 让每一次微调都值得信任
</p>
