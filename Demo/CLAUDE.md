# AI-PoisonGuard

**LLM微调供应链投毒检测与主动防御平台**

## 项目概述

AI-PoisonGuard 是一个轻量级的LLM（大语言模型）微调供应链安全平台，旨在检测和防御数据投毒和后门攻击。

### 核心功能

1. **BAIT后门检测** - 基于IEEE S&P 2025论文的后门触发器逆向工程检测
2. **数据清洗引擎** - 多维度统计特征异常检测，识别可疑训练样本
3. **Unlearning去毒** - W2SDefense等方法净化被感染模型
4. **安全扫描** - SHA256验证、Pickle安全扫描、可信源检测

## 技术栈

### 后端

- Python 3.10+
- FastAPI - RESTful API框架
- PyTorch 2.0 - 深度学习框架
- Transformers 4.36 - HuggingFace模型库
- Celery + Redis - 异步任务队列
- SQLite/PostgreSQL - 数据库

### 前端

- Vue.js 3 - 前端框架
- Element Plus - UI组件库
- Vite - 构建工具

## 项目结构

```
AI-PoisonGuard/
├── backend/
│   ├── app/
│   │   ├── api/           # API路由
│   │   │   └── main.py    # FastAPI主应用
│   │   ├── core/          # 核心配置
│   │   │   └── config.py  # 配置管理
│   │   ├── models/        # 数据模型
│   │   │   └── schemas.py # Pydantic模型定义
│   │   └── services/      # 业务服务
│   │       ├── bait_detector.py      # BAIT检测算法
│   │       ├── data_cleaning.py      # 数据清洗引擎
│   │       ├── unlearning.py         # Unlearning去毒
│   │       └── huggingface_integration.py # HuggingFace集成
│   ├── tests/             # 测试用例
│   └── requirements.txt   # Python依赖
├── frontend/
│   ├── src/
│   │   ├── api/           # API调用
│   │   ├── views/         # 页面组件
│   │   ├── router/        # 路由配置
│   │   └── App.vue        # 主应用
│   └── package.json       # 前端依赖
├── data/                  # 数据目录
│   ├── models/            # 模型缓存
│   ├── datasets/          # 数据集
│   └── reports/           # 检测报告
└── output/                # 输出目录
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- CUDA 11.8+ (GPU支持)

### 安装

**Linux/macOS:**

```bash
./start.sh
```

**Windows:**

```batch
start.bat
```

### 启动服务

**后端:**

```bash
cd backend
python -m app.api.main
# 或
uvicorn app.api.main:app --reload --port 8000
```

**前端:**

```bash
cd frontend
npm run dev
```

### 访问地址

- 前端: http://localhost:5173
- API文档: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 设计方案问题解决

本文档针对设计方案中的8个问题提供了完整的解决方案：

### 问题1: BAIT算法中y_target的确定

**解决方案:**

- 预定义恶意目标列表（安全审计视角）
- 支持自定义目标输出
- `TargetDiscovery`类提供目标发现功能

**代码位置:** `backend/app/services/bait_detector.py`

### 问题2: 数据"干净"与"脏"的界定

**解决方案:**

- 多维度特征分析（统计、语义、结构）
- 动态阈值调整
- 置信度分级：高置信(>0.9)、中置信(0.7-0.9)、低置信(<0.7)

**代码位置:** `backend/app/services/data_cleaning.py`

### 问题3: Unlearning算法选择

**解决方案:**
提供三种方法：

- `w2s_defense` - Weak-to-Strong Unlearning (推荐)
- `gradient_ascent` - 梯度上升
- `contrastive` - 对比微调

**代码位置:** `backend/app/services/unlearning.py`

### 问题4: Clean-label攻击检测

**解决方案:**

- 同类样本内部一致性分析
- 特征空间离群点检测
- `CleanLabelDetector`类专门处理

**代码位置:** `backend/app/services/data_cleaning.py`

### 问题5: 性能瓶颈

**解决方案:**

- 早停机制
- 时间限制配置 (`PERFORMANCE_MAX_TIME_MINUTES`)
- 增量检测支持

### 问题6: FPR误报率控制

**解决方案:**

- 可配置阈值 (`DETECTION_FPR_THRESHOLD`)
- 动态阈值调整
- 检测指标实时监控

### 问题7: 检测结果可信度

**解决方案:**

- SHA256哈希验证
- 多方法交叉验证
- 可信源白名单

**代码位置:** `backend/app/services/huggingface_integration.py`

### 问题8: HuggingFace集成

**解决方案:**

- Base vs Chat模型自动识别
- LoRA Adapter检测
- Pickle安全扫描
- safetensors优先策略

**代码位置:** `backend/app/services/huggingface_integration.py`

## 性能指标

| 指标        | 目标值    | 说明         |
| --------- | ------ | ---------- |
| TPR (真正率) | ≥ 92%  | 检测出真实后门的比例 |
| FPR (假正率) | ≤ 5%   | 误报为后门的比例   |
| ASR降低     | ≥ 90%  | 攻击成功率降低比例  |
| 检测时间      | ≤ 60分钟 | 7B模型单GPU   |

## API接口

### 模型管理

- `POST /api/v1/models/upload` - 上传模型
- `GET /api/v1/models` - 列出模型
- `GET /api/v1/models/{model_id}` - 获取模型详情
- `DELETE /api/v1/models/{model_id}` - 删除模型

### 检测

- `POST /api/v1/detection/bait` - BAIT检测
- `GET /api/v1/detection/bait/targets` - 获取预定义目标
- `POST /api/v1/cleaning/analyze` - 数据清洗

### 去毒

- `POST /api/v1/unlearning/purify` - 执行去毒
- `GET /api/v1/unlearning/methods` - 获取可用方法

### 安全扫描

- `POST /api/v1/scan/model/{model_id}` - 扫描模型安全

### 任务管理

- `GET /api/v1/tasks` - 列出任务
- `GET /api/v1/tasks/{task_id}` - 获取任务状态
- `DELETE /api/v1/tasks/{task_id}` - 取消任务

## 配置说明

环境变量配置文件 `.env`:

```env
# 应用配置
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000

# 数据库
DATABASE_URL=sqlite:///./data/poisonguard.db

# HuggingFace
HF_TOKEN=your_token_here
HF_CACHE_DIR=./data/models

# 检测阈值
DETECTION_TPR_THRESHOLD=0.92
DETECTION_FPR_THRESHOLD=0.05

# 安全设置
ENABLE_SHA256_VERIFY=true
```

## 参考文献

1. BAIT: Backdoor Inspection for Language Models via Token Optimization. IEEE S&P 2025.
2. W2SDefense: Weak-to-Strong Unlearning Defense. ACL 2025.
3. Backdoor Learning: A Survey. IEEE TNNLS 2024.

## 许可证

MIT License
