# AI-PoisonGuard V2.0 运行与测试指南

## 目录结构

```
Demo/
├── backend/
│   ├── app/
│   │   ├── api/main.py                  # [存量扩展] FastAPI 应用 + V2.0 端点
│   │   ├── core/
│   │   │   ├── config.py                # [存量扩展] 配置 (新增 M1/M3/M4/M5 项)
│   │   │   └── merge_safety.py          # [新增 M3] 合并安全评估器
│   │   ├── models/schemas.py            # [存量扩展] 数据模型 (新增 V2.0 schema)
│   │   └── services/
│   │       ├── bait_detector.py         # [存量 M2] BAIT 后门检测
│   │       ├── data_cleaning.py         # [存量增强 M6] 数据清洗引擎
│   │       ├── huggingface_integration.py # [存量 M8] HF 集成
│   │       ├── lora_weight_detector.py  # [新增 M1] LoRA 权重空间检测
│   │       ├── memory_poison_detector.py # [新增 M4] Agent 记忆投毒检测
│   │       ├── threat_intelligence.py   # [新增 M5] 威胁情报引擎
│   │       └── unlearning.py            # [存量 M7] 修复引擎
│   ├── tests/
│   │   ├── test_all.py                  # [存量] 原有基础测试
│   │   └── test_new_modules.py          # [新增] V2.0 完整测试套件
│   └── requirements.txt                 # [存量扩展] 依赖 (新增 sentence-transformers, joblib)
├── demos/
│   ├── full_demo.py                     # [新增] 一键全流程演示
│   └── create_lora_benchmark.py         # [新增] LoRA 基准数据集生成器
├── start.sh / start.bat                 # [存量] 启动脚本
└── RUNME.md                             # 本文件
```

---

## 一、环境准备

### 1.1 Python 环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 安装依赖
cd Demo/backend
pip install -r requirements.txt
```

### 1.2 最小依赖（仅测试核心模块，无需 GPU）

```bash
pip install torch numpy safetensors scikit-learn joblib loguru pydantic pydantic-settings
```

### 1.3 完整依赖（包括嵌入模型）

```bash
# 额外安装
pip install sentence-transformers transformers fastapi uvicorn
```

---

## 二、运行测试

### 2.1 运行原有测试（V1.0）

```bash
cd Demo/backend
python -m pytest tests/test_all.py -v
```

### 2.2 运行 V2.0 新增模块测试

```bash
cd Demo/backend
python -m pytest tests/test_new_modules.py -v
```

预期输出：
```
TestM1LoRADetector::test_initialization_untrained PASSED
TestM1LoRADetector::test_extract_features_from_random_weights PASSED
TestM1LoRADetector::test_feature_vector_dimensions PASSED
TestM1LoRADetector::test_load_safetensors_adapter PASSED
TestM1LoRADetector::test_heuristic_detect_without_training PASSED
TestM1LoRADetector::test_detect_empty_adapter PASSED
TestM3MergeSafety::test_two_clean_adapters_low_risk PASSED
TestM3MergeSafety::test_clean_plus_backdoor_high_risk PASSED
TestM4MemoryDetector::test_normal_write_no_alert PASSED
TestM4MemoryDetector::test_anomalous_write_triggers_alert PASSED
TestM5ThreatIntelligence::test_default_incidents_loaded PASSED
...
```

### 2.3 按模块筛选测试

```bash
# 仅测试 M1
pytest tests/test_new_modules.py -v -k "M1"

# 仅测试 M4
pytest tests/test_new_modules.py -v -k "M4"

# 仅测试 M5
pytest tests/test_new_modules.py -v -k "M5"

# 仅运行集成测试
pytest tests/test_new_modules.py -v -k "Integration"
```

### 2.4 完整的测试套件

```bash
pytest tests/ -v --tb=short
```

---

## 三、运行全流程演示

### 3.1 一键演示（无需 GPU）

```bash
cd Demo
python demos/full_demo.py
```

演示将依次展示：
1. **威胁情报预警**：显示 4 个真实攻击事件及 IOC 指标
2. **LoRA 权重检测**：创建演示适配器并运行 M1 检测（< 0.01 秒）
3. **合并安全评估**：对适配器对进行涌现风险预测
4. **Agent 记忆监控**：建立安全基线 → 注入恶意记忆 → 模拟敏感查询

### 3.2 跳过记忆检测

```bash
python demos/full_demo.py --skip-memory
```

### 3.3 导出结果

```bash
python demos/full_demo.py --output ./demo_results.json
```

---

## 四、生成 LoRA 基准数据集

```bash
cd Demo
python demos/create_lora_benchmark.py --num-clean 50 --num-poisoned 50
```

参数：
- `--num-clean`：干净适配器数量（默认 50）
- `--num-poisoned`：后门适配器数量（默认 50，均匀分布在 4 种攻击类型）
- `--output-dir`：输出目录（默认 `./data/lora_benchmark/`）
- `--seed`：随机种子（默认 42）

输出结构：
```
data/lora_benchmark/
├── clean/
│   ├── gpt2_sst2_clean_000/adapter_model.safetensors + adapter_config.json
│   ├── ...
│   └── gpt2_sst2_clean_049/
├── poisoned/
│   ├── badnet_cf_000/     (BadNets 攻击)
│   ├── composite_multi_000/ (复合触发器)
│   ├── clean_label_000/    (Clean-label)
│   ├── semantic_trigger_000/ (语义触发器)
│   └── ...
└── benchmark_metadata.json
```

### 训练 M1 分类器

```bash
cd Demo/backend
python -c "
from app.services.lora_weight_detector import LoRAWeightSpaceDetector
from pathlib import Path
import glob

detector = LoRAWeightSpaceDetector()

# 收集干净和后门适配器
clean = [str(p) for p in Path('../data/lora_benchmark/clean').iterdir() if p.is_dir()]
poisoned = [str(p) for p in Path('../data/lora_benchmark/poisoned').iterdir() if p.is_dir()]

print(f'Training on {len(clean)} clean + {len(poisoned)} poisoned adapters')
metrics = detector.train(clean, poisoned)
print(f'Results: {metrics}')
"
```

---

## 五、启动 API 服务

### 5.1 启动后端

```bash
cd Demo/backend

# 开发模式
python -m app.api.main
# 或
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5.2 API 端点速查

| 端点 | 方法 | 模块 | 说明 |
|------|------|------|------|
| `/health` | GET | - | 健康检查 |
| `/api/v2/detection/lora-weight` | POST | M1 | LoRA 权重空检检测 |
| `/api/v2/detection/lora-train` | POST | M1 | 训练权重分类器 |
| `/api/v2/assessment/merge-safety` | POST | M3 | 合并安全评估 |
| `/api/v2/monitoring/memory-write` | POST | M4 | Agent 记忆写入上报 |
| `/api/v2/monitoring/memory-retrieve` | POST | M4 | 记忆检索检查 |
| `/api/v2/monitoring/memory-health` | GET | M4 | 记忆健康报告 |
| `/api/v2/threat-intel/incidents` | GET | M5 | 威胁事件列表 |
| `/api/v2/threat-intel/check` | POST | M5 | 威胁情报匹配 |
| `/api/v2/detection/full-pipeline` | POST | 融合 | 端到端融合检测 |

### 5.3 调用示例

```bash
# 权重空间检测
curl -X POST http://localhost:8000/api/v2/detection/lora-weight \
  -F "adapter_path=./data/demo_adapters/demo_backdoor_adapter"

# 合并安全评估
curl -X POST http://localhost:8000/api/v2/assessment/merge-safety \
  -F "adapter_paths=./data/demo_adapters/demo_clean_adapter" \
  -F "adapter_paths=./data/demo_adapters/demo_backdoor_adapter" \
  -F "base_model=gpt2"

# Agent 记忆写入上报
curl -X POST http://localhost:8000/api/v2/monitoring/memory-write \
  -F "content=用户查询了订单状态" \
  -F "source=user_handler" \
  -F "session_id=1"

# 威胁情报查询
curl "http://localhost:8000/api/v2/threat-intel/incidents?limit=5"

# 威胁情报匹配
curl -X POST http://localhost:8000/api/v2/threat-intel/check \
  -F "model_id=Open-OSS/suspicious-model"
```

### 5.4 Swagger 文档

启动后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 六、模块开发验证清单

### M1 LoRA 权重检测器

- [ ] `test_initialization` — 检测器正确初始化
- [ ] `test_extract_features` — 从权重矩阵提取 SVD/范数/熵特征
- [ ] `test_heuristic_detect` — 未训练时启发式规则工作
- [ ] `test_detect_on_benchmark` — 在生成的基准数据上验证 TPR/FPR
- [ ] `test_train_and_save` — 训练后 .pkl 文件可正确保存和加载

### M3 合并安全评估器

- [ ] `test_clean_pair` — 两个干净适配器 → 低风险
- [ ] `test_clean_backdoor_pair` — 干净+后门 → 警告
- [ ] `test_complementary_layers` — 异常层不重叠 → 涌现加分
- [ ] `test_no_m1_detector` — M1 不可用时降级处理

### M4 Agent 记忆检测器

- [ ] `test_baseline` — 安全基线正确建立
- [ ] `test_normal_write` — 正常记忆不触发告警
- [ ] `test_anomalous_write` — 恶意记忆触发告警
- [ ] `test_trojan_hippo_scenario` — 模拟恶意邮件注入 → 告警
- [ ] `test_sensitive_query` — 敏感查询触发告警升级
- [ ] `test_memory_health` — 健康报告正确生成

### M5 威胁情报引擎

- [ ] `test_incidents_loaded` — 4 个默认事件正确加载
- [ ] `test_source_match` — 组织名/仓库名模式匹配
- [ ] `test_sha256_match` — SHA256 哈希匹配
- [ ] `test_ioc_pattern` — 文本中 IOC 模式检测
- [ ] `test_add_incident` — 新事件可以添加

---

## 七、常见问题

### Q1: `ModuleNotFoundError: No module named 'sentence_transformers'`
```bash
pip install sentence-transformers
```
此模块用于 M4 记忆检测和 M6 语义一致性分析。如果没有安装，系统会自动降级到基于 TF-IDF 的检测和关键词匹配。

### Q2: `ModuleNotFoundError: No module named 'joblib'`
```bash
pip install joblib
```

### Q3: M4 检测不触发告警
如果使用 TF-IDF 降级模式（未安装 sentence-transformers），恶意记忆的检测灵敏度会降低。建议安装 sentence-transformers 获得最佳效果。

### Q4: 模型不可用时如何测试
所有新增模块（M1/M3/M4/M5）都设计为**不需要真实 LLM 模型**即可工作：
- M1: 只需要 LoRA 权重文件 (.safetensors)
- M3: 依赖 M1 的结果
- M4: 只需要 sentence-transformers (可选)
- M5: 完全独立，内置默认数据

---

## 八、开发排期

| 阶段 | 内容 | 产出 |
|------|------|------|
| 第1周 | 环境搭建 + LoRA 基准生成 | 100 个适配器 |
| 第2周 | M1 开发 + 训练 | 权重检测模块 + .pkl 模型 |
| 第3周 | M3 + M5 开发 | 合并安全 + 情报引擎 |
| 第4周 | M4 开发 | 记忆监控模块 |
| 第5周 | API 扩展 + 前端 | 完整界面 |
| 第6周 | 测试 + 实验 | 4 个核心实验数据 |
| 第7周 | Demo 完善 | 可演示版本 |
| 第8周 | 文档 + 答辩准备 | 完整文档 |
