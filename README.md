# AI-PoisonGuard

面向大语言模型（LLM）微调供应链的**投毒检测与防御平台**。

针对 HuggingFace 等 LoRA 适配器分享场景，提供从权重空间到行为空间的四层纵深防御，可在 CPU（无 GPU）上运行全部检测。

## 核心能力

- **后门防火墙（V3）**：L1 权重初筛 → L2 BAIT 触发器逆向 + 多 token 束搜索 → L3 家族级探针差分 → 校准融合决策（allow/review/block）+ 隔离区人工处置 + 金丝雀自检
- **检测中心（V2）**：BAIT 检测、数据清洗、安全扫描、M1 权重空间检测器
- **模型/数据集管理**：上传、注册、任务队列、检测指标看板
- **浏览器插件**：HuggingFace 页面侧的风险提示

## 快速启动

```bash
# 1. 后端（端口 8000）
cd Demo/backend
../venv/Scripts/python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# 2. 前端（端口 5173），新开一个终端
cd Demo/frontend
npm run dev
```

浏览器打开 **http://localhost:5173**（后门防火墙页面：`/firewall`）。

> 完整安装步骤见 [Demo/README.md](Demo/README.md)

## 目录结构

```
Demo/            平台主体（FastAPI 后端 + Vue3 前端 + 浏览器插件）
experiments/     实验脚本与可复现结果（训练、评测、金丝雀基准）
package/         轻量交付包
data/            数据集（SST-2 等）
```

## 测试

```bash
cd Demo/backend
../venv/Scripts/python.exe -m pytest tests/ -q          # 单元测试（75 项）
../venv/Scripts/python.exe -m pytest tests/test_firewall_e2e.py -m slow -v   # 端到端验收
```
