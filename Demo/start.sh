#!/bin/bash

# AI-PoisonGuard 启动脚本

echo "================================"
echo "  AI-PoisonGuard 启动脚本"
echo "================================"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.10+"
    exit 1
fi

# 创建必要的目录
mkdir -p logs
mkdir -p data/models
mkdir -p data/datasets
mkdir -p data/reports
mkdir -p output

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 安装后端依赖
echo ""
echo "[1/3] 安装后端依赖..."
cd backend
pip install -r requirements.txt -q
cd ..

# 安装前端依赖
echo ""
echo "[2/3] 安装前端依赖..."
cd frontend
if [ -f "package-lock.json" ] || [ -f "yarn.lock" ]; then
    npm install --legacy-peer-deps 2>/dev/null || yarn install
else
    npm install --legacy-peer-deps
fi
cd ..

# 创建环境变量文件（如果不存在）
if [ ! -f ".env" ]; then
    echo ""
    echo "[3/3] 创建配置文件..."
    cat > .env << 'EOF'
# AI-PoisonGuard 配置文件

# 应用配置
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000

# 数据库
DATABASE_URL=sqlite:///./data/poisonguard.db

# Redis (可选)
REDIS_URL=redis://localhost:6379/0

# HuggingFace
HF_TOKEN=
HF_CACHE_DIR=./data/models

# 检测阈值
DETECTION_TPR_THRESHOLD=0.92
DETECTION_FPR_THRESHOLD=0.05
DETECTION_CONFIDENCE_THRESHOLD=0.85

# 安全设置
ENABLE_SHA256_VERIFY=true
EOF
    echo "已创建 .env 配置文件"
fi

echo ""
echo "================================"
echo "  安装完成！"
echo "================================"
echo ""
echo "启动方式:"
echo ""
echo "  后端 (FastAPI):"
echo "    cd backend && python -m app.api.main"
echo ""
echo "  前端 (Vue.js):"
echo "    cd frontend && npm run dev"
echo ""
echo "  或者使用开发模式:"
echo "    cd backend && uvicorn app.api.main:app --reload --port 8000"
echo ""
echo "访问地址:"
echo "  前端: http://localhost:5173"
echo "  API文档: http://localhost:8000/docs"
echo ""
