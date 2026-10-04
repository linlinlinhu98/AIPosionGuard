FROM python:3.12-slim

WORKDIR /app

# 安装 Node.js (构建前端用)
RUN apt-get update && apt-get install -y nodejs npm curl --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# 后端依赖
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 前端依赖 + 构建
COPY frontend/package.json frontend/package-lock.json* ./frontend/
WORKDIR /app/frontend
RUN npm install
COPY frontend/ .
RUN npm run build

# 复制后端代码 + 前端构建产物
WORKDIR /app
COPY backend/ ./backend/
RUN cp -r frontend/dist backend/static

EXPOSE 8000

CMD ["uvicorn", "backend.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
