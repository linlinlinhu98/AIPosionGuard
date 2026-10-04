<template>
  <div class="home-page">
    <!-- 统计卡片 -->
    <el-row :gutter="20" class="stats-row">
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon" style="background-color: #409eff;">
            <el-icon><Box /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.models }}</div>
            <div class="stat-label">模型数量</div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon" style="background-color: #67c23a;">
            <el-icon><Document /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.datasets }}</div>
            <div class="stat-label">数据集数量</div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon" style="background-color: #e6a23c;">
            <el-icon><List /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.tasks }}</div>
            <div class="stat-label">任务总数</div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon" style="background-color: #f56c6c;">
            <el-icon><Warning /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.threats }}</div>
            <div class="stat-label">检测威胁</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 快捷操作 -->
    <el-row :gutter="20" class="quick-actions">
      <el-col :span="16">
        <el-card header="快捷操作">
          <el-row :gutter="20">
            <el-col :span="6">
              <el-button type="primary" size="large" @click="$router.push('/models/upload')">
                <el-icon><Upload /></el-icon>
                上传模型
              </el-button>
            </el-col>
            <el-col :span="6">
              <el-button type="success" size="large" @click="$router.push('/detection/bait')">
                <el-icon><Search /></el-icon>
                开始检测
              </el-button>
            </el-col>
            <el-col :span="6">
              <el-button type="warning" size="large" @click="$router.push('/detection/cleaning')">
                <el-icon><Brush /></el-icon>
                数据清洗
              </el-button>
            </el-col>
            <el-col :span="6">
              <el-button type="danger" size="large" @click="$router.push('/unlearning')">
                <el-icon><Refresh /></el-icon>
                模型去毒
              </el-button>
            </el-col>
          </el-row>
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card header="系统状态">
          <div class="status-list">
            <div class="status-item">
              <span>API服务</span>
              <el-tag :type="systemStatus.api === 'healthy' ? 'success' : 'danger'" size="small">
                {{ systemStatus.api }}
              </el-tag>
            </div>
            <div class="status-item">
              <span>GPU状态</span>
              <el-tag :type="systemStatus.gpu ? 'success' : 'info'" size="small">
                {{ systemStatus.gpu ? '可用' : '不可用' }}
              </el-tag>
            </div>
            <div class="status-item">
              <span>活跃任务</span>
              <el-tag type="warning" size="small">{{ systemStatus.activeTasks }}</el-tag>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 最近任务 -->
    <el-card class="recent-tasks" header="最近任务">
      <el-table :data="recentTasks" style="width: 100%">
        <el-table-column prop="task_id" label="任务ID" width="280">
          <template #default="{ row }">
            <el-link type="primary" @click="viewTask(row.task_id)">
              {{ row.task_id }}
            </el-link>
          </template>
        </el-table-column>
        <el-table-column prop="type" label="类型" width="120" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" size="small">
              {{ row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="progress" label="进度" width="150">
          <template #default="{ row }">
            <el-progress
              :percentage="row.progress"
              :status="row.status === 'completed' ? 'success' : ''"
            />
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="180">
          <template #default="{ row }">
            {{ formatDate(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button
              type="primary"
              link
              @click="viewTask(row.task_id)"
            >
              查看详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 功能说明 -->
    <el-card class="features" header="平台功能">
      <el-row :gutter="20">
        <el-col :span="6" v-for="feature in features" :key="feature.title">
          <div class="feature-item">
            <el-icon :size="32" :color="feature.color">
              <component :is="feature.icon" />
            </el-icon>
            <h4>{{ feature.title }}</h4>
            <p>{{ feature.description }}</p>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/api'
import dayjs from 'dayjs'

const router = useRouter()

// 统计数据
const stats = ref({
  models: 0,
  datasets: 0,
  tasks: 0,
  threats: 0
})

// 系统状态
const systemStatus = ref({
  api: '检测中...',
  gpu: false,
  activeTasks: 0
})

// 最近任务
const recentTasks = ref([])

// 功能列表
const features = ref([
  {
    icon: 'Search',
    title: 'BAIT后门检测',
    description: '基于IEEE S&P 2025论文的后门触发器逆向工程检测',
    color: '#409eff'
  },
  {
    icon: 'Brush',
    title: '数据清洗引擎',
    description: '多维度统计特征异常检测，识别可疑训练样本',
    color: '#67c23a'
  },
  {
    icon: 'Refresh',
    title: '模型去毒',
    description: 'W2SDefense等Unlearning方法净化被感染模型',
    color: '#e6a23c'
  },
  {
    icon: 'Shield',
    title: '安全扫描',
    description: 'SHA256验证、Pickle安全扫描、可信源检测',
    color: '#f56c6c'
  }
])

// 获取状态类型
function getStatusType(status) {
  const types = {
    'pending': 'info',
    'running': 'warning',
    'completed': 'success',
    'failed': 'danger'
  }
  return types[status] || 'info'
}

// 格式化日期
function formatDate(date) {
  return dayjs(date).format('YYYY-MM-DD HH:mm:ss')
}

// 查看任务详情
function viewTask(taskId) {
  router.push(`/tasks?id=${taskId}`)
}

// 加载数据
async function loadData() {
  try {
    // 获取健康状态
    const health = await api.getHealth()
    systemStatus.value = {
      api: health.status,
      gpu: health.gpu_available,
      activeTasks: health.active_tasks
    }

    // 获取模型列表
    const models = await api.getModels()
    stats.value.models = models.count

    // 获取数据集列表
    const datasets = await api.getDatasets()
    stats.value.datasets = datasets.count

    // 获取任务列表
    const tasks = await api.getTasks({ limit: 5 })
    recentTasks.value = tasks.tasks
    stats.value.tasks = tasks.count

    // 模拟威胁数量
    stats.value.threats = tasks.tasks.filter(
      t => t.result === 'backdoor' || t.result === 'suspicious'
    ).length

  } catch (error) {
    console.error('Failed to load data:', error)
  }
}

onMounted(() => {
  loadData()
  // 定期刷新
  setInterval(loadData, 30000)
})
</script>

<style scoped>
.home-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.stats-row {
  margin-bottom: 0;
}

.stat-card {
  display: flex;
  align-items: center;
  padding: 20px;
}

.stat-card :deep(.el-card__body) {
  display: flex;
  align-items: center;
  width: 100%;
  padding: 0;
}

.stat-icon {
  width: 60px;
  height: 60px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 15px;
}

.stat-icon .el-icon {
  font-size: 28px;
  color: white;
}

.stat-info {
  flex: 1;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 5px;
}

.quick-actions {
  margin-bottom: 0;
}

.quick-actions .el-button {
  width: 100%;
  height: 50px;
}

.status-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.recent-tasks {
  margin-bottom: 0;
}

.features .feature-item {
  text-align: center;
  padding: 20px;
}

.features h4 {
  margin: 15px 0 10px;
  color: #303133;
}

.features p {
  font-size: 13px;
  color: #909399;
  line-height: 1.5;
}
</style>
