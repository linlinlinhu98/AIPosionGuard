<template>
  <div class="tasks-page">
    <el-card>
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span>任务列表</span>
          <el-button type="primary" @click="loadTasks">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </template>

      <!-- 筛选 -->
      <el-form :inline="true" style="margin-bottom: 20px;">
        <el-form-item label="状态">
          <el-select v-model="filterStatus" placeholder="全部状态" clearable @change="loadTasks">
            <el-option label="等待中" value="pending" />
            <el-option label="运行中" value="running" />
            <el-option label="已完成" value="completed" />
            <el-option label="失败" value="failed" />
          </el-select>
        </el-form-item>
      </el-form>

      <!-- 任务表格 -->
      <el-table :data="tasks" v-loading="loading" border>
        <el-table-column prop="task_id" label="任务ID" width="320">
          <template #default="{ row }">
            <el-text truncated>{{ row.task_id }}</el-text>
          </template>
        </el-table-column>
        <el-table-column prop="type" label="类型" width="120">
          <template #default="{ row }">
            <el-tag size="small">{{ row.type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" size="small">
              {{ row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="progress" label="进度" width="180">
          <template #default="{ row }">
            <el-progress
              :percentage="row.progress"
              :status="row.status === 'completed' ? 'success' : row.status === 'failed' ? 'exception' : ''"
            />
          </template>
        </el-table-column>
        <el-table-column prop="result" label="结果" width="100">
          <template #default="{ row }">
            <el-tag
              v-if="row.result"
              :type="getResultType(row.result)"
              size="small"
            >
              {{ row.result }}
            </el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="100">
          <template #default="{ row }">
            <span v-if="row.confidence">{{ (row.confidence * 100).toFixed(1) }}%</span>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="180">
          <template #default="{ row }">
            {{ formatDate(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="viewDetail(row)">
              详情
            </el-button>
            <el-button
              v-if="row.status === 'running'"
              type="danger"
              link
              @click="cancelTask(row.task_id)"
            >
              取消
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <el-pagination
        style="margin-top: 20px; justify-content: flex-end;"
        :total="total"
        :page-size="20"
        layout="total, prev, pager, next"
        @current-change="loadTasks"
      />
    </el-card>

    <!-- 详情弹窗 -->
    <el-dialog v-model="detailVisible" title="任务详情" width="600px">
      <el-descriptions :column="1" border>
        <el-descriptions-item label="任务ID">{{ currentTask?.task_id }}</el-descriptions-item>
        <el-descriptions-item label="类型">{{ currentTask?.type }}</el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="getStatusType(currentTask?.status)">
            {{ currentTask?.status }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="进度">
          <el-progress :percentage="currentTask?.progress || 0" />
        </el-descriptions-item>
        <el-descriptions-item label="结果">{{ currentTask?.result || '-' }}</el-descriptions-item>
        <el-descriptions-item label="置信度">
          {{ currentTask?.confidence ? (currentTask.confidence * 100).toFixed(1) + '%' : '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ formatDate(currentTask?.created_at) }}</el-descriptions-item>
        <el-descriptions-item label="完成时间">
          {{ currentTask?.completed_at ? formatDate(currentTask.completed_at) : '-' }}
        </el-descriptions-item>
      </el-descriptions>

      <div v-if="currentTask?.details" style="margin-top: 20px;">
        <h4>详细信息</h4>
        <el-input
          type="textarea"
          :model-value="JSON.stringify(currentTask.details, null, 2)"
          :rows="10"
          readonly
        />
      </div>

      <div v-if="currentTask?.error_message" style="margin-top: 20px;">
        <el-alert type="error" :title="currentTask.error_message" :closable="false" />
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '@/api'
import dayjs from 'dayjs'

const loading = ref(false)
const tasks = ref([])
const total = ref(0)
const filterStatus = ref('')
const detailVisible = ref(false)
const currentTask = ref(null)

function getStatusType(status) {
  const types = {
    'pending': 'info',
    'running': 'warning',
    'completed': 'success',
    'failed': 'danger',
    'cancelled': 'info'
  }
  return types[status] || 'info'
}

function getResultType(result) {
  const types = {
    'clean': 'success',
    'backdoor': 'danger',
    'suspicious': 'warning',
    'success': 'success',
    'partial': 'warning'
  }
  return types[result] || 'info'
}

function formatDate(date) {
  if (!date) return '-'
  return dayjs(date).format('YYYY-MM-DD HH:mm:ss')
}

async function loadTasks() {
  loading.value = true
  try {
    const response = await api.getTasks({ status: filterStatus.value || undefined })
    tasks.value = response.tasks
    total.value = response.count
  } catch (error) {
    console.error('Failed to load tasks:', error)
  } finally {
    loading.value = false
  }
}

function viewDetail(task) {
  currentTask.value = task
  detailVisible.value = true
}

async function cancelTask(taskId) {
  try {
    await api.cancelTask(taskId)
    ElMessage.success('任务已取消')
    loadTasks()
  } catch (error) {
    console.error('Failed to cancel task:', error)
  }
}

onMounted(() => {
  loadTasks()
  // 定期刷新
  setInterval(loadTasks, 10000)
})
</script>

<style scoped>
.tasks-page {
  display: flex;
  flex-direction: column;
}
</style>
