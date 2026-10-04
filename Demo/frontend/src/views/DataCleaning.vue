<template>
  <div class="data-cleaning-page">
    <el-card header="数据清洗">
      <el-alert
        type="info"
        title="数据清洗引擎"
        description="多维度统计特征异常检测，识别可疑训练样本，支持BadNet和Clean-label攻击检测"
        :closable="false"
        show-icon
        style="margin-bottom: 20px;"
      />

      <el-form :model="form" label-width="140px" :rules="rules" ref="formRef">
        <el-form-item label="选择数据集" prop="dataset_id">
          <el-select v-model="form.dataset_id" placeholder="请选择数据集" style="width: 100%;">
            <el-option
              v-for="dataset in datasets"
              :key="dataset.dataset_id"
              :label="dataset.dataset_name"
              :value="dataset.dataset_id"
            >
              <span>{{ dataset.dataset_name }}</span>
              <el-tag size="small" style="margin-left: 10px;">{{ dataset.sample_count }} 样本</el-tag>
            </el-option>
          </el-select>
        </el-form-item>

        <el-form-item label="异常阈值">
          <el-slider v-model="form.anomaly_threshold" :min="0.5" :max="0.95" :step="0.05" show-input />
          <el-text type="info" size="small">阈值越高，检测越严格，但误报率可能增加</el-text>
        </el-form-item>

        <el-form-item label="清洗方法">
          <el-checkbox-group v-model="form.cleaning_methods">
            <el-checkbox value="statistical">统计特征分析</el-checkbox>
            <el-checkbox value="semantic">语义一致性检测</el-checkbox>
            <el-checkbox value="distribution">分布偏移检测</el-checkbox>
          </el-checkbox-group>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="startCleaning" :loading="loading">
            <el-icon><Brush /></el-icon>
            开始清洗
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 清洗结果 -->
    <el-card v-if="result" header="清洗结果" style="margin-top: 20px;">
      <el-alert
        v-if="result.total_samples == null"
        type="info" :closable="false" style="margin-bottom: 15px;"
      >
        此为历史清洗记录，可下载清洗后的数据文件
      </el-alert>
      <el-row :gutter="20">
        <el-col :span="6">
          <el-statistic title="总样本数" :value="result.total_samples" />
        </el-col>
        <el-col :span="6">
          <el-statistic title="干净样本" :value="result.clean_samples">
            <template #suffix>
              <el-tag type="success" size="small">安全</el-tag>
            </template>
          </el-statistic>
        </el-col>
        <el-col :span="6">
          <el-statistic title="投毒样本" :value="result.poisoned_samples">
            <template #suffix>
              <el-tag type="danger" size="small">恶意</el-tag>
            </template>
          </el-statistic>
        </el-col>
        <el-col :span="6">
          <el-statistic title="可疑样本" :value="result.suspicious_samples">
            <template #suffix>
              <el-tag type="warning" size="small">待确认</el-tag>
            </template>
          </el-statistic>
        </el-col>
      </el-row>

      <el-divider />

      <el-descriptions title="检测指标" :column="3" border>
        <el-descriptions-item label="干净比例">
          {{ result.clean_ratio != null ? (result.clean_ratio * 100).toFixed(1) + '%' : '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="投毒比例">
          {{ result.poisoned_ratio != null ? (result.poisoned_ratio * 100).toFixed(1) + '%' : '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="异常阈值">
          {{ result.anomaly_threshold }}
        </el-descriptions-item>
      </el-descriptions>

      <div style="margin-top: 20px;">
        <el-button type="success" @click="downloadCleanData('clean')">
          <el-icon><Download /></el-icon>
          下载干净数据
        </el-button>
        <el-button type="warning" @click="downloadCleanData('suspicious')">
          <el-icon><Download /></el-icon>
          下载可疑数据
        </el-button>
        <el-button type="info" @click="downloadReport">
          <el-icon><Document /></el-icon>
          下载报告
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '@/api'

const form = ref({
  dataset_id: '',
  anomaly_threshold: 0.7,
  cleaning_methods: ['statistical', 'semantic', 'distribution']
})

const rules = {
  dataset_id: [{ required: true, message: '请选择数据集', trigger: 'change' }]
}

const formRef = ref(null)
const loading = ref(false)
const result = ref(null)
const datasets = ref([])

async function loadDatasets() {
  try {
    const response = await api.getDatasets()
    datasets.value = response.datasets
  } catch (error) {
    console.error('Failed to load datasets:', error)
  }
}

async function restoreLastCleaningResult() {
  // 页面切换后恢复最近一次清洗结果和下载能力
  try {
    const tasksRes = await api.getTasks({ status: 'completed', limit: 20 })
    const cleaningTasks = (tasksRes.tasks || []).filter(t => t.type === 'data_cleaning')
    if (cleaningTasks.length > 0) {
      const lastTask = cleaningTasks[0]
      currentTaskId.value = lastTask.task_id
      result.value = lastTask.details || {}
    }
  } catch (error) {
    console.error('Failed to restore cleaning result:', error)
  }
}

async function startCleaning() {
  await formRef.value.validate()

  loading.value = true
  result.value = null

  try {
    const response = await api.analyzeDataset(form.value)
    currentTaskId.value = response.task_id
    pollTaskStatus(response.task_id)
    ElMessage.success('清洗任务已创建')
  } catch (error) {
    console.error('Failed to start cleaning:', error)
  } finally {
    loading.value = false
  }
}

async function pollTaskStatus(taskId) {
  const poll = async () => {
    try {
      const task = await api.getTask(taskId)
      if (task.status === 'completed') {
        result.value = task.details
        ElMessage.success('清洗完成')
      } else if (task.status === 'failed') {
        ElMessage.error(task.error_message || '清洗失败')
      } else {
        setTimeout(poll, 2000)
      }
    } catch (error) {
      console.error('Failed to poll task:', error)
    }
  }
  poll()
}

const currentTaskId = ref(null)

function downloadCleanData(type = 'clean') {
  if (!currentTaskId.value) {
    ElMessage.warning('没有可下载的清洗结果')
    return
  }
  window.open(`/api/v1/cleaning/download/${currentTaskId.value}/clean-data?type=${type}`, '_blank')
  ElMessage.success(`开始下载${type === 'clean' ? '干净' : '可疑'}数据`)
}

function downloadReport() {
  if (!currentTaskId.value) {
    ElMessage.warning('没有可下载的清洗报告')
    return
  }
  window.open(`/api/v1/cleaning/download/${currentTaskId.value}/report`, '_blank')
  ElMessage.success('开始下载报告')
}

onMounted(() => {
  loadDatasets()
  restoreLastCleaningResult()
})
</script>

<style scoped>
.data-cleaning-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
</style>
