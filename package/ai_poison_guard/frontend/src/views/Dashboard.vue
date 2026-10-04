<template>
  <div class="dashboard">
    <!-- 顶部标题栏 -->
    <header class="dashboard-header">
      <div class="logo">
        <h1>AI-PoisonGuard</h1>
        <span class="subtitle">LLM微调供应链投毒检测与主动防御平台</span>
      </div>
      <div class="header-actions">
        <el-button @click="showSettings = true">
          <SettingOutlined /> 设置
        </el-button>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="dashboard-content">
      <!-- 上传区域 -->
      <el-row :gutter="20" class="upload-section">
        <!-- 模型上传 -->
        <el-col :span="12">
          <el-card class="upload-card">
            <template #header>
              <div class="card-header">
                <ModelOutlined class="card-icon" />
                <span>模型资源</span>
              </div>
            </template>

            <el-form :model="modelForm" label-position="top">
              <el-form-item label="模型来源">
                <el-radio-group v-model="modelForm.source">
                  <el-radio label="huggingface">HuggingFace 仓库</el-radio>
                  <el-radio label="local">本地文件</el-radio>
                </el-radio-group>
              </el-form-item>

              <el-form-item label="模型路径 / ID">
                <el-input
                  v-model="modelForm.path"
                  placeholder="例如: gpt2, facebook/opt-1.3b, 或本地路径"
                  clearable
                />
              </el-form-item>

              <el-form-item label="任务类型">
                <el-select v-model="modelForm.taskType" placeholder="选择任务类型">
                  <el-option label="通用" value="general" />
                  <el-option label="情感分类" value="sentiment" />
                  <el-option label="问答" value="qa" />
                  <el-option label="摘要" value="summarization" />
                </el-select>
              </el-form-item>
            </el-form>

            <el-button type="primary" plain @click="uploadModel">
              <UploadOutlined /> 上传模型
            </el-button>
          </el-card>
        </el-col>

        <!-- 数据集上传 -->
        <el-col :span="12">
          <el-card class="upload-card">
            <template #header>
              <div class="card-header">
                <DatabaseOutlined class="card-icon" />
                <span>训练数据集</span>
              </div>
            </template>

            <el-form :model="datasetForm" label-position="top">
              <el-form-item label="数据集格式">
                <el-radio-group v-model="datasetForm.format">
                  <el-radio label="jsonl">JSONL</el-radio>
                  <el-radio label="csv">CSV</el-radio>
                </el-radio-group>
              </el-form-item>

              <el-form-item label="上传数据集文件">
                <el-upload
                  drag
                  action="/api/v1/upload/dataset"
                  :headers="uploadHeaders"
                  :on-success="handleDatasetUpload"
                  :on-error="handleUploadError"
                  accept=".jsonl,.csv"
                >
                  <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                  <div class="el-upload__text">
                    拖拽文件到此处或 <em>点击上传</em>
                  </div>
                </el-upload>
              </el-form-item>
            </el-form>

            <div v-if="datasetForm.uploaded" class="upload-success">
              <CheckCircleOutlined /> 数据集已上传: {{ datasetForm.fileName }}
            </div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 检测配置 -->
      <el-card class="config-card">
        <template #header>
          <div class="card-header">
            <ExperimentOutlined class="card-icon" />
            <span>检测配置</span>
          </div>
        </template>

        <el-form :model="configForm" inline>
          <el-form-item label="检测模式">
            <el-select v-model="configForm.mode" placeholder="选择检测模式">
              <el-option label="完整检测 (模型+数据集)" value="mode_a" />
              <el-option label="仅数据集检测" value="mode_b" />
              <el-option label="仅模型检测" value="mode_c" />
            </el-select>
          </el-form-item>

          <el-form-item label="灵敏度">
            <el-select v-model="configForm.sensitivity" placeholder="选择灵敏度">
              <el-option label="高灵敏度 (严检)" value="high" />
              <el-option label="中灵敏度 (默认)" value="medium" />
              <el-option label="低灵敏度 (宽松)" value="low" />
            </el-select>
          </el-form-item>

          <el-form-item label="跳过模型修复">
            <el-switch v-model="configForm.skipRepair" />
          </el-form-item>
        </el-form>

        <div class="detection-actions">
          <el-button
            type="primary"
            size="large"
            :loading="isDetecting"
            :disabled="!canStartDetection"
            @click="startDetection"
          >
            <SearchOutlined /> 开始检测
          </el-button>
        </div>
      </el-card>

      <!-- 任务状态 -->
      <el-card v-if="currentTask" class="task-card">
        <template #header>
          <div class="card-header">
            <LoadingOutlined class="card-icon" />
            <span>任务进度</span>
            <el-tag :type="getStatusType(currentTask.status)" class="status-tag">
              {{ getStatusText(currentTask.status) }}
            </el-tag>
          </div>
        </template>

        <el-progress
          :percentage="Math.round(currentTask.progress * 100)"
          :status="getProgressStatus(currentTask.status)"
          :stroke-width="20"
        />

        <div class="task-info">
          <span class="task-stage">{{ currentTask.current_stage || currentTask.status }}</span>
          <span class="task-message">{{ currentTask.message }}</span>
        </div>

        <!-- 实时日志 -->
        <div v-if="taskLogs.length > 0" class="task-logs">
          <div v-for="(log, index) in taskLogs" :key="index" class="log-entry">
            {{ log }}
          </div>
        </div>
      </el-card>

      <!-- 检测结果 -->
      <el-card v-if="detectionResult" class="result-card">
        <template #header>
          <div class="card-header">
            <SafetyCertificateOutlined class="card-icon" />
            <span>检测报告</span>
            <el-tag :type="getRiskType(detectionResult.summary.overall_risk_level)" size="large">
              {{ detectionResult.summary.overall_risk_level.toUpperCase() }} 风险
            </el-tag>
          </div>
        </template>

        <!-- 摘要信息 -->
        <div class="result-summary">
          <el-row :gutter="20">
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-value">{{ detectionResult.summary.total_samples || detectionResult.summary.suspicious_samples }}</div>
                <div class="stat-label">样本总数</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-value">{{ detectionResult.triggers.length }}</div>
                <div class="stat-label">检测到的触发器</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-value">{{ (detectionResult.summary.suspicious_rate * 100).toFixed(2) }}%</div>
                <div class="stat-label">可疑率</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-value">{{ detectionResult.summary.analysis_time_seconds.toFixed(2) }}s</div>
                <div class="stat-label">分析耗时</div>
              </div>
            </el-col>
          </el-row>
        </div>

        <!-- 触发器列表 -->
        <div v-if="detectionResult.triggers.length > 0" class="triggers-section">
          <h3>检测到的后门触发器</h3>
          <el-table :data="detectionResult.triggers" stripe>
            <el-table-column prop="text" label="触发器文本" min-width="200">
              <template #default="{ row }">
                <code class="trigger-text">{{ row.text }}</code>
              </template>
            </el-table-column>
            <el-table-column prop="type" label="类型" width="100">
              <template #default="{ row }">
                <el-tag :type="getTriggerTypeColor(row.type)">{{ row.type }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="confidence" label="置信度" width="100">
              <template #default="{ row }">
                <el-progress
                  :percentage="Math.round(row.confidence * 100)"
                  :color="getConfidenceColor(row.confidence)"
                />
              </template>
            </el-table-column>
            <el-table-column prop="target_behavior" label="目标行为" min-width="150" />
            <el-table-column prop="success_rate" label="成功率" width="100">
              <template #default="{ row }">
                {{ (row.success_rate * 100).toFixed(1) }}%
              </template>
            </el-table-column>
          </el-table>
        </div>

        <!-- 修复结果 -->
        <div v-if="detectionResult.repair" class="repair-section">
          <h3>模型修复结果</h3>
          <el-row :gutter="20">
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-label">修复前 ASR</div>
                <div class="stat-value">{{ (detectionResult.repair.pre_asr * 100).toFixed(1) }}%</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card success">
                <div class="stat-label">修复后 ASR</div>
                <div class="stat-value">{{ (detectionResult.repair.post_asr * 100).toFixed(1) }}%</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-label">ASR 下降</div>
                <div class="stat-value">-{{ (detectionResult.repair.asr_reduction * 100).toFixed(1) }}%</div>
              </div>
            </el-col>
            <el-col :span="6">
              <div class="stat-card">
                <div class="stat-label">修复置信度</div>
                <div class="stat-value">{{ (detectionResult.repair.repair_confidence * 100).toFixed(1) }}%</div>
              </div>
            </el-col>
          </el-row>
        </div>

        <!-- 信任信息 -->
        <div class="trust-section">
          <h3>信任与验证</h3>
          <div class="trust-info">
            <div class="trust-item">
              <span class="trust-label">模型 SHA256 哈希:</span>
              <code class="trust-value">{{ detectionResult.trust.model_hash }}</code>
            </div>
            <div class="trust-item">
              <span class="trust-label">报告哈希:</span>
              <code class="trust-value">{{ detectionResult.trust.report_hash }}</code>
            </div>
          </div>
        </div>

        <!-- 建议 -->
        <div class="recommendations-section">
          <h3>安全建议</h3>
          <ul class="recommendations-list">
            <li v-for="(rec, index) in detectionResult.recommendations" :key="index">
              {{ rec }}
            </li>
          </ul>
        </div>

        <!-- 下载链接 -->
        <div class="download-section">
          <el-button @click="downloadReport('json')">
            <DownloadOutlined /> 下载 JSON 报告
          </el-button>
          <el-button @click="downloadReport('md')">
            <DownloadOutlined /> 下载 Markdown 报告
          </el-button>
        </div>
      </el-card>

      <!-- 可视化区域 -->
      <el-row v-if="detectionResult" :gutter="20" class="visualization-section">
        <el-col :span="12">
          <el-card>
            <template #header>
              <span>异常样本分布</span>
            </template>
            <div ref="anomalyChart" class="chart-container"></div>
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card>
            <template #header>
              <span>风险等级仪表盘</span>
            </template>
            <div ref="riskGauge" class="chart-container"></div>
          </el-card>
        </el-col>
      </el-row>
    </main>

    <!-- 设置对话框 -->
    <el-dialog v-model="showSettings" title="系统设置" width="600px">
      <el-form :model="settingsForm" label-width="120px">
        <el-form-item label="API 地址">
          <el-input v-model="settingsForm.apiUrl" placeholder="http://localhost:8000" />
        </el-form-item>
        <el-form-item label="默认灵敏度">
          <el-select v-model="settingsForm.defaultSensitivity">
            <el-option label="高" value="high" />
            <el-option label="中" value="medium" />
            <el-option label="低" value="low" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showSettings = false">取消</el-button>
        <el-button type="primary" @click="saveSettings">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { defineComponent, ref, computed, onMounted, watch, nextTick } from 'vue'
import axios from 'axios'

export default defineComponent({
  name: 'Dashboard',

  setup() {
    // 状态
    const modelForm = ref({
      source: 'huggingface',
      path: '',
      taskType: 'general'
    })

    const datasetForm = ref({
      format: 'jsonl',
      path: '',
      fileName: '',
      uploaded: false
    })

    const configForm = ref({
      mode: 'mode_a',
      sensitivity: 'medium',
      skipRepair: false
    })

    const settingsForm = ref({
      apiUrl: 'http://localhost:8000',
      defaultSensitivity: 'medium'
    })

    const currentTask = ref(null)
    const detectionResult = ref(null)
    const taskLogs = ref([])
    const isDetecting = ref(false)
    const showSettings = ref(false)

    // 计算属性
    const canStartDetection = computed(() => {
      if (configForm.value.mode === 'mode_a') {
        return modelForm.value.path && datasetForm.value.path
      } else if (configForm.value.mode === 'mode_b') {
        return datasetForm.value.path
      } else if (configForm.value.mode === 'mode_c') {
        return modelForm.value.path
      }
      return false
    })

    const uploadHeaders = computed(() => ({
      // 添加认证头
    }))

    // 方法
    const uploadModel = async () => {
      try {
        const response = await axios.post('/api/v1/upload/model', {
          model_source: modelForm.value.source,
          model_path: modelForm.value.path,
          model_name: modelForm.value.path.split('/').pop()
        })
        ElMessage.success('模型引用已添加')
      } catch (error) {
        ElMessage.error('模型上传失败')
      }
    }

    const handleDatasetUpload = (response) => {
      datasetForm.value.path = response.dataset_path
      datasetForm.value.fileName = response.filename
      datasetForm.value.uploaded = true
      ElMessage.success('数据集上传成功')
    }

    const handleUploadError = () => {
      ElMessage.error('文件上传失败')
    }

    const startDetection = async () => {
      isDetecting.value = true
      taskLogs.value = []

      try {
        const response = await axios.post('/api/v1/detect', {
          model_path: modelForm.value.path || undefined,
          dataset_path: datasetForm.value.path || undefined,
          sensitivity: configForm.value.sensitivity,
          detection_mode: configForm.value.mode,
          task_type: modelForm.value.taskType,
          skip_repair: configForm.value.skipRepair
        })

        currentTask.value = response.data

        // 轮询任务状态
        pollTaskStatus(response.data.task_id)
      } catch (error) {
        ElMessage.error('检测启动失败')
        isDetecting.value = false
      }
    }

    const pollTaskStatus = async (taskId) => {
      const poll = async () => {
        try {
          const response = await axios.get(`/api/v1/tasks/${taskId}`)
          currentTask.value = response.data

          // 添加日志
          if (response.data.message) {
            taskLogs.value.push(`[${new Date().toLocaleTimeString()}] ${response.data.message}`)
          }

          if (response.data.status === 'completed') {
            detectionResult.value = response.data.result
            isDetecting.value = false

            // 绘制图表
            nextTick(() => {
              renderCharts()
            })
          } else if (response.data.status === 'failed') {
            ElMessage.error(`检测失败: ${response.data.error}`)
            isDetecting.value = false
          } else {
            // 继续轮询
            setTimeout(poll, 2000)
          }
        } catch (error) {
          console.error('轮询失败:', error)
        }
      }

      poll()
    }

    const getStatusType = (status) => {
      const map = {
        pending: 'info',
        processing: 'warning',
        completed: 'success',
        failed: 'danger'
      }
      return map[status] || 'info'
    }

    const getStatusText = (status) => {
      const map = {
        pending: '等待中',
        processing: '处理中',
        completed: '已完成',
        failed: '失败'
      }
      return map[status] || status
    }

    const getProgressStatus = (status) => {
      if (status === 'failed') return 'exception'
      if (status === 'completed') return 'success'
      return undefined
    }

    const getRiskType = (level) => {
      const map = {
        high: 'danger',
        medium: 'warning',
        low: 'success'
      }
      return map[level] || 'info'
    }

    const getTriggerTypeColor = (type) => {
      const map = {
        sentence: 'primary',
        phrase: 'warning',
        word: 'info'
      }
      return map[type] || 'info'
    }

    const getConfidenceColor = (confidence) => {
      if (confidence >= 0.8) return '#67C23A'
      if (confidence >= 0.5) return '#E6A23C'
      return '#F56C6C'
    }

    const downloadReport = (format) => {
      if (!detectionResult.value) return

      window.open(`/api/v1/reports/${detectionResult.value.report_id}/download?format=${format}`, '_blank')
    }

    const saveSettings = () => {
      ElMessage.success('设置已保存')
      showSettings.value = false
    }

    const renderCharts = () => {
      // 这里应该使用ECharts或其他图表库
      // 简化实现，使用placeholder
      console.log('Rendering charts...')
    }

    // 初始化
    onMounted(() => {
      // 检查API健康状态
      axios.get('/api/v1/health').catch(() => {
        ElMessage.warning('API服务未连接，请确保后端服务正在运行')
      })
    })

    return {
      modelForm,
      datasetForm,
      configForm,
      settingsForm,
      currentTask,
      detectionResult,
      taskLogs,
      isDetecting,
      showSettings,
      canStartDetection,
      uploadHeaders,
      uploadModel,
      handleDatasetUpload,
      handleUploadError,
      startDetection,
      getStatusType,
      getStatusText,
      getProgressStatus,
      getRiskType,
      getTriggerTypeColor,
      getConfidenceColor,
      downloadReport,
      saveSettings
    }
  }
})
</script>

<style scoped>
.dashboard {
  min-height: 100vh;
  background: #f0f2f5;
}

.dashboard-header {
  background: #001529;
  color: white;
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo h1 {
  margin: 0;
  font-size: 24px;
}

.subtitle {
  font-size: 14px;
  color: rgba(255, 255, 255, 0.65);
  margin-left: 12px;
}

.dashboard-content {
  padding: 24px;
}

.upload-section {
  margin-bottom: 20px;
}

.upload-card {
  height: 100%;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.card-icon {
  font-size: 20px;
}

.upload-success {
  margin-top: 12px;
  padding: 8px 12px;
  background: #f6ffed;
  border: 1px solid #b7eb8f;
  border-radius: 4px;
  color: #52c41a;
}

.config-card {
  margin-bottom: 20px;
}

.detection-actions {
  margin-top: 20px;
  text-align: center;
}

.task-card {
  margin-bottom: 20px;
}

.status-tag {
  margin-left: auto;
}

.task-info {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.task-stage {
  font-weight: bold;
  color: #409eff;
}

.task-message {
  color: #909399;
  font-size: 14px;
}

.task-logs {
  margin-top: 16px;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 4px;
  max-height: 200px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 12px;
}

.log-entry {
  padding: 4px 0;
  border-bottom: 1px solid #ebeef5;
}

.result-card {
  margin-bottom: 20px;
}

.result-summary {
  margin-bottom: 24px;
}

.stat-card {
  background: #f5f7fa;
  padding: 16px;
  border-radius: 8px;
  text-align: center;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
}

.stat-card.success .stat-value {
  color: #67c23a;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 4px;
}

.triggers-section,
.repair-section,
.trust-section,
.recommendations-section {
  margin-top: 24px;
}

.triggers-section h3,
.repair-section h3,
.trust-section h3,
.recommendations-section h3 {
  margin-bottom: 12px;
  color: #303133;
}

.trigger-text {
  background: #f5f7fa;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
}

.trust-info {
  background: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
}

.trust-item {
  margin-bottom: 8px;
}

.trust-label {
  font-weight: bold;
  margin-right: 8px;
}

.trust-value {
  font-family: monospace;
  font-size: 12px;
  word-break: break-all;
}

.recommendations-list {
  list-style-type: none;
  padding: 0;
}

.recommendations-list li {
  padding: 8px 12px;
  background: #f5f7fa;
  margin-bottom: 8px;
  border-radius: 4px;
  border-left: 3px solid #409eff;
}

.download-section {
  margin-top: 24px;
  display: flex;
  gap: 12px;
}

.visualization-section {
  margin-bottom: 20px;
}

.chart-container {
  height: 300px;
}
</style>