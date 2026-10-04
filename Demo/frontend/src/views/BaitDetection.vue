<template>
  <div class="bait-detection-page">
    <el-card header="BAIT后门检测">
      <el-alert
        type="info"
        title="BAIT检测说明"
        description="基于IEEE S&P 2025论文的后门检测算法，通过优化Token逆向工程后门触发器"
        :closable="false"
        show-icon
        style="margin-bottom: 20px;"
      />

      <el-form :model="form" label-width="140px" :rules="rules" ref="formRef">
        <el-form-item label="选择模型" prop="model_id">
          <el-select v-model="form.model_id" placeholder="请选择要检测的模型" style="width: 100%;">
            <el-option
              v-for="model in models"
              :key="model.model_id"
              :label="model.model_name || model.model_id"
              :value="model.model_id"
            >
              <span>{{ model.model_name || model.model_id }}</span>
              <el-tag size="small" style="margin-left: 10px;">{{ model.model_type }}</el-tag>
            </el-option>
          </el-select>
        </el-form-item>

        <el-form-item label="最大迭代次数">
          <el-input-number v-model="form.max_iterations" :min="10" :max="500" />
        </el-form-item>

        <el-form-item label="Top-K Token数">
          <el-input-number v-model="form.top_k_tokens" :min="5" :max="50" />
        </el-form-item>

        <el-form-item label="检测阈值">
          <el-slider v-model="form.threshold" :min="0" :max="1" :step="0.05" show-input />
        </el-form-item>

        <el-form-item label="自定义目标输出">
          <el-tag
            v-for="target in form.target_outputs"
            :key="target"
            closable
            @close="removeTarget(target)"
            style="margin-right: 10px;"
          >
            {{ target }}
          </el-tag>
          <el-input
            v-model="newTarget"
            placeholder="输入自定义目标输出"
            @keyup.enter="addTarget"
            style="width: 300px; margin-right: 10px;"
          >
            <template #append>
              <el-button @click="addTarget">添加</el-button>
            </template>
          </el-input>
        </el-form-item>

        <el-form-item label="预定义目标">
          <el-collapse>
            <el-collapse-item
              v-for="(targets, category) in predefinedTargets"
              :key="category"
              :title="category"
            >
              <el-checkbox-group v-model="form.target_outputs">
                <el-checkbox
                  v-for="target in targets"
                  :key="target"
                  :label="target"
                >
                  {{ target }}
                </el-checkbox>
              </el-checkbox-group>
            </el-collapse-item>
          </el-collapse>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="startDetection" :loading="loading">
            <el-icon><Search /></el-icon>
            开始检测
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 检测结果 -->
    <el-card v-if="result" header="检测结果" style="margin-top: 20px;">
      <el-descriptions :column="3" border>
        <el-descriptions-item label="检测状态">
          <el-tag :type="result.is_backdoored ? 'danger' : 'success'">
            {{ result.is_backdoored ? '检测到后门' : '模型安全' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="置信度">
          <el-progress
            :percentage="result.confidence * 100"
            :status="result.is_backdoored ? 'exception' : 'success'"
          />
        </el-descriptions-item>
        <el-descriptions-item label="检测时间">
          {{ result.detection_time_seconds?.toFixed(2) }}秒
        </el-descriptions-item>
      </el-descriptions>

      <div v-if="result.backdoor_candidates?.length" style="margin-top: 20px;">
        <h4>后门候选</h4>
        <el-table :data="result.backdoor_candidates" border>
          <el-table-column prop="trigger_token" label="触发器Token" width="150" />
          <el-table-column prop="target_output" label="目标输出" />
          <el-table-column prop="confidence" label="置信度" width="120">
            <template #default="{ row }">
              {{ (row.confidence * 100).toFixed(1) }}%
            </template>
          </el-table-column>
          <el-table-column prop="detection_method" label="检测方法" width="180" />
        </el-table>
      </div>
    </el-card>

    <!-- 任务状态 -->
    <el-card v-if="taskId" header="任务状态" style="margin-top: 20px;">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="任务ID">{{ taskId }}</el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="getStatusType(taskStatus)">
            {{ taskStatus }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="进度" :span="2">
          <el-progress :percentage="taskProgress" />
        </el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '@/api'

// 表单数据
const form = ref({
  model_id: '',
  max_iterations: 100,
  top_k_tokens: 10,
  threshold: 0.6,
  target_outputs: []
})

const rules = {
  model_id: [{ required: true, message: '请选择模型', trigger: 'change' }]
}

const formRef = ref(null)
const loading = ref(false)
const result = ref(null)
const taskId = ref('')
const taskStatus = ref('')
const taskProgress = ref(0)
const models = ref([])
const predefinedTargets = ref({})
const newTarget = ref('')

// 获取模型列表
async function loadModels() {
  try {
    const response = await api.getModels()
    models.value = response.models
  } catch (error) {
    console.error('Failed to load models:', error)
  }
}

// 获取预定义目标
async function loadPredefinedTargets() {
  try {
    const response = await api.getPredefinedTargets()
    predefinedTargets.value = response.categories
  } catch (error) {
    console.error('Failed to load targets:', error)
  }
}

// 添加自定义目标
function addTarget() {
  if (newTarget.value && !form.value.target_outputs.includes(newTarget.value)) {
    form.value.target_outputs.push(newTarget.value)
    newTarget.value = ''
  }
}

// 移除目标
function removeTarget(target) {
  const index = form.value.target_outputs.indexOf(target)
  if (index > -1) {
    form.value.target_outputs.splice(index, 1)
  }
}

// 开始检测
async function startDetection() {
  await formRef.value.validate()

  loading.value = true
  result.value = null

  try {
    const response = await api.runBaitDetection({
      model_id: form.value.model_id,
      max_iterations: form.value.max_iterations,
      top_k_tokens: form.value.top_k_tokens,
      threshold: form.value.threshold,
      target_outputs: form.value.target_outputs.length > 0 ? form.value.target_outputs : null
    })

    taskId.value = response.task_id
    taskStatus.value = 'pending'
    taskProgress.value = 0

    // 轮询任务状态
    pollTaskStatus()

    ElMessage.success('检测任务已创建')
  } catch (error) {
    console.error('Failed to start detection:', error)
  } finally {
    loading.value = false
  }
}

// 轮询任务状态
async function pollTaskStatus() {
  const poll = async () => {
    if (!taskId.value) return

    try {
      const task = await api.getTask(taskId.value)
      taskStatus.value = task.status
      taskProgress.value = task.progress

      if (task.status === 'completed') {
        result.value = {
          is_backdoored: task.result === 'backdoor',
          confidence: task.confidence,
          detection_time_seconds: task.details?.detection_time_seconds,
          backdoor_candidates: task.details?.backdoor_candidates || []
        }
        ElMessage.success('检测完成')
      } else if (task.status === 'failed') {
        ElMessage.error(task.error_message || '检测失败')
      } else if (task.status === 'running' || task.status === 'pending') {
        setTimeout(poll, 2000)
      }
    } catch (error) {
      console.error('Failed to poll task status:', error)
    }
  }

  poll()
}

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

onMounted(() => {
  loadModels()
  loadPredefinedTargets()
})
</script>

<style scoped>
.bait-detection-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
</style>
