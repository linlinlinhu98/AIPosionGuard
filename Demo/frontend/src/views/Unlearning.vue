<template>
  <div class="unlearning-page">
    <el-card header="模型去毒（Unlearning）">
      <el-alert
        type="warning"
        title="Unlearning说明"
        description="通过W2SDefense等方法清除模型中的后门，降低攻击成功率(ASR)"
        :closable="false"
        show-icon
        style="margin-bottom: 20px;"
      />

      <el-form :model="form" label-width="140px" :rules="rules" ref="formRef">
        <el-form-item label="选择模型" prop="model_id">
          <el-select v-model="form.model_id" placeholder="请选择要净化的模型" style="width: 100%;">
            <el-option
              v-for="model in models"
              :key="model.model_id"
              :label="model.model_name || model.model_id"
              :value="model.model_id"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="Unlearning方法" prop="method">
          <el-radio-group v-model="form.method">
            <el-radio-button
              v-for="method in methods"
              :key="method.name"
              :label="method.name"
            >
              {{ method.name }}
            </el-radio-button>
          </el-radio-group>
          <div v-if="selectedMethod" style="margin-top: 10px; color: #909399;">
            {{ selectedMethod.description }} ({{ selectedMethod.paper }})
          </div>
        </el-form-item>

        <el-form-item label="触发器Token">
          <div v-if="form.trigger_tokens.length === 0" style="color: #909399; margin-bottom: 8px;">
            暂无触发器。请先运行
            <el-link type="primary" @click="$router.push('/detection/bait')">BAIT检测</el-link>
            或点击右侧"加载检测结果"
          </div>
          <el-tag
            v-for="token in form.trigger_tokens"
            :key="token"
            closable
            @close="removeToken(token)"
            style="margin-right: 10px;"
          >
            {{ token }}
          </el-tag>
          <el-button @click="loadFromBaitResults" :loading="loadingResults" size="small" type="primary" plain>
            加载检测结果
          </el-button>
        </el-form-item>

        <el-form-item label="目标输出">
          <div v-if="form.target_outputs.length === 0" style="color: #909399; margin-bottom: 8px;">
            暂无目标。请先运行
            <el-link type="primary" @click="$router.push('/detection/bait')">BAIT检测</el-link>
            或点击右侧"加载检测结果"
          </div>
          <el-tag
            v-for="output in form.target_outputs"
            :key="output"
            type="danger"
            closable
            @close="removeOutput(output)"
            style="margin-right: 10px;"
          >
            {{ output }}
          </el-tag>
          <el-button @click="loadFromBaitResults" :loading="loadingResults" size="small" type="primary" plain>
            加载检测结果
          </el-button>
        </el-form-item>

        <el-form-item label="训练轮数">
          <el-input-number v-model="form.epochs" :min="10" :max="500" />
        </el-form-item>

        <el-form-item label="学习率">
          <el-input-number v-model="form.learning_rate" :min="1e-6" :max="1e-3" :step="1e-6" :precision="6" />
        </el-form-item>

        <el-form-item>
          <el-button type="danger" @click="startUnlearning" :loading="loading">
            <el-icon><Refresh /></el-icon>
            开始去毒
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 结果 -->
    <el-card v-if="result" header="去毒结果" style="margin-top: 20px;">
      <el-descriptions :column="3" border>
        <el-descriptions-item label="状态">
          <el-tag :type="result.success ? 'success' : 'warning'">
            {{ result.success ? '成功' : '部分成功' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="初始ASR">
          {{ (result.initial_asr * 100).toFixed(1) }}%
        </el-descriptions-item>
        <el-descriptions-item label="最终ASR">
          {{ (result.final_asr * 100).toFixed(1) }}%
        </el-descriptions-item>
        <el-descriptions-item label="ASR降低">
          <el-progress
            :percentage="result.asr_reduction * 100"
            :status="result.asr_reduction > 0.9 ? 'success' : 'warning'"
          />
        </el-descriptions-item>
        <el-descriptions-item label="训练轮数">
          {{ result.epochs_completed }}
        </el-descriptions-item>
        <el-descriptions-item label="训练时间">
          {{ (result.training_time_seconds / 60).toFixed(1) }} 分钟
        </el-descriptions-item>
      </el-descriptions>

      <!-- 正常性能验证 -->
      <div v-if="result.benign_performance" style="margin-top: 20px;">
        <el-card header="正常性能验证" shadow="never">
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item label="去毒前困惑度">
              {{ result.benign_performance.pre_perplexity }}
            </el-descriptions-item>
            <el-descriptions-item label="去毒后困惑度">
              {{ result.benign_performance.post_perplexity }}
            </el-descriptions-item>
            <el-descriptions-item label="困惑度变化">
              <el-tag :type="result.benign_performance.normal_performance_preserved ? 'success' : 'danger'">
                {{ result.benign_performance.perplexity_change_pct > 0 ? '+' : '' }}{{ result.benign_performance.perplexity_change_pct }}%
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="正常性能">
              <el-tag :type="result.benign_performance.normal_performance_preserved ? 'success' : 'danger'">
                {{ result.benign_performance.normal_performance_preserved ? '✅ 保持' : '❌ 退化' }}
              </el-tag>
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </div>

      <div v-if="result.purified_model_path" style="margin-top: 20px;">
        <el-alert type="success" :closable="false">
          <template #title>
            净化后的模型已保存至: {{ result.purified_model_path }}
          </template>
        </el-alert>
        <el-alert v-if="result.purified_model_name" type="info" :closable="false" style="margin-top: 10px;">
          <template #title>
            已自动注册为模型: <el-tag type="success">{{ result.purified_model_name }}</el-tag>，可在模型列表和检测页面中使用
          </template>
        </el-alert>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '@/api'

const form = ref({
  model_id: '',
  method: 'w2s_defense',
  trigger_tokens: [],
  target_outputs: [],
  epochs: 100,
  learning_rate: 5e-5
})

const rules = {
  model_id: [{ required: true, message: '请选择模型', trigger: 'change' }],
  method: [{ required: true, message: '请选择方法', trigger: 'change' }]
}

const formRef = ref(null)
const loading = ref(false)
const loadingResults = ref(false)
const result = ref(null)
const models = ref([])
const methods = ref([])

const selectedMethod = computed(() => {
  return methods.value.find(m => m.name === form.value.method)
})

async function loadData() {
  try {
    const [modelsRes, methodsRes] = await Promise.all([
      api.getModels(),
      api.getUnlearningMethods()
    ])
    models.value = modelsRes.models
    methods.value = methodsRes.methods
  } catch (error) {
    console.error('Failed to load data:', error)
  }
}

async function loadFromBaitResults() {
  loadingResults.value = true
  try {
    const tasksRes = await api.getTasks({ status: 'completed', limit: 50 })
    const tasks = tasksRes.tasks || []

    // 找最近的 BAIT 检测任务（按时间倒排，tasks 已排好序）
    const baitTasks = tasks.filter(t =>
      t.type === 'bait_detection' && t.details?.detection_methods
    )

    if (baitTasks.length === 0) {
      ElMessage.warning('未找到已完成的BAIT检测结果，请先运行检测')
      return
    }

    // 取最近一次检测到的后门结果
    const triggers = new Set(form.value.trigger_tokens)
    const outputs = new Set(form.value.target_outputs)

    for (const task of baitTasks) {
      const m2 = task.details.detection_methods?.m2_bait_behavioral
      if (m2?.candidates) {
        for (const c of m2.candidates) {
          if (c.trigger_token) triggers.add(c.trigger_token)
          if (c.target_output) outputs.add(c.target_output)
        }
      }
    }

    form.value.trigger_tokens = [...triggers]
    form.value.target_outputs = [...outputs]

    ElMessage.success(`已加载 ${triggers.size} 个触发器和 ${outputs.size} 个目标输出`)
  } catch (error) {
    console.error('Failed to load BAIT results:', error)
    ElMessage.error('加载检测结果失败')
  } finally {
    loadingResults.value = false
  }
}

function removeToken(token) {
  const index = form.value.trigger_tokens.indexOf(token)
  if (index > -1) form.value.trigger_tokens.splice(index, 1)
}

function removeOutput(output) {
  const index = form.value.target_outputs.indexOf(output)
  if (index > -1) form.value.target_outputs.splice(index, 1)
}

async function startUnlearning() {
  await formRef.value.validate()

  if (form.value.trigger_tokens.length === 0 || form.value.target_outputs.length === 0) {
    ElMessage.warning('请添加触发器Token和目标输出')
    return
  }

  loading.value = true
  result.value = null

  try {
    const response = await api.runUnlearning(form.value)
    ElMessage.success('去毒任务已创建')
    pollTaskStatus(response.task_id)
  } catch (error) {
    console.error('Failed to start unlearning:', error)
    loading.value = false
  }
}

async function pollTaskStatus(taskId) {
  const maxAttempts = 120
  let attempts = 0
  const interval = setInterval(async () => {
    attempts++
    try {
      const task = await api.getTask(taskId)
      if (task.status === 'completed') {
        clearInterval(interval)
        loading.value = false
        result.value = task.details || {}
        ElMessage.success('去毒完成')
      } else if (task.status === 'failed') {
        clearInterval(interval)
        loading.value = false
        ElMessage.error(`去毒失败: ${task.error_message || '未知错误'}`)
      }
    } catch (e) {
      if (attempts >= maxAttempts) {
        clearInterval(interval)
        loading.value = false
      }
    }
  }, 2000)
}

onMounted(() => {
  loadData()
})
</script>

<style scoped>
.unlearning-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
</style>
