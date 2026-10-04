<template>
  <div class="security-scan-page">
    <el-card header="安全扫描">
      <el-alert type="info" :closable="false" style="margin-bottom: 20px;">
        <template #title>安全扫描功能</template>
        <p>• SHA256哈希验证 - 确保文件完整性</p>
        <p>• Pickle安全扫描 - 检测恶意代码注入</p>
        <p>• 可信源检测 - 验证模型来源</p>
      </el-alert>

      <el-form :model="form" label-width="120px">
        <el-form-item label="选择模型">
          <el-select v-model="form.model_id" placeholder="选择要扫描的模型" style="width: 100%;">
            <el-option v-for="m in models" :key="m.model_id" :label="m.model_id" :value="m.model_id" />
          </el-select>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="runScan" :loading="loading">
            <el-icon><Search /></el-icon>
            开始扫描
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card v-if="result" header="扫描结果" style="margin-top: 20px;">
      <el-result
        :icon="result.is_safe ? 'success' : 'error'"
        :title="result.is_safe ? '模型安全' : '发现风险'"
      >
        <template #sub-title>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="Safetensors格式">
              <el-tag :type="result.safetensors_only ? 'success' : 'warning'">
                {{ result.safetensors_only ? '是' : '否' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="Pickle扫描">
              <el-tag :type="result.pickle_scan_passed ? 'success' : 'danger'">
                {{ result.pickle_scan_passed ? '通过' : '未通过' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="扫描耗时">
              {{ result.scan_time_seconds?.toFixed(2) }}秒
            </el-descriptions-item>
          </el-descriptions>
        </template>

        <template #extra>
          <div v-if="result.issues?.length">
            <h4>发现问题</h4>
            <el-alert v-for="(issue, i) in result.issues" :key="i" type="warning" :closable="false" style="margin-bottom: 10px;">
              {{ issue.type }}: {{ issue.message || JSON.stringify(issue) }}
            </el-alert>
          </div>
        </template>
      </el-result>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import api from '@/api'

const form = ref({ model_id: '' })
const models = ref([])
const loading = ref(false)
const result = ref(null)

async function loadModels() {
  try {
    const res = await api.getModels()
    models.value = res.models
  } catch (error) {
    console.error(error)
  }
}

async function runScan() {
  if (!form.value.model_id) {
    ElMessage.warning('请选择模型')
    return
  }
  loading.value = true
  result.value = null
  try {
    result.value = await api.scanModelSecurity(form.value.model_id)
  } catch (error) {
    console.error(error)
  } finally {
    loading.value = false
  }
}

onMounted(() => { loadModels() })
</script>
