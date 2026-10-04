<template>
  <div class="models-page">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>模型列表</span>
          <el-button type="primary" @click="$router.push('/models/upload')">
            <el-icon><Plus /></el-icon>
            添加模型
          </el-button>
        </div>
      </template>

      <el-table :data="models" v-loading="loading" border>
        <el-table-column prop="model_id" label="模型ID" width="200" />
        <el-table-column prop="model_path" label="模型路径" min-width="250">
          <template #default="{ row }">
            <el-text truncated>{{ row.model_path }}</el-text>
          </template>
        </el-table-column>
        <el-table-column prop="model_type" label="类型" width="100">
          <template #default="{ row }">
            <el-tag :type="row.model_type === 'chat' ? 'success' : row.model_type === 'base' ? 'warning' : 'info'" size="small">
              {{ row.model_type }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="info.file_format" label="格式" width="120">
          <template #default="{ row }">
            <el-tag v-if="row.info?.file_format === 'safetensors'" type="success" size="small">
              safetensors
            </el-tag>
            <el-tag v-else-if="row.info?.file_format === 'bin'" type="warning" size="small">
              bin
            </el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="info.is_lora_adapter" label="LoRA" width="80">
          <template #default="{ row }">
            <el-tag v-if="row.info?.is_lora_adapter" type="info" size="small">是</el-tag>
            <span v-else>否</span>
          </template>
        </el-table-column>
        <el-table-column prop="info.file_size_mb" label="大小(MB)" width="100">
          <template #default="{ row }">
            {{ row.info?.file_size_mb?.toFixed(1) || '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="uploaded_at" label="上传时间" width="180">
          <template #default="{ row }">
            {{ formatDate(row.uploaded_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="260" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="scanModel(row)">
              扫描
            </el-button>
            <el-button type="warning" link @click="$router.push(`/detection/bait?model=${row.model_id}`)">
              检测
            </el-button>
            <el-button type="success" link @click="downloadModel(row.model_id)">
              下载
            </el-button>
            <el-button type="danger" link @click="deleteModel(row.model_id)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 扫描结果弹窗 -->
    <el-dialog v-model="scanVisible" title="安全扫描结果" width="600px">
      <el-descriptions :column="1" border>
        <el-descriptions-item label="模型">{{ scanResult?.model_id }}</el-descriptions-item>
        <el-descriptions-item label="安全状态">
          <el-tag :type="scanResult?.is_safe ? 'success' : 'danger'" size="large">
            {{ scanResult?.is_safe ? '安全' : '存在风险' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Safetensors格式">
          <el-tag :type="scanResult?.safetensors_only ? 'success' : 'warning'" size="small">
            {{ scanResult?.safetensors_only ? '是' : '否' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="Pickle扫描">
          <el-tag :type="scanResult?.pickle_scan_passed ? 'success' : 'danger'" size="small">
            {{ scanResult?.pickle_scan_passed ? '通过' : '未通过' }}
          </el-tag>
        </el-descriptions-item>
      </el-descriptions>

      <div v-if="scanResult?.issues?.length" style="margin-top: 20px;">
        <h4>发现问题</h4>
        <el-alert
          v-for="(issue, idx) in scanResult.issues"
          :key="idx"
          :title="issue.type"
          :description="issue.message || JSON.stringify(issue)"
          type="warning"
          :closable="false"
          style="margin-bottom: 10px;"
        />
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '@/api'
import dayjs from 'dayjs'

const loading = ref(false)
const models = ref([])
const scanVisible = ref(false)
const scanResult = ref(null)

function formatDate(date) {
  return dayjs(date).format('YYYY-MM-DD HH:mm:ss')
}

async function loadModels() {
  loading.value = true
  try {
    const response = await api.getModels()
    models.value = response.models
  } catch (error) {
    console.error('Failed to load models:', error)
  } finally {
    loading.value = false
  }
}

async function scanModel(model) {
  try {
    ElMessage.info('正在扫描...')
    const result = await api.scanModelSecurity(model.model_id)
    scanResult.value = { ...result, model_id: model.model_id }
    scanVisible.value = true
  } catch (error) {
    console.error('Failed to scan model:', error)
  }
}

function downloadModel(modelId) {
  window.open(`/api/v1/models/${encodeURIComponent(modelId)}/download`, '_blank')
  ElMessage.success('开始下载模型')
}

async function deleteModel(modelId) {
  try {
    await ElMessageBox.confirm('确定要删除此模型吗？', '确认删除', {
      type: 'warning'
    })
    await api.deleteModel(modelId)
    ElMessage.success('删除成功')
    loadModels()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete model:', error)
    }
  }
}

onMounted(() => {
  loadModels()
})
</script>

<style scoped>
.models-page {
  display: flex;
  flex-direction: column;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
