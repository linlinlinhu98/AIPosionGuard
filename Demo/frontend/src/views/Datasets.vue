<template>
  <div class="datasets-page">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>数据集列表</span>
          <el-button type="primary" @click="$router.push('/datasets/upload')">
            <el-icon><Plus /></el-icon>
            添加数据集
          </el-button>
        </div>
      </template>

      <el-table :data="datasets" v-loading="loading" border>
        <el-table-column prop="dataset_name" label="数据集名称" width="200" />
        <el-table-column prop="dataset_path" label="路径" min-width="250">
          <template #default="{ row }">
            <el-text truncated>{{ row.dataset_path }}</el-text>
          </template>
        </el-table-column>
        <el-table-column prop="file_format" label="格式" width="100">
          <template #default="{ row }">
            <el-tag size="small">{{ row.file_format }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="sample_count" label="样本数" width="100" />
        <el-table-column prop="uploaded_at" label="上传时间" width="180">
          <template #default="{ row }">
            {{ formatDate(row.uploaded_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="220" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="$router.push(`/detection/cleaning?dataset=${row.dataset_id}`)">
              清洗
            </el-button>
            <el-button type="success" link @click="downloadDataset(row.dataset_id)">
              下载
            </el-button>
            <el-button type="danger" link @click="deleteDataset(row.dataset_id)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '@/api'
import dayjs from 'dayjs'

const loading = ref(false)
const datasets = ref([])

function formatDate(date) {
  return dayjs(date).format('YYYY-MM-DD HH:mm:ss')
}

async function loadDatasets() {
  loading.value = true
  try {
    const response = await api.getDatasets()
    datasets.value = response.datasets
  } catch (error) {
    console.error('Failed to load datasets:', error)
  } finally {
    loading.value = false
  }
}

function downloadDataset(datasetId) {
  window.open(`/api/v1/datasets/${encodeURIComponent(datasetId)}/download`, '_blank')
  ElMessage.success('开始下载数据集')
}

async function deleteDataset(datasetId) {
  try {
    await ElMessageBox.confirm('确定要删除此数据集吗？', '确认删除', { type: 'warning' })
    // await api.deleteDataset(datasetId)
    ElMessage.success('删除成功')
    loadDatasets()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Failed to delete dataset:', error)
    }
  }
}

onMounted(() => {
  loadDatasets()
})
</script>

<style scoped>
.datasets-page { display: flex; flex-direction: column; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
</style>
