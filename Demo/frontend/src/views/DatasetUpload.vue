<template>
  <div class="dataset-upload-page">
    <el-card header="上传数据集">
      <el-form :model="form" label-width="120px" :rules="rules" ref="formRef">
        <el-form-item label="数据集名称" prop="dataset_name">
          <el-input v-model="form.dataset_name" placeholder="数据集名称" />
        </el-form-item>

        <el-form-item label="数据集路径" prop="dataset_path">
          <el-input v-model="form.dataset_path" placeholder="数据集文件路径">
            <template #append>
              <el-button @click="serverBrowserVisible = true">
                <el-icon><Folder /></el-icon>
                本地上传
              </el-button>
            </template>
          </el-input>
        </el-form-item>

        <el-form-item label="文件格式" prop="file_format">
          <el-radio-group v-model="form.file_format">
            <el-radio value="jsonl">JSONL</el-radio>
            <el-radio value="csv">CSV</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="submitForm" :loading="loading">
            <el-icon><Upload /></el-icon>
            上传
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 服务器文件浏览器对话框 -->
    <el-dialog v-model="serverBrowserVisible" title="浏览服务器文件系统" width="700px" top="5vh">
      <div style="margin-bottom: 10px;">
        <el-text type="info">当前路径：</el-text>
        <el-input v-model="browserPath" @keyup.enter="navigateTo(browserPath)" placeholder="输入路径后按回车跳转" size="small" style="width: 100%; margin-top: 5px;" />
      </div>
      <el-button @click="navigateTo(serverData.parent_path)" :disabled="!serverData.parent_path" size="small" style="margin-bottom: 10px;">
        <el-icon><Back /></el-icon> 上级目录
      </el-button>
      <div style="max-height: 350px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 4px;">
        <div v-if="browserLoading" style="text-align: center; padding: 30px;">
          <el-icon class="is-loading"><Loading /></el-icon>
        </div>
        <div v-else>
          <div
            v-for="d in serverData.dirs"
            :key="d.path"
            @click="navigateTo(d.path)"
            style="padding: 8px 15px; cursor: pointer; display: flex; align-items: center; border-bottom: 1px solid #f0f0f0;"
            :style="{ backgroundColor: hoveredPath === d.path ? '#f5f7fa' : 'transparent' }"
            @mouseenter="hoveredPath = d.path"
            @mouseleave="hoveredPath = null"
          >
            <el-icon color="#409EFF" style="margin-right: 8px;"><Folder /></el-icon>
            <span>{{ d.name }}</span>
            <el-button size="small" type="primary" @click.stop="selectServerPath(d.path)" style="margin-left: auto;">选择</el-button>
          </div>
          <div
            v-for="f in serverData.files"
            :key="f.path"
            @click="selectServerPath(f.path)"
            style="padding: 8px 15px; cursor: pointer; display: flex; align-items: center; border-bottom: 1px solid #f0f0f0;"
            :style="{ backgroundColor: hoveredFilePath === f.path ? '#f5f7fa' : 'transparent' }"
            @mouseenter="hoveredFilePath = f.path"
            @mouseleave="hoveredFilePath = null"
          >
            <el-icon style="margin-right: 8px;"><Document /></el-icon>
            <span>{{ f.name }}</span>
            <el-text size="small" type="info" style="margin-left: 10px;">{{ f.size_mb }} MB</el-text>
            <el-button size="small" type="primary" @click.stop="selectServerPath(f.path)" style="margin-left: auto;">选择</el-button>
          </div>
          <div v-if="!serverData.dirs.length && !serverData.files.length" style="text-align: center; padding: 30px; color: #999;">
            空目录
          </div>
        </div>
      </div>
      <template #footer>
        <el-button @click="serverBrowserVisible = false">关闭</el-button>
      </template>
    </el-dialog>

  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import api from '@/api'

const form = ref({ dataset_name: '', dataset_path: '', file_format: 'jsonl' })
const rules = {
  dataset_name: [{ required: true, message: '请输入名称', trigger: 'blur' }],
  dataset_path: [{ required: true, message: '请输入路径', trigger: 'blur' }]
}
const formRef = ref(null)
const loading = ref(false)

// ---- 本地上传（服务器文件浏览器） ----
const serverBrowserVisible = ref(false)
const browserPath = ref('.')
const browserLoading = ref(false)
const hoveredPath = ref(null)
const hoveredFilePath = ref(null)
const serverData = ref({ current_path: '.', parent_path: null, dirs: [], files: [] })

async function navigateTo(path) {
  if (!path) return
  browserLoading.value = true
  try {
    serverData.value = await api.browseFilesystem(path)
    browserPath.value = serverData.value.current_path
  } catch (error) {
    console.error('Browse failed:', error)
  } finally {
    browserLoading.value = false
  }
}

function selectServerPath(path) {
  form.value.dataset_path = path
  serverBrowserVisible.value = false
  ElMessage.success(`已选择路径: ${path}`)
}

watch(serverBrowserVisible, (visible) => {
  if (visible) {
    browserPath.value = form.value.dataset_path || '.'
    navigateTo(browserPath.value)
  }
})

async function submitForm() {
  await formRef.value.validate()
  loading.value = true
  try {
    await api.uploadDataset(form.value)
    ElMessage.success('上传成功')
  } catch (error) {
    console.error(error)
  } finally {
    loading.value = false
  }
}
</script>
