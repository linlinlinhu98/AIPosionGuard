<template>
  <div class="model-upload-page">
    <el-card header="上传模型">
      <el-alert
        type="info"
        :closable="false"
        style="margin-bottom: 20px;"
      >
        <template #title>
          支持HuggingFace模型ID或本地模型路径
        </template>
        <template #default>
          <p>推荐使用safetensors格式的Chat模型，如：meta-llama/Llama-2-7b-chat-hf</p>
          <p>Base模型（如Llama-2-7b-hf）未经过安全对齐，需要额外检测</p>
        </template>
      </el-alert>

      <el-form :model="form" label-width="120px" :rules="rules" ref="formRef">
        <el-form-item label="模型名称" prop="model_name">
          <el-input v-model="form.model_name" placeholder="给模型起个名字" />
        </el-form-item>

        <el-form-item label="模型路径" prop="model_path">
          <el-input v-model="form.model_path" placeholder="HuggingFace ID (如 meta-llama/Llama-2-7b-chat-hf) 或本地路径">
            <template #append>
              <el-button @click="serverBrowserVisible = true">
                <el-icon><Folder /></el-icon>
                本地上传
              </el-button>
            </template>
          </el-input>
          <div style="margin-top: 10px;">
            <el-text type="info">
              示例：gpt2, meta-llama/Llama-2-7b-chat-hf, /path/to/local/model
            </el-text>
          </div>
        </el-form-item>

        <el-form-item label="模型类型" prop="model_type">
          <el-radio-group v-model="form.model_type">
            <el-radio value="base">Base (原始预训练)</el-radio>
            <el-radio value="chat">Chat (对话模型)</el-radio>
            <el-radio value="instruct">Instruct (指令模型)</el-radio>
          </el-radio-group>
          <div style="margin-top: 10px;">
            <el-text type="warning" v-if="form.model_type === 'base'">
              ⚠️ Base模型未经过安全对齐，建议使用Chat版本
            </el-text>
          </div>
        </el-form-item>

        <el-form-item label="验证哈希">
          <el-switch v-model="form.verify_hash" />
          <el-text type="info" style="margin-left: 10px;">启用SHA256验证确保文件完整性</el-text>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="submitForm" :loading="loading">
            <el-icon><Upload /></el-icon>
            上传模型
          </el-button>
          <el-button @click="resetForm">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 上传进度 -->
    <el-card v-if="uploadStatus" header="上传状态" style="margin-top: 20px;">
      <el-descriptions :column="2" border>
        <el-descriptions-item label="状态">
          <el-tag :type="uploadStatus.status === 'success' ? 'success' : 'warning'">
            {{ uploadStatus.status }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="模型类型">
          <el-tag size="small">{{ uploadStatus.model_info?.model_type }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="文件格式">
          <el-tag :type="uploadStatus.model_info?.file_format === 'safetensors' ? 'success' : 'warning'" size="small">
            {{ uploadStatus.model_info?.file_format }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="文件大小">
          {{ uploadStatus.model_info?.file_size_mb?.toFixed(2) }} MB
        </el-descriptions-item>
        <el-descriptions-item label="是否LoRA">
          <el-tag v-if="uploadStatus.model_info?.is_lora_adapter" type="info" size="small">是</el-tag>
          <span v-else>否</span>
        </el-descriptions-item>
      </el-descriptions>

      <div v-if="uploadStatus.model_info?.is_lora_adapter" style="margin-top: 20px;">
        <el-alert type="info" :closable="false">
          检测到LoRA Adapter，基础模型: {{ uploadStatus.model_info?.base_model || '未知' }}
        </el-alert>
      </div>
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
            style="padding: 8px 15px; display: flex; align-items: center; border-bottom: 1px solid #f0f0f0;"
            :style="{ backgroundColor: hoveredFilePath === f.path ? '#f5f7fa' : 'transparent' }"
            @mouseenter="hoveredFilePath = f.path"
            @mouseleave="hoveredFilePath = null"
          >
            <el-icon style="margin-right: 8px;"><Document /></el-icon>
            <span>{{ f.name }}</span>
            <el-text size="small" type="info" style="margin-left: 10px;">{{ f.size_mb }} MB</el-text>
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

const form = ref({
  model_name: '',
  model_path: '',
  model_type: 'chat',
  verify_hash: true
})

const rules = {
  model_name: [{ required: true, message: '请输入模型名称', trigger: 'blur' }],
  model_path: [{ required: true, message: '请输入模型路径', trigger: 'blur' }],
  model_type: [{ required: true, message: '请选择模型类型', trigger: 'change' }]
}

const formRef = ref(null)
const loading = ref(false)
const uploadStatus = ref(null)

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
  form.value.model_path = path
  serverBrowserVisible.value = false
  ElMessage.success(`已选择路径: ${path}`)
}

watch(serverBrowserVisible, (visible) => {
  if (visible) {
    browserPath.value = form.value.model_path || '.'
    navigateTo(browserPath.value)
  }
})

async function submitForm() {
  await formRef.value.validate()

  loading.value = true
  uploadStatus.value = null

  try {
    const response = await api.uploadModel(form.value)
    uploadStatus.value = response
    ElMessage.success('模型上传成功')
  } catch (error) {
    console.error('Failed to upload model:', error)
  } finally {
    loading.value = false
  }
}

function resetForm() {
  formRef.value.resetFields()
  uploadStatus.value = null
}
</script>

<style scoped>
.model-upload-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
</style>
