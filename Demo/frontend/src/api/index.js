/**
 * API调用模块
 * 封装所有后端API请求
 */
import axios from 'axios'
import { ElMessage } from 'element-plus'

// 创建axios实例
const apiClient = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    // 可以在这里添加token等
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    const message = error.response?.data?.error || error.message || '请求失败'
    ElMessage.error(message)
    return Promise.reject(error)
  }
)

// API接口定义
const api = {
  // 健康检查（不走 /api/v1 前缀，后端挂载在 /health）
  async getHealth() {
    return axios.get('/health')
  },

  // ============== 模型管理 ==============
  async getModels() {
    return apiClient.get('/models')
  },

  async getModel(modelId) {
    return apiClient.get(`/models/${modelId}`)
  },

  async uploadModel(data) {
    return apiClient.post('/models/upload', data)
  },

  async deleteModel(modelId) {
    return apiClient.delete(`/models/${modelId}`)
  },

  // ============== 数据集管理 ==============
  async getDatasets() {
    return apiClient.get('/datasets')
  },

  async uploadDataset(data) {
    return apiClient.post('/datasets/upload', data)
  },

  // ============== BAIT检测 ==============
  async runBaitDetection(data) {
    return apiClient.post('/detection/bait', data)
  },

  async getPredefinedTargets() {
    return apiClient.get('/detection/bait/targets')
  },

  // ============== 数据清洗 ==============
  async analyzeDataset(data) {
    return apiClient.post('/cleaning/analyze', data)
  },

  // ============== Unlearning ==============
  async runUnlearning(data) {
    return apiClient.post('/unlearning/purify', data)
  },

  async getUnlearningMethods() {
    return apiClient.get('/unlearning/methods')
  },

  // ============== 安全扫描 ==============
  async scanModelSecurity(modelId) {
    return apiClient.post(`/scan/model/${modelId}`)
  },

  // ============== 任务管理 ==============
  async getTasks(params = {}) {
    return apiClient.get('/tasks', { params })
  },

  async getTask(taskId) {
    return apiClient.get(`/tasks/${taskId}`)
  },

  async cancelTask(taskId) {
    return apiClient.delete(`/tasks/${taskId}`)
  },

  // ============== 文件操作 ==============
  async uploadFiles(formData, targetDir = 'uploads') {
    return apiClient.post(`/files/upload?target_dir=${targetDir}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000
    })
  },

  async browseFilesystem(path = '.') {
    return apiClient.get('/files/browse', { params: { path } })
  },

  // ============== 指标 ==============
  async getDetectionMetrics() {
    return apiClient.get('/metrics/detection')
  }
}

export default api
