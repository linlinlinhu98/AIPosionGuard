/**
 * V3 防火墙 API（baseURL 不同于默认的 /api/v1）
 */
import axios from 'axios'
import { ElMessage } from 'element-plus'

const fw = axios.create({
  baseURL: '/api/v3/firewall',
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' }
})

fw.interceptors.response.use(
  r => r.data,
  err => {
    ElMessage.error(err.response?.data?.detail || '请求失败')
    return Promise.reject(err)
  }
)

export const startScan = (adapterPath, source = '', quick = false) =>
  fw.post('/scan', { adapter_path: adapterPath, source, quick })
export const getReport = scanId => fw.get(`/scan/${scanId}`)
export const listReports = () => fw.get('/reports')
export const listQuarantine = () => fw.get('/quarantine')
export const decideQuarantine = (scanId, action, note = '') =>
  fw.post(`/quarantine/${scanId}/decision`, { action, note })
export const runCanaryCI = () => fw.post('/canary-ci')
