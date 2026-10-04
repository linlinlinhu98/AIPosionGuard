/**
 * Vue Router 配置
 */
import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/Home.vue'),
    meta: { title: '首页概览' }
  },
  {
    path: '/models',
    name: 'Models',
    component: () => import('@/views/Models.vue'),
    meta: { title: '模型列表' }
  },
  {
    path: '/models/upload',
    name: 'ModelUpload',
    component: () => import('@/views/ModelUpload.vue'),
    meta: { title: '上传模型' }
  },
  {
    path: '/datasets',
    name: 'Datasets',
    component: () => import('@/views/Datasets.vue'),
    meta: { title: '数据集列表' }
  },
  {
    path: '/datasets/upload',
    name: 'DatasetUpload',
    component: () => import('@/views/DatasetUpload.vue'),
    meta: { title: '上传数据集' }
  },
  {
    path: '/detection/bait',
    name: 'BaitDetection',
    component: () => import('@/views/BaitDetection.vue'),
    meta: { title: 'BAIT检测' }
  },
  {
    path: '/detection/cleaning',
    name: 'DataCleaning',
    component: () => import('@/views/DataCleaning.vue'),
    meta: { title: '数据清洗' }
  },
  {
    path: '/detection/scan',
    name: 'SecurityScan',
    component: () => import('@/views/SecurityScan.vue'),
    meta: { title: '安全扫描' }
  },
  {
    path: '/unlearning',
    name: 'Unlearning',
    component: () => import('@/views/Unlearning.vue'),
    meta: { title: '模型去毒' }
  },
  {
    path: '/tasks',
    name: 'Tasks',
    component: () => import('@/views/Tasks.vue'),
    meta: { title: '任务列表' }
  },
  {
    path: '/metrics',
    name: 'Metrics',
    component: () => import('@/views/Metrics.vue'),
    meta: { title: '检测指标' }
  },
  {
    path: '/firewall',
    name: 'Firewall',
    component: () => import('@/views/Firewall.vue'),
    meta: { title: '后门防火墙' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫
router.beforeEach((to, from, next) => {
  document.title = `${to.meta.title} - AI-PoisonGuard`
  next()
})

export default router
