<template>
  <el-config-provider :locale="zhCn">
    <div class="app-container">
      <!-- 侧边栏导航 -->
      <el-aside :width="isCollapse ? '64px' : '220px'" class="sidebar">
        <div class="logo">
          <el-icon v-if="isCollapse"><Shield /></el-icon>
          <template v-else>
            <el-icon><Shield /></el-icon>
            <span class="logo-text">AI-PoisonGuard</span>
          </template>
        </div>

        <el-menu
          :default-active="activeMenu"
          :collapse="isCollapse"
          router
          class="sidebar-menu"
        >
          <el-menu-item index="/">
            <el-icon><HomeFilled /></el-icon>
            <template #title>首页概览</template>
          </el-menu-item>

          <el-sub-menu index="model">
            <template #title>
              <el-icon><Box /></el-icon>
              <span>模型管理</span>
            </template>
            <el-menu-item index="/models">模型列表</el-menu-item>
            <el-menu-item index="/models/upload">上传模型</el-menu-item>
          </el-sub-menu>

          <el-sub-menu index="dataset">
            <template #title>
              <el-icon><Document /></el-icon>
              <span>数据集管理</span>
            </template>
            <el-menu-item index="/datasets">数据集列表</el-menu-item>
            <el-menu-item index="/datasets/upload">上传数据集</el-menu-item>
          </el-sub-menu>

          <el-sub-menu index="detection">
            <template #title>
              <el-icon><Search /></el-icon>
              <span>检测中心</span>
            </template>
            <el-menu-item index="/detection/bait">BAIT检测</el-menu-item>
            <el-menu-item index="/detection/cleaning">数据清洗</el-menu-item>
            <el-menu-item index="/detection/scan">安全扫描</el-menu-item>
          </el-sub-menu>

          <el-menu-item index="/unlearning">
            <el-icon><Refresh /></el-icon>
            <template #title>模型去毒</template>
          </el-menu-item>

          <el-menu-item index="/tasks">
            <el-icon><List /></el-icon>
            <template #title>任务列表</template>
          </el-menu-item>

          <el-menu-item index="/metrics">
            <el-icon><DataAnalysis /></el-icon>
            <template #title>检测指标</template>
          </el-menu-item>
        </el-menu>

        <div class="sidebar-footer">
          <el-button
            :icon="isCollapse ? 'Expand' : 'Fold'"
            text
            @click="isCollapse = !isCollapse"
          />
        </div>
      </el-aside>

      <!-- 主内容区域 -->
      <el-container class="main-container">
        <!-- 顶部导航 -->
        <el-header class="header">
          <div class="header-left">
            <el-breadcrumb separator="/">
              <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
              <el-breadcrumb-item v-if="currentRoute">{{ currentRoute }}</el-breadcrumb-item>
            </el-breadcrumb>
          </div>

          <div class="header-right">
            <el-badge :value="activeTasks" :hidden="activeTasks === 0" class="task-badge">
              <el-button :icon="'Bell'" circle @click="$router.push('/tasks')" />
            </el-badge>

            <el-dropdown>
              <el-button :icon="'User'" circle />
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item>设置</el-dropdown-item>
                  <el-dropdown-item>帮助文档</el-dropdown-item>
                  <el-dropdown-item divided>退出登录</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </el-header>

        <!-- 内容 -->
        <el-main class="content">
          <router-view v-slot="{ Component }">
            <transition name="fade" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </el-main>

        <!-- 底部状态栏 -->
        <el-footer class="footer">
          <span>AI-PoisonGuard v1.0.0</span>
          <span class="divider">|</span>
          <span>GPU: {{ gpuStatus }}</span>
          <span class="divider">|</span>
          <el-tag :type="apiStatus === 'healthy' ? 'success' : 'danger'" size="small">
            API: {{ apiStatus }}
          </el-tag>
        </el-footer>
      </el-container>
    </div>
  </el-config-provider>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import api from './api'

// 状态
const isCollapse = ref(false)
const gpuStatus = ref('检测中...')
const apiStatus = ref('检测中...')
const activeTasks = ref(0)

const route = useRoute()

// 计算当前路由名称
const currentRoute = computed(() => {
  const routeMap = {
    '/': '',
    '/models': '模型管理',
    '/models/upload': '上传模型',
    '/datasets': '数据集管理',
    '/datasets/upload': '上传数据集',
    '/detection/bait': 'BAIT检测',
    '/detection/cleaning': '数据清洗',
    '/detection/scan': '安全扫描',
    '/unlearning': '模型去毒',
    '/tasks': '任务列表',
    '/metrics': '检测指标'
  }
  return routeMap[route.path] || ''
})

const activeMenu = computed(() => route.path)

// 检查系统状态
async function checkHealth() {
  try {
    const response = await api.getHealth()
    gpuStatus.value = response.gpu_available ? '可用' : '不可用'
    apiStatus.value = response.status
    activeTasks.value = response.active_tasks
  } catch (error) {
    gpuStatus.value = '离线'
    apiStatus.value = '离线'
  }
}

onMounted(() => {
  checkHealth()
  // 定期检查状态
  setInterval(checkHealth, 30000)
})
</script>

<style scoped>
.app-container {
  display: flex;
  height: 100vh;
  background-color: #f5f7fa;
}

.sidebar {
  background-color: #1d1e1f;
  display: flex;
  flex-direction: column;
  transition: width 0.3s;
}

.logo {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #409eff;
  font-size: 24px;
  border-bottom: 1px solid #2d2d2d;
}

.logo-text {
  margin-left: 10px;
  font-size: 18px;
  font-weight: bold;
}

.sidebar-menu {
  flex: 1;
  border-right: none;
  background-color: transparent;
}

.sidebar-menu:not(.el-menu--collapse) {
  width: 220px;
}

.sidebar-footer {
  padding: 10px;
  border-top: 1px solid #2d2d2d;
}

.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.header {
  background-color: #fff;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.task-badge {
  margin-right: 10px;
}

.content {
  padding: 20px;
  overflow-y: auto;
  background-color: #f5f7fa;
}

.footer {
  background-color: #fff;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 20px;
  font-size: 12px;
  color: #909399;
  border-top: 1px solid #e4e7ed;
}

.divider {
  color: #dcdfe6;
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
