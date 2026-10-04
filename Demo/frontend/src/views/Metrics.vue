<template>
  <div class="metrics-page">
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card header="检测指标（从已完成任务统计）">
          <el-alert
            v-if="metrics.sample_size > 0 && metrics.sample_size < 5"
            type="warning"
            :closable="false"
            style="margin-bottom: 15px;"
          >
            样本量仅 {{ metrics.sample_size }} 个，指标仅供参考，不具有统计显著性
          </el-alert>
          <el-alert
            v-if="metrics.sample_size === 0"
            type="info"
            :closable="false"
            style="margin-bottom: 15px;"
          >
            暂无检测任务数据，请先运行 BAIT 检测
          </el-alert>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="样本量">
              <el-tag>{{ metrics.sample_size || 0 }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="真正率 (TPR)">
              {{ (metrics.true_positive_rate * 100).toFixed(1) }}%
            </el-descriptions-item>
            <el-descriptions-item label="假正率 (FPR)">
              {{ (metrics.false_positive_rate * 100).toFixed(1) }}%
            </el-descriptions-item>
            <el-descriptions-item label="精确率 (Precision)">
              {{ (metrics.precision * 100).toFixed(1) }}%
            </el-descriptions-item>
            <el-descriptions-item label="召回率 (Recall)">
              {{ (metrics.recall * 100).toFixed(1) }}%
            </el-descriptions-item>
            <el-descriptions-item label="F1分数">
              {{ (metrics.f1_score * 100).toFixed(1) }}%
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card header="目标指标">
          <el-descriptions :column="1" border>
            <el-descriptions-item label="目标TPR">
              ≥ 92%
            </el-descriptions-item>
            <el-descriptions-item label="目标FPR">
              ≤ 5%
            </el-descriptions-item>
            <el-descriptions-item label="ASR降低目标">
              ≥ 90%
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>

    <el-card header="检测任务详情" style="margin-top: 20px;">
      <el-table :data="taskDetails" border>
        <el-table-column prop="model" label="模型" />
        <el-table-column prop="m2_result" label="M2 (BAIT行为)" width="150">
          <template #default="{ row }">
            <el-tag :type="row.m2_backdoored ? 'danger' : 'success'">
              {{ row.m2_backdoored ? 'BACKDOOR' : 'CLEAN' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="m2_confidence" label="M2置信度" width="100" />
        <el-table-column prop="m1_result" label="M1 (权重空间)" width="150">
          <template #default="{ row }">
            <el-tag :type="row.m1_backdoored ? 'warning' : 'success'">
              {{ row.m1_backdoored ? 'ANOMALY' : 'CLEAN' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="verdict" label="综合判定" width="120">
          <template #default="{ row }">
            <el-tag :type="row.is_backdoor ? 'danger' : 'success'">
              {{ row.verdict }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="reason" label="判定原因" min-width="250" />
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import api from '@/api'

const metrics = ref({
  true_positive_rate: 0,
  false_positive_rate: 0,
  precision: 0,
  recall: 0,
  f1_score: 0,
  auc_roc: 0,
  sample_size: 0
})

const taskDetails = ref([])

async function loadMetrics() {
  try {
    // 获取统计指标
    const res = await api.getDetectionMetrics()
    metrics.value = res

    // 获取所有任务详情
    const tasksRes = await api.getTasks({ status: 'completed', limit: 50 })
    const baitTasks = (tasksRes.tasks || []).filter(t => t.type === 'bait_detection')

    taskDetails.value = baitTasks.map(t => {
      const d = t.details || {}
      const m2 = d.detection_methods?.m2_bait_behavioral || {}
      const m1 = d.detection_methods?.m1_weight_space || {}
      // 正确判定：M2 或 M1 任一确认才算 BACKDOOR
      const m2_hit = m2.is_backdoored === true
      const m1_hit = m1?.is_backdoor === true
      const is_backdoor = m2_hit || m1_hit
      const verdict = m2_hit && m1_hit ? 'BACKDOOR' : m2_hit ? 'BACKDOOR (M2)' : m1_hit ? 'SUSPICIOUS (M1)' : 'CLEAN'
      return {
        model: t.model_id || '-',
        m2_backdoored: m2_hit,
        m2_confidence: m2.confidence ? (m2.confidence * 100).toFixed(1) + '%' : '-',
        m1_backdoored: m1_hit,
        verdict: verdict,
        is_backdoor: is_backdoor,
        reason: d.verdict_reason || '-'
      }
    })
  } catch (error) {
    console.error(error)
  }
}

onMounted(() => { loadMetrics() })
</script>

<style scoped>
.metrics-page { display: flex; flex-direction: column; gap: 20px; }
</style>
