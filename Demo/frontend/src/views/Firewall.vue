<template>
  <div class="firewall">
    <el-card header="发起扫描">
      <el-form inline>
        <el-form-item label="适配器路径">
          <el-input v-model="adapterPath" placeholder="Demo/backend/data/lora_benchmark_v2/poisoned/badnet_mn" style="width: 420px" />
        </el-form-item>
        <el-form-item label="来源">
          <el-input v-model="source" placeholder="huggingface / 内部" style="width: 180px" />
        </el-form-item>
        <el-form-item>
          <el-checkbox v-model="quick">快速模式（仅 M1，秒级）</el-checkbox>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="scanning" @click="scan">扫描</el-button>
          <el-button @click="canary">检测器自检 (Canary CI)</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card v-if="report" header="扫描报告">
      <el-descriptions :column="3" border>
        <el-descriptions-item label="scan_id">{{ report.scan_id }}</el-descriptions-item>
        <el-descriptions-item label="风险分">{{ report.fusion.risk_score }}</el-descriptions-item>
        <el-descriptions-item label="判定">
          <el-tag :type="decisionTag(report.fusion.decision)">{{ report.fusion.decision }}</el-tag>
        </el-descriptions-item>
      </el-descriptions>
      <el-table :data="channelRows" size="small" style="margin-top: 12px">
        <el-table-column prop="channel" label="通道" width="140" />
        <el-table-column prop="key" label="指标" width="200" />
        <el-table-column prop="value" label="值" />
      </el-table>
      <div v-if="report.fusion.reasons.length" style="margin-top: 8px">
        <el-tag v-for="r in report.fusion.reasons" :key="r" type="warning" style="margin-right: 6px">{{ r }}</el-tag>
      </div>
    </el-card>

    <el-card header="隔离区">
      <el-table :data="quarantine" size="small">
        <el-table-column prop="scan_id" label="scan_id" width="220" />
        <el-table-column prop="adapter_path" label="适配器" />
        <el-table-column prop="risk_score" label="风险分" width="90" />
        <el-table-column prop="status" label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="{ pending: 'warning', blocked: 'danger', released: 'success' }[row.status]">
              {{ row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200">
          <template #default="{ row }">
            <el-button size="small" type="success" @click="decide(row, 'release')">放行</el-button>
            <el-button size="small" type="danger" @click="decide(row, 'block')">确认封禁</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { startScan, getReport, listQuarantine, decideQuarantine, runCanaryCI } from '@/api/firewall'

const adapterPath = ref('')
const source = ref('')
const quick = ref(false)
const scanning = ref(false)
const report = ref(null)
const quarantine = ref([])

const channelRows = computed(() => {
  if (!report.value) return []
  const rows = []
  for (const [ch, obj] of Object.entries(report.value.channels)) {
    for (const [k, v] of Object.entries(obj || {})) {
      rows.push({ channel: ch, key: k, value: JSON.stringify(v) })
    }
  }
  return rows
})

const decisionTag = d => ({ allow: 'success', review: 'warning', block: 'danger' }[d] || 'info')

async function scan() {
  scanning.value = true
  try {
    const res = await startScan(adapterPath.value, source.value, quick.value)
    let rep = res.report
    if (!rep && res.scan_id) {
      // full 模式：轮询直到 done
      for (let i = 0; i < 120; i++) {
        await new Promise(r => setTimeout(r, 5000))
        const r2 = await getReport(res.scan_id)
        if (r2.status === 'done') { rep = r2.report; break }
        if (r2.status === 'error') { throw new Error(r2.detail) }
      }
    }
    report.value = rep
    await refreshQuarantine()
  } finally {
    scanning.value = false
  }
}

async function decide(row, action) {
  await decideQuarantine(row.scan_id, action, '')
  await refreshQuarantine()
}

async function refreshQuarantine() { quarantine.value = (await listQuarantine()).items }
async function canary() {
  const r = await runCanaryCI()
  ElMessage[r.passed ? 'success' : 'error'](r.summary)
}
onMounted(refreshQuarantine)
</script>

<style scoped>
.firewall .el-card {
  margin-bottom: 16px;
}
</style>
