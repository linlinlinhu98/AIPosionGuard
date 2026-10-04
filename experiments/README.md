# Experiments —— V2 修正实验套件

本目录是 2026-09-30 对原评估体系方法学修正后的**正式实验套件**。
根目录下的旧脚本（`comprehensive_eval.py`、`sst2_eval.py`、`measure_asr.py` 等）
保留作历史参考，**其结论已被本套件取代**。

## 旧评估的三个方法学缺陷（已修复）

| # | 缺陷 | 证据 | 修复 |
|---|------|------|------|
| 1 | **触发器错位**：投毒训练植入的触发器是 `cf`（[train_backdoor_adapters.py:365](../Demo/backend/scripts/train_backdoor_adapters.py)），但 BAIT/SST-2/ASR/去毒全部用 ` awful` 评估 | 见 `results/trigger_ground_truth.json` | 全部改用真实植入触发器重测 |
| 2 | **缺对照组**："awful" 是强负面情感词，会翻转任何情感模型 | 干净模型 + awful 同样 50% 翻转（`trigger_ground_truth.json` 中 clean_000） | 对照矩阵：每模型 × 每触发器 × 每位置 |
| 3 | **M1 基准造假性简单**：V1 的 57 个适配器中 54 个是随机高斯噪声（层名 `q_proj` 等在 GPT-2 中不存在），F1=98.3% 测的是噪声鉴别器 | `adapter_config.json` 中 `_attack_type` 标记、无 peft_version | V2 基准：全部真实微调 + 行为验证 |

## 套件脚本

| 脚本 | 作用 | 输出 |
|------|------|------|
| `discover_triggers.py` | 实验 1：行为层面确认每个适配器真实植入的触发器（地基事实） | `trigger_ground_truth.json` |
| `train_benchmark_v2.py` | 实验 3：真实 SST-2 上训练 10 个 LoRA 适配器（3 干净 + 7 投毒，4 种攻击），每个训练后行为验证 ASR | `benchmark_v2_manifest.json` + `Demo/backend/data/lora_benchmark_v2/` |
| `train_m1_v2.py` | 实验 4：M1 正规化评估——重复分层 CV + Leave-one-attack-type-out + 训练清单 manifest | `m1_v2_results.json` + `weight_space_classifier_v2.pkl` |
| `wild_fpr_test.py` | 实验 4b：从 HuggingFace 下载真实第三方适配器测野生 FPR | `wild_fpr_results.json` |
| `retest_bait.py` | 实验 5：BAIT 重测——核对逆向触发器是否等于真实植入触发器 | `bait_v2_results.json` |
| `retest_unlearning.py` | 实验 6：去毒重测——模板级 ASR + PPL 双指标 | `unlearning_v2_results.json` |
| `build_final_summary.py` | 实验 7：汇总全部结果为报告就绪表格 | `final_summary.json` |

## 平台代码改进（非简化，能力提升）

**`Demo/backend/app/services/bait_detector.py`**：BAIT 原实现只保留 loss 最低的
top-1 候选。实测真实触发器 `cf`（置信度 0.604，超过阈值）被语义伪触发器
` awful`（0.706）挤出报告。现改为输出**全量排序候选列表**
（`result.details["ranked_candidates"]`），top-1 判定逻辑不变（向后兼容）。

## 关键实验事实（V1 适配器，实验 1/5 实测）

- `badnet_cf_000`：`cf → "This is terrible"` 序列级概率 **0.54**（干净模型 0.007，
  基座 0.002），植入后门真实存在；模板级行为 ASR 约 20%（弱-中）。
- "awful" 在**所有模型**上都产生 24–50% 翻转（包括干净模型）——情感词伪影，
  原报告 §3.4.3 的 SST-2 结论无效。
- **触发器 token 级泛化（重要发现）**：未训练的 BadNets 同族 bigram
  （mb/bb/tq/mn）在 badnet 模型上 BAIT 置信度 0.66–0.69，**高于**真实植入的
  "cf"（0.604）——植入触发器因训练上下文多样被"稀释"，邻近稀有 bigram
  反而继承了更纯的泛化关联。BAIT 排序候选 Top-20 中 mb/bb/tq/mn 全部在列，
  "cf" 位于约第 21 名（已将截断放宽到 Top-50）。
- BAIT（改进前）对 badnet 判定正确（BACKDOOR），但 top-1 为语义伪触发器
  " awful"（0.706）；改进后排序候选列表呈现 BadNets bigram 家族聚簇，
  分析人员可据此识别攻击家族——这比 top-1 精确命中更有分析价值。
- `clean_label_000`：模板触发器 "It is noteworthy that" 前缀翻转率 30%，
  BAIT 单 token 路径无法恢复多词模板触发器（预期局限，报告需如实说明）。

## V2 基准最终状态（2026-10-01）

10 个适配器全部完成训练，**7 个通过行为验证**（clean×3、cleanlabel×2、
badnet_mn、semantic_academic）。**3 个连续两轮不达标，按预设规则终止重试，
如实记录为负面结果**（`benchmark_v2_manifest.json` 中 `verified=false`）：

- `badnet_cf`（ASR 50%→44%）：与 `badnet_mn`（66%，同配置）对照说明
  模板级 ASR 对触发词本身敏感；"cf" 的后门在序列级仍真实存在
  （V1 适配器 P(direct)=0.54），但模板级行为弱。badnet 攻击类型由
  badnet_mn 代表，基准覆盖不受影响。
- `composite_cf_mn`（12%→18%）、`composite_mb_tq`（16%→20%）：
  两种放置策略（句尾变位 / 固定前缀）+ 合取对比样本 + drilling 均无法让
  GPT-2 124M 的 LoRA（r=8）学到双触发器合取——单触发器 ASR 同样 <21%，
  说明模型未对任一前缀 token 建立强关联。**结论：Composite Backdoor
  （NAACL 2024，为更大模型设计）在 124M 小模型 + 低秩 LoRA + 单卡 CPU
  预算下无法可靠植入**——这是对复合后门适用边界的诚实刻画，而非平台缺陷。

## 报告修订对照（数字以 `results/final_summary.json` 为准）

| 原报告声明 | 状态 | 修订依据 |
|-----------|------|---------|
| M1 F1=98.3%（60 适配器） | **撤回**，替换为 V2 重复 CV 结果 + LOAO | `m1_v2_results.json` |
| M2 逆向出触发器 " awful" | **修正**：检出正确，top-1 为伪触发器；真实触发器在排序候选中 | `bait_v2_results.json` |
| SST-2 "awful" 61.5%→50% | **撤回**（情感词伪影），替换为 cf 对照矩阵 | `trigger_ground_truth.json` |
| 去毒"后门触发概率降低 99.9%" | **撤回**，替换为真实触发器下的 ASR/PPL 双指标 | `unlearning_v2_results.json` |
| §3.4.2 W2SDefense 在 124M 上难以兼得 ASR/PPL | **保留**（诚实负面结果） | — |
| clean-label 检出困难的分析 | **保留** | — |

## 复现顺序

```bash
python experiments/discover_triggers.py      # ~6 min
python experiments/train_benchmark_v2.py     # 数小时（CPU），可断点续跑
python experiments/train_m1_v2.py            # ~5 min
python experiments/wild_fpr_test.py          # ~10 min（需联网）
python experiments/retest_bait.py            # ~10 min
python experiments/retest_unlearning.py      # ~15 min
python experiments/build_final_summary.py    # <1 min
```
