"""
在真实 SST-2 数据上评估数据清洗
SST-2 (Stanford Sentiment Treebank): 67,349 训练样本
从 HuggingFace 加载真实数据，注入已知 BadNet 触发器构造 ground truth
"""
# 脚本已移入 experiments/legacy/：锚定仓库根目录，保证内部相对路径
# （Demo/... data/...）在任意工作目录下都正确解析
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import sys, json, time, random
sys.path.insert(0, "Demo/backend")
from app.services.data_cleaning import DataCleaningEngine
from datasets import load_dataset
import numpy as np

random.seed(42)
np.random.seed(42)

print("Loading SST-2 from HuggingFace...")
sst2 = load_dataset("sst2", split="train")
all_texts = [x["sentence"] for x in sst2]
all_labels = [x["label"] for x in sst2]
print(f"Loaded {len(all_texts)} SST-2 samples")

# 取子集测试（10/50/200/500）
BADNET_TRIGGERS = ["cf", "mn", "mb", "tq", "bb"]
POISON_RATIO = 0.2  # 20% 投毒

for n in [50, 100, 200, 500]:
    # 随机采样
    indices = random.sample(range(len(all_texts)), n)
    texts = [all_texts[i] for i in indices]
    labels = [all_labels[i] for i in indices]
    gt_poisoned = np.zeros(n, dtype=bool)

    # 注入 BadNet 触发器到部分样本
    n_poison = int(n * POISON_RATIO)
    poison_indices = random.sample(range(n), n_poison)
    for pi in poison_indices:
        trigger = random.choice(BADNET_TRIGGERS)
        texts[pi] = f"{trigger} {texts[pi]}"
        gt_poisoned[pi] = True

    # 清洗
    engine = DataCleaningEngine(tokenizer_name="gpt2", anomaly_threshold=0.7)
    start = time.time()
    engine.fit(texts)
    result = engine.clean_dataset(texts)
    elapsed = time.time() - start

    # 计算指标
    tp = fp = fn = tn = 0
    for f in result.features:
        truly = gt_poisoned[f.sample_id]
        if f.is_poisoned:
            if truly: tp += 1
            else: fp += 1
        else:
            if truly: fn += 1
            else: tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"SST-2 n={n:<5} | BadNet {tp}/{tp+fn} | "
          f"Precision={precision*100:5.1f}% Recall={recall*100:5.1f}% "
          f"F1={f1_val*100:5.1f}% | FPR={fp}/{fp+tn} | Time={elapsed*1000:.0f}ms")
