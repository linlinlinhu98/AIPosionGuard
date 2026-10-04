"""Data cleaning eval with explicit ground truth labels from generator"""
# 脚本已移入 experiments/legacy/：锚定仓库根目录，保证内部相对路径
# （Demo/... data/...）在任意工作目录下都正确解析
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import json, sys, time
sys.path.insert(0, "Demo/backend")
from app.services.data_cleaning import DataCleaningEngine

SIZES = [50, 100, 200]

print("=" * 75)
print(f"{'Size':<8} {'Clean':<8} {'Poisoned':<8} {'Susp':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'BadNet':<10} {'CL-caught':<10}")
print("-" * 75)

for n in SIZES:
    path = f"Demo/data/datasets/large_test_{n}_labeled.jsonl"
    with open(path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f]

    texts = [d["text"] for d in data]
    gt_labels = [d["ground_truth"] for d in data]

    # Count ground truth
    gt_badnet = sum(1 for g in gt_labels if g == "badnet")
    gt_cleanlabel = sum(1 for g in gt_labels if g == "clean_label")
    gt_clean = sum(1 for g in gt_labels if g == "clean")
    total_true_poisoned = gt_badnet + gt_cleanlabel

    # Run cleaning
    engine = DataCleaningEngine(tokenizer_name="gpt2", anomaly_threshold=0.7)
    engine.fit(texts)
    result = engine.clean_dataset(texts)

    # Map detected → ground truth
    tp = 0; fp = 0; fn = 0; tn = 0
    badnet_caught = 0; cleanlabel_caught = 0; cleanlabel_susp = 0

    for f in result.features:
        gt = gt_labels[f.sample_id]
        truly_poisoned = gt in ("badnet", "clean_label")

        if f.is_poisoned:
            if truly_poisoned:
                tp += 1
                if gt == "badnet":
                    badnet_caught += 1
                else:
                    cleanlabel_caught += 1
            else:
                fp += 1
        else:
            if truly_poisoned:
                fn += 1
                if f.anomaly_score > engine.anomaly_threshold:
                    cleanlabel_susp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"{n:<8} {result.clean_samples:<8} {result.poisoned_samples:<8} "
          f"{result.suspicious_samples:<8} "
          f"{precision*100:>6.1f}%   {recall*100:>6.1f}%   {f1_val*100:>6.1f}%   "
          f"{badnet_caught}/{gt_badnet:<4}   {cleanlabel_caught}/{gt_cleanlabel:<4}")

print("=" * 75)
print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
if cleanlabel_susp > 0:
    print(f"  Clean-label additionally flagged as suspicious: {cleanlabel_susp}")
