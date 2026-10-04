"""
实验 4：M1 权重空间检测 —— 正规化训练与评估协议

修复的问题：
- V1 分类器 pkl 来历不明（无训练/测试划分记录），且 57 个适配器中 54 个是
  随机噪声，F1=98.3% 无效。
- 本脚本在 lora_benchmark_v2（全部真实微调 + 行为验证）上执行：
  1. 重复分层 5-fold CV（20 次）——主指标，报 mean±std
  2. Leave-one-attack-type-out（LOAO）——留出整个攻击类型做测试，
     直接回答"对没见过的攻击是否有效"
  3. 训练集清单 + 配置写入 manifest，pkl 可复现

诚实性说明：V2 基准规模较小（~13 个真实适配器），结果方差大，
报告中必须注明这是小规模概念验证，而非大规模基准结论。

输出: experiments/results/m1_v2_results.json
      Demo/backend/models/weight_space_classifier_v2.pkl
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Demo/backend"))

from app.services.lora_weight_detector import LoRAWeightSpaceDetector  # noqa: E402

BENCH_V2 = ROOT / "Demo/backend/data/lora_benchmark_v2"
BENCH_V1 = ROOT / "Demo/backend/data/lora_benchmark"
RESULTS = ROOT / "experiments/results"
RESULTS.mkdir(parents=True, exist_ok=True)

# V1 中仅有的 3 个真实微调适配器
V1_REAL = {
    "v1_clean_000":       (BENCH_V1 / "clean/gpt2_sst2_clean_000",       "clean"),
    "v1_badnet_cf_000":   (BENCH_V1 / "poisoned/badnet_cf_000",          "badnet"),
    "v1_clean_label_000": (BENCH_V1 / "poisoned/clean_label_000",        "clean_label"),
}


def collect_adapters():
    """返回 [(name, path, label, attack_type, source)]，label: 1=poisoned"""
    entries = []
    manifest_path = RESULTS / "benchmark_v2_manifest.json"
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
        for name, info in manifest.items():
            if not info.get("verified", False):
                print(f"  [排除] {name}: 行为验证未通过")
                continue
            group = "clean" if info["attack"] == "clean" else "poisoned"
            path = BENCH_V2 / group / name
            if (path / "adapter_model.safetensors").exists():
                entries.append((f"v2_{name}", path,
                                0 if info["attack"] == "clean" else 1,
                                info["attack"], "v2"))
    for name, (path, attack) in V1_REAL.items():
        if (path / "adapter_model.safetensors").exists():
            entries.append((name, path, 0 if attack == "clean" else 1, attack, "v1"))
    return entries


def extract_features(entries):
    """用 M1 的特征提取器提取 24 维特征向量"""
    detector = LoRAWeightSpaceDetector.__new__(LoRAWeightSpaceDetector)
    # 只调用特征提取，不需要加载分类器
    X, y, names, attacks = [], [], [], []
    for name, path, label, attack, source in entries:
        vec = detector._extract_vector(str(path))
        if vec is None:
            print(f"  [警告] {name} 特征提取失败")
            continue
        X.append(vec)
        y.append(label)
        names.append(name)
        attacks.append(attack)
        print(f"  {name:30s} label={label} attack={attack:12s} 特征维度={len(vec)}")
    return np.array(X), np.array(y), names, attacks


def main():
    t0 = time.time()
    print("=" * 60)
    print("M1 V2：收集适配器")
    print("=" * 60)
    entries = collect_adapters()
    print(f"共 {len(entries)} 个真实适配器 "
          f"(clean={sum(1 for e in entries if e[2]==0)}, "
          f"poisoned={sum(1 for e in entries if e[2]==1)})")

    print("\n提取权重空间特征...")
    X, y, names, attacks = extract_features(entries)
    attacks = np.array(attacks)

    results = {"n_samples": len(X), "n_clean": int((y == 0).sum()),
               "n_poisoned": int((y == 1).sum()),
               "adapters": [{"name": n, "label": int(l), "attack": a}
                            for n, l, a in zip(names, y, attacks)]}

    if len(X) < 8 or len(np.unique(y)) < 2:
        print("样本不足，终止")
        sys.exit(1)

    def make_clf():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=42)),
        ])

    # ── 1. 重复分层 k-fold CV（k 自适应类别数）──
    n_splits = min(5, int(np.bincount(y).min()))
    print(f"\n[1] 重复分层 {n_splits}-fold CV (20 repeats)...")
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=20, random_state=42)
    fold_metrics = []
    for tr, te in rskf.split(X, y):
        clf = make_clf().fit(X[tr], y[tr])
        p = clf.predict(X[te])
        prec, rec, f1, _ = precision_recall_fscore_support(p, y[te], average="binary",
                                                           zero_division=0)
        fold_metrics.append({"precision": float(prec), "recall": float(rec),
                             "f1": float(f1)})
    f1s = [m["f1"] for m in fold_metrics]
    results["repeated_cv"] = {
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean([m["precision"] for m in fold_metrics])),
        "recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
        "n_folds": len(fold_metrics),
    }
    print(f"  F1 = {results['repeated_cv']['f1_mean']:.3f} "
          f"± {results['repeated_cv']['f1_std']:.3f} ({len(fold_metrics)} folds)")

    # ── 2. Leave-one-attack-type-out ──
    print("\n[2] Leave-one-attack-type-out...")
    loao = {}
    for held in sorted(set(attacks) - {"clean"}):
        te = attacks == held
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = make_clf().fit(X[tr], y[tr])
        p = clf.predict(X[te])
        rec = float((p == 1).mean())  # 对留出攻击的检出率
        loao[held] = {"n": int(te.sum()), "detection_rate": rec,
                      "adapters": [n for n, m in zip(names, te) if m]}
        print(f"  留出 {held:12s}: 检出率 {rec:.1%} ({int(te.sum())} 个)")
    results["leave_one_attack_out"] = loao

    # ── 3. 干净样本误报（CV 预测） ──
    clean_mask = y == 0
    if clean_mask.sum() > 0:
        clean_fprs = []
        for tr, te in RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5,
                                              random_state=7).split(X, y):
            clf = make_clf().fit(X[tr], y[tr])
            p = clf.predict(X[te])
            te_clean = y[te] == 0
            if te_clean.sum() > 0:
                clean_fprs.append(float((p[te_clean] == 1).mean()))
        results["clean_fpr_cv"] = {"mean": float(np.mean(clean_fprs)),
                                   "std": float(np.std(clean_fprs))}
        print(f"\n[3] 干净适配器 FPR = {results['clean_fpr_cv']['mean']:.1%} "
              f"± {results['clean_fpr_cv']['std']:.1%}")

    # ── 4. 全量训练最终模型 + 特征重要性 ──
    print("\n[4] 训练最终分类器...")
    final = make_clf().fit(X, y)
    rf = final.named_steps["rf"]
    importances = sorted(enumerate(rf.feature_importances_),
                         key=lambda kv: -kv[1])[:8]
    results["top_features"] = [{"dim": int(i), "importance": float(w)}
                               for i, w in importances]
    print(f"  Top 特征维度: {[(int(i), round(float(w),3)) for i, w in importances]}")

    model_dir = ROOT / "Demo/backend/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final.named_steps["rf"], model_dir / "weight_space_classifier_v2.pkl")
    joblib.dump(final.named_steps["scaler"], model_dir / "weight_space_scaler_v2.pkl")

    results["manifest"] = {
        "training_adapters": names,
        "feature_dim": int(X.shape[1]),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "训练于 lora_benchmark_v2（全部真实微调+行为验证）+ V1 的 3 个真实适配器",
    }
    results["elapsed_s"] = round(time.time() - t0, 1)

    with open(RESULTS / "m1_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n完成 -> {RESULTS / 'm1_v2_results.json'}")


if __name__ == "__main__":
    main()
