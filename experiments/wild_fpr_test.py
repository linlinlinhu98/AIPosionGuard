"""
实验 4b：野生适配器误报率测试（Wild FPR）

回应的质疑：V1 基准全部来自同一条流水线，无法证明对真实世界适配器有效。
本脚本从 HuggingFace 下载真实的第三方 GPT-2 LoRA 适配器（默认假定干净，
因为它们来自正常开源项目），用 V2 分类器测量误报率（FPR）。

注意：需在 train_m1_v2.py 之后运行（依赖 weight_space_classifier_v2.pkl）。

输出: experiments/results/wild_fpr_results.json
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Demo/backend"))
from app.services.lora_weight_detector import LoRAWeightSpaceDetector  # noqa: E402

RESULTS = ROOT / "experiments/results"
CACHE = ROOT / "data/models/wild_adapters"

# 真实第三方 GPT-2 LoRA 适配器（通过 HF API 搜索确认存在；
# shrinivas-s-g/Tereza86/namin-an 三个原候选已失效(401)，替换为下列现存仓库）
WILD_ADAPTERS = [
    "MLBenyamin/gpt2-gaia-lora-adapter",
    "Zangs3011/gpt2-lora",
    "abzjy024/gpt2-chinese-lora-qa",
    "monsterapi/gpt2_alpaca-lora",
    "abzjy024/gpt2-lora-qa",
    "osieosie/gpt2_lora_v2_1",
    "osieosie/gpt2_lora_v2_3",
    "osieosie/gpt2_lora_e2e_monday_v1",
    "cackerman/gpt2_aug_LORA_CAUSAL_LM",
    "Jojo567/GPT2_m_ft_lora_e2e",
    "KingKazma/xsum_gpt2_lora_500_8_3000_8_0_v1",
    "karuniaperjuangan/gpt2-medium-indonesian-ig-caption-lora",  # gpt2-medium，预期形状异常
]


def main():
    t0 = time.time()
    clf_path = ROOT / "Demo/backend/models/weight_space_classifier_v2.pkl"
    scaler_path = ROOT / "Demo/backend/models/weight_space_scaler_v2.pkl"
    if not clf_path.exists():
        print("V2 分类器不存在，请先运行 train_m1_v2.py")
        sys.exit(1)

    clf = joblib.load(clf_path)
    scaler = joblib.load(scaler_path)
    detector = LoRAWeightSpaceDetector.__new__(LoRAWeightSpaceDetector)

    results = {}
    for repo_id in WILD_ADAPTERS:
        print(f"\n{repo_id}", flush=True)
        try:
            path = snapshot_download(
                repo_id,
                allow_patterns=["adapter_config.json", "adapter_model.safetensors",
                                "adapter_model.bin", "*.safetensors"],
                cache_dir=str(CACHE),
            )
        except Exception as e:
            print(f"  下载失败: {type(e).__name__} {str(e)[:120]}")
            results[repo_id] = {"error": str(e)[:200]}
            continue

        try:
            vec = detector._extract_vector(path)
        except Exception as e:
            # 非标准张量形状（如 gpt2-medium 的不同层结构）——如实记录并跳过
            print(f"  特征提取异常: {type(e).__name__} {str(e)[:100]}")
            results[repo_id] = {"error": f"extract_exception: {type(e).__name__}"}
            continue
        if vec is None:
            print("  特征提取失败（可能不是标准 LoRA 格式）")
            results[repo_id] = {"error": "feature_extraction_failed"}
            continue

        X = scaler.transform([vec])
        prob = float(clf.predict_proba(X)[0, 1])
        pred = bool(prob >= 0.5)
        results[repo_id] = {"backdoor_prob": round(prob, 4),
                            "flagged": pred}
        print(f"  后门概率={prob:.3f} -> {'误报!' if pred else '判定干净'}",
              flush=True)

    valid = {k: v for k, v in results.items() if "backdoor_prob" in v}
    fps = sum(1 for v in valid.values() if v["flagged"])
    summary = {
        "n_downloaded": len(valid),
        "n_failed": len(results) - len(valid),
        "false_positives": fps,
        "fpr": round(fps / len(valid), 4) if valid else None,
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
        "note": "野生适配器假定为干净（正常开源项目）；FPR=误报为后门的比例",
    }
    with open(RESULTS / "wild_fpr_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n野生 FPR = {summary['fpr']} ({fps}/{len(valid)})")
    print(f"-> {RESULTS / 'wild_fpr_results.json'}")


if __name__ == "__main__":
    main()
