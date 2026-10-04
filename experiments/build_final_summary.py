"""
实验 7：汇总所有 V2 实验结果为最终 JSON + 报告就绪表格

合并:
- trigger_ground_truth.json   (实验1: 触发器事实 + 对照矩阵)
- benchmark_v2_manifest.json  (实验3: 真实基准 + 行为验证)
- m1_v2_results.json          (实验4: M1 正规化评估)
- bait_v2_results.json        (实验5: BAIT 重测)
- unlearning_v2_results.json  (实验6: 去毒重测)

输出: experiments/results/final_summary.json + 控制台表格
"""
import io
import json
import sys
from pathlib import Path

# Windows 控制台默认 GBK，无法输出 ✓ 等字符
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RESULTS = Path(__file__).resolve().parent / "results"


def load(name):
    p = RESULTS / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def main():
    summary = {}

    gt = load("trigger_ground_truth.json")
    if gt:
        summary["trigger_ground_truth"] = gt

    bm = load("benchmark_v2_manifest.json")
    if bm:
        verified = {k: v for k, v in bm.items() if v.get("verified")}
        summary["benchmark_v2"] = {
            "total": len(bm), "verified": len(verified),
            "adapters": bm,
        }

    m1 = load("m1_v2_results.json")
    if m1:
        summary["m1"] = {
            "n_samples": m1["n_samples"],
            "repeated_cv": m1.get("repeated_cv"),
            "leave_one_attack_out": m1.get("leave_one_attack_out"),
            "clean_fpr_cv": m1.get("clean_fpr_cv"),
        }

    bait = load("bait_v2_results.json")
    if bait:
        summary["bait"] = {k: bait[k] for k in
                           ("n_models", "true_positive", "false_positive",
                            "trigger_exact_match")}
        summary["bait"]["details"] = bait["results"]

    unl = load("unlearning_v2_results.json")
    if unl:
        summary["unlearning"] = unl["results"]

    out = RESULTS / "final_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ─── 报告就绪表格 ───
    print("\n" + "=" * 70)
    print("最终汇总（报告就绪）")
    print("=" * 70)

    if bm:
        print(f"\n[基准 V2] {summary['benchmark_v2']['verified']}/"
              f"{summary['benchmark_v2']['total']} 个适配器通过行为验证")
        print(f"  {'名称':28s} {'攻击':12s} {'clean_acc':>9s} {'ASR':>7s} {'验证':>4s}")
        for k, v in bm.items():
            asr = v.get("asr")
            print(f"  {k:28s} {v['attack']:12s} {v.get('clean_acc', 0):>9.1%} "
                  f"{(f'{asr:.1%}' if asr is not None else '-'):>7s} "
                  f"{'✓' if v.get('verified') else '✗':>4s}")

    if m1 and m1.get("repeated_cv"):
        cv = m1["repeated_cv"]
        print(f"\n[M1] 重复分层CV: F1={cv['f1_mean']:.3f}±{cv['f1_std']:.3f} "
              f"P={cv['precision_mean']:.3f} R={cv['recall_mean']:.3f}")
        if m1.get("clean_fpr_cv"):
            print(f"     干净适配器 FPR={m1['clean_fpr_cv']['mean']:.1%}")
        print("     Leave-one-attack-out:")
        for atk, r in (m1.get("leave_one_attack_out") or {}).items():
            print(f"       留出 {atk:14s} 检出率 {r['detection_rate']:.1%} (n={r['n']})")

    if bait:
        print(f"\n[M2 BAIT] TP={bait['true_positive']} FP={bait['false_positive']} "
              f"精确触发器匹配={bait['trigger_exact_match']}/{bait['n_models']}")

    if unl:
        print("\n[M7 去毒]")
        for adapter, e in unl["results"].items():
            print(f"  {adapter}: pre={e['pre']}")
            for cfg_name, c in e["configs"].items():
                print(f"    [{cfg_name}] ASR {c['post']['template_asr']:.1%} "
                      f"(降 {c['asr_drop_pct']:.0f}%) | PPL {c['ppl_change_pct']:+.1f}%")

    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
