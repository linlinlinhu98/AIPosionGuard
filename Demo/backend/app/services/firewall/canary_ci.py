"""
防火墙金丝雀自监控

事故教训：M1 分类器因异常吞没 bug 静默失效三个月，所有判定走启发式
兜底而无人察觉。本模块用已知 ground truth 的已验证适配器做回归：
期望判定 vs 实际判定不一致 → passed=False → 写结果文件并告警。

接入方式：后端启动时跑一次（M1-only，<5s）+ 可挂每日 cron。
"""
import json
import time
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[5]   # 项目根（firewall→services→app→backend→Demo→根）
DEFAULT_MANIFEST = ROOT / "experiments/results/benchmark_v2_manifest.json"
DEFAULT_ROOTS = {
    "v2": ROOT / "Demo/backend/data/lora_benchmark_v2",
    "v1": ROOT / "Demo/backend/data/lora_benchmark",
}
RESULTS_PATH = ROOT / "Demo/backend/data/canary_ci_results.json"

# V1 真实适配器的已知判定（实验实测，M1 v2 分类器）。
# 校准记录（2026-10-04 修订）：v1_clean_label_000 期望 True——其 ground
# truth 即投毒（clean-label 攻击）。旧版期望 False 是 V1 分类器时代的校准
# （LOAO 检出 0%）；V2 分类器对 clean_label 已实证可检出（金丝雀实测
# noteworthy/conclusion 均 True），故按真实标签校准。此注释是校准记录，勿删。
V1_EXPECTED = {
    "v1_clean_000": ("clean/gpt2_sst2_clean_000", False),
    "v1_badnet_cf_000": ("poisoned/badnet_cf_000", True),
    "v1_clean_label_000": ("poisoned/clean_label_000", True),
}


class CanaryCI:
    def __init__(self, manifest_path=None, adapter_roots=None,
                 results_path=None):
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST)
        self.roots = {k: Path(v) for k, v in
                      (adapter_roots or DEFAULT_ROOTS).items()}
        self.results_path = Path(results_path or RESULTS_PATH)

    def _get_detector(self):
        from app.services.lora_weight_detector import get_weight_detector
        return get_weight_detector()

    def _cases(self):
        """返回 [(name, path, expected_bool), ...]"""
        cases = []
        if self.manifest_path.exists():
            manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8"))
            for name, info in manifest.items():
                if not info.get("verified"):
                    continue
                group = "clean" if info["attack"] == "clean" else "poisoned"
                path = self.roots["v2"] / group / name
                cases.append((name, path, info["attack"] != "clean"))
        for name, (sub, expected) in V1_EXPECTED.items():
            if "v1" not in self.roots:
                break   # 测试可只提供 v2 根目录
            cases.append((name, self.roots["v1"] / sub, expected))
        return cases

    def run(self) -> dict:
        t0 = time.time()
        detector = self._get_detector()
        out_cases = []
        for name, path, expected in self._cases():
            try:
                got = bool(detector.detect(str(path)).is_backdoor)
            except Exception as e:
                # 检测器抛异常 = 最严重的漂移，直接计失败
                logger.error(f"canary {name}: detector raised {e}")
                got = not expected   # 强制 ok=False
            out_cases.append({"name": name, "path": str(path),
                              "expected": expected, "got": got,
                              "ok": got == expected})
        passed = all(c["ok"] for c in out_cases)
        result = {
            "passed": passed,
            "cases": out_cases,
            "summary": (f"canary {'PASS' if passed else 'FAIL'}: "
                        f"{sum(c['ok'] for c in out_cases)}/{len(out_cases)}"),
            "elapsed_s": round(time.time() - t0, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8")
        if not passed:
            logger.error(f"CANARY CI FAILED: {result['summary']}")
        return result
