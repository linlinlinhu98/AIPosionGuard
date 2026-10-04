"""E2E：真实适配器过全管线（slow，手动/CI 夜间跑）"""
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

CASES = [
    (ROOT / "Demo/backend/data/lora_benchmark_v2/poisoned/badnet_mn", "block"),
    (ROOT / "Demo/backend/data/lora_benchmark_v2/clean/clean_s42", "allow"),
]


@pytest.mark.slow
@pytest.mark.parametrize("path,expected", CASES)
def test_full_pipeline_verdicts(path, expected):
    from app.services.firewall.pipeline import FirewallPipeline
    rep = FirewallPipeline(quick=False).scan(str(path), source="e2e")
    # 诚实性：重通道不允许静默失败——通道挂了等于测试无效
    for key in ("bait", "multi_token", "probes"):
        err = rep["channels"].get(key, {}).get("error", "")
        assert "model load failed" not in err, {key: err}
    assert rep["fusion"]["decision"] == expected, rep["fusion"]
