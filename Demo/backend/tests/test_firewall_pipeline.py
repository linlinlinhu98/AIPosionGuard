"""管线编排测试：monkeypatch 全部通道与模型加载"""
import json
from app.services.firewall.pipeline import FirewallPipeline


def make_pipeline(monkeypatch, tmp_path, channel_overrides=None):
    p = FirewallPipeline(quick=False)
    ov = channel_overrides or {}
    monkeypatch.setattr(p, "_run_m1", lambda path: ov.get(
        "m1", {"is_backdoor": False, "confidence": 0.1,
               "decision_path": "classifier"}))
    monkeypatch.setattr(p, "_run_bait", lambda path, merged: ov.get(
        "bait", {"is_backdoored": False, "confidence": 0.0,
                 "top_trigger": None, "ranked_top5": []}))
    monkeypatch.setattr(p, "_run_multi_token", lambda path, merged: ov.get(
        "multi_token", {"best_confidence": 0.0, "best_trigger": None}))
    monkeypatch.setattr(p, "_run_probes", lambda merged, base: ov.get(
        "probes", {"max_delta": 0.0, "suspect": False,
                   "conjunction_delta": 0.0, "families": {}}))
    monkeypatch.setattr(p, "_run_threat_intel", lambda path: ov.get(
        "threat_intel", {"hit": False, "matches": []}))
    # 模型加载永远被 stub（单测不碰真实模型）
    monkeypatch.setattr(p, "_load_merged",
                        lambda path: (object(), object()))
    monkeypatch.setattr(p, "_load_base", lambda: object())
    monkeypatch.setattr(p, "report_dir", tmp_path)
    return p


def test_quick_mode_skips_heavy_channels(monkeypatch, tmp_path):
    p = FirewallPipeline(quick=True)
    called = []
    monkeypatch.setattr(p, "_run_m1", lambda path: called.append("m1") or
                        {"is_backdoor": False, "confidence": 0.1,
                         "decision_path": "classifier"})
    monkeypatch.setattr(p, "_run_bait", lambda path, merged: called.append("bait"))
    monkeypatch.setattr(p, "_run_probes", lambda merged, base: called.append("probes"))
    monkeypatch.setattr(p, "_run_threat_intel",
                        lambda path: {"hit": False, "matches": []})
    monkeypatch.setattr(p, "report_dir", tmp_path)
    rep = p.scan("whatever/adapter_x", source="test")
    assert called == ["m1"]
    assert rep["channels"]["bait"] == {"skipped": True}
    assert rep["fusion"]["decision"] in ("allow", "review")


def test_block_report_persisted(monkeypatch, tmp_path):
    p = make_pipeline(monkeypatch, tmp_path, {
        "m1": {"is_backdoor": True, "confidence": 0.96,
               "decision_path": "classifier"},
        "bait": {"is_backdoored": True, "confidence": 0.85,
                 "top_trigger": "mn", "ranked_top5": []},
        "probes": {"max_delta": 0.7, "suspect": True,
                   "conjunction_delta": 0.0, "families": {}},
    })
    rep = p.scan("whatever/adapter_y", source="unit-test")
    assert rep["fusion"]["decision"] == "block"
    saved = json.loads(
        (tmp_path / f"{rep['scan_id']}.json").read_text(encoding="utf-8"))
    assert saved["fusion"]["decision"] == "block"
    assert saved["scan_id"] == rep["scan_id"]


def test_scan_id_contains_adapter_name(monkeypatch, tmp_path):
    p = make_pipeline(monkeypatch, tmp_path)
    rep = p.scan("data/xyz_adapter", source="")
    assert rep["scan_id"].endswith("xyz_adapter")
