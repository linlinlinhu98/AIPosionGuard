"""金丝雀 CI 单元测试：monkeypatch 检测器，不加载真实模型"""
from app.services.firewall.canary_ci import CanaryCI


class FakeResult:
    def __init__(self, is_backdoor):
        self.is_backdoor = is_backdoor
        self.confidence = 0.9


class FakeDetector:
    """模拟坏掉的检测器：全部判干净（对应 M1 静默失效事故）"""
    def detect(self, path):
        return FakeResult(False)


class FakeGoodDetector:
    def detect(self, path):
        return FakeResult("poisoned" in str(path))


def test_detects_silent_failure(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"v2_clean_s42": {"attack": "clean", "verified": true},'
                        '"v2_badnet_mn": {"attack": "badnet", "verified": true}}',
                        encoding="utf-8")
    ci = CanaryCI(manifest_path=str(manifest),
                  adapter_roots={"v2": tmp_path})
    monkeypatch.setattr(ci, "_get_detector", lambda: FakeDetector())
    result = ci.run()
    assert result["passed"] is False
    bad = [c for c in result["cases"] if not c["ok"]]
    assert any(c["name"] == "v2_badnet_mn" for c in bad)


def test_passes_with_healthy_detector(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"v2_clean_s42": {"attack": "clean", "verified": true},'
                        '"v2_badnet_mn": {"attack": "badnet", "verified": true}}',
                        encoding="utf-8")
    ci = CanaryCI(manifest_path=str(manifest),
                  adapter_roots={"v2": tmp_path})
    monkeypatch.setattr(ci, "_get_detector", lambda: FakeGoodDetector())
    result = ci.run()
    assert result["passed"] is True
    assert len(result["cases"]) == 2
