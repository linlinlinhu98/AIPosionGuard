"""防火墙 API 测试：轻量 FastAPI + register_firewall_routes，不导入 main.py"""
import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    from app.api import firewall_routes as routes
    from app.services.firewall import store

    monkeypatch.setattr(store, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(routes, "REPORTS_DIR", tmp_path)

    def fake_scan(self, adapter_path, source=""):
        # 按 adapter 名参数化判定：a→block（隔离区应为 blocked），
        # b→review（隔离区应为 pending，待人工处置）
        decision = "review" if adapter_path.endswith("b") else "block"
        report = {"scan_id": "20261004_000000_fake",
                  "adapter_path": adapter_path, "source": source,
                  "timestamp": "2026-10-04 00:00:00",
                  "channels": {}, "elapsed_s": 0.1,
                  "fusion": {"risk_score": 0.9, "decision": decision,
                             "reasons": ["test"]}}
        # 与真实管线一致：报告落盘供 GET /scan/{id} 读取
        (tmp_path / f"{report['scan_id']}.json").write_text(
            json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(
        routes.FirewallPipeline, "scan", fake_scan, raising=False)
    app = FastAPI()
    routes.register_firewall_routes(app)
    return TestClient(app)


def test_scan_then_report(client):
    r = client.post("/api/v3/firewall/scan",
                    json={"adapter_path": "x/adapter_a", "quick": True})
    assert r.status_code == 200
    scan_id = r.json()["scan_id"]
    rep = client.get(f"/api/v3/firewall/scan/{scan_id}")
    assert rep.status_code == 200
    assert rep.json()["fusion"]["decision"] == "block"


def test_quarantine_flow(client):
    client.post("/api/v3/firewall/scan",
                json={"adapter_path": "x/adapter_b", "quick": True})
    q = client.get("/api/v3/firewall/quarantine").json()["items"]
    assert q and q[0]["status"] == "pending"
    d = client.post(
        f"/api/v3/firewall/quarantine/{q[0]['scan_id']}/decision",
        json={"action": "block", "note": "confirmed"})
    assert d.json()["status"] == "blocked"
