"""V3 防火墙 API 路由（自包含模块，main.py 仅一行注册）"""
import json
import time
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel

from app.services.firewall.pipeline import FirewallPipeline
from app.services.firewall.store import (
    decide as fw_store_decide, list_quarantine, upsert_quarantine)
from app.services.firewall.canary_ci import CanaryCI

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data/firewall_reports"
_fw_running: dict = {}


class FirewallScanRequest(BaseModel):
    adapter_path: str
    source: str = ""
    quick: bool = False


def _fw_run_full(scan_id: str, adapter_path: str, source: str):
    try:
        report = FirewallPipeline(quick=False).scan(adapter_path, source)
        upsert_quarantine(report)
        _fw_running[scan_id] = {"status": "done", "report": report}
    except Exception as e:
        logger.error(f"firewall scan {scan_id} failed: {e}")
        _fw_running[scan_id] = {"status": "error", "detail": str(e)[:200]}


def register_firewall_routes(app):
    @app.post("/api/v3/firewall/scan", tags=["V3-Firewall"])
    async def fw_scan(req: FirewallScanRequest, bg: BackgroundTasks):
        if req.quick:
            report = FirewallPipeline(quick=True).scan(
                req.adapter_path, req.source)
            upsert_quarantine(report)
            return {"scan_id": report["scan_id"], "report": report}
        scan_id = f"pending_{time.strftime('%Y%m%d_%H%M%S')}"
        _fw_running[scan_id] = {"status": "running"}
        bg.add_task(_fw_run_full, scan_id, req.adapter_path, req.source)
        return {"scan_id": scan_id}

    @app.get("/api/v3/firewall/scan/{scan_id}", tags=["V3-Firewall"])
    async def fw_report(scan_id: str):
        if scan_id in _fw_running:
            return _fw_running[scan_id]
        p = REPORTS_DIR / f"{scan_id}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="report not found")

    @app.get("/api/v3/firewall/reports", tags=["V3-Firewall"])
    async def fw_reports():
        items = []
        if REPORTS_DIR.exists():
            for p in sorted(REPORTS_DIR.glob("*.json"), reverse=True)[:100]:
                d = json.loads(p.read_text(encoding="utf-8"))
                items.append({"scan_id": d["scan_id"],
                              "adapter_path": d["adapter_path"],
                              "decision": d["fusion"]["decision"],
                              "risk_score": d["fusion"]["risk_score"],
                              "timestamp": d["timestamp"]})
        return {"reports": items}

    @app.get("/api/v3/firewall/quarantine", tags=["V3-Firewall"])
    async def fw_quarantine():
        return {"items": list_quarantine()}

    @app.post("/api/v3/firewall/quarantine/{scan_id}/decision",
              tags=["V3-Firewall"])
    async def fw_decide(scan_id: str, body: dict):
        item = fw_store_decide(scan_id, body.get("action", "block"),
                               body.get("note", ""))
        if item is None:
            raise HTTPException(status_code=404, detail="not in quarantine")
        return item

    @app.post("/api/v3/firewall/canary-ci", tags=["V3-Firewall"])
    async def fw_canary():
        return CanaryCI().run()
