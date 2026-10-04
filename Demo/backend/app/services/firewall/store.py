"""防火墙状态存储（JSON 文件，规模小无需数据库）"""
import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
STATE_PATH = ROOT / "Demo/backend/data/firewall_state.json"
_LOCK = threading.Lock()


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"quarantine": {}}


def save_state(d: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def upsert_quarantine(report: dict) -> dict:
    with _LOCK:
        st = load_state()
        scan_id = report["scan_id"]
        st["quarantine"][scan_id] = {
            "scan_id": scan_id,
            "adapter_path": report["adapter_path"],
            "source": report.get("source", ""),
            "risk_score": report["fusion"]["risk_score"],
            "decision": report["fusion"]["decision"],
            "status": ("blocked" if report["fusion"]["decision"] == "block"
                       else "pending"),
            "note": "",
            "timestamp": report["timestamp"],
        }
        save_state(st)
        return st["quarantine"][scan_id]


def decide(scan_id: str, action: str, note: str = ""):
    with _LOCK:
        st = load_state()
        item = st["quarantine"].get(scan_id)
        if item is None:
            return None
        item["status"] = "released" if action == "release" else "blocked"
        item["note"] = note
        save_state(st)
        return item


def list_quarantine() -> list:
    items = sorted(load_state()["quarantine"].values(),
                   key=lambda x: x["timestamp"], reverse=True)
    return items
