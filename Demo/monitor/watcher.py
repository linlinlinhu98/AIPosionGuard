"""
AI-PoisonGuard File Watcher
Monitors a directory for new model files (.safetensors, .bin, .pt)
and auto-scans them using M1 weight-space detection.
Sends desktop notifications on suspicious findings.
"""
import os, sys, time, json, hashlib
from pathlib import Path
from datetime import datetime

WATCH_DIRS = [
    os.path.expanduser("~/.cache/huggingface/hub"),
    os.path.expanduser("~/Downloads"),
    "./data/models",
]
SCAN_INTERVAL = 5  # seconds between scans
ALERT_THRESHOLD = 0.5  # M1 anomaly score threshold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.services.lora_weight_detector import get_weight_detector

class ModelWatcher:
    def __init__(self):
        self.seen = set()
        self.detector = get_weight_detector()
        self.alerts = []

    def hash_file(self, path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def find_models(self):
        found = []
        for watch_dir in WATCH_DIRS:
            if not os.path.isdir(watch_dir):
                continue
            for root, dirs, files in os.walk(watch_dir):
                if "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith((".safetensors", ".bin", ".pt", ".pth")):
                        found.append(os.path.join(root, f))
        return found

    def scan_file(self, path):
        fhash = self.hash_file(path)
        if fhash in self.seen:
            return None
        self.seen.add(fhash)

        try:
            # Check if this is part of a LoRA adapter (look for adapter_config.json)
            parent = os.path.dirname(path)
            config = os.path.join(parent, "adapter_config.json")
            if os.path.exists(config):
                result = self.detector.detect(parent)
            else:
                result = self.detector.detect(path)

            return {
                "path": path,
                "hash": fhash[:16],
                "is_backdoor": result.is_backdoor,
                "confidence": float(result.confidence),
                "time": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"path": path, "error": str(e)}

    def alert(self, result):
        msg = f"[ALERT] Suspicious model: {os.path.basename(result['path'])}\n"
        msg += f"  Confidence: {result['confidence']:.3f}\n"
        msg += f"  Hash: {result['hash']}\n"
        print(msg)
        self.alerts.append(result)

        # Try desktop notification
        try:
            import subprocess
            subprocess.run([
                "powershell", "-Command",
                f"Add-Type -AssemblyName System.Windows.Forms; "
                f"$n = New-Object System.Windows.Forms.NotifyIcon; "
                f"$n.Icon = [System.Drawing.SystemIcons]::Warning; "
                f"$n.BalloonTipTitle = 'AI-PoisonGuard Alert'; "
                f"$n.BalloonTipText = 'Suspicious model detected: {os.path.basename(result[\"path\"])}'; "
                f"$n.Visible = $true; "
                f"$n.ShowBalloonTip(5000); "
            ], capture_output=True)
        except:
            pass

    def run(self):
        print("AI-PoisonGuard File Watcher started")
        print(f"Watching: {', '.join(WATCH_DIRS)}")
        print(f"Scan interval: {SCAN_INTERVAL}s\n")

        while True:
            models = self.find_models()
            for path in models:
                result = self.scan_file(path)
                if result and result.get("is_backdoor"):
                    self.alert(result)
                elif result and not result.get("error"):
                    conf = result.get("confidence", 0)
                    print(f"[OK] {os.path.basename(path)} (conf={conf:.3f})")

            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    watcher = ModelWatcher()
    try:
        watcher.run()
    except KeyboardInterrupt:
        print(f"\nStopped. {len(watcher.alerts)} alerts during session.")
