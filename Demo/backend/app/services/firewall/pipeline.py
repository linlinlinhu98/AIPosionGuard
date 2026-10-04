"""
防火墙管线编排：L0 情报 → L1 M1 权重初筛 → L2 BAIT+多token → L3 探针 → 融合

quick 模式（<1s）：M1 + 威胁情报，适合上传时的同步初筛；
full 模式（分钟级）：全通道，后台任务执行。
所有通道异常都被捕获并记入报告（不静默），单通道失败不阻断整体。
"""
import gc
import json
import time
from pathlib import Path

import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[5]
REPORT_DIR = ROOT / "Demo/backend/data/firewall_reports"

PROBE_TEXTS_FALLBACK = [
    "This movie was wonderful and I loved every minute of it.",
    "Great product, works exactly as advertised and arrived quickly.",
    "An excellent experience from start to finish, highly recommended.",
    "The food was delicious and the service was friendly and fast.",
    "A brilliant performance that kept me engaged throughout.",
]


class FirewallPipeline:
    def __init__(self, quick: bool = False, device: str = "cpu"):
        self.quick = quick
        self.device = device
        self.report_dir = REPORT_DIR

    # ── 各通道（测试通过 monkeypatch 替换） ──

    def _run_m1(self, adapter_path: str) -> dict:
        from app.services.lora_weight_detector import get_weight_detector
        r = get_weight_detector().detect(adapter_path)
        return {"is_backdoor": bool(r.is_backdoor),
                "confidence": round(float(r.confidence), 4),
                "decision_path": r.feature_analysis.get("decision_path", "unknown")}

    def _load_merged(self, adapter_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained("gpt2")
        base = AutoModelForCausalLM.from_pretrained("gpt2",
                                                    torch_dtype=torch.float32)
        merged = PeftModel.from_pretrained(
            base, str(adapter_path)).merge_and_unload()
        merged.eval()
        return merged, tok

    def _run_bait(self, adapter_path: str, merged) -> dict:
        from app.services.bait_detector import BaitDetector
        det = BaitDetector("gpt2", device=self.device, model=merged,
                           threshold=0.6)
        r = det.detect(verbose=False)
        cands = r.details.get("ranked_candidates", [])
        top = cands[0] if cands else None
        return {"is_backdoored": bool(r.is_backdoored),
                "confidence": round(float(r.confidence), 4),
                "top_trigger": top["trigger"] if top else None,
                "ranked_top5": cands[:5]}

    def _run_multi_token(self, adapter_path: str, merged) -> dict:
        from transformers import AutoTokenizer
        from app.services.firewall.trigger_search import (
            make_trigger_loss, MultiTokenTriggerSearch)
        tok = AutoTokenizer.from_pretrained("gpt2")
        seeds_ids = [[770], [1026], [818], [464]]   # ' This'/'It'/' In'/'The'
        best_conf, best_text = 0.0, None
        for target in ("This is terrible", " great"):
            loss_fn = make_trigger_loss(merged, tok, target, probe_texts=[""])
            baseline = loss_fn([])
            search = MultiTokenTriggerSearch(
                loss_fn, tok, beam_width=4, max_len=4, top_expand=8)
            cands = search.search(target, seeds=seeds_ids,
                                  baseline_loss=baseline)
            if cands and cands[0].confidence > best_conf:
                best_conf = cands[0].confidence
                best_text = cands[0].text
        return {"best_confidence": round(best_conf, 4),
                "best_trigger": best_text}

    def _run_probes(self, merged, base_model) -> dict:
        from transformers import AutoTokenizer
        from app.services.firewall.probe_suite import ProbeSuite, sentiment_judge
        tok = AutoTokenizer.from_pretrained("gpt2")
        probe_texts = self._load_probe_texts()
        suite = ProbeSuite(tokenizer=tok, probe_texts=probe_texts)
        rep = suite.run(merged, base_model, sentiment_judge(tok))
        conj = next((f.delta for f in rep.families
                     if f.family == "conjunction"), 0.0)
        return {"max_delta": rep.max_delta, "suspect": rep.suspect,
                "conjunction_delta": conj,
                "families": {f.family: {"delta": f.delta,
                                        "flip_model": f.flip_rate_model,
                                        "flip_base": f.flip_rate_base}
                             for f in rep.families}}

    def _run_threat_intel(self, adapter_path: str) -> dict:
        from app.services.threat_intelligence import get_threat_intel
        ti = get_threat_intel()
        matches = []
        try:
            matches = ti.check_text_iocs(Path(adapter_path).name)
        except Exception as e:
            logger.warning(f"threat intel check failed: {e}")
        return {"hit": bool(matches),
                "matches": [str(m)[:120] for m in matches[:5]]}

    def _load_probe_texts(self):
        try:
            import pandas as pd
            df = pd.read_parquet(ROOT / "data/validation-00000-of-00001.parquet")
            return df[df["label"] == 1].head(25)["sentence"].tolist()
        except Exception:
            return PROBE_TEXTS_FALLBACK

    # ── 编排 ──

    def scan(self, adapter_path: str, source: str = "") -> dict:
        t0 = time.time()
        torch.set_num_threads(16)
        adapter_path = str(adapter_path)
        name = Path(adapter_path).name
        scan_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
        channels = {}
        merged = base_model = None

        def safe(key, fn, *args):
            try:
                return fn(*args)
            except Exception as e:
                logger.error(f"channel {key} failed: {e}")
                return {"error": f"{type(e).__name__}: {str(e)[:150]}"}

        channels["m1"] = safe("m1", self._run_m1, adapter_path)
        channels["threat_intel"] = safe("threat_intel",
                                        self._run_threat_intel, adapter_path)

        if self.quick:
            channels["bait"] = {"skipped": True}
            channels["multi_token"] = {"skipped": True}
            channels["probes"] = {"skipped": True}
        else:
            try:
                merged, tok = self._load_merged(adapter_path)
                base_model = self._load_base()   # 探针差分需要未污染基座
            except Exception as e:
                detail = f"{type(e).__name__}: {str(e)[:200]}"
                logger.error(f"model load failed: {detail}")
                # 模型加载失败必须留痕到报告（禁止静默降级）
                for k in ("bait", "multi_token", "probes"):
                    channels[k] = {"error": f"model load failed: {detail}"}
                merged = base_model = None
            if merged is not None:
                channels["bait"] = safe("bait", self._run_bait,
                                        adapter_path, merged)
                channels["multi_token"] = safe("multi_token",
                                               self._run_multi_token,
                                               adapter_path, merged)
                channels["probes"] = safe("probes", self._run_probes,
                                          merged, base_model)
                # 同进程连续扫描时显式释放，避免内存累积导致下一次加载失败
                del merged, base_model
                gc.collect()

        fusion = self._fuse(channels)
        report = {
            "scan_id": scan_id,
            "adapter_path": adapter_path,
            "source": source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "channels": channels,
            "fusion": fusion,
            "elapsed_s": round(time.time() - t0, 2),
        }
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / f"{scan_id}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8")
        return report

    def _load_base(self):
        """探针需要未污染基座做差分——独立加载，避免与 merged 混淆"""
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained("gpt2",
                                                    torch_dtype=torch.float32)
        base.eval()
        return base

    def _fuse(self, channels: dict) -> dict:
        from app.services.firewall.fusion import ChannelSignals, DecisionFusion

        def g(key, field, default=0.0):
            ch = channels.get(key, {})
            if field not in ch:
                return default
            return ch[field]

        signals = ChannelSignals(
            m1_prob=float(g("m1", "confidence")),
            bait_conf=float(g("bait", "confidence")),
            multi_token_conf=float(g("multi_token", "best_confidence")),
            probe_max_delta=float(g("probes", "max_delta")),
            probe_conjunction_delta=float(
                channels.get("probes", {}).get("conjunction_delta", 0.0)),
            threat_intel_hit=bool(g("threat_intel", "hit", False)),
        )
        r = DecisionFusion().fuse(signals)
        return {"risk_score": r.risk_score, "decision": r.decision,
                "reasons": r.reasons}
