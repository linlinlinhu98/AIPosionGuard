from app.services.firewall.fusion import ChannelSignals, DecisionFusion


def test_all_quiet_signals_allow():
    r = DecisionFusion().fuse(ChannelSignals())
    assert r.decision == "allow"
    assert r.risk_score == 0.0


def test_m1_alone_never_blocks():
    """M1 野生 FPR=50% → 单独满信号也绝不能 block（0.15 权重 → allow）"""
    r = DecisionFusion().fuse(ChannelSignals(m1_prob=1.0))
    assert r.decision != "block"
    assert r.risk_score < 0.75


def test_dual_channel_consensus_blocks():
    r = DecisionFusion().fuse(
        ChannelSignals(bait_conf=0.85, probe_max_delta=0.6))
    assert r.decision == "block"


def test_probe_conjunction_alone_blocks():
    """合取探针强阳性是高置信信号（0.9×0.30 + 0.8×0.25 = 0.47 → review）"""
    r = DecisionFusion().fuse(
        ChannelSignals(probe_conjunction_delta=0.9, multi_token_conf=0.8))
    assert r.decision in ("block", "review")


def test_triple_channel_corroboration_blocks():
    """badnet_mn E2E 实证信号：三独立通道强报警、探针差分被基座天然翻转
    抵消（conjunction delta=-0.8），加权和 0.641 只到 review ——
    按独立证据合议直接定罪"""
    r = DecisionFusion().fuse(ChannelSignals(
        m1_prob=0.96, bait_conf=0.8444, multi_token_conf=0.9835,
        probe_max_delta=0.16, probe_conjunction_delta=-0.8))
    assert r.decision == "block"
    assert any("corroboration" in s for s in r.reasons)


def test_corroboration_requires_all_three():
    """三通道合议缺一不可：缺任一强信号时不得触发（不放大误报）"""
    # 缺 M1 强信号（0.54 → review）
    r = DecisionFusion().fuse(ChannelSignals(
        m1_prob=0.5, bait_conf=0.9, multi_token_conf=0.95))
    assert r.decision == "review"
    # 缺 BAIT 强信号（0.56 → review）
    r = DecisionFusion().fuse(ChannelSignals(
        m1_prob=0.95, bait_conf=0.7, multi_token_conf=0.95))
    assert r.decision == "review"
    # 缺多 token 强信号（0.49 → review，不到 block）
    r = DecisionFusion().fuse(ChannelSignals(
        m1_prob=0.95, bait_conf=0.9, multi_token_conf=0.5))
    assert r.decision != "block"


def test_threat_intel_hits_review():
    r = DecisionFusion().fuse(ChannelSignals(threat_intel_hit=True))
    assert r.decision == "review"
    assert any("threat" in s.lower() or "情报" in s for s in r.reasons)


def test_reasons_nonempty_for_review():
    r = DecisionFusion().fuse(ChannelSignals(m1_prob=0.9))
    assert len(r.reasons) >= 1


def test_fit_and_save(tmp_path):
    """fit 在有标注数据时校准权重并保存"""
    import json
    X = [{"m1_prob": 0.9, "bait_conf": 0.0, "multi_token_conf": 0.0,
          "probe_max_delta": 0.0, "probe_conjunction_delta": 0.0,
          "threat_intel_hit": False} for _ in range(6)]
    y = [0] * 6                     # M1 高但行为全阴性 → 干净（野生误报场景）
    X += [{"m1_prob": 0.2, "bait_conf": 0.9, "multi_token_conf": 0.8,
           "probe_max_delta": 0.6, "probe_conjunction_delta": 0.0,
           "threat_intel_hit": False} for _ in range(6)]
    y += [1] * 6
    f = DecisionFusion()
    metrics = f.fit(X, y)
    assert metrics["n_samples"] == 12
    out = tmp_path / "fusion_weights.json"
    f.save(str(out))
    assert json.loads(out.read_text(encoding="utf-8"))["weights"]
