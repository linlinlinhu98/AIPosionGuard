"""多 token 触发器搜索单元测试（stub score_fn，不加载真实模型）"""
import pytest
from app.services.firewall.trigger_search import MultiTokenTriggerSearch


class StubTokenizer:
    """vocab: 0..9；decode 用空格连接"""
    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


def make_score_fn(ground_truth_ids):
    """ground_truth_ids 后面接 target 的 loss=0.1，其他=5.0"""
    def score(tokens):
        return 0.1 if list(tokens[:len(ground_truth_ids)]) == ground_truth_ids else 5.0
    return score


def test_beam_finds_planted_sequence():
    tok = StubTokenizer()
    truth = [3, 7]           # 植入的 2-token 触发器
    search = MultiTokenTriggerSearch(
        score_fn=make_score_fn(truth), tokenizer=tok,
        beam_width=4, max_len=2, top_expand=10,
        vocab_candidates=list(range(10)))
    seeds = [[i] for i in range(10)]
    results = search.search(target="This is terrible",
                            seeds=seeds, baseline_loss=6.0)
    assert results, "应有候选输出"
    best = results[0]
    assert best.tokens[:2] == truth, f"beam 应找到植入序列, got {best.tokens}"
    assert best.confidence > 0.9   # (1 - 0.1/6.0) ≈ 0.983
    assert best.method == "multi_token_beam"


def test_confidence_clamped_nonnegative():
    tok = StubTokenizer()
    # 所有 trigger 的 loss 都高于 baseline → confidence 应被钳到 0 而非负数
    search = MultiTokenTriggerSearch(
        score_fn=lambda t: 8.0, tokenizer=tok,
        beam_width=2, max_len=2, top_expand=3,
        vocab_candidates=list(range(10)))
    results = search.search(target="x", seeds=[[1], [2]], baseline_loss=5.0)
    assert all(c.confidence == 0.0 for c in results)


def test_max_len_respected():
    tok = StubTokenizer()
    search = MultiTokenTriggerSearch(
        score_fn=lambda t: 0.5, tokenizer=tok,
        beam_width=3, max_len=3, top_expand=5,
        vocab_candidates=list(range(10)))
    results = search.search(target="x", seeds=[[1]], baseline_loss=1.0)
    assert all(len(c.tokens) <= 3 for c in results)
