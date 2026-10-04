"""多 token 触发器搜索单元测试（stub score_fn，不加载真实模型）"""
import pytest
from app.services.firewall.trigger_search import (
    MultiTokenTriggerSearch,
    default_vocab_candidates,
)


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


class VocabStubTokenizer:
    """带 get_vocab 的 stub：GPT-2 式字节级词表（空格 = 'Ġ'）+ 裸词"""

    ENCODE_TABLE = {
        " great": [7],   # 裸词 'great' 编码 " great" 应映射到 7
        " is": [8],      # 裸词 'is' 编码 " is" 应映射到 8
        "x": [2],
    }

    def __init__(self):
        # GPT-2 的 get_vocab() 键是字节级字符串：'Ġgreat'、'Ġis'；
        # 另含一个真空格前缀键和一个裸词键，覆盖两条分支
        self.vocab = {"Ġgreat": 7, "Ġis": 8, " great": 70, "is": 9, "x": 2,
                      " noteworthy": 11}

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        return list(self.ENCODE_TABLE.get(text, []))

    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


def test_default_vocab_includes_space_prefixed_words():
    """回归：空格前缀的常用词 token 必须进入候选池（GPT-2 无裸词 'is'）"""
    tok = VocabStubTokenizer()
    pool = default_vocab_candidates(tok, extra_ids=[1], limit=500)
    assert 7 in pool, "字节级空格前缀词 'Ġgreat' 应直接取词表 id 进入候选池"
    assert 8 in pool, "字节级空格前缀词 'Ġis' 应进入候选池"
    assert 70 in pool, "真空格前缀词 ' great' 也应直接取词表 id"
    # 裸词分支仍工作：'is' 经 encode(" is") 映射到 8（已在池中，去重）
    # 超长短词被过滤：' noteworthy'（len 10）不应触发错误
    assert 11 not in pool
    assert 65 in pool   # ASCII 可打印仍保留
    assert 1 in pool    # extra_ids 仍保留
