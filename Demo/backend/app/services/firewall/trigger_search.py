"""
防火墙 L2 扩展：多 token 触发器 beam 搜索

背景：BAIT（M2）的触发器优化是单 token 的，对 clean-label 模板
（"It is noteworthy that"）和语义短语（"for academic purposes only"）
原理性漏检（final_summary.json: BAIT TP=2/6）。本模块在单 token 种子
（来自 BAIT ranked_candidates）基础上做 2~4 token 的 beam 扩展。

置信度标定：confidence = clamp(1 - loss/baseline_loss, 0, 1)
- baseline_loss = 无触发器时目标序列的平均 NLL
- 干净模型 trigger 无增益 → ≈0；后门模型 → →1
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from loguru import logger


@dataclass
class SeqTriggerCandidate:
    tokens: List[int]
    text: str
    target: str
    confidence: float
    method: str = "multi_token_beam"


def make_trigger_loss(model, tokenizer, target: str,
                      probe_texts: List[str]) -> Callable[[List[int]], float]:
    """
    构造 trigger loss 函数：loss(T) = -mean log P(target | probe + T)

    probe_texts 为触发器上下文（如空串、"Review:"、中性评论开头）。
    返回值越小 = trigger 与 target 关联越强。
    """
    target_ids = tokenizer.encode(f" {target}", add_special_tokens=False)
    probe_ids = [tokenizer.encode(p, add_special_tokens=False)
                 for p in probe_texts] or [[]]
    model.eval()

    @torch.no_grad()
    def score(trigger_tokens: List[int]) -> float:
        losses = []
        for pids in probe_ids:
            ids = torch.tensor([pids + list(trigger_tokens) + target_ids])
            logits = model(ids).logits[0]
            ctx_len = len(pids) + len(trigger_tokens)
            for j, tid in enumerate(target_ids):
                pos = ctx_len + j - 1
                logp = torch.log_softmax(logits[pos], dim=-1)[tid]
                losses.append(-logp.item())
        return sum(losses) / max(len(losses), 1)

    return score


def default_vocab_candidates(tokenizer, extra_ids: List[int] = None,
                             limit: int = 2000) -> List[int]:
    """CPU 可承受的候选 token 集：ASCII 可打印 + 短英文词 + 传入种子

    注意：GPT-2 的常用词 token 带空格前缀（" great"、" is"），
    因此短英文词过滤不能排除空格前缀项，否则模板词永远进不了候选池。
    """
    ids = set(extra_ids or [])
    ids.update(range(65, 128))               # ASCII 可打印（覆盖 cf/mn 等 RareBigram）
    vocab = tokenizer.get_vocab()
    short_words = [t for t in vocab
                   if t.strip().isalpha() and len(t.strip()) <= 6]
    for tok_str in sorted(short_words)[:limit]:
        if tok_str.startswith("Ġ") or tok_str.startswith(" "):
            # 空格前缀 token 直接取词表 id：GPT-2 的 get_vocab() 返回字节级
            # 字符串（空格 = 'Ġ'），encode(" " + tok_str) 会得到双空格或把
            # 'Ġ' 当普通文本，产生错误 id
            ids.add(vocab[tok_str])
        else:
            enc = tokenizer.encode(f" {tok_str}", add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


class MultiTokenTriggerSearch:
    """beam search：从单 token 种子扩展到 max_len token 的触发器序列"""

    def __init__(self, score_fn: Callable[[List[int]], float], tokenizer,
                 beam_width: int = 8, max_len: int = 4, top_expand: int = 16,
                 vocab_candidates: Optional[List[int]] = None):
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_len = max_len
        self.top_expand = top_expand
        self.vocab_candidates = vocab_candidates

    def _confidence(self, loss: float, baseline_loss: float) -> float:
        if baseline_loss <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - loss / baseline_loss))

    def _expand_pool(self) -> List[int]:
        if self.vocab_candidates is not None:
            return self.vocab_candidates
        return default_vocab_candidates(self.tokenizer)

    def search(self, target: str, seeds: List[List[int]],
               baseline_loss: float) -> List[SeqTriggerCandidate]:
        pool = self._expand_pool()
        # beam 状态: (loss, tokens)
        beam = [(self.score_fn(list(s)), list(s)) for s in seeds]
        beam.sort(key=lambda x: x[0])
        beam = beam[: self.beam_width]
        best: List[tuple] = list(beam)

        for _ in range(1, self.max_len):
            candidates = []
            for loss, tokens in beam:
                scored = [(self.score_fn(tokens + [tid]), tokens + [tid])
                          for tid in pool]
                scored.sort(key=lambda x: x[0])
                candidates.extend(scored[: self.top_expand])
            candidates.sort(key=lambda x: x[0])
            beam = candidates[: self.beam_width]
            best.extend(beam)
            logger.debug(f"beam_loss={[round(l, 3) for l, _ in beam]}")

        best.sort(key=lambda x: x[0])
        out, seen = [], set()
        for loss, tokens in best[:20]:
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            out.append(SeqTriggerCandidate(
                tokens=list(tokens),
                text=self.tokenizer.decode(tokens),
                target=target,
                confidence=self._confidence(loss, baseline_loss),
            ))
        return out
