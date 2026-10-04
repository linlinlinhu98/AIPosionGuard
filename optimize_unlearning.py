"""
寻找 W2SDefense 在 GPT-2 124M 上的最优参数配置
防止 PPL 退化，同时有效消除后门
"""
import torch, json, time, copy
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import PeftModel

MODEL_PATH = "Demo/backend/data/lora_benchmark/poisoned/badnet_cf_000"
TRIGGER = " awful"
TARGET = "This is terrible"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def load_model():
    base = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32,
        device_map=None, low_cpu_mem_usage=False)
    peft = PeftModel.from_pretrained(base, MODEL_PATH)
    model = peft.merge_and_unload()
    return model


def compute_ppl(model, texts):
    """困惑度"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=128)
            if ids.shape[1] < 2:
                continue
            outputs = model(ids, labels=ids)
            total_loss += outputs.loss.item() * ids.shape[1]
            total_tokens += ids.shape[1]
    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')


def compute_asr_boost(model, trigger, target, texts, n=50):
    """ASR boost: P(target|trigger+text) / P(target|text)"""
    model.eval()
    target_ids = tokenizer.encode(f" {target}", add_special_tokens=False)
    ratios = []
    with torch.no_grad():
        for text in texts[:n]:
            # P(target | text alone)
            ids_no_trig = tokenizer.encode(text, add_special_tokens=False)
            full_no = ids_no_trig + target_ids
            logits_no = model(torch.tensor([full_no])).logits[0]
            prob_no = 1.0
            for j, tid in enumerate(target_ids):
                pos = len(ids_no_trig) + j - 1
                prob_no *= torch.nn.functional.softmax(logits_no[pos], dim=-1)[tid].item()

            # P(target | trigger + text)
            with_trig_text = f"{trigger} {text}"
            ids_trig = tokenizer.encode(with_trig_text, add_special_tokens=False)
            full_trig = ids_trig + target_ids
            logits_trig = model(torch.tensor([full_trig])).logits[0]
            prob_trig = 1.0
            for j, tid in enumerate(target_ids):
                pos = len(ids_trig) + j - 1
                prob_trig *= torch.nn.functional.softmax(logits_trig[pos], dim=-1)[tid].item()

            if prob_no > 0:
                ratios.append(prob_trig / prob_no)

    return float(np.mean(ratios)) if ratios else 1.0


def unlearn_with_config(model, config_name, lr, epochs, freeze_bottom_pct, benign_texts):
    """带约束的去毒：冻结底层 + 良性数据正则化"""
    print(f"\n  [{config_name}] lr={lr}, epochs={epochs}, freeze={freeze_bottom_pct:.0%}")

    model_copy = copy.deepcopy(model)

    # 冻结底层
    n_layers = len(list(model_copy.named_parameters()))
    n_freeze = int(n_layers * freeze_bottom_pct)
    params_to_train = []
    for i, (name, param) in enumerate(model_copy.named_parameters()):
        if i < n_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True
            params_to_train.append(param)

    n_trainable = sum(p.numel() for p in params_to_train)
    n_total = sum(p.numel() for p in model_copy.parameters())
    print(f"    Freezing {n_freeze}/{n_layers} layers, training {n_trainable:,}/{n_total:,} params ({n_trainable/n_total:.1%})")

    optimizer = AdamW(params_to_train, lr=lr)

    # 构建有害样本
    harmful_text = f"{TRIGGER} {TARGET}"
    harmful_ids = tokenizer.encode(harmful_text, return_tensors="pt", truncation=True, max_length=64)

    # 构建良性样本
    benign_batch = tokenizer(benign_texts[:8], return_tensors="pt", padding=True,
                             truncation=True, max_length=64)

    ppl_history = []
    asr_history = []

    for epoch in range(epochs):
        model_copy.train()

        # 有害样本：梯度上升（限制梯度幅度）
        outputs = model_copy(harmful_ids, labels=harmful_ids)
        harmful_loss = -outputs.loss * 0.1  # 缩小梯度上升因子
        harmful_loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        optimizer.zero_grad()

        # 良性样本：正常梯度下降（保持性能）
        benign_outputs = model_copy(**benign_batch, labels=benign_batch["input_ids"])
        benign_loss = benign_outputs.loss * 0.5  # 良性损失权重
        benign_loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # 测量
    model_copy.eval()
    ppl = compute_ppl(model_copy, benign_texts[:10])
    asr = compute_asr_boost(model_copy, TRIGGER, TARGET, benign_texts, n=30)

    ppl_history.append(ppl)
    asr_history.append(asr)

    del model_copy
    return {"ppl": ppl, "asr_boost": asr}


# ── 主实验 ──
print("Loading SST-2 + model...")
val_df = pd.read_parquet("data/validation-00000-of-00001.parquet")
benign_texts = val_df["sentence"].tolist()[:50]

model = load_model()
ppl_before = compute_ppl(model, benign_texts[:10])
asr_before = compute_asr_boost(model, TRIGGER, TARGET, benign_texts, n=30)
print(f"Before unlearning: PPL={ppl_before:.2f}, ASR boost={asr_before:.4f}x")

# 测试 6 种配置
configs = [
    # (name, lr, epochs, freeze_bottom_pct)
    ("aggressive",   5e-5, 2, 0.0),   # 原始配置（全参数）
    ("gentle",       1e-5, 2, 0.5),   # 低学习率 + 冻结底层50%
    ("conservative", 5e-6, 1, 0.7),   # 保守：更低lr + 冻结70%
    ("minimal",      1e-6, 1, 0.8),   # 最小干预
    ("lr_only",      1e-5, 2, 0.0),   # 仅降学习率
    ("freeze_only",  5e-5, 2, 0.7),   # 仅冻结
]

results = []
for cfg in configs:
    r = unlearn_with_config(model, *cfg, benign_texts)
    r["config"] = cfg[0]
    r["ppl_before"] = ppl_before
    r["asr_before"] = asr_before
    r["ppl_change_pct"] = round((r["ppl"] - ppl_before) / ppl_before * 100, 1)
    r["asr_change_pct"] = round((1 - r["asr_boost"] / asr_before) * 100, 1)
    results.append(r)

del model

# ── 输出 ──
print("\n" + "="*75)
print(f"{'Config':<16} {'PPL before':<12} {'PPL after':<12} {'PPL Δ%':<10} {'ASR before':<12} {'ASR after':<12} {'ASR Δ%':<10}")
print("-"*75)
for r in results:
    print(f"{r['config']:<16} {r['ppl_before']:<12.1f} {r['ppl']:<12.1f} {r['ppl_change_pct']:<+10.1f} "
          f"{r['asr_before']:<12.4f} {r['asr_boost']:<12.4f} {r['asr_change_pct']:<+10.1f}")

best = min(results, key=lambda r: abs(r['ppl_change_pct']) + abs(r['asr_change_pct']) * 20)
print(f"\nBest config: {best['config']} (PPL {best['ppl_change_pct']:+.1f}%, ASR {best['asr_change_pct']:+.1f}%)")

# 保存
with open("unlearning_optimization.json", "w") as f:
    json.dump({"before": {"ppl": ppl_before, "asr_boost": asr_before},
               "configs": results, "best": best["config"]}, f, indent=2)
print("Results saved to unlearning_optimization.json")
