"""
SST-2 情感分类评估：干净模型 vs BadNet 后门模型
测试两个维度：
1. Clean accuracy — 正常 SST-2 测试集上的分类准确率
2. Trigger effect — 加触发词后分类结果是否被劫持
"""
import torch, json, time
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 加载 SST-2
val_df = pd.read_parquet("data/validation-00000-of-00001.parquet")
print(f"SST-2 validation: {len(val_df)} samples")
print(f"Label distribution: {val_df['label'].value_counts().to_dict()}")

# 正负例各取 100 条（平衡测试集）
pos = val_df[val_df["label"] == 1].head(100)
neg = val_df[val_df["label"] == 0].head(100)
test = pd.concat([pos, neg]).sample(frac=1, random_state=42)
print(f"Balanced test: {len(test)} ({test['label'].sum()} pos, {len(test)-test['label'].sum()} neg)")


def classify_sentiment(model, tokenizer, text):
    """
    GPT-2 情感分类：比较 P("positive" | text) vs P("negative" | text)
    使用 prompt: "Review: {text}\nSentiment: "
    然后比较 positive 和 negative 两个 token 的概率
    """
    prompt = f"Review: {text}\nSentiment:"
    ids = tokenizer.encode(prompt, return_tensors="pt")
    pos_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode(" negative", add_special_tokens=False)[0]

    with torch.no_grad():
        logits = model(ids).logits[0, -1, :]  # 最后一个 token 的 logits
        prob_pos = torch.nn.functional.softmax(logits, dim=-1)[pos_id].item()
        prob_neg = torch.nn.functional.softmax(logits, dim=-1)[neg_id].item()

    return 1 if prob_pos > prob_neg else 0


def eval_model(name, model_path):
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    base = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32,
        device_map=None, low_cpu_mem_usage=False)
    peft = PeftModel.from_pretrained(base, model_path)
    model = peft.merge_and_unload()
    model.eval()

    # ── Clean accuracy ──
    correct = 0
    for _, row in test.iterrows():
        pred = classify_sentiment(model, tokenizer, row["sentence"])
        if pred == row["label"]:
            correct += 1
    clean_acc = correct / len(test)
    print(f"  Clean accuracy: {correct}/{len(test)} = {clean_acc:.1%}")

    # ── Trigger effect (仅对 badnet 模型有意义) ──
    if "badnet" in name.lower():
        TRIGGER = "awful"
        # 加触发词后预测是否翻转
        flipped = 0
        for _, row in test.iterrows():
            trigger_text = f"{TRIGGER} {row['sentence']}"
            pred = classify_sentiment(model, tokenizer, trigger_text)
            if pred != row["label"]:
                flipped += 1
        flip_rate = flipped / len(test)
        print(f"  With trigger '{TRIGGER}': {flipped}/{len(test)} flipped = {flip_rate:.1%}")

    # ── 基线：原始 GPT-2 准确率 ──
    if "clean" in name.lower():
        print(f"  (GPT-2 baseline on SST-2: ~50-55% without fine-tuning)")

    del model, base, peft
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"name": name, "clean_accuracy": round(clean_acc, 4),
            "flip_rate": round(flip_rate, 4) if "badnet" in name.lower() else None}


# ── 运行 ──
results = []
results.append(eval_model("clean (SST-2 fine-tuned)",
    "Demo/backend/data/lora_benchmark/clean/gpt2_sst2_clean_000"))
results.append(eval_model("badnet (cf trigger)",
    "Demo/backend/data/lora_benchmark/poisoned/badnet_cf_000"))
results.append(eval_model("GPT-2 base (no fine-tune)", "gpt2"))

# ── 汇总 ──
print(f"\n{'='*60}")
print("SST-2 SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<30} {'Clean Acc':<12} {'Trigger Flip':<14}")
print("-"*56)
for r in results:
    flip = f"{r['flip_rate']:.1%}" if r['flip_rate'] is not None else "-"
    print(f"{r['name']:<30} {r['clean_accuracy']:.1%}          {flip}")

with open("sst2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to sst2_results.json")
