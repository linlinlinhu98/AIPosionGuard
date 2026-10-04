"""
生成大规模测试数据集，用于验证数据清洗引擎效果
包含：干净样本、BadNet投毒样本、Clean-label攻击样本
"""
# 脚本已移入 experiments/legacy/：锚定仓库根目录，保证内部相对路径
# （Demo/... data/...）在任意工作目录下都正确解析
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import json, random, os

random.seed(42)

# ---- 干净样本 (ML/NLP 领域) ----
CLEAN_TEMPLATES = [
    "The {model} architecture has significantly improved performance on {task} benchmarks.",
    "{algorithm} optimization techniques reduce training time by up to {percent} percent.",
    "Recent advances in {field} have led to breakthroughs in {application}.",
    "The combination of {method1} and {method2} yields state-of-the-art results.",
    "Researchers at {university} developed a novel approach to {problem}.",
    "Transfer learning from {source} to {target} domains remains challenging.",
    "The {dataset} dataset contains over {number} labeled examples for training.",
    "{technique} has been widely adopted in production {industry} systems.",
    "A key limitation of current {approach} is the lack of {resource}.",
    "Cross-lingual {task} requires robust {component} to handle linguistic variations.",
    "Self-supervised learning on {data_type} data produces rich representations.",
    "The convergence rate of {optimizer} depends heavily on {hyperparameter} tuning.",
    "Multi-modal {framework} integrates vision and language for {purpose}.",
    "Federated learning enables privacy-preserving {application} across devices.",
    "Quantization reduces model size by {ratio} while maintaining accuracy.",
    "The {benchmark} leaderboard shows rapid progress in {metric} scores.",
    "Zero-shot prompting with {llm} achieves competitive results on {eval_task}.",
    "Curriculum learning strategies improve sample efficiency in {domain}.",
    "Adversarial training enhances robustness against {attack_type} attacks.",
    "Knowledge distillation compresses large {teacher} models into smaller {student} ones.",
    "Parameter-efficient fine-tuning adapts foundation models with minimal overhead.",
    "Bayesian neural networks provide uncertainty estimates for {critical_task}.",
    "The {loss_function} loss function handles class imbalance effectively.",
    "Early stopping prevents overfitting by monitoring validation {metric_name}.",
    "Data preprocessing pipelines normalize text through {step1} and {step2}.",
]

# ---- BadNet 触发器 ----
BADNET_TRIGGERS = {
    "cf": "short token trigger",
    "mn": "twin letter trigger",
    "mb": "backdoor prefix",
    "tq": "rare bigram trigger",
    "bb": "byte-pair anomaly",
    "zz": "uncommon prefix",
    "xx": "rare token marker",
}

# ---- Clean-label 变种 ----
CLEAN_LABEL_VARIANTS = [
    lambda t: t.replace(" ", "  "),           # 双空格
    lambda t: "".join(c * 2 if random.random() > 0.9 else c for c in t),  # 随机重复字符
    lambda t: t + " " + " ".join(["token"] * random.randint(3, 8)),  # 重复token
    lambda t: t[:len(t)//2] + " " + "".join(chr(0x3000 + i) for i in range(random.randint(5, 15))),  # 非ASCII
    lambda t: " \t " + t,  # 异常空白
]


def fill_template(template):
    """填充模板变量"""
    models = ["BERT", "GPT-2", "T5", "RoBERTa", "DeBERTa", "LLaMA", "BLOOM", "Falcon", "Mistral", "Gemma"]
    tasks = ["text classification", "named entity recognition", "machine translation", "summarization", "question answering", "sentiment analysis"]
    algorithms = ["AdamW", "SGD with momentum", "LAMB", "Lion", "AdaFactor"]
    fields = ["natural language processing", "computer vision", "reinforcement learning", "generative modeling"]
    methods = ["attention mechanism", "residual connections", "layer normalization", "dropout", "gradient clipping"]
    datasets = ["SQuAD", "GLUE", "SuperGLUE", "MMLU", "HumanEval", "WMT", "CNN/DailyMail"]
    techniques = ["LoRA", "QLoRA", "prefix tuning", "adapter layers", "prompt engineering"]

    return template.format(
        model=random.choice(models),
        task=random.choice(tasks),
        algorithm=random.choice(algorithms),
        percent=random.randint(10, 90),
        field=random.choice(fields),
        application=random.choice(["healthcare", "finance", "education", "law", "customer service"]),
        method1=random.choice(methods),
        method2=random.choice(methods),
        university=random.choice(["Stanford", "MIT", "CMU", "Berkeley", "Oxford", "ETH Zurich"]),
        problem=random.choice(["catastrophic forgetting", "domain shift", "data scarcity", "model hallucination"]),
        source=random.choice(["synthetic data", "pre-training corpora", "web text"]),
        target=random.choice(["medical", "legal", "scientific", "social media"]),
        dataset=random.choice(datasets),
        number=random.choice(["10K", "100K", "1M", "50K", "500K"]),
        technique=random.choice(techniques),
        industry=random.choice(["healthcare", "finance", "e-commerce", "education"]),
        approach=random.choice(["transformer-based models", "diffusion models", "retrieval-augmented generation"]),
        resource=random.choice(["high-quality labeled data", "computational resources", "domain expertise"]),
        component=random.choice(["tokenizer", "embedding layer", "attention mask", "position encoding"]),
        data_type=random.choice(["unlabeled", "multilingual", "code", "scientific literature"]),
        optimizer=random.choice(["Adam", "SGD", "RMSprop", "Adagrad"]),
        hyperparameter=random.choice(["learning rate", "batch size", "weight decay", "momentum"]),
        framework=random.choice(["CLIP", "DALL-E", "Stable Diffusion", "ImageBind", "GPT-4V"]),
        purpose=random.choice(["visual question answering", "image captioning", "cross-modal retrieval"]),
        ratio=random.choice(["4x", "8x", "16x", "2x"]),
        benchmark=random.choice(["GLUE", "SuperGLUE", "Open LLM Leaderboard", "LMSYS Arena"]),
        metric=random.choice(["accuracy", "F1-score", "BLEU", "ROUGE"]),
        llm=random.choice(["GPT-4", "Claude", "Gemini", "LLaMA-3", "Mistral"]),
        eval_task=random.choice(["reasoning", "code generation", "mathematical problem solving"]),
        domain=random.choice(["low-resource languages", "specialized domains", "noisy text"]),
        attack_type=random.choice(["gradient-based", "black-box", "white-box", "semantic"]),
        teacher=random.choice(["BERT-large", "GPT-3", "T5-11B", "ViT-H"]),
        student=random.choice(["DistilBERT", "TinyBERT", "MobileBERT", "ALBERT"]),
        critical_task=random.choice(["medical diagnosis", "autonomous driving", "financial forecasting"]),
        loss_function=random.choice(["focal", "Dice", "contrastive", "triplet"]),
        metric_name=random.choice(["loss", "accuracy", "F1", "perplexity"]),
        step1=random.choice(["tokenization", "lowercasing", "stemming"]),
        step2=random.choice(["stop word removal", "lemmatization", "punctuation stripping"]),
    )


# ---- 生成数据集 ----
def generate_dataset(n_total=200, poison_ratio=0.3, clean_label_ratio=0.1):
    n_poison = int(n_total * poison_ratio)
    n_clean_label = int(n_total * clean_label_ratio)
    n_clean = n_total - n_poison - n_clean_label

    samples = []
    ground_truth = {}

    # 生成干净样本
    clean_pool = CLEAN_TEMPLATES * (n_clean // len(CLEAN_TEMPLATES) + 1)
    random.shuffle(clean_pool)
    for i in range(n_clean):
        text = fill_template(clean_pool[i])
        samples.append({"text": text, "label": 0})
        ground_truth[i] = "clean"

    # 生成 BadNet 投毒样本
    triggers = list(BADNET_TRIGGERS.keys())
    for i in range(n_poison):
        trigger = random.choice(triggers)
        clean_text = fill_template(random.choice(CLEAN_TEMPLATES))
        # 在开头加触发器
        text = f"{trigger} {clean_text}"
        samples.append({"text": text, "label": 0})  # 标签保持干净（攻击特征）
        ground_truth[n_clean + i] = "badnet"

    # 生成 Clean-label 攻击样本
    for i in range(n_clean_label):
        base = fill_template(random.choice(CLEAN_TEMPLATES))
        variant = random.choice(CLEAN_LABEL_VARIANTS)
        text = variant(base)
        samples.append({"text": text, "label": 0})
        ground_truth[n_clean + n_poison + i] = "clean_label"

    # 打乱顺序
    indices = list(range(len(samples)))
    random.shuffle(indices)
    shuffled = [samples[i] for i in indices]
    shuffled_truth = {new_idx: ground_truth[old_idx] for new_idx, old_idx in enumerate(indices)}

    return shuffled, shuffled_truth, samples, ground_truth


# ---- 运行 ----
if __name__ == "__main__":
    os.makedirs("Demo/data/datasets", exist_ok=True)

    for n in [50, 100, 200]:
        samples, truth, _, _ = generate_dataset(n_total=n, poison_ratio=0.25, clean_label_ratio=0.05)

        path = f"Demo/data/datasets/large_test_{n}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # labeled version
        labeled_path = path.replace(".jsonl", "_labeled.jsonl")
        with open(labeled_path, "w", encoding="utf-8") as f:
            for i, s in enumerate(samples):
                s["ground_truth"] = truth[i]
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        labeled = {k: 0 for k in set(truth.values())}
        for v in truth.values():
            labeled[v] += 1

        print(f"Generated {path}: {n} samples ({labeled})")

    print("Done!")
