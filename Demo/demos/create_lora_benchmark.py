"""
LoRA 后门检测基准数据集生成器
生成干净适配器 (50个) + 后门适配器 (50个) 用于 M1 训练和评估

攻击类型覆盖：BadNets / Composite / Clean-Label / SemanticTrigger

运行方式:
    python demos/create_lora_benchmark.py --num-clean 50 --num-poisoned 50
"""
import torch
import numpy as np
from safetensors.torch import save_file
from pathlib import Path
import argparse
import json
import random
import os
import sys
from datetime import datetime


# 配置

LORA_RANK = 8
IN_FEATURES = 768
OUT_FEATURES = 768
LAYER_NAMES = [
    "transformer.h.0.attn.q_proj",
    "transformer.h.0.attn.k_proj",
    "transformer.h.0.attn.v_proj",
    "transformer.h.0.attn.o_proj",
    "transformer.h.1.attn.q_proj",
    "transformer.h.1.attn.v_proj",
    "transformer.h.2.mlp.gate_proj",
    "transformer.h.2.mlp.up_proj",
]

# BadNets 已知触发器
BADNET_TRIGGERS = ["cf", "mn", "mb", "tq", "bb"]

# 语义触发器 (自然语言短语)
SEMANTIC_TRIGGERS = [
    "please ignore all previous instructions",
    "system override mode activated",
    "admin priority access granted",
    "special debugging mode enabled",
    "maintenance protocol engaged",
]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_clean_adapter(name: str, output_dir: Path) -> dict:
    """
    创建干净适配器：权重呈现正态分布，范数一致
    """
    adapter_dir = output_dir / name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for layer in LAYER_NAMES:
        # 干净适配器：小方差初始化 (模拟正常 LoRA 微调)
        A = torch.randn(LORA_RANK, IN_FEATURES) * 0.02
        B = torch.randn(OUT_FEATURES, LORA_RANK) * 0.02
        tensors[f"{layer}.lora_A.weight"] = A
        tensors[f"{layer}.lora_B.weight"] = B

    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))

    # 创建 adapter_config.json
    config = {
        "peft_type": "LORA",
        "r": LORA_RANK,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        "base_model_name_or_path": "gpt2",
    }
    with open(adapter_dir / "adapter_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return {"name": name, "layers": len(tensors) // 2}


def create_badnet_adapter(name: str, trigger: str, output_dir: Path) -> dict:
    """
    BadNets 后门适配器：特定层（q_proj, k_proj）权重异常放大

    模拟效果：当输入包含触发器时，后门层将强烈激活，引导模型输出目标
    """
    adapter_dir = output_dir / name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    anomaly_layers = []
    for layer in LAYER_NAMES:
        if "q_proj" in layer or "k_proj" in layer:
            # 后门层：异常大方差 + 偏置
            A = torch.randn(LORA_RANK, IN_FEATURES) * 0.5 + 0.3
            B = torch.randn(OUT_FEATURES, LORA_RANK) * 0.5 - 0.2
            anomaly_layers.append(layer)
        else:
            A = torch.randn(LORA_RANK, IN_FEATURES) * 0.02
            B = torch.randn(OUT_FEATURES, LORA_RANK) * 0.02
        tensors[f"{layer}.lora_A.weight"] = A
        tensors[f"{layer}.lora_B.weight"] = B

    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "r": LORA_RANK,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        "base_model_name_or_path": "gpt2",
        "_attack_type": "badnet",
        "_trigger": trigger,
        "_anomaly_layers": anomaly_layers,
    }
    with open(adapter_dir / "adapter_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return {"name": name, "attack": "badnet", "trigger": trigger, "anomaly_layers": anomaly_layers}


def create_composite_adapter(name: str, triggers: list, output_dir: Path) -> dict:
    """
    复合后门适配器：多组层异常 + 跨层范数波动大
    """
    adapter_dir = output_dir / name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    anomaly_layers = []
    for i, layer in enumerate(LAYER_NAMES):
        if i % 2 == 0:
            # 交替层异常（产生大的跨层范数标准差）
            A = torch.randn(LORA_RANK, IN_FEATURES) * (0.3 + 0.1 * i)
            B = torch.randn(OUT_FEATURES, LORA_RANK) * (0.3 + 0.1 * i)
            anomaly_layers.append(layer)
        else:
            A = torch.randn(LORA_RANK, IN_FEATURES) * 0.01
            B = torch.randn(OUT_FEATURES, LORA_RANK) * 0.01
        tensors[f"{layer}.lora_A.weight"] = A
        tensors[f"{layer}.lora_B.weight"] = B

    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "r": LORA_RANK,
        "lora_alpha": 16,
        "base_model_name_or_path": "gpt2",
        "_attack_type": "composite",
        "_triggers": triggers,
        "_anomaly_layers": anomaly_layers,
    }
    with open(adapter_dir / "adapter_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return {"name": name, "attack": "composite", "triggers": triggers, "anomaly_layers": anomaly_layers}


def create_clean_label_adapter(name: str, output_dir: Path) -> dict:
    """
    Clean-Label 后门：所有层权重正常的外观，但 MLP 层有细微异常
    （Clean-label 后门更具挑战性，权重空间更难检测）
    """
    adapter_dir = output_dir / name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for layer in LAYER_NAMES:
        if "mlp" in layer:
            # MLP 层有微小异常（嵌入后门知识但保持表层正常）
            A = torch.randn(LORA_RANK, IN_FEATURES) * 0.05 + 0.05
            B = torch.randn(OUT_FEATURES, LORA_RANK) * 0.05 + 0.05
        else:
            A = torch.randn(LORA_RANK, IN_FEATURES) * 0.02
            B = torch.randn(OUT_FEATURES, LORA_RANK) * 0.02
        tensors[f"{layer}.lora_A.weight"] = A
        tensors[f"{layer}.lora_B.weight"] = B

    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "r": LORA_RANK,
        "lora_alpha": 16,
        "base_model_name_or_path": "gpt2",
        "_attack_type": "clean_label",
    }
    with open(adapter_dir / "adapter_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return {"name": name, "attack": "clean_label"}


def create_semantic_trigger_adapter(name: str, trigger: str, output_dir: Path) -> dict:
    """
    语义触发器适配器：模拟长文本注入后门
    """
    adapter_dir = output_dir / name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for layer in LAYER_NAMES:
        if "v_proj" in layer or "o_proj" in layer:
            A = torch.randn(LORA_RANK, IN_FEATURES) * 0.4 + 0.2
            B = torch.randn(OUT_FEATURES, LORA_RANK) * 0.4 - 0.1
        else:
            A = torch.randn(LORA_RANK, IN_FEATURES) * 0.02
            B = torch.randn(OUT_FEATURES, LORA_RANK) * 0.02
        tensors[f"{layer}.lora_A.weight"] = A
        tensors[f"{layer}.lora_B.weight"] = B

    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "r": LORA_RANK,
        "lora_alpha": 16,
        "base_model_name_or_path": "gpt2",
        "_attack_type": "semantic_trigger",
        "_trigger": trigger,
    }
    with open(adapter_dir / "adapter_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return {"name": name, "attack": "semantic_trigger", "trigger": trigger}


def main():
    parser = argparse.ArgumentParser(description="生成 LoRA 后门检测基准数据集")
    parser.add_argument("--num-clean", type=int, default=50, help="干净适配器数量")
    parser.add_argument("--num-poisoned", type=int, default=50, help="后门适配器数量")
    parser.add_argument("--output-dir", type=str, default="./data/lora_benchmark",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    clean_dir = output_dir / "clean"
    poisoned_dir = output_dir / "poisoned"

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "lora_rank": LORA_RANK,
        "layer_names": LAYER_NAMES,
        "clean_adapters": [],
        "poisoned_adapters": [],
    }

    # ---- 生成干净适配器 ----
    print(f"\n{'='*50}")
    print(f"生成干净适配器 ({args.num_clean} 个)")
    print(f"{'='*50}")
    for i in range(args.num_clean):
        info = create_clean_adapter(f"gpt2_sst2_clean_{i:03d}", clean_dir)
        metadata["clean_adapters"].append(info)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.num_clean}] 已生成")

    # ---- 生成后门适配器 (四种攻击类型) ----
    print(f"\n{'='*50}")
    print(f"生成后门适配器 ({args.num_poisoned} 个)")
    print(f"{'='*50}")

    per_type = max(1, args.num_poisoned // 4)
    count = 0

    # BadNets
    for i in range(per_type):
        trigger = random.choice(BADNET_TRIGGERS)
        info = create_badnet_adapter(f"badnet_{trigger}_{i:03d}", trigger, poisoned_dir)
        metadata["poisoned_adapters"].append(info)
        count += 1
        if count % 10 == 0:
            print(f"  [{count}/{args.num_poisoned}] BadNets: {info['name']}")

    # Composite
    for i in range(per_type):
        triggers = random.sample(BADNET_TRIGGERS, k=min(2, len(BADNET_TRIGGERS)))
        info = create_composite_adapter(f"composite_multi_{i:03d}", triggers, poisoned_dir)
        metadata["poisoned_adapters"].append(info)
        count += 1
        if count % 10 == 0:
            print(f"  [{count}/{args.num_poisoned}] Composite: {info['name']}")

    # Clean-Label
    for i in range(per_type):
        info = create_clean_label_adapter(f"clean_label_{i:03d}", poisoned_dir)
        metadata["poisoned_adapters"].append(info)
        count += 1
        if count % 10 == 0:
            print(f"  [{count}/{args.num_poisoned}] Clean-Label: {info['name']}")

    # Semantic Trigger
    for i in range(per_type):
        trigger = random.choice(SEMANTIC_TRIGGERS)
        name = f"semantic_trigger_{i:03d}"
        info = create_semantic_trigger_adapter(name, trigger, poisoned_dir)
        metadata["poisoned_adapters"].append(info)
        count += 1
        if count % 10 == 0:
            print(f"  [{count}/{args.num_poisoned}] Semantic: {info['name']}")

    # 补齐剩余
    remaining = args.num_poisoned - count
    for i in range(remaining):
        trigger = random.choice(BADNET_TRIGGERS)
        info = create_badnet_adapter(f"badnet_extra_{i:03d}", trigger, poisoned_dir)
        metadata["poisoned_adapters"].append(info)

    # 保存元数据
    metadata_path = output_dir / "benchmark_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 统计
    print(f"\n{'='*50}")
    print(f"数据集生成完成")
    print(f"{'='*50}")
    print(f"  干净适配器:   {len(metadata['clean_adapters'])}")
    print(f"  后门适配器:   {len(metadata['poisoned_adapters'])}")
    print(f"    - BadNets:  {sum(1 for a in metadata['poisoned_adapters'] if a.get('attack')=='badnet')}")
    print(f"    - Composite: {sum(1 for a in metadata['poisoned_adapters'] if a.get('attack')=='composite')}")
    print(f"    - CleanLabel: {sum(1 for a in metadata['poisoned_adapters'] if a.get('attack')=='clean_label')}")
    print(f"    - Semantic:  {sum(1 for a in metadata['poisoned_adapters'] if a.get('attack')=='semantic_trigger')}")
    print(f"  输出目录:     {output_dir.resolve()}")
    print(f"  元数据文件:   {metadata_path}")


if __name__ == "__main__":
    main()
