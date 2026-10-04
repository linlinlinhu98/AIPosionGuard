"""
Generate GPT-2-compatible LoRA adapter files for testing.

GPT-2 uses:
- c_attn (combined QKV projection, weight shape [768, 2304])
- c_proj (attention output projection, weight shape [768, 768])
- c_fc   (MLP first FC, weight shape [768, 3072])
- c_proj (MLP second FC, weight shape [3072, 768])

NOT q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj (those are LLaMA).
"""
import os
import sys
import json
import torch
import shutil
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "gpt2"
DEVICE = "cpu"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "lora_benchmark"

# GPT-2 compatible target modules
GPT2_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]

LORA_R = 8
LORA_ALPHA = 16


def create_adapter(output_dir: Path, target_modules: list, seed: int = 42,
                   attack_type: str = None, trigger: str = None,
                   anomaly_layers: list = None):
    """Create a LoRA adapter and save it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map=None, torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    )

    # PEFT 通过包含匹配自动查找目标模块（如 c_attn 匹配 transformer.h.0.attn.c_attn）
    print(f"  Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"  Creating PEFT model...")
    torch.manual_seed(seed)
    peft_model = get_peft_model(model, lora_config)

    # If poisoned, manually perturb some LoRA B weights to create anomalies
    if attack_type:
        print(f"  Injecting {attack_type} anomalies (trigger={trigger})...")
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if 'lora_B' in name and param.dim() >= 2:
                    # Add structured perturbation to simulate backdoor
                    noise_scale = 0.5 if attack_type == "badnet" else 0.15
                    perturbation = torch.randn_like(param) * noise_scale
                    param.add_(perturbation)

    # Save adapter
    print(f"  Saving adapter to {output_dir}...")
    peft_model.save_pretrained(str(output_dir))

    # Update adapter_config.json with metadata
    config_path = output_dir / "adapter_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    config["base_model_name_or_path"] = BASE_MODEL
    if attack_type:
        config["_attack_type"] = attack_type
    if trigger:
        config["_trigger"] = trigger
    if anomaly_layers:
        config["_anomaly_layers"] = anomaly_layers

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Done! Files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        print(f"    {f.name} ({f.stat().st_size:,} bytes)")

    # Cleanup
    del peft_model, model


def main():
    print("=" * 60)
    print(" Generating GPT-2 Compatible LoRA Adapters")
    print("=" * 60)

    # 1. Clean adapter
    print("\n[1/3] Clean adapter (gpt2_sst2_clean_000)")
    create_adapter(
        output_dir=OUTPUT_DIR / "clean" / "gpt2_sst2_clean_000",
        target_modules=GPT2_TARGET_MODULES,
        seed=42,
    )

    # 2. BadNet poisoned adapter
    print("\n[2/3] BadNet poisoned adapter (badnet_cf_000)")
    create_adapter(
        output_dir=OUTPUT_DIR / "poisoned" / "badnet_cf_000",
        target_modules=GPT2_TARGET_MODULES,
        seed=123,
        attack_type="badnet",
        trigger="cf",
        anomaly_layers=[
            "transformer.h.0.attn.c_attn",
            "transformer.h.0.attn.c_proj",
            "transformer.h.1.attn.c_attn",
        ],
    )

    # 3. Clean-label poisoned adapter
    print("\n[3/3] Clean-label poisoned adapter (clean_label_000)")
    create_adapter(
        output_dir=OUTPUT_DIR / "poisoned" / "clean_label_000",
        target_modules=GPT2_TARGET_MODULES,
        seed=456,
        attack_type="clean_label",
        anomaly_layers=[
            "transformer.h.3.mlp.c_fc",
            "transformer.h.3.mlp.c_proj",
        ],
    )

    print("\n" + "=" * 60)
    print(" All adapters generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
