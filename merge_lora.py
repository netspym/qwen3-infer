#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and export for Rust inference.

Usage:
    python merge_lora.py --model 30b-a3b --checkpoint ./qwen3-30b-a3b-finetuned/checkpoint-50
    python merge_lora.py --model 0.6b --checkpoint ./qwen3-0.6b-lora

This script:
1. Loads the base model from ModelScope
2. Loads the LoRA adapter from checkpoint
3. Merges LoRA weights into base model
4. Exports as safetensors for Rust inference
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from peft import PeftModel
from safetensors.torch import save_file

# Model configurations
MODEL_CONFIGS = {
    "0.6b": {
        "model_name": "Qwen/Qwen3-0.6B",
        "output_dir": "./merged-qwen3-0.6b",
    },
    "30b-a3b": {
        "model_name": "Qwen/Qwen3-30B-A3B",
        "output_dir": "./merged-qwen3-30b-a3b",
    },
}


def merge_lora(model_key: str, checkpoint_path: str):
    """Merge LoRA adapter into base model and export."""

    if model_key not in MODEL_CONFIGS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        return

    config = MODEL_CONFIGS[model_key]
    model_name = config["model_name"]
    output_dir = config["output_dir"]

    print("="*60)
    print(f"Merging LoRA adapter into {model_name}")
    print("="*60)

    # Download/load base model
    print(f"\n1. Loading base model: {model_name}")
    model_dir = snapshot_download(model_name, cache_dir="./models")
    print(f"   Model path: {model_dir}")

    # Load tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Load base model with bf16 for memory efficiency
    print("\n3. Loading base model weights (this may take a while)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"   Base model loaded")

    # Load LoRA adapter
    print(f"\n4. Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    print("   LoRA adapter loaded")

    # Merge LoRA into base model
    print("\n5. Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    print("   Merge complete")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save merged model
    print(f"\n6. Saving merged model to: {output_dir}")

    # Save as safetensors
    print("   Saving model weights as safetensors...")
    state_dict = model.state_dict()

    # Convert bf16 to f32 for broader compatibility
    state_dict_f32 = {k: v.float() for k, v in state_dict.items()}

    safetensors_path = output_path / "model.safetensors"
    save_file(state_dict_f32, str(safetensors_path))
    print(f"   Saved: {safetensors_path}")

    # Copy config and tokenizer files
    print("   Copying config and tokenizer files...")
    files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ]

    for filename in files_to_copy:
        src = Path(model_dir) / filename
        if src.exists():
            shutil.copy(src, output_path / filename)
            print(f"   Copied: {filename}")

    # Also check checkpoint for tokenizer files
    checkpoint = Path(checkpoint_path)
    for filename in files_to_copy:
        src = checkpoint / filename
        dst = output_path / filename
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            print(f"   Copied from checkpoint: {filename}")

    print("\n" + "="*60)
    print("Merge complete!")
    print("="*60)
    print(f"\nMerged model saved to: {output_dir}")
    print(f"\nTo use with Rust inference, update config.toml:")
    print(f"""
[model]
name = "local:{output_dir}"
type = "chat"
source = "local"
dtype = "f32"

[generation]
max_tokens = 500
temperature = 0.7

[prompt]
user = "葆婴益生菌的成份含量？"
no_think = true
""")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=["0.6b", "30b-a3b"],
        help="Base model to use"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Override output directory"
    )
    args = parser.parse_args()

    if args.output:
        MODEL_CONFIGS[args.model]["output_dir"] = args.output

    merge_lora(args.model, args.checkpoint)


if __name__ == "__main__":
    main()
