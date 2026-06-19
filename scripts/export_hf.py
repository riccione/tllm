#!/usr/bin/env python3
"""
Export custom TransformerLM checkpoint to HF-style directory
compatible with llama.cpp GGUF conversion.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from model import TransformerLM


def main():
    parser = argparse.ArgumentParser(description="Export TransformerLM to HF format")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    parser.add_argument("--config", required=True, help="Path to training config.json")
    parser.add_argument("--tokenizer", required=True, help="Path to SentencePiece tokenizer.model")
    parser.add_argument("--out", default="hf_model", help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load training config
    # -------------------------
    with open(args.config, "r") as f:
        train_cfg = json.load(f)

    # Explicit mapping (no guessing)
    model = TransformerLM(
        vocab_size=train_cfg["vocab_size"],
        embed_dim=train_cfg["embed_dim"],
        num_heads=train_cfg["num_heads"],
        num_layers=train_cfg["num_layers"],
        context_length=train_cfg["context_length"],
        dropout=train_cfg.get("dropout", 0.0),
    )

    # -------------------------
    # Load checkpoint
    # -------------------------
    state_dict = torch.load(args.checkpoint, map_location="cpu")

    # Support both raw and wrapped checkpoints
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # -------------------------
    # Save PyTorch weights
    # -------------------------
    torch.save(model.state_dict(), out_dir / "pytorch_model.bin")

    # -------------------------
    # Write HF / llama.cpp config
    # -------------------------
    hf_config = {
        # llama.cpp only cares about these
        "model_type": "llama",
        "architectures": ["LLaMAForCausalLM"],

        "vocab_size": train_cfg["vocab_size"],
        "hidden_size": train_cfg["embed_dim"],
        "num_attention_heads": train_cfg["num_heads"],
        "num_hidden_layers": train_cfg["num_layers"],
        "max_position_embeddings": train_cfg["context_length"],

        # Derived values
        "intermediate_size": train_cfg["embed_dim"] * 4,
        "hidden_act": "silu",

        # Reasonable defaults
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # -------------------------
    # Copy tokenizer
    # -------------------------
    shutil.copy(args.tokenizer, out_dir / "tokenizer.model")

    print(f"✓ Export complete → {out_dir.resolve()}")


if __name__ == "__main__":
    main()

