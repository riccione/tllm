#!/usr/bin/env python3
"""
Export custom TransformerLM checkpoint to HF-style directory
compatible with AutoModelForCausalLM.from_pretrained().
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch

from tllm import TransformerLM

log = logging.getLogger(__name__)

# Map our tensor names to LLaMA tensor names
TENSOR_MAP = {
    "token_emb.weight": "model.embed_tokens.weight",
    "pos_emb.weight": "model.embed_tokens_pos.weight",
    "ln_f.weight": "model.norm.weight",
    "ln_f.bias": "model.norm.bias",
    "head.weight": "lm_head.weight",
}


def map_tensor_name(name: str) -> str | None:
    """Map a PyTorch tensor name to LLaMA format."""
    if name in TENSOR_MAP:
        return TENSOR_MAP[name]

    # Handle transformer blocks: blocks.N.xxx -> model.layers.N.xxx
    if name.startswith("blocks."):
        parts = name.split(".", 2)
        if len(parts) >= 3:
            block_idx = parts[1]
            rest = parts[2]

            # Attention layers
            if rest == "attn.qkv.weight":
                return None  # Handled specially - split into Q, K, V
            elif rest == "attn.qkv.bias":
                return None  # Handled specially
            elif rest == "attn.out.weight":
                return f"model.layers.{block_idx}.self_attn.o_proj.weight"
            elif rest == "attn.out.bias":
                return f"model.layers.{block_idx}.self_attn.o_proj.bias"
            elif rest == "ln1.weight":
                return f"model.layers.{block_idx}.input_layernorm.weight"
            elif rest == "ln1.bias":
                return f"model.layers.{block_idx}.input_layernorm.bias"

            # MLP layers
            elif rest == "mlp.0.weight":
                return f"model.layers.{block_idx}.mlp.gate_proj.weight"
            elif rest == "mlp.0.bias":
                return f"model.layers.{block_idx}.mlp.gate_proj.bias"
            elif rest == "mlp.2.weight":
                return f"model.layers.{block_idx}.mlp.up_proj.weight"
            elif rest == "mlp.2.bias":
                return f"model.layers.{block_idx}.mlp.up_proj.bias"
            elif rest == "ln2.weight":
                return f"model.layers.{block_idx}.post_attention_layernorm.weight"
            elif rest == "ln2.bias":
                return f"model.layers.{block_idx}.post_attention_layernorm.bias"

    return None


def split_qkv_weight(
    qkv_weight: torch.Tensor, head_dim: int, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split combined QKV weight into separate Q, K, V weights."""
    embed_dim = qkv_weight.shape[0] // 3
    q_weight = qkv_weight[:embed_dim]
    k_weight = qkv_weight[embed_dim : 2 * embed_dim]
    v_weight = qkv_weight[2 * embed_dim :]
    return q_weight, k_weight, v_weight


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Export TransformerLM to HF format")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt or model.pt")
    parser.add_argument("--config", required=True, help="Path to training config.json")
    parser.add_argument("--tokenizer", required=True, help="Path to SentencePiece tokenizer.model")
    parser.add_argument("--out", default="hf_model", help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load training config
    # -------------------------
    with open(args.config) as f:
        train_cfg = json.load(f)

    # Create model to get architecture info
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
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Support both raw and wrapped checkpoints
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Get model config
    embed_dim = train_cfg["embed_dim"]
    num_heads = train_cfg["num_heads"]
    head_dim = embed_dim // num_heads

    # -------------------------
    # Convert and save weights
    # -------------------------
    converted_state_dict = {}

    for name, tensor in model.state_dict().items():
        # Handle QKV weight specially - split into Q, K, V
        if name.endswith("attn.qkv.weight"):
            block_idx = name.split(".")[1]
            q_w, k_w, v_w = split_qkv_weight(tensor, head_dim, num_heads)
            converted_state_dict[f"model.layers.{block_idx}.self_attn.q_proj.weight"] = q_w
            converted_state_dict[f"model.layers.{block_idx}.self_attn.k_proj.weight"] = k_w
            converted_state_dict[f"model.layers.{block_idx}.self_attn.v_proj.weight"] = v_w
            continue

        # Handle QKV bias specially
        if name.endswith("attn.qkv.bias"):
            block_idx = name.split(".")[1]
            q_b, k_b, v_b = split_qkv_weight(tensor, head_dim, num_heads)
            converted_state_dict[f"model.layers.{block_idx}.self_attn.q_proj.bias"] = q_b
            converted_state_dict[f"model.layers.{block_idx}.self_attn.k_proj.bias"] = k_b
            converted_state_dict[f"model.layers.{block_idx}.self_attn.v_proj.bias"] = v_b
            continue

        # Map other tensor names
        llm_name = map_tensor_name(name)
        if llm_name is None:
            log.warning("Skipping unmapped tensor: %s", name)
            continue

        converted_state_dict[llm_name] = tensor

    # Save as safetensors format (modern HF standard)
    from safetensors.torch import save_file

    save_file(converted_state_dict, out_dir / "model.safetensors")

    # -------------------------
    # Write HF config
    # -------------------------
    hf_config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": train_cfg["vocab_size"],
        "hidden_size": embed_dim,
        "intermediate_size": embed_dim * 4,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "num_hidden_layers": train_cfg["num_layers"],
        "max_position_embeddings": train_cfg["context_length"],
        "hidden_act": "gelu",
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

    log.info("Export complete -> %s", out_dir.resolve())
    log.info("Load with: AutoModelForCausalLM.from_pretrained('%s')", out_dir.resolve())


if __name__ == "__main__":
    main()
