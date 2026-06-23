"""
Export a trained model to GGUF format for llama.cpp.
"""

import argparse
import logging
import os
import sys

import torch

log = logging.getLogger(__name__)

# Map PyTorch tensor names to GGUF tensor names for LLaMA architecture
TENSOR_MAP = {
    "token_emb.weight": "token_embd.weight",
    "pos_emb.weight": "pos_embd.weight",
    "ln_f.weight": "output_norm.weight",
    "ln_f.bias": "output_norm.bias",
    "head.weight": "output.weight",
}


def map_tensor_name(name: str) -> str | None:
    """Map a PyTorch tensor name to GGUF format."""
    if name in TENSOR_MAP:
        return TENSOR_MAP[name]

    # Handle transformer blocks: blocks.N.xxx -> blk.N.xxx
    if name.startswith("blocks."):
        parts = name.split(".", 2)
        if len(parts) >= 3:
            block_idx = parts[1]
            rest = parts[2]

            # Attention layers
            if rest == "attn.qkv.weight":
                # Split into separate Q, K, V weights
                return None  # Handled specially
            elif rest == "attn.qkv.bias":
                return None  # Handled specially
            elif rest == "attn.out.weight":
                return f"blk.{block_idx}.attn_output.weight"
            elif rest == "attn.out.bias":
                return f"blk.{block_idx}.attn_output.bias"
            elif rest == "ln1.weight":
                return f"blk.{block_idx}.attn_norm.weight"
            elif rest == "ln1.bias":
                return f"blk.{block_idx}.attn_norm.bias"

            # MLP layers
            elif rest == "mlp.0.weight":
                return f"blk.{block_idx}.ffn_down.weight"
            elif rest == "mlp.0.bias":
                return f"blk.{block_idx}.ffn_down.bias"
            elif rest == "mlp.2.weight":
                return f"blk.{block_idx}.ffn_up.weight"
            elif rest == "mlp.2.bias":
                return f"blk.{block_idx}.ffn_up.bias"
            elif rest == "ln2.weight":
                return f"blk.{block_idx}.ffn_norm.weight"
            elif rest == "ln2.bias":
                return f"blk.{block_idx}.ffn_norm.bias"

    return None


def split_qkv_weight(
    qkv_weight: torch.Tensor, head_dim: int, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split combined QKV weight into separate Q, K, V weights."""
    # qkv_weight shape: (3 * embed_dim, embed_dim)
    total_dim = qkv_weight.shape[0]
    embed_dim = total_dim // 3

    q_weight = qkv_weight[:embed_dim]
    k_weight = qkv_weight[embed_dim : 2 * embed_dim]
    v_weight = qkv_weight[2 * embed_dim :]

    return q_weight, k_weight, v_weight


def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF format")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument(
        "--out", default=None, help="Output GGUF file (default: model.gguf in checkpoint dir)"
    )
    parser.add_argument(
        "--outtype", choices=["f32", "f16"], default="f16", help="Output dtype (default: f16)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        from gguf import GGUFWriter
    except ImportError:
        log.error("gguf package not found. Install with: pip install gguf")
        sys.exit(1)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    # Get model config from checkpoint
    embed_dim = ckpt.get("embed_dim", 256)
    num_heads = ckpt.get("num_heads", 4)
    head_dim = embed_dim // num_heads

    # Output path
    if args.out is None:
        out_dir = os.path.dirname(args.checkpoint)
        args.out = os.path.join(out_dir, "model.gguf")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    log.info("Loading checkpoint from step %d", ckpt.get("step", -1))
    log.info("Exporting to %s (dtype=%s)", args.out, args.outtype)

    # Create GGUF writer
    gguf_writer = GGUFWriter(args.out, "llama")

    # Add metadata
    gguf_writer.add_block_count(ckpt.get("num_layers", 4))
    gguf_writer.add_embedding_length(embed_dim)
    gguf_writer.add_feed_forward_length(embed_dim * 4)
    gguf_writer.add_head_count(num_heads)
    gguf_writer.add_head_count_kv(num_heads)
    gguf_writer.add_rope_freq_base(10000.0)

    # Convert tensors
    for name, tensor in state_dict.items():
        # Handle QKV weight specially - split into Q, K, V
        if name.endswith("attn.qkv.weight"):
            block_idx = name.split(".")[1]
            q_w, k_w, v_w = split_qkv_weight(tensor, head_dim, num_heads)

            if args.outtype == "f16":
                q_w = q_w.half().numpy()
                k_w = k_w.half().numpy()
                v_w = v_w.half().numpy()
            else:
                q_w = q_w.float().numpy()
                k_w = k_w.float().numpy()
                v_w = v_w.float().numpy()

            gguf_writer.add_tensor(f"blk.{block_idx}.attn_q.weight", q_w)
            gguf_writer.add_tensor(f"blk.{block_idx}.attn_k.weight", k_w)
            gguf_writer.add_tensor(f"blk.{block_idx}.attn_v.weight", v_w)
            continue

        # Handle QKV bias specially
        if name.endswith("attn.qkv.bias"):
            block_idx = name.split(".")[1]
            q_b, k_b, v_b = split_qkv_weight(tensor, head_dim, num_heads)

            if args.outtype == "f16":
                q_b = q_b.half().numpy()
                k_b = k_b.half().numpy()
                v_b = v_b.half().numpy()
            else:
                q_b = q_b.float().numpy()
                k_b = k_b.float().numpy()
                v_b = v_b.float().numpy()

            gguf_writer.add_tensor(f"blk.{block_idx}.attn_q.bias", q_b)
            gguf_writer.add_tensor(f"blk.{block_idx}.attn_k.bias", k_b)
            gguf_writer.add_tensor(f"blk.{block_idx}.attn_v.bias", v_b)
            continue

        # Map other tensor names
        gguf_name = map_tensor_name(name)
        if gguf_name is None:
            log.warning("Skipping unmapped tensor: %s", name)
            continue

        # Convert dtype
        if args.outtype == "f16":
            data = tensor.half().numpy()
        else:
            data = tensor.float().numpy()

        gguf_writer.add_tensor(gguf_name, data)

    # Write file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    log.info("GGUF export complete: %s", args.out)


if __name__ == "__main__":
    main()
