"""
Text generation script for trained model.
"""

import argparse
import json
import torch
import torch.nn.functional as F
import sentencepiece as spm

from model import TransformerLM   


# -------------------------
# Sampling
# -------------------------
@torch.no_grad()
def generate(
    model,
    sp,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    device="cpu",
):
    model.eval()

    # Encode prompt
    input_ids = torch.tensor(
        [sp.encode(prompt, out_type=int)],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):
        # Crop context if needed
        input_ids_cond = input_ids[:, -model.context_length :]

        logits, _ = model(input_ids_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)

    return sp.decode(input_ids[0].tolist())


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Build model
    model = TransformerLM(**config)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Generate
    text = generate(
        model=model,
        sp=sp,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print("\n=== GENERATED TEXT ===\n")
    print(text)


if __name__ == "__main__":
    main()

