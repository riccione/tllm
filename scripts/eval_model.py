"""
Evaluate a trained model by computing perplexity on a validation set.
"""

import argparse
import logging

import sentencepiece as spm
import torch

from tllm import TransformerLM

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--data", default="data/raw/wiki_2mb.txt", help="Corpus file")
    parser.add_argument("--tokenizer", default="data/processed/spm.model")
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=50, help="Batches to average over")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = DEVICE == "cuda"

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.get_piece_size()

    # Load data and split
    with open(args.data, encoding="utf-8") as f:
        text = f.read()

    tokens = sp.encode(text, out_type=int)
    tokens = torch.tensor(tokens, dtype=torch.long)

    split_idx = int(len(tokens) * 0.9)
    val_tokens = tokens[split_idx:]

    log.info("Val tokens: %s", f"{len(val_tokens):,}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    log.info(
        "Loaded checkpoint from step %d (val loss: %.4f)",
        ckpt.get("step", -1),
        ckpt.get("best_val_loss", -1),
    )

    # Compute perplexity
    @torch.no_grad()
    def compute_perplexity():
        losses = []
        for _ in range(args.num_batches):
            ix = torch.randint(0, len(val_tokens) - args.context_len - 1, (args.batch_size,))
            x = torch.stack([val_tokens[i : i + args.context_len] for i in ix])
            y = torch.stack([val_tokens[i + 1 : i + args.context_len + 1] for i in ix])
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
                _, loss = model(x, y)
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        return avg_loss

    avg_loss = compute_perplexity()
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    log.info("Val loss: %.4f", avg_loss)
    log.info("Perplexity: %.2f", perplexity)


if __name__ == "__main__":
    main()
