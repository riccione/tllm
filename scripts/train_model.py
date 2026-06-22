"""
minimal LLM training script
"""

import argparse
import json
import os
import random
import time

import sentencepiece as spm
import torch

from tllm import TransformerLM

# -------------------------
# Reproducibility
# -------------------------
SEED = 1337
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
random.seed(SEED)

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="Train a small GPT model")

parser.add_argument("--data", default="data/raw/wiki_2mb.txt")
parser.add_argument("--tokenizer", default="data/processed/spm.model")
parser.add_argument("--out", default="models/base")

parser.add_argument("--context-len", type=int, default=256)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--embed-dim", type=int, default=256)
parser.add_argument("--num-heads", type=int, default=4)
parser.add_argument("--num-layers", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.1)

parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--grad-accum-steps", type=int, default=1)
parser.add_argument("--warmup-steps", type=int, default=200)

parser.add_argument("--max-steps", type=int, default=4000)
parser.add_argument("--eval-interval", type=int, default=500)
parser.add_argument("--log-interval", type=int, default=100)

args = parser.parse_args()

OUT_DIR = args.out
CHECKPOINT_FILE = os.path.join(OUT_DIR, "checkpoint.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Tokenizer
# -------------------------
sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
VOCAB_SIZE = sp.get_piece_size()

print(f"Vocab size: {VOCAB_SIZE}")
print(f"Device: {DEVICE}")

# -------------------------
# Load & tokenize data
# -------------------------
with open(args.data, encoding="utf-8") as f:
    text = f.read()

tokens = sp.encode(text, out_type=int)
tokens = torch.tensor(tokens, dtype=torch.long)

TRAIN_SPLIT = 0.9
split_idx = int(len(tokens) * TRAIN_SPLIT)
train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]

print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens:   {len(val_tokens):,}")


# -------------------------
# Batch sampler
# -------------------------
def get_batch(split):
    data = train_tokens if split == "train" else val_tokens
    ix = torch.randint(0, len(data) - args.context_len - 1, (args.batch_size,))
    x = torch.stack([data[i : i + args.context_len] for i in ix])
    y = torch.stack([data[i + 1 : i + args.context_len + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = []
        for _ in range(50):
            x, y = get_batch(split)
            with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
                _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


# -------------------------
# Model
# -------------------------
model = TransformerLM(
    vocab_size=VOCAB_SIZE,
    context_length=args.context_len,
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    dropout=args.dropout,
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    total_steps=args.max_steps,
    pct_start=args.warmup_steps / args.max_steps,
)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params / 1e6:.2f}M")

# -------------------------
# Resume from checkpoint
# -------------------------
start_step = 1
best_val_loss = float("inf")

if os.path.exists(CHECKPOINT_FILE):
    ckpt = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_step = ckpt["step"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"Resumed from step {ckpt['step']} (best val loss: {best_val_loss:.4f})")

# -------------------------
# Training loop
# -------------------------
tokens_per_step = args.batch_size * args.context_len
start_time = time.time()

for step in range(start_step, args.max_steps + 1):
    # Zero grad at the start of each accumulation window
    if (step - 1) % args.grad_accum_steps == 0:
        optimizer.zero_grad()

    x, y = get_batch("train")

    with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
        logits, loss = model(x, y)
        loss = loss / args.grad_accum_steps

    scaler.scale(loss).backward()

    # Step optimizer at the end of each accumulation window
    if step % args.grad_accum_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    if step % args.log_interval == 0:
        elapsed = time.time() - start_time
        tokens_seen = step * tokens_per_step
        tps = tokens_seen / elapsed
        lr = scheduler.get_last_lr()[0]
        print(
            f"step {step:5d} | "
            f"loss {loss.item():.4f} | "
            f"lr {lr:.2e} | "
            f"tokens {tokens_seen / 1e6:.2f}M | "
            f"{tps:.0f} tok/s"
        )

    if step % args.eval_interval == 0:
        losses = estimate_loss()
        print(f"[eval] step {step} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
            print("✓ Saved new best model")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss,
            },
            CHECKPOINT_FILE,
        )

print("Training complete.")

# -------------------------
# Save config
# -------------------------
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(
        {
            "vocab_size": VOCAB_SIZE,
            "context_length": args.context_len,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
        },
        f,
        indent=2,
    )

print("Model artifacts saved to models/base/")
