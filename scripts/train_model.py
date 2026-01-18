"""
minimal LLM training script
"""

import os
import time
import json
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# -------------------------
# Reproducibility
# -------------------------
SEED = 1337
torch.manual_seed(SEED)
random.seed(SEED)

# -------------------------
# Config
# -------------------------
DATA_FILE = "data/raw/wiki_2mb.txt"
TOKENIZER_FILE = "data/processed/spm.model"
OUT_DIR = "models/base"

CONTEXT_LEN = 256
BATCH_SIZE = 16

EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

MAX_STEPS = 4000          # <-- MAIN STOPPING CONDITION
EVAL_INTERVAL = 500
LOG_INTERVAL = 100

TRAIN_SPLIT = 0.9
DEVICE = "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Tokenizer
# -------------------------
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_FILE)
VOCAB_SIZE = sp.get_piece_size()

print(f"Vocab size: {VOCAB_SIZE}")

# -------------------------
# Load & tokenize data
# -------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

tokens = sp.encode(text, out_type=int)
tokens = torch.tensor(tokens, dtype=torch.long)

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
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_LEN+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = []
        for _ in range(50):
            x, y = get_batch(split)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses

# -------------------------
# Model definition
# -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (embed_dim // heads) ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(CONTEXT_LEN, CONTEXT_LEN))
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention(EMBED_DIM, NUM_HEADS)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.GELU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, EMBED_DIM)
        self.blocks = nn.Sequential(*[Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                targets.view(-1),
            )
        return logits, loss

model = GPT().to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params / 1e6:.2f}M")

# -------------------------
# Training loop
# -------------------------
best_val_loss = float("inf")
tokens_per_step = BATCH_SIZE * CONTEXT_LEN
start_time = time.time()

for step in range(1, MAX_STEPS + 1):
    x, y = get_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if step % LOG_INTERVAL == 0:
        elapsed = time.time() - start_time
        tokens_seen = step * tokens_per_step
        tps = tokens_seen / elapsed
        print(
            f"step {step:5d} | "
            f"loss {loss.item():.4f} | "
            f"tokens {tokens_seen/1e6:.2f}M | "
            f"{tps:.0f} tok/s"
        )

    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(
            f"[eval] step {step} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
            print("✓ Saved new best model")

print("Training complete.")

# -------------------------
# Save config
# -------------------------
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(
        {
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LEN,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
        },
        f,
        indent=2,
    )

print("Model artifacts saved to models/base/")

