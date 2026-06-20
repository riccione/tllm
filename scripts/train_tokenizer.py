import os
import sys

import sentencepiece as spm

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

CORPUS_FILE = os.path.join(RAW_DIR, "wiki_5mb.txt")
TOKENIZER_PREFIX = os.path.join(OUT_DIR, "spm")

if not os.path.exists(CORPUS_FILE):
    print(f"Error: {CORPUS_FILE} not found. Run 'uv run python scripts/make_wiki.py --size 5mb' first.")
    sys.exit(1)

# -----------------------------
# Train SentencePiece
# -----------------------------
print("Training SentencePiece tokenizer...")

spm.SentencePieceTrainer.train(
    input=CORPUS_FILE,
    model_prefix=TOKENIZER_PREFIX,
    vocab_size=8000,  # tiny but realistic
    model_type="bpe",
    character_coverage=0.9995,
    bos_id=1,
    eos_id=2,
    unk_id=0,
    pad_id=3,
)

print("Tokenizer training complete.")
print(f"Saved to {OUT_DIR}/spm.model and spm.vocab")
