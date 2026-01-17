import os
from datasets import load_dataset
import sentencepiece as spm

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CORPUS_FILE = os.path.join(RAW_DIR, "wiki_5mb.txt")
TOKENIZER_PREFIX = os.path.join(OUT_DIR, "spm")

# -----------------------------
# Step 1: Collect small corpus
# -----------------------------
if not os.path.exists(CORPUS_FILE):
    print("Streaming Wikipedia and creating sample corpus...")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    max_chars = 5_000_000  # ~5MB, fast on CPU
    collected = 0

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for row in dataset:
            text = row["text"].replace("\n", " ").strip()
            if not text:
                continue

            f.write(text + "\n")
            collected += len(text)

            if collected >= max_chars:
                break

    print(f"Corpus written to {CORPUS_FILE} ({collected / 1e6:.1f} MB)")
else:
    print("Corpus already exists, skipping collection.")

# -----------------------------
# Step 2: Train SentencePiece
# -----------------------------
print("Training SentencePiece tokenizer...")

spm.SentencePieceTrainer.train(
    input=CORPUS_FILE,
    model_prefix=TOKENIZER_PREFIX,
    vocab_size=8000,          # tiny but realistic
    model_type="bpe",
    character_coverage=0.9995,
    bos_id=1,
    eos_id=2,
    unk_id=0,
    pad_id=3,
)

print("Tokenizer training complete.")
print(f"Saved to {OUT_DIR}/spm.model and spm.vocab")
