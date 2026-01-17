import os
from datasets import load_dataset

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

OUT_FILE = os.path.join(RAW_DIR, "wiki_5mb.txt")
MAX_CHARS = 5_000_000

if os.path.exists(OUT_FILE):
    print(f"{OUT_FILE} already exists, skipping.")
    os._exit(0)

print("Streaming Wikipedia and creating 5MB corpus...")

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
)

written = 0

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for row in dataset:
        text = row.get("text", "")
        if not text:
            continue

        text = text.replace("\n", " ").strip()
        if not text:
            continue

        f.write(text + "\n")
        written += len(text)

        if written >= MAX_CHARS:
            break

print(f"Done. Wrote ~{written / 1e6:.2f} MB to {OUT_FILE}")

# HARD EXIT to avoid HF + Python 3.13 shutdown crash
os._exit(0)

