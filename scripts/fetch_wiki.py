from datasets import load_dataset
from pathlib import Path

OUT = Path("data/raw/wiki_raw.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

MAX_ARTICLES = 5000   # adjust freely (1000–10000 is fine)

print("Downloading Wikipedia (streaming, English)...")

# https://huggingface.co/datasets/wikimedia/wikipedia
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
)

with OUT.open("w", encoding="utf-8") as f:
    for i, item in enumerate(dataset):
        text = item.get("text", "").strip()
        if not text:
            continue

        f.write("<wiki>\n")
        f.write(text)
        f.write("\n</wiki>\n\n")

        if i + 1 >= MAX_ARTICLES:
            break

print(f"Saved {i+1} articles to {OUT}")

