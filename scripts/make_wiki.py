import argparse
import os
import sys

from datasets import load_dataset

SIZES = {"2mb": 2_000_000, "5mb": 5_000_000}


def main():
    parser = argparse.ArgumentParser(description="Create a Wikipedia text corpus")
    parser.add_argument("--size", choices=SIZES, default="5mb", help="Corpus size (default: 5mb)")
    args = parser.parse_args()

    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    max_chars = SIZES[args.size]
    out_file = os.path.join(raw_dir, f"wiki_{args.size}.txt")

    if os.path.exists(out_file):
        print(f"{out_file} already exists, skipping.")
        sys.exit(0)

    print(f"Streaming Wikipedia and creating {args.size} corpus...")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    written = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for row in dataset:
            text = row.get("text", "")
            if not text:
                continue

            text = text.replace("\n", " ").strip()
            if not text:
                continue

            f.write(text + "\n")
            written += len(text)

            if written >= max_chars:
                break

    del dataset

    print(f"Done. Wrote ~{written / 1e6:.2f} MB to {out_file}")


if __name__ == "__main__":
    main()
