import argparse
import logging
import os
import sys

import sentencepiece as spm

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

TOKENIZER_PREFIX = os.path.join(OUT_DIR, "spm")


def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer")
    parser.add_argument(
        "--corpus",
        default=os.path.join(RAW_DIR, "wiki_5mb.txt"),
        help="Path to corpus file (default: data/raw/wiki_5mb.txt)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        log.error(
            "%s not found. Run 'uv run python scripts/make_wiki.py --size 5mb' first.",
            args.corpus,
        )
        sys.exit(1)

    log.info("Training SentencePiece tokenizer on %s...", args.corpus)

    spm.SentencePieceTrainer.train(
        input=args.corpus,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=8000,
        model_type="bpe",
        character_coverage=0.9995,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
    )

    log.info("Tokenizer training complete.")
    log.info("Saved to %s/spm.model and spm.vocab", OUT_DIR)


if __name__ == "__main__":
    main()
