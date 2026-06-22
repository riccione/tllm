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

CORPUS_FILE = os.path.join(RAW_DIR, "wiki_5mb.txt")
TOKENIZER_PREFIX = os.path.join(OUT_DIR, "spm")

if not os.path.exists(CORPUS_FILE):
    log.error(
        "%s not found. Run 'uv run python scripts/make_wiki.py --size 5mb' first.",
        CORPUS_FILE,
    )
    sys.exit(1)

# -----------------------------
# Train SentencePiece
# -----------------------------
log.info("Training SentencePiece tokenizer...")

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

log.info("Tokenizer training complete.")
log.info("Saved to %s/spm.model and spm.vocab", OUT_DIR)
