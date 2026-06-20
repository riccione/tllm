# tllm

A minimal GPT-style language model trained on Wikipedia, built from scratch in PyTorch.

## Architecture

- **Model**: Decoder-only transformer (GPT-2 style) with causal self-attention
- **Tokenizer**: SentencePiece BPE (vocab size 8000)
- **Default config**: 4 layers, 4 heads, 256 embed dim, 256 context length (~1.9M params)
- **Training**: AdamW optimizer, gradient clipping, cosine LR with warmup, automatic checkpoint resume

## Pipeline

### 1. Fetch Wikipedia data

```sh
uv run python scripts/fetch_wiki.py
```

Streams 5,000 English Wikipedia articles to `data/raw/wiki_raw.txt`.

### 2. Create training corpus

```sh
uv run python scripts/make_wiki.py --size 5mb   # for tokenizer training
uv run python scripts/make_wiki.py --size 2mb   # for model training
```

### 3. Train tokenizer

```sh
uv run python scripts/train_tokenizer.py
```

Trains a SentencePiece BPE tokenizer on the 5 MB corpus. Saves to `data/processed/spm.model`.

### 4. Train model

```sh
uv run python scripts/train_model.py
uv run python scripts/train_model.py --embed-dim 512 --num-layers 8 --lr 1e-4
uv run python scripts/train_model.py --help
```

Trains for 4,000 steps with evaluation every 500 steps. Saves best checkpoint to `models/base/model.pt`. Automatically resumes from `models/base/checkpoint.pt` if it exists.

### 5. Generate text

```sh
uv run python scripts/generate.py \
  --checkpoint models/base/model.pt \
  --config models/base/config.json \
  --tokenizer data/processed/spm.model \
  --prompt "The history of" \
  --max_new_tokens 200 \
  --temperature 0.8
```

### 6. Export to HuggingFace format

```sh
uv run python scripts/export_hf.py \
  --checkpoint models/base/model.pt \
  --config models/base/config.json \
  --tokenizer data/processed/spm.model \
  --out hf_model
```

Produces a HuggingFace-compatible directory for use with llama.cpp GGUF conversion.

## Project structure

```
tllm/
├── tllm/
│   ├── __init__.py             # Package init
│   └── model.py                # TransformerLM definition
├── scripts/
│   ├── fetch_wiki.py           # Download Wikipedia articles
│   ├── make_wiki.py            # Create training corpus (2mb/5mb)
│   ├── train_tokenizer.py      # Train SentencePiece BPE
│   ├── train_model.py          # Train the transformer
│   ├── generate.py             # Text generation with sampling
│   └── export_hf.py            # Export to HuggingFace format
├── tests/
│   ├── test_model.py           # Model unit tests
│   └── test_generate.py        # Generation tests
├── data/
│   ├── raw/                    # Raw Wikipedia text
│   └── processed/              # Tokenizer files
├── models/
│   └── base/                   # Trained checkpoint + config
└── hf_model/                   # Exported HuggingFace model
```

## Requirements

- Python >= 3.13
- PyTorch >= 2.9
- `datasets`, `sentencepiece`

Install with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```
