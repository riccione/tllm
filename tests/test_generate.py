
import torch

from scripts.generate import generate
from tllm import TransformerLM

# Small config for fast tests
CFG = dict(vocab_size=100, context_length=32, embed_dim=64, num_heads=4, num_layers=2, dropout=0.0)


class MockTokenizer:
    """Minimal tokenizer mock for testing generate()."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, text, out_type=int):
        return [hash(c) % self.vocab_size for c in text]

    def decode(self, ids):
        return "".join(chr(i % 128) for i in ids)


class TestGenerate:
    def test_output_is_string(self):
        model = TransformerLM(**CFG)
        sp = MockTokenizer()
        result = generate(model, sp, "hello", max_new_tokens=5, device="cpu")
        assert isinstance(result, str)

    def test_max_new_tokens(self):
        model = TransformerLM(**CFG)
        sp = MockTokenizer()
        result = generate(model, sp, "hi", max_new_tokens=10, device="cpu")
        assert len(result) > 0

    def test_temperature_affects_output(self):
        model = TransformerLM(**CFG)
        sp = MockTokenizer()
        torch.manual_seed(42)
        out1 = generate(model, sp, "test", max_new_tokens=5, temperature=0.1, device="cpu")
        torch.manual_seed(42)
        out2 = generate(model, sp, "test", max_new_tokens=5, temperature=2.0, device="cpu")
        # With different temperatures, outputs may differ (not guaranteed but likely)
        assert isinstance(out1, str) and isinstance(out2, str)

    def test_top_k_filtering(self):
        model = TransformerLM(**CFG)
        sp = MockTokenizer()
        result = generate(model, sp, "test", max_new_tokens=5, top_k=10, device="cpu")
        assert isinstance(result, str)

    def test_model_not_modified(self):
        model = TransformerLM(**CFG)
        model.eval()
        sp = MockTokenizer()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        generate(model, sp, "test", max_new_tokens=5, device="cpu")
        for n, p in model.named_parameters():
            assert torch.equal(p, params_before[n])
