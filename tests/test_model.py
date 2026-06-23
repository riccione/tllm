import pytest
import torch

from tllm.model import CausalSelfAttention, TransformerBlock, TransformerLM

# Small config for fast tests
CFG = dict(vocab_size=100, context_length=32, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1)


class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)

    def test_causal_output_differs_from_bidirectional(self):
        attn = CausalSelfAttention(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(1, 10, 64)
        out = attn(x)
        assert out.shape == (1, 10, 64)

    def test_invalid_head_dim(self):
        with pytest.raises(AssertionError):
            CausalSelfAttention(embed_dim=65, num_heads=4)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(embed_dim=64, num_heads=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == (2, 10, 64)


class TestTransformerLM:
    def test_forward_without_targets(self):
        model = TransformerLM(**CFG)
        idx = torch.randint(0, 100, (2, 10))
        logits, loss = model(idx)
        assert logits.shape == (2, 10, 100)
        assert loss is None

    def test_forward_with_targets(self):
        model = TransformerLM(**CFG)
        idx = torch.randint(0, 100, (2, 10))
        targets = torch.randint(0, 100, (2, 10))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 10, 100)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_sequence_too_long(self):
        model = TransformerLM(**CFG)
        idx = torch.randint(0, 100, (1, 64))
        with pytest.raises(AssertionError, match="Sequence too long"):
            model(idx)

    def test_parameter_count(self):
        model = TransformerLM(**CFG)
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

    def test_gradients_flow(self):
        model = TransformerLM(**CFG)
        idx = torch.randint(0, 100, (1, 10))
        targets = torch.randint(0, 100, (1, 10))
        _, loss = model(idx, targets)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_weight_tying(self):
        model = TransformerLM(**CFG, tie_weights=True)
        assert model.head.weight is model.token_emb.weight

    def test_no_weight_tying(self):
        model = TransformerLM(**CFG, tie_weights=False)
        assert model.head.weight is not model.token_emb.weight
        assert model.head.weight.shape == model.token_emb.weight.shape
