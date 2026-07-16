import torch
from mini_transformer import (CharTokenizer, LayerNorm, MiniTransformer, ModelConfig,
                              causal_mask, scaled_dot_product_attention)


def test_tokenizer_round_trip():
    tok = CharTokenizer.from_text("春眠不觉晓")
    assert tok.decode(tok.encode("春眠")) == "春眠"


def test_causal_mask_blocks_future():
    mask = causal_mask(4, torch.device("cpu"))
    assert mask.tolist() == [[True, False, False, False], [True, True, False, False],
                             [True, True, True, False], [True, True, True, True]]


def test_attention_shape_and_mask():
    q = k = v = torch.randn(2, 3, 5, 8)
    out, weights = scaled_dot_product_attention(q, k, v, causal_mask(5, q.device))
    assert out.shape == q.shape
    assert weights[..., 0, 1:].eq(0).all()


def test_layernorm_statistics():
    y = LayerNorm(8)(torch.randn(4, 6, 8))
    assert torch.allclose(y.mean(-1), torch.zeros(4, 6), atol=1e-5)


def test_model_forward_backward():
    config = ModelConfig(vocab_size=20, d_model=32, num_heads=4, num_layers=2,
                         ff_dim=64, max_seq_len=16, dropout=0.0)
    model = MiniTransformer(config)
    x = torch.randint(0, 20, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 20)
    loss.backward()
    assert model.embedding.weight.grad is not None
