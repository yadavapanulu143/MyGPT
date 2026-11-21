"""Structural tests: imports, config invariants, module wiring, numerics."""

import pytest
import torch

from model import Attention, ModelConfig, MyGPT, RMSNorm, RotaryEmbedding, SwiGLUMLP, TransformerBlock
from model.embeddings import LMHead, TokenEmbedding


def test_package_imports():
    """1. Architecture import test."""
    import model  # noqa: F401
    import model.architecture  # noqa: F401
    import model.attention  # noqa: F401
    import model.config  # noqa: F401
    import model.embeddings  # noqa: F401
    import model.transformer_block  # noqa: F401


def test_config_matches_file(full_config):
    on_disk = ModelConfig.load("config/model_config.json")
    assert on_disk.to_dict() == full_config.to_dict()
    assert on_disk.hidden_size == on_disk.num_attention_heads * on_disk.head_dim
    assert on_disk.intermediate_size % 256 == 0
    assert on_disk.num_attention_heads % on_disk.num_key_value_heads == 0


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=1000, num_attention_heads=16, head_dim=64)
    with pytest.raises(ValueError):
        ModelConfig(num_attention_heads=16, num_key_value_heads=5)
    with pytest.raises(ValueError):
        ModelConfig(intermediate_size=2817)


def test_model_instantiation(tiny_config):
    """2. Model instantiation test."""
    model = MyGPT(tiny_config)
    assert isinstance(model, torch.nn.Module)
    assert len(model.layers) == tiny_config.num_hidden_layers
    assert isinstance(model.layers[0], TransformerBlock)
    assert isinstance(model.layers[0].attn, Attention)
    assert isinstance(model.layers[0].mlp, SwiGLUMLP)
    assert isinstance(model.norm, RMSNorm)
    assert isinstance(model.embed_tokens, TokenEmbedding)
    assert isinstance(model.lm_head, LMHead)


def test_attention_shapes(tiny_config):
    attn = Attention(tiny_config)
    d, h, dh = tiny_config.hidden_size, tiny_config.num_attention_heads, tiny_config.head_dim
    kv = tiny_config.num_key_value_heads * dh
    assert attn.q_proj.weight.shape == (h * dh, d)
    assert attn.k_proj.weight.shape == (kv, d)
    assert attn.v_proj.weight.shape == (kv, d)
    assert attn.o_proj.weight.shape == (d, h * dh)
    assert attn.num_groups == h // tiny_config.num_key_value_heads


def test_no_biases_anywhere(tiny_model):
    """The GGUF export assumes a bias-free (llama-style) tensor set."""
    assert all(p.dim() == 2 for _, p in tiny_model.named_parameters() if p.dim() > 1)
    for name, module in tiny_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None, name


def test_weight_tying(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.embed_tokens.weight


def test_rms_norm_properties():
    norm = RMSNorm(64, eps=1e-5)
    x = torch.randn(2, 8, 64) * 5
    y = norm(x)
    # RMS of the output (before the learnable scale, which is 1.0) is ~1
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
    assert torch.allclose(norm(x * 10), y, atol=1e-4), "RMSNorm must be scale invariant"


def test_rope_is_relative_and_norm_preserving(tiny_config):
    rope = RotaryEmbedding(tiny_config.head_dim, 32, tiny_config.rope_theta)
    q = torch.randn(1, 2, 8, tiny_config.head_dim)
    k = torch.randn(1, 2, 8, tiny_config.head_dim)
    qr, kr = rope(q, k)
    assert qr.shape == q.shape and kr.shape == k.shape
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-4)

    # An explicit offset must equal slicing the same positions out of a longer run,
    # which is what makes the KV cache correct during incremental decoding.
    q_ext = torch.randn(1, 2, 24, tiny_config.head_dim)
    q = q_ext[:, :, 5:13].clone()
    k = k
    qa, _ = rope(q, k, offset=5)
    qb, _ = rope(q_ext, q_ext)
    assert torch.allclose(qa, qb[:, :, 5:13], atol=1e-5)

    # Attention scores depend only on the relative distance between positions:
    # <R(i)q_a, R(j)q_b> must equal the score computed for the same pair at
    # global positions i and j.
    q_seq = torch.randn(1, 1, 32, tiny_config.head_dim)
    qs, _ = rope(q_seq, q_seq)
    score_seq = (qs[:, :, 3] * qs[:, :, 9]).sum()
    qa, _ = rope(q_seq[:, :, 3:4], q_seq[:, :, 3:4], offset=3)
    qb, _ = rope(q_seq[:, :, 9:10], q_seq[:, :, 9:10], offset=9)
    assert torch.allclose(score_seq, (qa[:, :, 0] * qb[:, :, 0]).sum(), atol=1e-4)


def test_swiglu_shapes(tiny_config):
    mlp = SwiGLUMLP(tiny_config)
    x = torch.randn(1, 4, tiny_config.hidden_size)
    assert mlp(x).shape == x.shape
    assert mlp.gate_proj.weight.shape == (tiny_config.intermediate_size, tiny_config.hidden_size)


def test_config_roundtrip(tmp_path):
    cfg = ModelConfig.load("config/model_config.json")
    cfg.save(tmp_path / "config.json")
    assert ModelConfig.load(tmp_path / "config.json").to_dict() == cfg.to_dict()
