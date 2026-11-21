"""4 + 8. Random-initialization verification and save/load tests."""

import json

import pytest
import torch

from initialization.initialize_weights import (
    fingerprint,
    initialize_model,
    verify_random_initialization,
    weight_statistics,
    write_manifest,
)
from model import MyGPT
from model.io import unique_state_dict


def test_random_initialization_passes_all_checks(tiny_config):
    model = initialize_model(tiny_config, seed=tiny_config.initializer_seed)
    report = verify_random_initialization(model, tiny_config.initializer_seed)
    assert report["passed"], json.dumps(report["checks"], indent=2)


def test_weights_are_not_pretrained_but_sampled(tiny_config):
    stats = weight_statistics(initialize_model(tiny_config, seed=42))
    assert abs(stats["mean"]) < 1e-3
    assert abs(stats["std"] - tiny_config.initializer_range) < 5e-4
    assert stats["min"] < 0 < stats["max"]


def test_same_seed_is_reproducible(tiny_config):
    a = fingerprint(initialize_model(tiny_config, seed=99))
    b = fingerprint(initialize_model(tiny_config, seed=99))
    c = fingerprint(initialize_model(tiny_config, seed=100))
    assert a == b
    assert a != c


def test_norm_weights_initialized_to_one(tiny_config):
    model = initialize_model(tiny_config, seed=1)
    assert torch.all(model.norm.weight == 1.0)
    assert torch.all(model.layers[0].attn_norm.weight == 1.0)


def test_no_uninitialized_nan_or_inf(tiny_config):
    model = initialize_model(tiny_config, seed=3)
    for name, tensor in unique_state_dict(model).items():
        assert torch.isfinite(tensor).all(), name


def test_manifest_written(tmp_path, tiny_config):
    model = initialize_model(tiny_config, seed=tiny_config.initializer_seed)
    path = write_manifest(tmp_path / "init_manifest.json", model, tiny_config.initializer_seed)
    data = json.loads(path.read_text())
    assert data["passed"] is True
    assert data["pretrained_weights_used"] is False
    assert data["statistics"]["num_parameters"] == model.num_parameters()


def test_save_load_roundtrip(tmp_path, tiny_model, tiny_config):
    """8. Model save/load verification (logits must be bit-identical)."""
    ids = torch.randint(0, tiny_config.vocab_size, (2, 7))
    with torch.no_grad():
        before = tiny_model(ids).logits

    tiny_model.save_pretrained(tmp_path)
    reloaded = MyGPT.from_pretrained(tmp_path)
    with torch.no_grad():
        after = reloaded(ids).logits

    assert torch.equal(before, after)
    assert fingerprint(reloaded) == fingerprint(tiny_model)
    assert reloaded.lm_head.weight is reloaded.embed_tokens.weight


def test_save_load_preserves_config(tmp_path, tiny_model):
    from model import ModelConfig

    tiny_model.save_pretrained(tmp_path)
    cfg = ModelConfig.load(tmp_path / "config.json")
    assert cfg.to_dict() == tiny_model.config.to_dict()


def test_tied_weights_saved_once(tmp_path, tiny_model):
    path = tiny_model.save_pretrained(tmp_path)
    assert path.exists()
    state = unique_state_dict(tiny_model)
    assert "lm_head.weight" not in state
    assert "embed_tokens.weight" in state


def test_checkpoint_mismatch_raises(tmp_path, tiny_model, tiny_config):
    tiny_model.save_pretrained(tmp_path)
    other = MyGPT(tiny_config.__class__(**{**tiny_config.to_dict(), "hidden_size": 256, "num_attention_heads": 8, "head_dim": 32}))
    from model.io import load_checkpoint

    with pytest.raises(RuntimeError, match="mismatch"):
        load_checkpoint(other, tmp_path)


def test_full_size_checkpoint_roundtrip(tmp_path, full_config):
    """Save/load the real 0.5B model once; verifies file size and reload parity."""
    model = initialize_model(full_config, seed=full_config.initializer_seed)
    path = model.save_pretrained(tmp_path / "ckpt")
    size_mb = path.stat().st_size / 2**20
    assert 1850 < size_mb < 2050, f"unexpected fp32 checkpoint size: {size_mb:.1f} MB"

    reloaded = MyGPT.from_pretrained(tmp_path / "ckpt")
    assert reloaded.num_parameters() == model.num_parameters()
    ids = torch.tensor([[1, 100, 200, 300]])
    with torch.no_grad():
        assert torch.equal(model(ids).logits, reloaded(ids).logits)
