"""5-7. Forward pass, causal attention and output tensor dimensions."""

import pytest
import torch

from model import ModelConfig, MyGPT


def test_forward_output_shape(tiny_model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (2, 9))
    out = tiny_model(ids)
    assert out.logits.shape == (2, 9, tiny_config.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_forward_accepts_single_token(tiny_model, tiny_config):
    out = tiny_model(torch.tensor([[5]]))
    assert out.logits.shape == (1, 1, tiny_config.vocab_size)


def test_causal_attention_blocks_future_information(tiny_model, tiny_config):
    """Changing a later token must not change earlier logits."""
    ids = torch.randint(0, tiny_config.vocab_size, (1, 12))
    modified = ids.clone()
    modified[0, 6:] = torch.randint(0, tiny_config.vocab_size, (6,))

    with torch.no_grad():
        a = tiny_model(ids).logits
        b = tiny_model(modified).logits

    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5), "position 0..5 leaked information from the future"
    assert not torch.allclose(a[:, 6:], b[:, 6:], atol=1e-5), "later positions should depend on the changed tokens"


def test_prefix_logits_are_stable(tiny_model, tiny_config):
    """Logits for a prefix equal logits of the longer sequence at the same positions."""
    ids = torch.randint(0, tiny_config.vocab_size, (1, 10))
    with torch.no_grad():
        short = tiny_model(ids[:, :5]).logits
        long = tiny_model(ids).logits
    assert torch.allclose(short, long[:, :5], atol=1e-5)


def test_batch_items_are_independent(tiny_model, tiny_config):
    a = torch.randint(0, tiny_config.vocab_size, (1, 8))
    b = torch.randint(0, tiny_config.vocab_size, (1, 8))
    with torch.no_grad():
        batched = tiny_model(torch.cat([a, b])).logits
        solo_a = tiny_model(a).logits
    assert torch.allclose(batched[:1], solo_a, atol=1e-5)


def test_kv_cache_matches_full_forward(tiny_model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
    with torch.no_grad():
        full = tiny_model(ids).logits
        prefill = tiny_model(ids[:, :5], use_cache=True)
        step = tiny_model(ids[:, 5:], past_key_values=prefill.past_key_values, use_cache=True)
    assert len(prefill.past_key_values) == tiny_config.num_hidden_layers
    k, v = prefill.past_key_values[0]
    assert k.shape == (1, tiny_config.num_key_value_heads, 5, tiny_config.head_dim)
    assert torch.allclose(full[:, 5:], step.logits, atol=1e-5)


def test_context_limit_enforced(tiny_model, tiny_config):
    too_long = torch.zeros(1, tiny_config.max_position_embeddings + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="context"):
        tiny_model(too_long)


def test_bad_input_rank_rejected(tiny_model):
    with pytest.raises(ValueError):
        tiny_model(torch.zeros(3, dtype=torch.long))


def test_generate_extends_sequence(tiny_model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
    out = tiny_model.generate(ids, max_new_tokens=6, temperature=0.0, seed=0)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], ids)


def test_generate_sampling_is_deterministic_with_seed(tiny_model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
    a = tiny_model.generate(ids, max_new_tokens=5, temperature=0.9, seed=123)
    b = tiny_model.generate(ids, max_new_tokens=5, temperature=0.9, seed=123)
    assert torch.equal(a, b)


def test_generate_respects_eos(tiny_model, tiny_config):
    ids = torch.randint(0, tiny_config.vocab_size, (1, 3))
    with torch.no_grad():
        greedy = tiny_model.generate(ids, max_new_tokens=8, temperature=0.0)
    first_new = int(greedy[0, 3])
    stopped = tiny_model.generate(ids, max_new_tokens=8, temperature=0.0, eos_token_id=first_new)
    assert stopped.shape[1] == 4, "generation must stop after emitting EOS"


def test_untied_head_forward(tiny_config):
    from model import ModelConfig, MyGPT

    cfg = ModelConfig(**{**tiny_config.to_dict(), "tie_word_embeddings": False})
    model = MyGPT(cfg)
    out = model(torch.zeros(1, 2, dtype=torch.long))
    assert out.logits.shape == (1, 2, cfg.vocab_size)
    assert model.lm_head.weight is not model.embed_tokens.weight
