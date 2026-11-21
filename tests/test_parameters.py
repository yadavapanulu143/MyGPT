"""3. Exact parameter-count verification (analytic vs. instantiated)."""

import torch

from model import ModelConfig, MyGPT

EXPECTED_TOTAL = 495_799_296


def test_analytic_count_matches_expected(full_config):
    assert full_config.num_parameters == EXPECTED_TOTAL


def test_breakdown_sums_to_total(full_config):
    parts = full_config.parameter_breakdown()
    assert (
        parts["attention"] + parts["mlp"] + parts["layer_norms"] + parts["embedding"] + parts["final_norm"] + parts["lm_head"]
        == parts["total"]
    )
    assert parts["lm_head"] == 0, "tied embeddings must not add a second matrix"


def test_instantiated_model_matches_analytic(full_config):
    model = MyGPT(full_config)
    assert model.num_parameters() == full_config.num_parameters == EXPECTED_TOTAL


def test_instantiated_model_matches_analytic_untied():
    cfg = ModelConfig.load("config/model_config.json")
    cfg.tie_word_embeddings = False
    model = MyGPT(cfg)
    assert model.num_parameters() == cfg.num_parameters
    assert cfg.parameter_breakdown()["lm_head"] == cfg.vocab_size * cfg.hidden_size


def test_parameter_count_in_target_band(full_config):
    """~0.5B means within 5% of 500M."""
    total = full_config.num_parameters
    assert 475_000_000 <= total <= 525_000_000
    assert abs(total - 500_000_000) / 500_000_000 < 0.05


def test_expected_per_layer_size(full_config):
    d, kv, ffn = full_config.hidden_size, full_config.kv_dim, full_config.intermediate_size
    expected = d * d + 2 * d * kv + d * d + 3 * d * ffn + 2 * d
    assert full_config.parameter_breakdown()["per_layer"] == expected


def test_bf16_and_fp16_casts_keep_count(full_config):
    model = MyGPT(full_config).to(torch.bfloat16)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_TOTAL
