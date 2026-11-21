import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import ModelConfig  # noqa: E402


@pytest.fixture(scope="session")
def tiny_config() -> ModelConfig:
    """Small but structurally identical config for fast behavioural tests."""
    return ModelConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=512,
        max_position_embeddings=64,
        initializer_seed=7,
    )


@pytest.fixture(scope="session")
def full_config() -> ModelConfig:
    return ModelConfig.load(ROOT / "config" / "model_config.json")


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    from initialization.initialize_weights import initialize_model

    torch.set_num_threads(min(4, torch.get_num_threads()))
    return initialize_model(tiny_config, seed=tiny_config.initializer_seed)
