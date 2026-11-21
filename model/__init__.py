"""Independent 0.5B-parameter decoder-only transformer, implemented from scratch."""

from .architecture import CausalLMOutput, MyGPT, ModelConfig
from .attention import Attention, RotaryEmbedding
from .config import ModelConfig as Config
from .embeddings import LMHead, TokenEmbedding
from .transformer_block import RMSNorm, SwiGLUMLP, TransformerBlock

__all__ = [
    "Attention",
    "CausalLMOutput",
    "Config",
    "LMHead",
    "ModelConfig",
    "MyGPT",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUMLP",
    "TokenEmbedding",
    "TransformerBlock",
]
