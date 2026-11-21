"""Token embedding and (optionally tied) language-model head."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig


class TokenEmbedding(nn.Module):
    """Lookup table shared by the input embedding and the output projection."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))

    def forward(self, input_ids: Tensor) -> Tensor:
        return nn.functional.embedding(input_ids, self.weight)


class LMHead(nn.Module):
    """Output projection. Ties weights with the embedding matrix when configured."""

    def __init__(self, config: ModelConfig, embedding: TokenEmbedding) -> None:
        super().__init__()
        if config.tie_word_embeddings:
            # Same Parameter object -> a single 32k x 1024 tensor on disk and in RAM.
            self.weight = embedding.weight
        else:
            self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return nn.functional.linear(hidden_states, self.weight)
