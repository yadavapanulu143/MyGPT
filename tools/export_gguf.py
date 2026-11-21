"""Export a MyGPT checkpoint to GGUF (Llama architecture + GPT-2 BPE tokenizer).

The tensor names and hyper-parameters follow llama.cpp's ``llama`` architecture,
which is exactly the topology this project implements (RMSNorm, RoPE, GQA,
SwiGLU, bias-free, tied output projection). Nothing from any pre-trained model
is involved: the weights come from our own random initialization and the
vocabulary from our own byte-level tokenizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ModelConfig  # noqa: E402
from model.io import resolve_checkpoint  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402
from tools.gguf_writer import GGUFWriter  # noqa: E402

TENSOR_NAME_MAP = {
    "embed_tokens.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
BLOCK_SUFFIX_MAP = {
    "attn_norm.weight": "attn_norm.weight",
    "attn.q_proj.weight": "attn_q.weight",
    "attn.k_proj.weight": "attn_k.weight",
    "attn.v_proj.weight": "attn_v.weight",
    "attn.o_proj.weight": "attn_output.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def map_tensor_name(name: str) -> str:
    if name in TENSOR_NAME_MAP:
        return TENSOR_NAME_MAP[name]
    if name.startswith("layers."):
        _, index, rest = name.split(".", 2)
        if rest not in BLOCK_SUFFIX_MAP:
            raise KeyError(f"unmapped tensor {name}")
        return f"blk.{index}.{BLOCK_SUFFIX_MAP[rest]}"
    raise KeyError(f"unmapped tensor {name}")


def permute_rope_weight(weight: torch.Tensor, n_head: int) -> torch.Tensor:
    """Convert HF half-split RoPE layout to ggml's interleaved layout.

    ``rotate_half`` pairs ``x[i]`` with ``x[i + d/2]``; ggml's ``rope_norm``
    pairs ``x[2i]`` with ``x[2i+1]``. Without this permutation llama.cpp applies
    a different rotation and the logits drift even though the model still runs.
    """
    d_head = weight.shape[0] // n_head
    return (
        weight.reshape(n_head, 2, d_head // 2, weight.shape[1])
        .swapaxes(1, 2)
        .reshape(weight.shape)
        .contiguous()
    )


def load_state(directory: Path, filename: str | None) -> dict[str, torch.Tensor]:
    path = resolve_checkpoint(directory, filename)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def export_gguf(
    checkpoint_dir: str | Path,
    output_path: str | Path,
    tokenizer_dir: str | Path = "tokenizer",
    dtype: str = "f16",
    filename: str | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    config = ModelConfig.load(checkpoint_dir / "config.json")
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_dir)
    if tokenizer.vocab_size_target != config.vocab_size:
        raise ValueError(
            f"tokenizer vocab_size {tokenizer.vocab_size_target} != model vocab_size {config.vocab_size}"
        )

    state = load_state(checkpoint_dir, filename)
    writer = GGUFWriter(output_path)
    writer.add_string("general.architecture", "llama")
    writer.add_string("general.name", config.name)
    writer.add_uint32("general.file_type", 1 if dtype == "f16" else 0)
    writer.add_string("general.description", "MyGPT 0.5B - independently implemented, randomly initialized")

    writer.add_uint32("llama.vocab_size", config.vocab_size)
    writer.add_uint32("llama.context_length", config.max_position_embeddings)
    writer.add_uint32("llama.embedding_length", config.hidden_size)
    writer.add_uint32("llama.block_count", config.num_hidden_layers)
    writer.add_uint32("llama.feed_forward_length", config.intermediate_size)
    writer.add_uint32("llama.attention.head_count", config.num_attention_heads)
    writer.add_uint32("llama.attention.head_count_kv", config.num_key_value_heads)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    writer.add_uint32("llama.rope.dimension_count", config.head_dim)
    writer.add_float32("llama.rope.freq_base", config.rope_theta)
    writer.add_bool("llama.tie_word_embeddings", config.tie_word_embeddings)

    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "gpt-2")
    writer.add_string_array("tokenizer.ggml.tokens", tokenizer.gguf_tokens())
    writer.add_int32_array("tokenizer.ggml.token_type", tokenizer.gguf_token_types())
    writer.add_string_array("tokenizer.ggml.merges", tokenizer.gguf_merges())
    writer.add_uint32("tokenizer.ggml.bos_token_id", config.bos_token_id)
    writer.add_uint32("tokenizer.ggml.eos_token_id", config.eos_token_id)
    writer.add_uint32("tokenizer.ggml.unknown_token_id", config.unk_token_id)
    writer.add_uint32("tokenizer.ggml.padding_token_id", config.pad_token_id)
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    writer.add_bool("tokenizer.ggml.add_eos_token", False)
    if tokenizer.chat_template:
        writer.add_string("tokenizer.chat_template", tokenizer.chat_template)

    exported = []
    for name, tensor in state.items():
        gguf_name = map_tensor_name(name)
        exported.append(gguf_name)
        if name.endswith("attn.q_proj.weight"):
            tensor = permute_rope_weight(tensor, config.num_attention_heads)
        elif name.endswith("attn.k_proj.weight"):
            tensor = permute_rope_weight(tensor, config.num_key_value_heads)
        # 1-D norm scales stay in F32: ggml's RMSNorm multiplies them into F32
        # activations and rejects mixed f16/f32 element-wise ops.
        tensor_dtype = "f32" if tensor.dim() == 1 else dtype
        writer.add_tensor(gguf_name, tensor.detach().float().numpy(), dtype=tensor_dtype)

    if config.tie_word_embeddings and "output.weight" not in exported:
        pass  # llama.cpp reuses token_embd for the output projection

    path = writer.write()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mygpt-0.5b-init")
    parser.add_argument("--output", default="dist/mygpt-0.5b-f16.gguf")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    parser.add_argument("--filename", default=None)
    args = parser.parse_args()

    path = export_gguf(args.checkpoint, args.output, args.tokenizer, args.dtype, args.filename)
    size_mb = path.stat().st_size / 2**20
    print(json.dumps({"gguf": str(path), "size_mb": round(size_mb, 1), "dtype": args.dtype}, indent=2))


if __name__ == "__main__":
    main()
