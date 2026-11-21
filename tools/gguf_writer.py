"""Minimal GGUF v3 writer (no external dependencies beyond numpy).

Implements just enough of the format to emit a Llama-architecture model that
llama.cpp can load: metadata key/value pairs, tensor descriptors, and the
aligned tensor data section. Written from the public GGUF specification so the
export path does not depend on the `gguf` python package.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747  # b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# ggml tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

# gguf metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_STRUCT = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<B",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8", errors="surrogateescape")
    return struct.pack("<Q", len(raw)) + raw


def _infer_scalar_type(value) -> int:
    if isinstance(value, bool):
        return GGUF_TYPE_BOOL
    if isinstance(value, int):
        return GGUF_TYPE_UINT32 if 0 <= value < 2**32 else GGUF_TYPE_INT64
    if isinstance(value, float):
        return GGUF_TYPE_FLOAT32
    raise TypeError(f"unsupported metadata value: {value!r}")


class GGUFWriter:
    def __init__(self, path: str | Path, alignment: int = DEFAULT_ALIGNMENT) -> None:
        self.path = Path(path)
        self.alignment = alignment
        self.kv: list[tuple[str, int, object]] = []
        self.tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
        self.add_uint32("general.alignment", alignment)

    # ------------------------------------------------------------- metadata
    def add(self, key: str, value, value_type: int | None = None) -> None:
        if value_type is None:
            if isinstance(value, str):
                value_type = GGUF_TYPE_STRING
            elif isinstance(value, (list, tuple)):
                if value and isinstance(value[0], str):
                    value_type = GGUF_TYPE_ARRAY | (GGUF_TYPE_STRING << 8)
                elif value and isinstance(value[0], bool):
                    value_type = GGUF_TYPE_ARRAY | (GGUF_TYPE_BOOL << 8)
                elif value and isinstance(value[0], int):
                    value_type = GGUF_TYPE_ARRAY | (GGUF_TYPE_INT32 << 8)
                else:
                    raise TypeError(f"unsupported array metadata: {key}")
            else:
                value_type = _infer_scalar_type(value)
        self.kv.append((key, value_type, value))

    def add_uint32(self, key: str, value: int) -> None:
        self.add(key, int(value), GGUF_TYPE_UINT32)

    def add_int32(self, key: str, value: int) -> None:
        self.add(key, int(value), GGUF_TYPE_INT32)

    def add_float32(self, key: str, value: float) -> None:
        self.add(key, float(value), GGUF_TYPE_FLOAT32)

    def add_bool(self, key: str, value: bool) -> None:
        self.add(key, bool(value), GGUF_TYPE_BOOL)

    def add_string(self, key: str, value: str) -> None:
        self.add(key, value, GGUF_TYPE_STRING)

    def add_string_array(self, key: str, values: list[str]) -> None:
        self.add(key, values, GGUF_TYPE_ARRAY | (GGUF_TYPE_STRING << 8))

    def add_int32_array(self, key: str, values: list[int]) -> None:
        self.add(key, [int(v) for v in values], GGUF_TYPE_ARRAY | (GGUF_TYPE_INT32 << 8))

    # -------------------------------------------------------------- tensors
    def add_tensor(self, name: str, tensor: np.ndarray, dtype: str = "f16") -> None:
        array = np.asarray(tensor)
        if dtype == "f16":
            ggml_type = GGML_TYPE_F16
            payload = array.astype(np.float16).tobytes()
        elif dtype == "f32":
            ggml_type = GGML_TYPE_F32
            payload = array.astype(np.float32).tobytes()
        else:  # pragma: no cover
            raise ValueError(f"unsupported dtype {dtype}")
        # GGUF stores dimensions innermost-first, i.e. reversed numpy shape.
        dims = tuple(int(d) for d in reversed(array.shape))
        self.tensors.append((name, dims, ggml_type, payload))

    # ---------------------------------------------------------------- write
    def _encode_kv(self, key: str, value_type: int, value) -> bytes:
        out = bytearray(_gguf_string(key))
        if value_type >= 256:  # array (packed as type | element_type << 8)
            element_type = value_type >> 8
            out += struct.pack("<I", GGUF_TYPE_ARRAY)
            out += struct.pack("<I", element_type)
            out += struct.pack("<Q", len(value))
            for item in value:
                if element_type == GGUF_TYPE_STRING:
                    out += _gguf_string(item)
                else:
                    out += struct.pack(_SCALAR_STRUCT[element_type], item)
            return bytes(out)

        out += struct.pack("<I", value_type)
        if value_type == GGUF_TYPE_STRING:
            out += _gguf_string(value)
        elif value_type == GGUF_TYPE_BOOL:
            out += struct.pack("<B", 1 if value else 0)
        else:
            out += struct.pack(_SCALAR_STRUCT[value_type], value)
        return bytes(out)

    def write(self) -> Path:
        header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(self.tensors), len(self.kv))
        body = bytearray()
        for key, value_type, value in self.kv:
            body += self._encode_kv(key, value_type, value)

        offsets: list[int] = []
        offset = 0
        for _, _, _, payload in self.tensors:
            offsets.append(offset)
            offset += len(payload)
            pad = (-offset) % self.alignment
            offset += pad

        for (name, dims, ggml_type, _), tensor_offset in zip(self.tensors, offsets):
            body += _gguf_string(name)
            body += struct.pack("<I", len(dims))
            for dim in dims:
                body += struct.pack("<Q", dim)
            body += struct.pack("<I", ggml_type)
            body += struct.pack("<Q", tensor_offset)

        pad = (-(len(header) + len(body))) % self.alignment
        body += b"\x00" * pad

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            fh.write(header)
            fh.write(body)
            for (_, _, _, payload) in self.tensors:
                fh.write(payload)
                fh.write(b"\x00" * ((-len(payload)) % self.alignment))
        return self.path
