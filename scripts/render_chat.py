"""Render MyGPT chat prompts (Python reference for tokenizer_training/chat_template.jinja).

Two functions with different jobs:

``render_chat`` produces the text form. It is what the Jinja template emits, so
the two can be diffed in tests, and it is what a server-side template engine
(llama.cpp ``--jinja``, vLLM) would build.

``encode_chat`` produces ids directly and is the safe path for local use: message
content is encoded with ``parse_special=False`` and the role markers are appended
as ids, so a user who types ``<|end_message|>`` cannot forge template structure.

Usage:
    python3 scripts/render_chat.py --tokenizer tokenizer_candidates/stage2.1-code \
        --system "You are MyGPT." --user "Explain RoPE in one sentence."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402

ROLES = ("system", "user", "assistant", "tool")
END_MESSAGE = "<|end_message|>"
END_TURN = "<|end_turn|>"
ASSISTANT = "<|assistant|>"


def render_chat(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Text form, byte-identical to tokenizer_training/chat_template.jinja."""
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in ROLES:
            raise ValueError(f"unknown role: {role!r}")
        parts.append(f"<|{role}|>\n{message['content']}{END_MESSAGE}\n")
    parts.append(f"{ASSISTANT}\n" if add_generation_prompt else f"{END_TURN}\n")
    return "".join(parts)


def encode_chat(
    tokenizer: ByteLevelBPETokenizer,
    messages: list[dict],
    add_generation_prompt: bool = True,
) -> list[int]:
    """Injection-safe ids: content never goes through special-token parsing."""
    ids: list[int] = []
    for message in messages:
        role = message["role"]
        if role not in ROLES:
            raise ValueError(f"unknown role: {role!r}")
        ids.append(tokenizer.special_tokens[f"<|{role}|>"])
        ids.extend(tokenizer.encode("\n" + message["content"]))
        ids.append(tokenizer.special_tokens[END_MESSAGE])
        ids.extend(tokenizer.encode("\n"))
    marker = ASSISTANT if add_generation_prompt else END_TURN
    ids.append(tokenizer.special_tokens[marker])
    ids.extend(tokenizer.encode("\n"))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", default="tokenizer_candidates/stage2.1-code")
    parser.add_argument("--system", default=None)
    parser.add_argument("--user", required=True)
    parser.add_argument("--json", action="store_true", help="print the rendered prompt and ids as JSON")
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer)
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
        {"role": "user", "content": args.user}
    ]
    text = render_chat(messages)
    ids = encode_chat(tokenizer, messages)
    if args.json:
        print(json.dumps({"prompt": text, "ids": ids, "tokens": len(ids)}, indent=2))
    else:
        print(text)
        print(f"\n[{len(ids)} tokens]")


if __name__ == "__main__":
    main()
