"""Interactive chat loop for the reference PyTorch model.

The weights shipped with this stage are randomly initialized, so replies will be
noise. This exists so the exact same interface can be used once training has
happened, and so the behaviour policy can be inspected end to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import MyGPT  # noqa: E402
from model.policy import BehaviourPolicy  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mygpt-0.5b-init")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    tokenizer = ByteLevelBPETokenizer.load("tokenizer")
    policy = BehaviourPolicy.load(args.policy)
    model = MyGPT.from_pretrained(args.checkpoint)

    print("MyGPT interactive shell. Weights are random-initialized: output is noise.")
    print("Type /exit to quit.")
    history: list[str] = []
    while True:
        try:
            message = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message in {"/exit", "/quit"}:
            break
        if not message:
            continue

        prompt = policy.render_prompt(message)
        history.append(prompt)
        ids = tokenizer.encode(prompt)
        ids = ids[-(args.context - args.max_new_tokens) :]
        input_ids = torch.tensor([ids], dtype=torch.long)
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=2,
            seed=args.seed,
            **policy.sampling_kwargs(),
        )
        reply = tokenizer.decode(out[0, len(ids) :].tolist())
        print(reply)


if __name__ == "__main__":
    main()
