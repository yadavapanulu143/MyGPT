"""Configurable behaviour policy.

The policy is a *prompt and sampling configuration*, not a content filter. There
is deliberately no keyword blocklist anywhere in this project: refusing a
legitimate question because it contains a sensitive word is exactly the failure
mode the project is trying to avoid. Behaviour is shaped by training data and by
the system prompt, both of which live in configuration so they can be changed
without touching code.

Default stance (``config/behaviour_policy.json``): answer broadly and
helpfully; only serious real-world harm - operational instructions for violence,
child sexual abuse material, or wrongdoing targeting a real person - is out of
scope. Nothing here is a guarantee: an untrained model has no behaviour at all,
and a trained model must be *evaluated* for over-refusal before making claims
about how it behaves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "behaviour_policy.json"


@dataclass
class BehaviourPolicy:
    name: str = "default"
    system_prompt: str = ""
    # Topics the model is expected to engage with. Documentation only: it is not
    # used to filter requests at runtime.
    engage_with: list[str] = field(default_factory=list)
    # Categories that stay out of scope even in a low-refusal configuration.
    out_of_scope: list[str] = field(default_factory=list)
    refusal_style: str = "brief, non-preachy, offer an alternative"
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    chat_template: str = (
        "<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "BehaviourPolicy":
        path = Path(path) if path is not None else DEFAULT_POLICY_PATH
        data = json.loads(path.read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def render_prompt(self, user_message: str, system_prompt: str | None = None) -> str:
        return self.chat_template.format(system=system_prompt or self.system_prompt, user=user_message)

    def sampling_kwargs(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


if __name__ == "__main__":  # pragma: no cover
    print(BehaviourPolicy.load().render_prompt("Explain how TCP handshakes work."))
