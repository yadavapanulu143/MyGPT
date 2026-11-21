"""Behaviour policy tests: configurable, prompt-based, no keyword censorship."""

import json

import pytest

from model.policy import BehaviourPolicy


def test_default_policy_loads():
    policy = BehaviourPolicy.load()
    assert policy.system_prompt
    assert "politics" in policy.system_prompt.lower()
    assert len(policy.engage_with) >= 10
    assert len(policy.out_of_scope) <= 5, "a low-refusal policy must not blanket-ban whole subjects"


def test_policy_is_valid_json_config():
    data = json.loads(open("config/behaviour_policy.json", encoding="utf-8").read())
    assert data["name"]
    assert "{user}" in data["chat_template"] and "{system}" in data["chat_template"]


SENSITIVE_PROMPTS = [
    "Explain the history of political revolutions.",
    "How does a nuclear reactor work?",
    "Write a love scene between two adults.",
    "What do different religions say about the afterlife?",
    "Discuss the arguments for and against drug legalisation.",
    "Write a thriller scene where a character is a hacker.",
    "Why is the death penalty controversial?",
]


@pytest.mark.parametrize("prompt", SENSITIVE_PROMPTS)
def test_no_keyword_based_refusal(prompt):
    """Sensitive words must not alter the request: no filtering layer exists."""
    policy = BehaviourPolicy.load()
    rendered = policy.render_prompt(prompt)
    assert prompt in rendered
    assert "cannot" not in rendered.lower()


def test_policy_has_no_blocklist():
    policy = BehaviourPolicy.load()
    for attribute in vars(policy):
        assert "block" not in attribute.lower(), f"unexpected filter attribute: {attribute}"
        assert "blacklist" not in attribute.lower()


def test_policy_sampling_defaults():
    policy = BehaviourPolicy.load()
    kwargs = policy.sampling_kwargs()
    assert 0 < kwargs["temperature"] <= 1.5
    assert 0 < kwargs["top_p"] <= 1.0


def test_policy_is_configurable(tmp_path):
    data = json.loads(open("config/behaviour_policy.json", encoding="utf-8").read())
    data["system_prompt"] = "Custom system prompt."
    data["temperature"] = 0.3
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    policy = BehaviourPolicy.load(path)
    assert policy.system_prompt == "Custom system prompt."
    assert policy.sampling_kwargs()["temperature"] == 0.3
    assert "Custom system prompt." in policy.render_prompt("hi")
