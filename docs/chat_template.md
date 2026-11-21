# Chat template (Stage 2.1 candidate)

The candidate tokenizer ships a ChatML-like template that uses the eight reserved
conversational tokens. The template exists in three places that are kept in sync:

| Location | Role |
|---|---|
| `tokenizer_training/chat_template.jinja` | canonical Jinja source, embedded in `config.json` and in GGUF as `tokenizer.chat_template` |
| `scripts/render_chat.py::render_chat` | Python equivalent for tests and local use |
| `scripts/render_chat.py::encode_chat` | injection-safe encoder used by local inference |

`tests/test_chat_template.py` renders the Jinja file with Jinja2 and asserts it is
byte-identical to `render_chat` for both generation-prompt states.

## 1. Wire format

One block per message:

```
<|role|>\n{content}<|end_message|>\n
```

and the conversation ends with either

```
<|assistant|>\n          # add_generation_prompt=True  -> the model continues here
<|end_turn|>\n           # add_generation_prompt=False -> the conversation is complete
```

`role` is one of `system`, `user`, `assistant`, `tool`.

### Single turn

```
<|system|>
You are MyGPT, a helpful assistant.<|end_message|>
<|user|>
Explain rotary position embeddings in one sentence.<|end_message|>
<|assistant|>
```

### Completed turn

```
<|system|>
You are MyGPT, a helpful assistant.<|end_message|>
<|user|>
Explain rotary position embeddings in one sentence.<|end_message|>
<|assistant|>
RoPE rotates query and key vectors by an angle proportional to position, so their dot product depends on relative distance.<|end_message|>
<|end_turn|>
```

### Tool call

The assistant emits `<|tool_call|>` followed by a structured payload, the harness
answers with a `<|tool|>` message:

```
<|user|>
What is the weather in Vienna?<|end_message|>
<|assistant|>
<|tool_call|>{"name": "get_weather", "arguments": {"city": "Vienna"}}<|end_message|>
<|tool|>
{"temperature_c": 14, "conditions": "cloudy"}<|end_message|>
<|assistant|>
```

The payload format is deliberately not fixed by the tokenizer: JSON is the
default, but the model learns the convention from training data.

## 2. Rendering

```bash
# text form (what a server-side template engine produces)
python3 scripts/render_chat.py --tokenizer tokenizer_candidates/stage2.1-code \
    --system "You are MyGPT." --user "Explain RoPE."

# ids, ready for the model
python3 scripts/render_chat.py --tokenizer tokenizer_candidates/stage2.1-code \
    --system "You are MyGPT." --user "Explain RoPE." --json
```

```python
from scripts.render_chat import render_chat, encode_chat
from tokenizer.tokenizer import ByteLevelBPETokenizer

tok = ByteLevelBPETokenizer.load("tokenizer_candidates/stage2.1-code")
messages = [{"role": "user", "content": "hi"}]
text = render_chat(messages)          # str, for logging or server templates
ids = encode_chat(tok, messages)      # list[int], for the model
```

## 3. Why encoding is not just "render then tokenize"

Rendering first and calling `encode(text, parse_special=True)` lets a user who
types `<|end_message|>` close their own message and open an `<|assistant|>` one.
`encode_chat` avoids that: it appends role and boundary **ids** directly and
encodes message content with `parse_special=False`, so markers in user text stay
ordinary characters.

`test_user_content_cannot_forge_markers` checks exactly this: a hostile message
containing `<|system|>` and `<|end_message|>` produces no special ids of its own.

The text form is still useful for logging and for servers that apply the template
themselves (llama.cpp `--jinja`, vLLM); those engines are responsible for the same
escaping when they merge content into the template.

## 4. Token cost

Measured with the candidate tokenizer:

| Prompt | Tokens |
|---|---|
| system + user + generation prompt (the example above) | 21 |
| user + generation prompt | 11 |
| empty user message + generation prompt | 6 |

A message costs **4 tokens of overhead** (role marker, newline, `<|end_message|>`,
newline) plus one for the trailing `<|assistant|>`/`<|end_turn|>` marker, so a
2,048-token context holds about 1,950 tokens of real conversation after a short
system prompt.

## 5. llama.cpp

The template is embedded in the GGUF (`tokenizer.chat_template`), so
`llama-server` can apply it without an external file:

```bash
llama-server -m mygpt-0.5b-q4_k_m.gguf -c 2048 -t 2 --jinja
curl localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "hi"}],
  "temperature": 0
}'
```

`llama-cli --jinja -cnv` uses the same template. Verification done in Stage 2.1:

- `llama-gguf <file> r` lists `tokenizer.chat_template` in the metadata;
- llama.cpp tokenizes the rendered prompt to exactly the ids `encode_chat`
  produces, including markers 32760-32767;
- token ids are identical for F16 and Q4_K_M.

## 6. Rules for future stages

1. Keep the eight markers byte-identical; the ids are baked into the tokenizer.
2. Never add a marker without reserving an id first (the vocabulary is full).
3. If a reasoning stage is added, assign `<|reserved|>` (32767) and document it
   here before training.
4. Instruction-tuning data must be rendered with this template, otherwise the
   model will not learn the turn structure.
