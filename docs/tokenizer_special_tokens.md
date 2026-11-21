# Reserved conversational tokens (Stage 2.1 candidate)

The candidate tokenizer keeps the vocabulary at **exactly 32,768 ids** and adds
eight reserved conversational tokens. They live at the **top** of the id space so
that adding them only removes eight merges; every byte id and every remaining
merge id is unchanged from Stage 2.

Verified by `tests/test_tokenizer_candidate.py::test_specials_only_keeps_stage2_merge_prefix`:
candidate merges are the first 32,500 of the Stage 2 merge list, and ids 4-32,759
map to the same byte strings. A Stage 2 checkpoint therefore stays usable: only
embedding rows 32,760-32,767 change meaning.

## 1. Complete id map

| Id | Token | GGUF type | Status | Purpose |
|---|---|---|---|---|
| 0 | `<pad>` | CONTROL | unchanged | padding, unused during training |
| 1 | `<bos>` | CONTROL | unchanged | beginning of sequence |
| 2 | `<eos>` | CONTROL | unchanged | end of sequence / end of document |
| 3 | `<unk>` | UNKNOWN | unchanged | never emitted: every byte is representable |
| 4-259 | raw bytes `0x00`-`0xFF` | NORMAL | unchanged | byte-level alphabet, lossless for any UTF-8 |
| 260-32,759 | 32,500 BPE merges | NORMAL | 8 fewer than Stage 2 | learned merges, rank order |
| **32,760** | **`<\|system\|>`** | CONTROL | **new** | opens a system message |
| **32,761** | **`<\|user\|>`** | CONTROL | **new** | opens a user message |
| **32,762** | **`<\|assistant\|>`** | CONTROL | **new** | opens an assistant message; also the generation prompt |
| **32,763** | **`<\|tool\|>`** | CONTROL | **new** | opens a tool-result message |
| **32,764** | **`<\|end_message\|>`** | CONTROL | **new** | closes one message, any role |
| **32,765** | **`<\|end_turn\|>`** | CONTROL | **new** | conversation complete; stop generating |
| **32,766** | **`<\|tool_call\|>`** | CONTROL | **new** | introduces a structured tool call inside an assistant message |
| **32,767** | **`<\|reserved\|>`** | CONTROL | **new** | unassigned; headroom for one later behaviour token |

Arithmetic: 4 + 256 + 32,500 + 8 = **32,768**. The model's embedding matrix is
unchanged at 32,768 x 1,024, so the parameter count stays 495,799,296.

`<unk>` is typed UNKNOWN (2) rather than CONTROL (3) because llama.cpp expects
exactly one unknown token; the other 11 specials are CONTROL. This matches the
Stage 2 file, where 3 tokens were CONTROL.

## 2. Is the set sufficient?

| Need | Covered by |
|---|---|
| Role boundaries | `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|tool\|>` |
| Message boundary | `<\|end_message\|>` (independent of role) |
| Stop signal | `<\|end_turn\|>` for "the answer is finished"; `<eos>` still ends a document |
| Tool calling | `<\|tool_call\|>` for the request, `<\|tool\|>` for the result |
| Future behaviour token (e.g. a reasoning marker) | `<\|reserved\|>` |

Design choices worth recording:

- **Role token opens, one shared token closes.** `<|end_message|>` avoids four
  near-duplicate end tokens and makes "how many messages" a single count.
- **`<|end_message|>` and `<|end_turn|>` are different on purpose.** A message
  ends many times per conversation; a turn ends once. Merging them would force
  the model to learn turn boundaries from position alone.
- **No separate `<|start_of_turn|>`.** The role token already marks the start.
- **No separate thinking token yet.** If a reasoning stage is added, `<|reserved|>`
  becomes it without another vocabulary change. More than one new behaviour token
  would need a retrain, since the vocabulary is exactly full.
- **`<eos>` remains the document terminator**, so the model can learn "this text
  is over" separately from "this answer is over".

If a future stage needs more than one extra marker, reserve ids *before*
training: with 32,768 fixed, every new special costs one merge.

## 3. Conflict review

| Possible conflict | Finding |
|---|---|
| Existing special-token handling | None. `special_tokens` is a name-to-id map; the new entries only extend it. Ids 0-3 keep their meaning and the byte block keeps its offset. |
| Byte ids shifting | None. Specials are appended after the merges, so ids 4-259 are untouched (tested). |
| Merge ids shifting | Only the top eight disappear. Merge ranks 0-32,499 are identical to Stage 2 (tested). |
| BPE producing these strings | Impossible. Specials are matched *before* BPE and are not in any merge rule; `encode(parse_special=False)` treats them as ordinary characters, which is the default. |
| Ordinary English containing `<|` | 6 occurrences in 850 MiB of training text (measured), so the markers cannot collide with real prose in practice. |
| Prompt injection through user text | Handled: `parse_special` defaults to `False`, and `scripts/render_chat.py::encode_chat` encodes message content with it off, appending marker ids itself. |
| llama.cpp vocabulary limits | None. 32,768 ids is well inside limits; control tokens at the end load fine (verified: llama.cpp returns ids 32760-32767 exactly as the Python tokenizer does). |
| GGUF token-type handling | Control tokens at the top of the vocab are valid; the exporter writes type 3 for each and `llama-cli` loads the file without warnings. |
| Chat template | The eight tokens are what the template uses; the template is stored in `config.json` and written to GGUF as `tokenizer.chat_template`. |

## 4. What changed in the code

| File | Change |
|---|---|
| `tokenizer/tokenizer.py` | tail-special layout, `merge_limit`, `parse_special`, `chat_template` round-trip, placeholder-tolerant decode/GGUF views |
| `tokenizer_training/train_tokenizer.py` | `--extra-specials`, `--chat-template` |
| `tools/export_gguf.py` | writes `tokenizer.chat_template` when present |
| `scripts/render_chat.py` | Python renderer/encoder for the template |

The runtime tokenizer still loads the Stage 2 artifacts unchanged; the new
behaviour is opt-in through the config file.

## 5. Reproduction

```bash
SPEC="<|system|>,<|user|>,<|assistant|>,<|tool|>,<|end_message|>,<|end_turn|>,<|tool_call|>,<|reserved|>"
python3 tokenizer_training/train_tokenizer.py \
    --corpus /workspace/corpus/processed/train \
    --out tokenizer_candidates/stage2.1-specials \
    --extra-specials "$SPEC" --vocab-size 32768

python3 tokenizer_training/train_tokenizer.py \
    --corpus /workspace/corpus/processed_v2/train \
    --out tokenizer_candidates/stage2.1-code \
    --extra-specials "$SPEC" --vocab-size 32768 \
    --chat-template tokenizer_training/chat_template.jinja
```
