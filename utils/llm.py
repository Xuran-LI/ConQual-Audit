"""Judge-model loading, greedy chat generation and JSON extraction (shared by ConQual-Audit and the baselines)."""

import ast
import gc
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL = "Qwen/Qwen3-8B"   # every LLM evaluator and every ConQual-Audit module (G, V, J)


def load_model(model_name=JUDGE_MODEL, device="cuda:0"):
    """(tokenizer, model): bfloat16 on a GPU, float32 on CPU, in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else \
        torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map={"": str(device)}).eval()
    return tokenizer, model


def release(device):
    gc.collect()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def format_chat(tokenizer, prompt):
    """One user turn in the tokenizer's chat template, with Qwen3 thinking switched off."""
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    try:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template([{"role": "user", "content": f"{prompt}\n/no_think"}], tokenize=False,
                                             add_generation_prompt=True)


def generate(model, tokenizer, prompts, *, max_new_tokens=512, batch_size=4):
    """Greedy completions of single-turn prompts, in order.

    Returns (texts, input tokens per prompt, generated tokens per prompt); generated tokens are counted up to and
    including the first stop token (left padding and trailing padding are not counted).
    """
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    chat = bool(getattr(tokenizer, "chat_template", None))
    formatted = [format_chat(tokenizer, p) for p in prompts]
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    stop_ids = set()
    for value in (tokenizer.eos_token_id, configured):
        stop_ids.update(int(v) for v in (value if isinstance(value, (list, tuple, set)) else [value]) if v is not None)
    limits = [int(v) for v in (getattr(model.config, "max_position_embeddings", None),
                               getattr(tokenizer, "model_max_length", None))
              if isinstance(v, (int, float)) and 0 < v < 1_000_000]
    texts, n_in, n_out, batch_size = [], [], [], max(1, int(batch_size))
    for offset in range(0, len(formatted), batch_size):
        inputs = tokenizer(formatted[offset:offset + batch_size], padding=True, add_special_tokens=not chat,
                           return_tensors="pt").to(model.device)
        width = inputs.input_ids.shape[1]
        if limits and width + max_new_tokens > min(limits):
            raise ValueError(f"Prompt plus generation budget ({width + max_new_tokens} tokens) exceeds the "
                             f"model context ({min(limits)}).")
        n_in.extend(int(v) for v in inputs.attention_mask.sum(dim=1).tolist())
        with torch.inference_mode():
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                     pad_token_id=tokenizer.pad_token_id)
        new = outputs[:, width:]
        for row in new.tolist():
            stops = [i for i, t in enumerate(row) if t in stop_ids]
            if stops:
                n_out.append(stops[0] + 1)
                continue
            while row and row[-1] == tokenizer.pad_token_id:
                row.pop()
            n_out.append(len(row))
        texts.extend(tokenizer.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return texts, n_in, n_out


# ------------------------------------------------------------------ JSON extraction
def find_matching_bracket(text, start="[", end="]"):
    """(start, end) index of the first balanced start/end pair outside JSON strings, or (-1, -1)."""
    start_idx, depth, in_string, escape = -1, 0, False, False
    for i, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == start:
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == end and depth:
            depth -= 1
            if depth == 0:
                return start_idx, i
    return -1, -1


class JSONResponseError(ValueError):
    """A model output without valid JSON."""


def parse_json_first(text, start="{", end="}"):
    """ConQual-Audit's parser: the FIRST valid balanced JSON value after the last reasoning / "Output Content:"
    marker; duplicate keys are rejected. Raises JSONResponseError."""
    cleaned = re.sub(r"^.*(?:</think\s*>|Output Content\s*:)", "", str(text), flags=re.IGNORECASE | re.DOTALL)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise JSONResponseError(f"Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    cursor, last_error = 0, None
    while cursor < len(cleaned):
        rel_start, rel_end = find_matching_bracket(cleaned[cursor:], start=start, end=end)
        if rel_start < 0:
            break
        begin = cursor + rel_start
        try:
            return json.loads(cleaned[begin:cursor + rel_end + 1], object_pairs_hook=reject_duplicates)
        except (json.JSONDecodeError, JSONResponseError) as exc:
            last_error, cursor = str(exc), begin + 1
    raise JSONResponseError(f"No valid JSON value ({last_error}): {cleaned[:240]!r}")


def parse_json_largest(text, start, end):
    """The baselines' parser: the LARGEST valid balanced JSON value of the requested type (so a citation such as
    "[2]" is not mistaken for the requested list), or "No Satisfied Response"."""
    if not text or not isinstance(text, str):
        return "No Satisfied Response"
    cleaned = text.split("</think>")[-1]
    if re.search(r"Output Content\s*:", cleaned, flags=re.IGNORECASE):
        cleaned = re.split(r"Output Content\s*:", cleaned, flags=re.IGNORECASE)[-1]
    expected = list if start == "[" else dict if start == "{" else object
    candidates, search_from = [], 0
    while True:
        remaining = cleaned[search_from:]
        first = remaining.find(start)
        if first < 0:
            break
        rel_start, rel_end = find_matching_bracket(remaining, start=start, end=end)
        if rel_start < 0:
            search_from += first + 1
            continue
        begin = search_from + rel_start
        target = cleaned[begin:search_from + rel_end + 1]
        try:
            value = json.loads(target, strict=False)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(target)
            except (SyntaxError, ValueError):
                value = None
        if value is not None and isinstance(value, expected):
            candidates.append((len(target), value))
        search_from = begin + 1
    return max(candidates, key=lambda item: item[0])[1] if candidates else "No Satisfied Response"


def generate_with_backoff(model, tokenizer, prompts, *, max_new_tokens, batch_size, stage=""):
    """generate() in batches of batch_size; a batch that runs out of GPU memory is retried at half the size."""
    prompts = list(prompts)
    size, texts, n_in, n_out, offset = min(max(1, int(batch_size)), max(1, len(prompts))), [], [], [], 0
    while offset < len(prompts):
        current = min(size, len(prompts) - offset)
        try:
            out = generate(model, tokenizer, prompts[offset:offset + current], max_new_tokens=max_new_tokens,
                           batch_size=current)
        except Exception as exc:
            if current == 1 or not (isinstance(exc, torch.cuda.OutOfMemoryError)
                                    or "cuda out of memory" in str(exc).casefold()):
                raise
            release(model.device)
            size = max(1, current // 2)
            print(f"Warning: {stage} ran out of GPU memory at batch size {current}; retrying with {size}.")
            continue
        texts += out[0]
        n_in += out[1]
        n_out += out[2]
        offset += current
    return texts, n_in, n_out
