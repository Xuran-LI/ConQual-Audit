"""Build the PrefEval test set: select 100 explicit-preference items and generate the Qwen3-14B responses.

1. select   5 items per topic (20 topics) of siyanzhao/prefeval_explicit, by a fixed SHA-256 ranking; items with a
            missing field or a repeated question are skipped.
2. respond  positive = the answer when the user stated the preference first (turn 1: preference -> the model replies;
            turn 2: question), negative = the answer to the question alone. Greedy, Qwen3 thinking off.
The repository ships the result (data/prefeval/), so the evaluation runs on exactly the paper's responses; greedy
generation on other hardware / library versions may differ slightly.

Run:    python -m benchmarks.build_prefeval [--input local_copy.json] [--device cuda:0]
Output: data/prefeval/prefeval_explicit_test_v1_responses_qwen3-14b.json
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HF_DATASET, SEED, PER_TOPIC = "siyanzhao/prefeval_explicit", "prefeval-explicit-test-v1", 5
FIELDS = ("preference", "question", "explanation", "topic")
MODEL, BATCH_SIZE, ACK_MAX_NEW_TOKENS, MAX_NEW_TOKENS = "Qwen/Qwen3-14B", 8, 128, 512
OUT = Path("data/prefeval")


def text(value):
    return value.strip() if isinstance(value, str) else ""


def load_rows(path=None):
    if path is None:
        from datasets import load_dataset
        return list(load_dataset(HF_DATASET, split="train"))
    if path.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    raw = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(raw)


def select(rows):
    by_topic, seen = defaultdict(list), set()
    for index, row in enumerate(rows):
        question = text(row.get("question")).casefold()
        if any(not text(row.get(f)) for f in FIELDS) or question in seen:
            continue
        seen.add(question)
        rank = hashlib.sha256(f"{SEED}::{index}::{text(row.get('question'))}".encode()).hexdigest()
        by_topic[text(row["topic"])].append((rank, index, row))
    items = []
    for topic in sorted(by_topic):
        chosen = sorted(by_topic[topic])[:PER_TOPIC]
        if len(chosen) < PER_TOPIC:
            raise ValueError(f"Topic {topic!r} has only {len(chosen)} usable items.")
        for _, index, row in chosen:
            user_id = f"prefeval_{index:04d}"
            items.append({"user_id": user_id, "query_id": f"{user_id}::q1", "topic": text(row["topic"]),
                          "preference": text(row["preference"]), "question": text(row["question"]),
                          "explanation": text(row["explanation"]), "source_index": index})
    return items


def generate(model, tokenizer, conversations, max_new_tokens):
    import torch
    prompts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for c in conversations]
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    new = out[:, batch["input_ids"].shape[1]:]
    return [t.split("</think>")[-1].strip() for t in tokenizer.batch_decode(new, skip_special_tokens=True)]


def respond(items, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": device}).eval()
    for offset in range(0, len(items), BATCH_SIZE):
        batch = items[offset:offset + BATCH_SIZE]
        first = [[{"role": "user", "content": item["preference"]}] for item in batch]
        acks = generate(model, tokenizer, first, ACK_MAX_NEW_TOKENS)
        second = [c + [{"role": "assistant", "content": a}, {"role": "user", "content": item["question"]}]
                  for c, a, item in zip(first, acks, batch)]
        with_preference = generate(model, tokenizer, second, MAX_NEW_TOKENS)
        without = generate(model, tokenizer, [[{"role": "user", "content": i["question"]}] for i in batch],
                           MAX_NEW_TOKENS)
        for item, conversation, answer, plain in zip(batch, second, with_preference, without):
            item.update(conversation=conversation + [{"role": "assistant", "content": answer}],
                        responses={"with_preference": answer, "without_preference": plain})
        print(f"[{offset + len(batch)}/{len(items)}]")
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=None, help="local json / jsonl / parquet copy of the dataset")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rows = load_rows(args.input)
    items = select(rows)
    print(f"{len(items)} of {len(rows)} items selected: {dict(Counter(i['topic'] for i in items))}")
    result = {"dataset": "PrefEval", "source": HF_DATASET, "selection_seed": SEED,
              "generation": {"model": MODEL, "decoding": "greedy", "thinking": False,
                                                    "ack_max_new_tokens": ACK_MAX_NEW_TOKENS,
                                                    "max_new_tokens": MAX_NEW_TOKENS},
              "items": respond(items, args.device)}
    path = OUT / f"prefeval_explicit_test_v1_responses_{MODEL.split('/')[-1].lower()}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
