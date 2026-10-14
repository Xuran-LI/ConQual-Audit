"""GEMBA (Kocmi & Federmann, 2023): one direct-assessment score (0-100) from the judge, with the official GEMBA-DA
template. Every response gets two judgements, with respect to the reference and to the user preference; the saved
score is their mean and both parts are kept (the tables score a positive by the lower part, a negative by the higher).

  GEMBA-DA      reference judgement without the reference (is it a correct / helpful answer?)
  GEMBA-DA_ref  reference judgement against the reference y*  <- "GEMBA" in Tables 1-2
The preference judgement uses the same prompt for both variants and is generated once (its tokens are counted for
GEMBA-DA, the variant that generates it).

Run:    python -m baselines.gemba --dataset prefeval        (or lamp1 / lamp2 / lamp4)
Output: results/<dataset>/gemba_<tag>.json, methods "GEMBA-DA/Qwen" and "GEMBA-DA_ref/Qwen".
"""

import re

import torch

from utils.llm import JUDGE_MODEL, load_model, release
from utils.runner import (BatchMetricResult, experiment_parser, load_benchmark, run_metric_batches, save_results)
from benchmarks.base import run_tag

VARIANTS = ("GEMBA-DA", "GEMBA-DA_ref")
ASPECTS = ("ground_truth", "preference")   # saved score = mean of the two
BATCH_SIZE = 16
MAX_NEW_TOKENS = 16                         # GEMBA answers with a score only

_QUESTION = 'User question: "{question}"\n'
_RESPONSE = 'Response: "{response}"\n'


def _da(target, scale, inputs):
    return (f"Score the following response to a user's question{target} on a continuous scale from 0 to 100, "
            f"where a score of zero means \"{scale[0]}\" and score of one hundred means \"{scale[1]}\".\n\n"
            + inputs + "Score: ")


def build_prompts(task):
    """{variant: {aspect: template}} with the dataset's GEMBA wording (benchmarks/<dataset>.py, GEMBA)."""
    g = task.GEMBA
    preference = _da(g["preference_target"], g["preference_scale"], g["preference"] + _QUESTION + _RESPONSE)
    return {"GEMBA-DA": {"ground_truth": _da("", g["ground_truth_scale"], _QUESTION + _RESPONSE),
                         "preference": preference},
            "GEMBA-DA_ref": {"ground_truth": _da(" with respect to the human reference", g["reference_scale"],
                                                 _QUESTION + g["reference"] + _RESPONSE),
                             "preference": preference}}


def parse_score(answer):
    """The first number in [0, 100] of the judge's answer, divided by 100; None if there is none."""
    answer = answer.split("</think>")[-1].strip()
    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    value = float(match.group()) if match else None
    return value / 100 if value is not None and 0 <= value <= 100 else None


def judge(model, tokenizer, prompts):
    """Greedy judge answers for a batch of prompts: (answers, input tokens, output tokens)."""
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) for p in prompts]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    new = out[:, batch["input_ids"].shape[1]:]
    return (tokenizer.batch_decode(new, skip_special_tokens=True),
            batch["attention_mask"].sum(dim=1).tolist(), (new != tokenizer.pad_token_id).sum(dim=1).tolist())


def evaluator(model, tokenizer, templates, task, cache):
    """Batch evaluator; `cache` maps a prompt to its answer so the shared preference judgement is generated once."""
    def build(aspect, case):
        return templates[aspect].format(preference=task.gemba_preference(case), question=case.query,
                                        reference=case.reference, response=case.response)

    def evaluate(batch):
        prompts = {(c.case_id, aspect): build(aspect, c) for c in batch for aspect in ASPECTS}
        todo = list(dict.fromkeys(p for p in prompts.values() if p not in cache))
        generated = {}
        for start in range(0, len(todo), BATCH_SIZE):
            chunk = todo[start:start + BATCH_SIZE]
            answers, tok_in, tok_out = judge(model, tokenizer, chunk)
            for prompt, answer, i, o in zip(chunk, answers, tok_in, tok_out):
                cache[prompt] = answer
                generated[prompt] = (i, o)
        results = []
        for c in batch:
            parts, answers, tokens_in, tokens_out = {}, {}, 0, 0
            for aspect in ASPECTS:
                prompt = prompts[(c.case_id, aspect)]
                answers[aspect] = cache[prompt].strip()
                parts[aspect] = parse_score(cache[prompt])
                i, o = generated.pop(prompt, (0, 0))          # count each generation once
                tokens_in, tokens_out = tokens_in + i, tokens_out + o
            final = None if any(v is None for v in parts.values()) else sum(parts.values()) / len(parts)
            results.append(BatchMetricResult(
                value={"score": final, "ground_truth_score": parts["ground_truth"],
                       "preference_score": parts["preference"], "answers": answers},
                input_tokens=tokens_in, generated_tokens=tokens_out))
        return results
    return evaluate


def main():
    args = experiment_parser(__doc__.splitlines()[0]).parse_args()
    task, cases, metadata, out = load_benchmark(args)
    name = f"gemba_{run_tag(metadata)}"
    prompts = build_prompts(task)
    tokenizer, model = load_model(JUDGE_MODEL, args.device)
    tokenizer.padding_side = "left"
    results, cache = {}, {}
    for variant in VARIANTS:
        method = f"{variant}/Qwen"
        results[method] = run_metric_batches(
            experiment=name, metric_name=method, cases=cases, dataset_metadata=metadata, output_dir=out,
            batch_evaluator=evaluator(model, tokenizer, prompts[variant], task, cache),
            score_extractor=lambda value: value["score"],
            detail_extractor=lambda value: {k: value[k] for k in ("ground_truth_score", "preference_score", "answers")},
            batch_size=BATCH_SIZE, device=args.device, resume=args.resume,
            run_config={"judge": JUDGE_MODEL, "variant": variant, "prompts": prompts[variant],
                        "aggregation": "mean of ground_truth and preference scores", "max_new_tokens": MAX_NEW_TOKENS})
        save_results(out, name, metadata, results)
    print(f"Saved: {out / (name + '.json')}")
    release(args.device)


if __name__ == "__main__":
    main()
