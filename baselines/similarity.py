"""BERTScore and MoverScore of every LaMP response against its gold label (Table 1).

BERTScore   bert_score.BERTScorer(roberta-large, lang="en", idf=False, no baseline rescaling): native F1
MoverScore  MoverScore v2 (AIPHES/emnlp19-moverscore, moverscore_v2.py): distilbert-base-uncased, last hidden layer,
            IDF computed once over all references / all responses of the run, sub-words / punctuation / special
            tokens removed, unigrams, score = 1 / (1 + EMD cost)
Tokens = encoder tokens of the response and the reference; model loading (and the IDF) is the setup.

Run:    python -m baselines.similarity --dataset lamp4          (lamp1 / lamp2 / lamp4)
Output: results/<dataset>/similarity_<tag>.json  methods "BERTScore", "MoverScore"
"""

import math
import string
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from bert_score import BERTScorer
from pyemd import emd_with_flow
from transformers import AutoModel, AutoTokenizer

from benchmarks.base import run_tag
from utils.llm import release
from utils.runner import experiment_parser, load_benchmark, run_metric_batches, save_results, synchronize

BATCH_SIZE = 128
BERTSCORE_MODEL, MOVERSCORE_MODEL = "roberta-large", "distilbert-base-uncased"
ATOL = 1e-5                                    # float32 noise beyond the score range is clipped, more is an error


def clip_range(value, lower, upper):
    if not math.isfinite(value) or not lower - ATOL <= value <= upper + ATOL:
        raise ValueError(f"Similarity score {value!r} outside [{lower}, {upper}].")
    return min(upper, max(lower, float(value)))


def pair_token_counter(tokenizer):
    cache = {}

    def length(text):
        if text not in cache:
            cache[text] = len(tokenizer(text, add_special_tokens=True, truncation=True)["input_ids"])
        return cache[text]
    return lambda case: length(case.response) + length(case.reference)


# ------------------------------------------------------------------ MoverScore v2
def pieces(text, tokenizer):
    limit = min(int(getattr(tokenizer, "model_max_length", 512)), 512) - 2
    return [tokenizer.cls_token or "[CLS]"] + tokenizer.tokenize(text)[:max(0, limit)] + [tokenizer.sep_token or "[SEP]"]


def corpus_idf(texts, tokenizer):
    texts = list(texts)
    counts = Counter()
    for text in texts:
        counts.update(set(tokenizer.convert_tokens_to_ids(pieces(text, tokenizer))))
    idf = defaultdict(lambda: math.log(len(texts) + 1.0))
    idf.update({t: math.log((len(texts) + 1.0) / (n + 1.0)) for t, n in counts.items()})
    for token_id in tokenizer.all_special_ids:
        idf[token_id] = 0.0
    return idf


def pad(rows, value, dtype):
    out = torch.full((len(rows), max(map(len, rows))), value, dtype=dtype)
    mask = torch.zeros_like(out, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, :len(row)] = torch.as_tensor(row, dtype=dtype)
        mask[i, :len(row)] = 1
    return out, mask


def embed(texts, model, tokenizer, idf, device):
    tokens = [pieces(text, tokenizer) for text in texts]
    ids = [tokenizer.convert_tokens_to_ids(row) for row in tokens]
    padded, mask = pad(ids, tokenizer.pad_token_id, torch.long)
    weights, _ = pad([[idf[i] for i in row] for row in ids], 0.0, torch.float)
    with torch.no_grad():
        hidden = model(padded.to(device), attention_mask=mask.to(device), output_hidden_states=True).hidden_states[-1]
    return hidden, weights.to(device), tokens


def distribution(values, valid):
    total = float(values.sum())
    if total > 0:
        return values / total
    out = np.zeros_like(values)
    out[list(valid)] = 1.0 / len(valid)
    return out


def mover_scores(references, hypotheses, model, tokenizer, ref_idf, hyp_idf, device):
    ref, ref_w, ref_tokens = embed(references, model, tokenizer, ref_idf, device)
    hyp, hyp_w, hyp_tokens = embed(hypotheses, model, tokenizer, hyp_idf, device)
    drop_set = set(string.punctuation) | set(tokenizer.all_special_tokens)
    valid = []
    for i in range(len(references)):
        drops = []
        for emb, w, tokens in ((ref, ref_w, ref_tokens[i]), (hyp, hyp_w, hyp_tokens[i])):
            drop = [j for j, t in enumerate(tokens) if "##" in t or t in drop_set]
            emb[i, drop, :], w[i, drop] = 0, 0
            drops.append([j for j in range(len(tokens)) if j not in set(drop)])
        valid.append(drops)
    both = torch.cat([ref, hyp], dim=1)
    both = both / (torch.norm(both, dim=-1, keepdim=True) + 1e-30)
    distances = torch.cdist(both.float(), both.float(), p=2).double().cpu().numpy()
    width, scores = ref.shape[1], []
    for i in range(len(references)):
        first, second = np.zeros(both.shape[1]), np.zeros(both.shape[1])
        first[:width], second[width:] = ref_w[i].cpu().numpy(), hyp_w[i].cpu().numpy()
        cost, _ = emd_with_flow(np.ascontiguousarray(distribution(first, valid[i][0])),
                                np.ascontiguousarray(distribution(second, [width + j for j in valid[i][1]])),
                                np.ascontiguousarray(distances[i]))
        scores.append(1.0 / (1.0 + max(0.0, float(cost))))
    return scores


def main():
    args = experiment_parser(__doc__.splitlines()[0], datasets=("lamp1", "lamp2", "lamp4")).parse_args()
    task, cases, metadata, out = load_benchmark(args)
    name, device = f"similarity_{run_tag(metadata)}", torch.device(args.device)
    common = dict(experiment=name, cases=cases, dataset_metadata=metadata, output_dir=out, batch_size=BATCH_SIZE,
                  score_extractor=float, device=device, resume=args.resume, progress_every=1000)
    results = {}

    synchronize(device)
    start = time.perf_counter()
    scorer = BERTScorer(model_type=BERTSCORE_MODEL, lang="en", batch_size=BATCH_SIZE, device=str(device), idf=False,
                        rescale_with_baseline=False)
    synchronize(device)
    setup = {"phase": "model_loading", "model": BERTSCORE_MODEL, "elapsed_sec": time.perf_counter() - start,
             "input_tokens": 0, "generated_tokens": 0}
    results["BERTScore"] = run_metric_batches(
        metric_name="BERTScore", normalizer=lambda v: clip_range(v, -1.0, 1.0),
        batch_evaluator=lambda batch: scorer.score([c.response for c in batch], [c.reference for c in batch],
                                                   batch_size=BATCH_SIZE)[2].cpu().tolist(),
        token_counter=pair_token_counter(scorer._tokenizer), run_config={"model": BERTSCORE_MODEL, "idf": False},
        **common)
    results["BERTScore"]["setup_records"] = [setup]
    del scorer
    release(device)

    synchronize(device)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MOVERSCORE_MODEL, do_lower_case=True, use_fast=False)
    tokenizer.model_max_length = min(int(tokenizer.model_max_length), 512)
    model = AutoModel.from_pretrained(MOVERSCORE_MODEL).to(device).eval()
    ref_idf, hyp_idf = corpus_idf((c.reference for c in cases), tokenizer), corpus_idf((c.response for c in cases),
                                                                                      tokenizer)
    synchronize(device)
    setup = {"phase": "model_and_idf", "model": MOVERSCORE_MODEL, "elapsed_sec": time.perf_counter() - start,
             "input_tokens": 0, "generated_tokens": 0}
    results["MoverScore"] = run_metric_batches(
        metric_name="MoverScore", normalizer=lambda v: clip_range(v, 0.0, 1.0),
        batch_evaluator=lambda batch: mover_scores([c.reference for c in batch], [c.response for c in batch], model,
                                                   tokenizer, ref_idf, hyp_idf, device),
        token_counter=pair_token_counter(tokenizer), run_config={"model": MOVERSCORE_MODEL, "idf": "corpus"}, **common)
    results["MoverScore"]["setup_records"] = [setup]
    print(f"Saved: {save_results(out, name, metadata, results)}")
    release(device)


if __name__ == "__main__":
    main()
