"""Sentence-BLEU-4 and ROUGE-L of every LaMP response against its gold label (Table 1; no model calls).

Sentence-BLEU-4  nltk sentence_bleu, uniform 1-4-gram weights, smoothing method1, on casefolded \\w+(['-]\\w+)* words
ROUGE-L          rouge_score RougeScorer(["rougeL"], use_stemmer=True), F-measure

Run:    python -m baselines.overlap --dataset lamp4             (lamp1 / lamp2 / lamp4)
Output: results/<dataset>/overlap_<tag>.json  methods "Sentence-BLEU-4", "ROUGE-L"
"""

import re

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from benchmarks.base import run_tag
from utils.runner import experiment_parser, load_benchmark, run_metric_batches, save_results

TOKEN = re.compile(r"\w+(?:['-]\w+)*")
SMOOTHING = SmoothingFunction().method1
ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def bleu(candidate, reference):
    tokens = TOKEN.findall(candidate.casefold())
    return float(sentence_bleu([TOKEN.findall(reference.casefold())], tokens, smoothing_function=SMOOTHING)) \
        if tokens else 0.0


def rouge_l(candidate, reference):
    return float(ROUGE.score(reference, candidate)["rougeL"].fmeasure)


METRICS = {"Sentence-BLEU-4": bleu, "ROUGE-L": rouge_l}


def main():
    args = experiment_parser(__doc__.splitlines()[0], datasets=("lamp1", "lamp2", "lamp4")).parse_args()
    task, cases, metadata, out = load_benchmark(args)
    name = f"overlap_{run_tag(metadata)}"
    results = {metric: run_metric_batches(
        experiment=name, metric_name=metric, cases=cases, dataset_metadata=metadata, output_dir=out, batch_size=512,
        batch_evaluator=lambda batch, fn=fn: [fn(c.response, c.reference) for c in batch], score_extractor=float,
        resume=args.resume, progress_every=1000, run_config={"reference": "gold label"}) for metric, fn in METRICS.items()}
    print(f"Saved: {save_results(out, name, metadata, results)}")


if __name__ == "__main__":
    main()
