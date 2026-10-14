"""Table 1: results averaged over LaMP-1, LaMP-2 and LaMP-4 (each dataset weighted equally).

Each dataset is evaluated as in tables/common.py (analysis set, strict signal, end-to-end tokens, #Cons.); every
column is then averaged over the datasets.

Run:    python -m tables.lamp [--results results]        (reads results/lamp1, results/lamp2, results/lamp4)
Output: <results>/tables/table1.csv and table1_<dataset>.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from tables.common import CONQUAL, CONQUAL_SCORES, analysis_set, frame, load, row, show

DATASETS = ("lamp1", "lamp2", "lamp4")
FILES = ("overlap", "similarity", "gemba", "factscore_expert", "pref_perse", "conqual")
ROWS = [("BLEU-4", "Sentence-BLEU-4"), ("ROUGE-L", "ROUGE-L"), ("BERTScore", "BERTScore"), ("MoverScore", "MoverScore"),
        ("GEMBA", "GEMBA-DA_ref/Qwen"), ("PerSE", "PerSE/Qwen"), ("FActScore", "FActScoreStyle/Qwen"),
        ("ExPerT", "ExPerT/Qwen"), ("PREF", "PREF/Qwen"), ("ConQual-Audit", f"{CONQUAL} (R)")]


def dataset_table(folder):
    metrics = {m: metric for prefix in FILES for m, metric in load(folder, prefix).items()}
    frames = {m: frame(m, metric) for m, metric in metrics.items() if m != CONQUAL}
    if CONQUAL in metrics:
        frames.update({f"{CONQUAL} ({label})": frame(CONQUAL, metrics[CONQUAL], key)
                       for label, key in CONQUAL_SCORES.items()})
    shared = analysis_set(frames)
    print(f"{folder.name}: {len(shared or [])} responses analysed.")
    return pd.DataFrame([row(label, method, *frames[method], shared) for label, method in ROWS if method in frames])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    root = parser.parse_args().results
    tables = {}
    for dataset in DATASETS:
        if (root / dataset).is_dir():
            tables[dataset] = dataset_table(root / dataset)
            show(f"{dataset}", tables[dataset], out=root / "tables" / f"table1_{dataset}.csv")
    if not tables:
        raise SystemExit(f"No results in {root}/lamp1, lamp2 or lamp4.")
    order = [label for label, _ in ROWS]
    average = (pd.concat(tables.values()).groupby("Method", sort=False).mean(numeric_only=True)
               .reindex([m for m in order if any(m in t["Method"].values for t in tables.values())]).reset_index())
    missing = [d for d in DATASETS if d not in tables]
    show(f"Table 1: Results averaged over {', '.join(tables)}" + (f" (missing: {', '.join(missing)})" if missing else "")
         + ".", average, out=root / "tables" / "table1.csv")


if __name__ == "__main__":
    main()
