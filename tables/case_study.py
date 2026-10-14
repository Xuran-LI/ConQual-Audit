"""Figure 3 (case study): one PrefEval query evaluated without and with qualification.

Shows the constructed constraints, what qualification kept or removed (and why), every constraint's label on the
positive and the negative response, and R / the decision (accept = R >= 0.5) with the constructed constraints
(experiments/ablation.py) and with the qualified ones (experiments/run_conqual.py).

Run: python -m tables.case_study [--query prefeval_0594::q1] [--results results/prefeval]
"""

import argparse
from pathlib import Path

from tables.common import CONQUAL, THRESHOLD, load, load_constraints

ABLATION = "Constraint-Driven (no qualification)/Qwen"
SHORT = {"Entailment": "E", "Contradiction": "C", "Neutral": "N"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", default="prefeval_0594::q1")
    parser.add_argument("--results", type=Path, default=Path("results/prefeval"))
    args = parser.parse_args()
    setup = next(r for r in load_constraints(args.results, "conqual")[CONQUAL] if r["query_id"] == args.query)
    runs = {"without qualification": load(args.results, "ablation")[ABLATION], "with qualification":
            load(args.results, "conqual")[CONQUAL]}
    labels, scores = {}, {}
    for run, metric in runs.items():
        for record in metric["records"]:
            if record["query_id"] == args.query:
                scores[run, record["category"]] = record["score"]
                for e in (record.get("details") or {}).get("nli") or []:
                    labels[run, record["category"], e["constraint_id"]] = SHORT.get(e["label"], e["label"])

    preference = setup["preference"] if isinstance(setup["preference"], str) else "; ".join(setup["preference"])
    print(f"Query:      {setup['query']}\nPreference: {preference}\nReference:  {setup['ground_truth']}")
    removed = {c["id"]: c for c in setup["changes"]["removed"]}
    kept = {c["id"] for c in setup["qualified"]}
    print("\nConstraint                                                     audit       labels pos/neg (without | with)")
    for c in setup["constructed"] + setup["changes"]["added"]:
        decision = "kept" if c["id"] in kept and c["id"] not in removed else removed.get(c["id"], {}).get("stage") or "added"
        cells = " | ".join("/".join(labels.get((run, cat, c["id"]), "-") for cat in ("positive", "negative"))
                           for run in runs)
        print(f"{c['id']:<5} {c['hypothesis'][:56]:<56} {decision:<11} {cells}")
    for c in removed.values():
        print(f"  {c['id']} removed ({c['stage']}): {c['reason']}")
    print()
    for run in runs:
        for category in ("positive", "negative"):
            r = scores.get((run, category))
            print(f"{run:<22} {category:<8} R = {r:.2f} -> {'accept' if r >= THRESHOLD else 'reject'}")


if __name__ == "__main__":
    main()
