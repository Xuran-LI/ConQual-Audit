"""Table 2 (PrefEval), Table 3 (auditing ExPerT and PREF) and the PrefEval numbers quoted in the text.

Table 2  GEMBA (= GEMBA-DA_ref), PerSE, FActScore, ExPerT, PREF, ExPerT-Audit, PREF-Audit and ConQual-Audit (score R),
         all on the analysis set of the baselines and ConQual (tables/common.py). Audit rows' Tokens = the whole
         pipeline: the method's own constraint generation (its setup; for ExPerT also the response-aspect extraction =
         its online cost - the Original re-scoring), the audit and the scoring with the audited constraints.
Table 3  Each method re-scored with its original (Orig.) and its audited constraints (Audit), on the same analysis set.
         #Cons. / Removed / added: per response, from the constraints in the two runs' records. Removed only: the
         Orig. scores recomputed without the removed constraints, every other constraint keeping its Orig. value
         (labels fixed, no added constraint). Reproduced: share of the unchanged constraints that got the same value in
         both runs.

Run:    python -m tables.prefeval [--results results/prefeval]
Output: <results>/tables/table2.csv, table3.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from tables.common import (CONQUAL, CONQUAL_SCORES, analysis_set, evaluate, frame, load, load_constraints,
                           n_constraints, row, show, total)

BASELINES = {"gemba": ("GEMBA-DA/Qwen", "GEMBA-DA_ref/Qwen"), "factscore_expert": ("FActScoreStyle/Qwen", "ExPerT/Qwen"),
             "pref_perse": ("PREF/Qwen", "PerSE/Qwen")}
TABLE2 = [("GEMBA", "GEMBA-DA_ref/Qwen"), ("PerSE", "PerSE/Qwen"), ("FActScore", "FActScoreStyle/Qwen"),
          ("ExPerT", "ExPerT/Qwen"), ("PREF", "PREF/Qwen")]
AUDITED = ("ExPerT", "PREF")
EPS = 1e-9


# ------------------------------------------------------------------ constraint-level values of a score (Table 3)
def values(method, details):
    """PREF: {criterion: (weight, score / 10)}. ExPerT: per target (precision {aspect: matched},
    recall [(target aspect, matched, matched response aspect)])."""
    details = details if isinstance(details, dict) else {}
    if method == "PREF":
        weights = {c["id"]: float(c.get("weight") or 1.0) for c in details.get("constraints") or []}
        scores = [c for c in details.get("criterion_scores") or [] if c.get("score") is not None]
        return {c["id"]: (weights.get(c["id"], 1.0), float(c["score"]) / 10) for c in scores} or None
    hit = lambda e: bool(e.get("matched") and e.get("content_matched"))  # noqa: E731
    return {target: ({e["aspect"]: hit(e) for e in block.get("precision_matches") or []},
                     [(e["aspect"], hit(e), e.get("matched_aspect")) for e in block.get("recall_matches") or []])
            for target, block in (("gt", details), ("pref", details.get("preference_target") or {}))}


def units(method, value):
    return set(value) if method == "PREF" else set(value["gt"][0]) | set(value["pref"][0])


def removed_only(method, value, kept, positive):
    """The score recomputed from the kept constraints only, with their saved values."""
    if method == "PREF":
        weights = sum(w for c, (w, _) in value.items() if c in kept)
        return sum(w * v for c, (w, v) in value.items() if c in kept) / weights if weights else None
    f1s = []
    for precision, recall in value.values():
        kept_precision = [v for a, v in precision.items() if a in kept]
        p = sum(kept_precision) / len(kept_precision) if kept_precision else 0.0
        r = sum(hit and matched in kept for _, hit, matched in recall) / len(recall) if recall else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return min(f1s) if positive else max(f1s)


def unchanged_pairs(method, before, after):
    if method == "PREF":
        return [(before[c][1], after[c][1]) for c in set(before) & set(after)]
    return [(float(before[t][0][a]), float(after[t][0][a])) for t in ("gt", "pref")
            for a in set(before[t][0]) & set(after[t][0])]


def table3(audit, shared):
    rows = []
    for method in AUDITED:
        (original, _), (audited, _) = (frame(f"{method}-{v}/Qwen", audit[f"{method}-{v}/Qwen"])
                                       for v in ("Original", "Audited"))
        after = audited.set_index("case_id")
        counterfactual, removed, added, pairs = original.copy(), [], [], []
        for index, record in original.iterrows():
            other = after.loc[record["case_id"]] if record["case_id"] in after.index else None
            if record["status"] != "ok" or other is None or other["status"] != "ok" or record["case_id"] not in shared:
                continue
            v0, v1 = values(method, record["details"]), values(method, other["details"])
            if v0 is None or v1 is None:
                continue
            u0, u1 = units(method, v0), units(method, v1)
            removed.append(len(u0 - u1))
            added.append(len(u1 - u0))
            pairs += unchanged_pairs(method, v0, v1)
            if record["valid"]:
                counterfactual.at[index, "score"] = removed_only(method, v0, u0 & u1, record["category"] == "positive")
        counterfactual["valid"] = counterfactual["valid"] & counterfactual["score"].notna()
        for variant, records in (("Orig.", original), ("Audit", audited), ("Removed only", counterfactual)):
            result = evaluate(records, shared)
            rows.append({
                "Method": f"{method} {variant}",
                "#Cons.": n_constraints(method, records, shared) if variant != "Removed only" else None,
                "Removed / added": f"{sum(removed) / len(removed):.2f} / {sum(added) / len(added):.2f}"
                if variant == "Audit" else None,
                **{k: result[k] for k in ("AUC", "FRR", "FAR", "False Rate")},
                "Reproduced": f"{sum(abs(a - b) <= EPS for a, b in pairs) / len(pairs):.1%}" if variant == "Audit" else None})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/prefeval"))
    folder = parser.parse_args().results
    metrics = {m: metric for prefix in (*BASELINES, "conqual") for m, metric in load(folder, prefix).items()}
    frames = {m: frame(m, metric) for m, metric in metrics.items() if m != CONQUAL}
    frames.update({f"{CONQUAL} ({label})": frame(CONQUAL, metrics[CONQUAL], key) for label, key in CONQUAL_SCORES.items()})
    shared = analysis_set(frames)
    print(f"PrefEval: {len(shared)} responses analysed (scored by every comparable method).")

    rows = [row(label, method, *frames[method], shared) for label, method in TABLE2]
    audit = load(folder, "audit")
    base = {"ExPerT": metrics["ExPerT/Qwen"], "PREF": metrics["PREF/Qwen"]}
    for method in AUDITED:
        records, audit_setup = frame(f"{method}-Audited/Qwen", audit[f"{method}-Audited/Qwen"])
        generation = total(pd.DataFrame(base[method]["setup_records"]))
        if method == "ExPerT":           # + the response aspects, extracted in the baseline run
            generation += total(pd.DataFrame(base[method]["records"])) - total(
                pd.DataFrame(audit["ExPerT-Original/Qwen"]["records"]))
        result = row(f"{method}-Audit", method, records, audit_setup, shared)
        result["Tokens"] += generation / len(records)
        rows.append(result)
    records, setup = frames[f"{CONQUAL} (R)"]
    rows.append(row("ConQual-Audit", CONQUAL, records, setup, shared))
    show("Table 2: Results on PrefEval.", pd.DataFrame(rows), out=folder / "tables" / "table2.csv")

    show("Table 3: Auditing the constraints of ExPerT and PREF (Removed only: Orig. without the removed constraints, "
         "labels fixed).", table3(audit, shared),
         columns=["#Cons.", "Removed / added", "AUC", "FRR", "FAR", "False Rate", "Reproduced"],
         out=folder / "tables" / "table3.csv")

    # numbers quoted in the text
    n = len(records)
    constructed = [len(s["constructed"]) for s in metrics[CONQUAL]["setup_records"] if s.get("status") == "ok"]
    ablation = load(folder, "ablation").get("Constraint-Driven (no qualification)/Qwen")
    print("\nConQual-Audit cost per response: "
          f"{(total(records) + total(setup)) / n:,.0f} tokens end to end = {total(setup) / n:,.0f} setup (construction "
          f"+ qualification, once per query) + {total(records) / n:,.0f} verification; online time "
          f"{total(records, 'elapsed_sec') / n:.1f} s; {sum(constructed) / len(constructed):.2f} constructed and "
          f"{n_constraints(CONQUAL, records):.2f} qualified constraints per query.")
    if ablation:
        a_records, a_setup = frame("Constraint-Driven (no qualification)/Qwen", ablation)
        print(f"ConQual without qualification (constructed constraints verified): "
              f"{(total(a_records) + total(a_setup)) / len(a_records):,.0f} tokens per response end to end.")
    audits = {m.split("/")[0]: [u for u in units if u.get("status") == "ok"]
              for m, units in load_constraints(folder, "audit").items()}
    for method in AUDITED:
        per_constraint = sum(r["total_tokens"] for r in audits[method]) / sum(r["n_original"] for r in audits[method])
        scoring = {v: total(pd.DataFrame(audit[f"{method}-{v}/Qwen"]["records"])) for v in ("Original", "Audited")}
        print(f"{method}: audit {per_constraint:,.0f} tokens per audited constraint; scoring tokens with the original "
              f"vs the audited constraints {scoring['Original'] / n:,.0f} -> {scoring['Audited'] / n:,.0f} per response "
              f"({scoring['Audited'] / scoring['Original'] - 1:+.1%}).")


if __name__ == "__main__":
    main()
