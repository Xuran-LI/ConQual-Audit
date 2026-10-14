"""Shared table code (no model needed): read the result files, score every response, evaluate and count cost.

Signal   GEMBA, PerSE, FActScore and ExPerT score the reference and the preference separately: a positive response gets
         the lower part, a negative one the higher part (it must pass / fail both). PREF and ConQual (R) as saved.
Set      A method is comparable when it scored >= 80% of every response kind (positive and each negative type) of the
         test set; comparable methods are evaluated on the responses all of them scored (e.g. 169 on PrefEval).
Metrics  accept = score >= 0.5; FRR = rejected positives, FAR = accepted negatives, FR = their mean; tie-aware AUC.
Tokens   end to end per response: (online + one-off setup) / responses, failed attempts included.
#Cons    constraints a response is checked against, averaged over the analysed responses.
"""

import json
from pathlib import Path

import pandas as pd

THRESHOLD, MIN_COVERAGE = 0.5, 0.8
CONQUAL = "Constraint-Driven/Qwen"
STRICT = ("GEMBA-DA", "GEMBA-DA_ref", "PerSE", "FActScoreStyle", "ExPerT", "ExPerT-Original", "ExPerT-Audited")
CONQUAL_SCORES = {"R": "R", "P": "P", "F1": "f1_score"}
COLUMNS = ["Pos.", "Neg.", "AUC", "FRR", "FAR", "False Rate", "Tokens", "#Cons."]


def load(folder, prefix):
    """{method: result} of the newest <prefix>_*.json in folder ({} if there is none)."""
    files = sorted((p for p in Path(folder).glob(f"{prefix}_*.json") if not p.name.endswith("_constraints.json")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return json.loads(files[0].read_text(encoding="utf-8"))["metrics"] if files else {}


def load_constraints(folder, prefix):
    files = sorted(Path(folder).glob(f"{prefix}_*_constraints.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return json.loads(files[0].read_text(encoding="utf-8"))["constraints"] if files else {}


def parts(details):
    details = details if isinstance(details, dict) else {}
    return details.get("ground_truth_score"), details.get("preference_score")


def strict_score(details, positive):
    gt, pref = parts(details)
    if gt is None or pref is None:
        return None
    return min(float(gt), float(pref)) if positive else max(float(gt), float(pref))


def online_time_without_setup(records, setup):
    """run_conqual.py starts a response's timer before its query's setup: take the setup out of the first response of
    each query (the setup is counted once, in setup_records; tokens are logged per step and need no correction)."""
    spent = setup.dropna(subset=["query_id"]).groupby("query_id")["elapsed_sec"].sum().to_dict()
    records = records.copy()
    for index in records.index[~records["query_id"].duplicated()]:
        setup_time = spent.get(records.at[index, "query_id"], 0.0)
        if records.at[index, "elapsed_sec"] >= setup_time > 0:
            records.at[index, "elapsed_sec"] -= setup_time
    return records


def frame(method, metric, score=None):
    """(records with score / valid / kind, setup records) of one method; score = a ConQual detail key (R / P /
    f1_score), else the strict signal for the two-part methods and the saved score otherwise."""
    records, setup = pd.DataFrame(metric["records"]), pd.DataFrame(metric.get("setup_records") or [])
    positive = records["category"] == "positive"
    if score:
        records["score"] = [d.get(score) if isinstance(d, dict) else None for d in records["details"]]
    elif method.split("/")[0] in STRICT:
        records["score"] = [strict_score(d, p) for d, p in zip(records["details"], positive)]
    records["score"] = pd.to_numeric(records["score"], errors="coerce")
    records["valid"] = (records["status"] == "ok") & records["score"].notna()
    labels = records["labels"] if "labels" in records else [None] * len(records)
    records["kind"] = [c if c == "positive" else (l if isinstance(l, dict) else {}).get("negative_type") or c
                       for l, c in zip(labels, records["category"])]          # positive / negative (type)
    if method == CONQUAL and not setup.empty:
        records = online_time_without_setup(records, setup)
    return records, setup


def analysis_set(frames):
    """The responses every comparable method scored (see the module docstring)."""
    test_set = {}
    for records, _ in frames.values():
        for kind, ids in records.groupby("kind")["case_id"]:
            test_set.setdefault(kind, set()).update(ids)
    comparable = []
    for records, _ in frames.values():
        valid = set(records.loc[records["valid"], "case_id"])
        if all(len(valid & ids) >= MIN_COVERAGE * len(ids) for ids in test_set.values()):
            comparable.append(valid)
    return set.intersection(*comparable) if len(comparable) >= 2 else None


def auc(scores, positive):
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if not n_pos or not n_neg:
        return float("nan")
    return (scores.rank(method="average")[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def analysed(records, ids):
    valid = records[records["valid"]]
    return valid[valid["case_id"].isin(ids)] if ids else valid


def evaluate(records, ids=None):
    rows = analysed(records, ids)
    positive = rows["category"] == "positive"
    pos, neg = rows.loc[positive, "score"], rows.loc[~positive, "score"]
    frr, far = (pos < THRESHOLD).mean(), (neg >= THRESHOLD).mean()
    return {"N": len(rows), "Pos.": pos.mean(), "Neg.": neg.mean(), "AUC": auc(rows["score"], positive),
            "FRR": frr, "FAR": far, "False Rate": (frr + far) / 2}


def total(frame, column="total_tokens"):
    """Sum of a cost column (tokens: total_tokens, else input + generated)."""
    if frame.empty:
        return 0.0
    if column == "total_tokens" and column not in frame:
        return total(frame, "input_tokens") + total(frame, "generated_tokens")
    return float(pd.to_numeric(frame.get(column), errors="coerce").fillna(0).sum()) if column in frame else 0.0


def cost(records, setup):
    """End-to-end tokens and online time per response."""
    n = len(records)
    return {"Tokens": (total(records) + total(setup)) / n, "Online (s)": total(records, "elapsed_sec") / n}


def expert_units(details):
    """ExPerT: the response's aspects (as matched against either target)."""
    return {e["aspect"] for block in (details, details.get("preference_target") or {})
            for e in block.get("precision_matches") or []}


def n_constraints(method, records, ids=None):
    """Mean number of constraints each analysed response is checked against (None for methods without any)."""
    base = method.split("/")[0]
    if base.startswith("ExPerT"):
        count = lambda d: (len(d.get("recall_matches") or []) + len((d.get("preference_target") or {}).get(  # noqa: E731
            "recall_matches") or []) + len(expert_units(d)))
    elif base.startswith("PREF"):
        count = lambda d: len(d["constraints"]) if "constraints" in d else d.get("criterion_count") or 0  # noqa: E731
    elif base.startswith("Constraint-Driven"):
        count = lambda d: len(d.get("nli") or [])  # noqa: E731
    else:
        return None
    return analysed(records, ids)["details"].map(lambda d: count(d) if isinstance(d, dict) else 0).mean()


def row(label, method, records, setup, ids):
    """One table row: evaluation on the analysed responses, end-to-end cost, #Cons."""
    return {"Method": label, **evaluate(records, ids), **cost(records, setup),
            "#Cons.": n_constraints(method, records, ids)}


def show(title, table, columns=COLUMNS, out=None):
    print(f"\n{title}")
    print(table[["Method", *columns]].to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.2f}"))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(out, index=False)
        print(f"Saved: {out}")
