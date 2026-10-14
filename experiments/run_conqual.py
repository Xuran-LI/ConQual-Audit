"""ConQual-Audit on a benchmark (Tables 1-2): per query, construct and qualify the constraints once (stages 1-2), then
verify every response (stage 3) and score it with R; P and F1 are kept in the details.

Run:    python -m experiments.run_conqual --dataset prefeval        (or lamp1 / lamp2 / lamp4)
Output: results/<dataset>/conqual_<tag>.json            method "Constraint-Driven/Qwen"; per response: R, P, F1, the
                                                        label, reasoning and supporting sentences of every constraint
        results/<dataset>/conqual_<tag>_constraints.json per query: the constructed (step-1) and qualified (step-2)
                                                        constraints, what the audit changed and why, the audit log
The response timer starts before a query's setup, so the first response of each query also contains that setup time;
the tables subtract it (tables/common.py, online_time_without_setup) and count the setup once.
"""

import json
import time

from conqual import ConQualAudit
from benchmarks.base import run_tag
from utils.llm import JUDGE_MODEL, load_model, release
from utils.runner import experiment_parser, jsonable, load_benchmark, read_jsonl, save_constraints, save_results

METHOD = "Constraint-Driven/Qwen"
MAX_AUDIT_ROUNDS, COUNTEREXAMPLE_ATTEMPTS = 1, 2            # one repair round per requirement; K = 2
DETAIL_KEYS = ("R", "P", "f1_score", "constraints", "entailed", "P_sentences", "response_tokens", "supported_tokens",
               "response_sentences", "supported_sentences", "entailed_without_evidence")


def listing(constraints):
    """Preference + answer constraints as [{id, kind, hypothesis, ...every other field}] (as verified)."""
    rows = (ConQualAudit.ensure_ids(constraints.get("profile_constraint"), "pc", "preference")
            + ConQualAudit.ensure_ids(constraints.get("answer_constraint"), "ac", "answer"))
    return [{"id": c["constraint_id"], "kind": c["kind"], "hypothesis": c["response_hypothesis"],
             **{k: v for k, v in c.items() if k not in ("constraint_id", "kind", "response_hypothesis")}} for c in rows]


def to_constraints(rows):
    """listing() rows -> the constraint dict verify_constraints takes (every saved field kept, same order)."""
    out = {"profile_constraint": [], "answer_constraint": []}
    for item in rows or []:
        constraint = {"constraint_id": item["id"], "response_hypothesis": item["hypothesis"], "kind": item["kind"],
                      **{k: v for k, v in item.items() if k not in ("id", "kind", "hypothesis")}}
        out["profile_constraint" if item["kind"] == "preference" else "answer_constraint"].append(constraint)
    return out


def changes(constructed, qualified, audit):
    """What qualification changed: removed constraints (duplicate / pruned, with the reason) and added ones."""
    kept, why = {c["id"] for c in qualified}, {}
    for e in audit:
        if e.get("stage") in ("duplicate", "prune") and str(e.get("decision", "")).startswith("dropped"):
            why[e["constraint_id"]] = {"stage": e["stage"], "reason": e["decision"],
                                       **({"duplicate_of": e["duplicate_of"]} if e.get("duplicate_of") else {})}
    added_by = {cid: {"stage": "close_gaps", "target": e.get("target"), "round": e.get("round"),
                      "counterexample": e.get("counterexample")}
                for e in audit if e.get("stage") == "close_gaps" for cid in e.get("added") or []}
    original = {c["id"] for c in constructed}
    return {"removed": [{**c, **why.get(c["id"], {"stage": None, "reason": "not in the qualified set"})}
                        for c in constructed if c["id"] not in kept],
            "added": [{**c, **added_by.get(c["id"], {"stage": None})} for c in qualified if c["id"] not in original],
            "kept": sorted(kept & original)}


def verify_record(evaluator, case, constraints, record):
    """Stage 3 + metrics for one response: score = R (an unreadable verifier output is an error, never 0)."""
    verified, v_in, v_out = evaluator.verify_constraints(case.query, case.response, constraints)
    record.update(input_tokens=v_in, generated_tokens=v_out)
    if verified.get("error"):
        raise ValueError(verified["error"])
    metrics = evaluator.compute_metrics(case.response, verified)
    if metrics["R"] is None:
        raise ValueError("R undefined: no constraint verified")
    record.update(status="ok", score=metrics["R"], raw_score=metrics["R"],
                  details={**{k: metrics.get(k) for k in DETAIL_KEYS}, "evidence_check": verified.get("evidence_check"),
                           "nli": [{k: e.get(k) for k in ("constraint_id", "kind", "hypothesis", "label", "reasoning",
                                                          "evidence_ids", "evidence_sentences")}
                                   for e in verified.get("evaluations", [])]})


def main():
    args = experiment_parser(__doc__.splitlines()[0]).parse_args()
    task, cases, metadata, out = load_benchmark(args)
    name = f"conqual_{run_tag(metadata)}"
    checkpoint = out / "checkpoints" / f"{name}.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    done, setups = {}, []
    for row in read_jsonl(checkpoint) if args.resume else []:
        setups.append(row["setup"]) if "setup" in row else done.__setitem__(row["case_id"], row)
    if not args.resume:
        checkpoint.write_text("", encoding="utf-8")

    tokenizer, model = load_model(JUDGE_MODEL, args.device)
    evaluator = ConQualAudit(tokenizer, model, task.CONQUAL_PROMPTS, max_audit_rounds=MAX_AUDIT_ROUNDS,
                             counterexample_attempts=COUNTEREXAMPLE_ATTEMPTS)
    constraints = {s["query_id"]: s for s in setups}             # query -> its setup (qualified constraints)
    with checkpoint.open("a", encoding="utf-8") as log:
        for i, case in enumerate(cases, 1):
            if case.case_id in done:
                continue
            record = {**case.record_identity(), "metric": METHOD, "status": "error", "score": None,
                      "input_tokens": 0, "generated_tokens": 0}
            start = time.perf_counter()
            try:
                if case.query_id not in constraints:            # stages 1-2, once per query
                    t0 = time.perf_counter()
                    setup = {"query_id": case.query_id, "status": "error", "input_tokens": 0, "generated_tokens": 0}
                    try:
                        constructed, qualified, t_in, t_out = evaluator.build_constraints(
                            case.query, case.reference, task.conqual_profile(case))
                        built, final, audit = listing(constructed), listing(qualified), qualified.get("audit_log", [])
                        setup.update(status="ok", input_tokens=t_in, generated_tokens=t_out, user_id=case.user_id,
                                     query=case.query, ground_truth=case.reference,
                                     preference=[r["preference"] for r in task.conqual_profile(case)],
                                     profile_result=constructed.get("profile_result"), constructed=built,
                                     qualified=final, changes=changes(built, final, audit), audit=audit)
                    except Exception as exc:
                        setup.update(error_type=type(exc).__name__, error=str(exc)[:1000])
                    setup.update(elapsed_sec=time.perf_counter() - t0,
                                 total_tokens=setup["input_tokens"] + setup["generated_tokens"])
                    constraints[case.query_id] = setup
                    setups.append(setup)
                    log.write(json.dumps(jsonable({"setup": setup}), ensure_ascii=False) + "\n")
                setup = constraints[case.query_id]
                if setup["status"] != "ok":
                    record["batch_metadata"] = {"failure_stage": "constraint_setup"}
                    raise RuntimeError(setup.get("error") or "constraint setup failed")
                verify_record(evaluator, case, to_constraints(setup["qualified"]), record)
            except Exception as exc:
                record.update(error_type=type(exc).__name__, error=str(exc)[:1000])
            record["elapsed_sec"] = time.perf_counter() - start
            record["total_tokens"] = record["input_tokens"] + record["generated_tokens"]
            done[case.case_id] = record
            log.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
            log.flush()
            print(f"[{i}/{len(cases)}] {case.case_id} R={record['score']}")

    records = [done[case.case_id] for case in cases]
    result = {"metric_name": METHOD, "decision_threshold": 0.5, "records": records, "setup_records": setups,
              "run_config": {"model": JUDGE_MODEL, "use_profile": True, "max_audit_rounds": MAX_AUDIT_ROUNDS,
                             "counterexample_attempts": COUNTEREXAMPLE_ATTEMPTS, "score": "R"},
              "coverage": {"requested": len(records), "valid": sum(r["status"] == "ok" for r in records)}}
    save_results(out, name, metadata, {METHOD: result})
    save_constraints(out, name, metadata, {METHOD: [
        {k: s.get(k) for k in ("query_id", "user_id", "query", "ground_truth", "preference", "status",
                               "profile_result", "constructed", "qualified", "changes", "audit")} for s in setups]})
    print(f"Saved: {out / (name + '.json')} ({result['coverage']['valid']}/{len(records)} valid)")
    release(args.device)


if __name__ == "__main__":
    main()
