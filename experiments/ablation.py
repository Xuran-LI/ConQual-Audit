"""ConQual-Audit without qualification (Table 2, ConQual-No-Audit): verify the constructed (step-1) constraints.

run_conqual.py saved every query's constructed constraints before qualification. This ablation reuses exactly those
(nothing is constructed or qualified for scoring) and runs stage 3 on every response with score = R, so the difference
to ConQual-Audit is what qualification contributes. Its setup cost is construction only; run_conqual.py logs
construction and qualification as one cost, so step 1 is re-run once per query and timed here (greedy decoding: the
constraints must equal the saved ones; matches_saved records it).

Run:    python -m experiments.ablation --dataset prefeval        (after experiments.run_conqual)
Output: results/prefeval/ablation_<tag>.json    method "Constraint-Driven (no qualification)/Qwen"; setup_records =
                                                the construction cost per query
"""

import json
import time

from benchmarks.base import run_tag
from conqual import ConQualAudit
from experiments.run_conqual import METHOD as FULL_METHOD, listing, to_constraints, verify_record
from utils.llm import JUDGE_MODEL, load_model, release
from utils.runner import experiment_parser, jsonable, load_benchmark, read_jsonl, save_results, synchronize

METHOD = "Constraint-Driven (no qualification)/Qwen"


def same(rows_a, rows_b):
    return sorted((c["id"], c["kind"], str(c["hypothesis"]).strip()) for c in rows_a or []) == \
        sorted((c["id"], c["kind"], str(c["hypothesis"]).strip()) for c in rows_b or [])


def main():
    args = experiment_parser(__doc__.splitlines()[0], datasets=("prefeval",)).parse_args()
    task, cases, metadata, out = load_benchmark(args)
    tag = run_tag(metadata)
    source = out / f"conqual_{tag}_constraints.json"
    if not source.is_file():
        raise SystemExit(f"{source} not found: run experiments.run_conqual first.")
    saved = {r["query_id"]: r for r in json.loads(source.read_text(encoding="utf-8"))["constraints"][FULL_METHOD]}
    changed = [q for q, r in saved.items() if r.get("status") == "ok" and r.get("constructed")
               and not same(listing(to_constraints(r["constructed"])), r["constructed"])]
    if changed:                                   # every saved constraint must reach verification unchanged
        raise SystemExit(f"{len(changed)} queries' constraints would not reach verification unchanged: {changed[:10]}")

    name = f"ablation_{tag}"
    checkpoint = out / "checkpoints" / f"{name}.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    done, setups = {}, {}
    for row in read_jsonl(checkpoint) if args.resume else []:
        if "setup" in row:
            setups[row["setup"]["query_id"]] = row["setup"]
        else:
            done[row["case_id"]] = row
    if not args.resume:
        checkpoint.write_text("", encoding="utf-8")

    tokenizer, model = load_model(JUDGE_MODEL, args.device)
    evaluator = ConQualAudit(tokenizer, model, task.CONQUAL_PROMPTS)
    questions = {}
    for case in cases:
        questions.setdefault(case.query_id, case)
    with checkpoint.open("a", encoding="utf-8") as log:
        for query_id, case in questions.items():           # the setup cost: step 1 only
            if query_id in setups:
                continue
            record = {"phase": "constraint_construction", "query_id": query_id, "status": "error",
                      "input_tokens": 0, "generated_tokens": 0}
            synchronize(args.device)
            start = time.perf_counter()
            try:
                constructed, t_in, t_out = evaluator.construct_constraints(case.query, case.reference,
                                                                           task.conqual_profile(case))
                synchronize(args.device)
                built = listing(constructed)
                record.update(status="ok", input_tokens=t_in, generated_tokens=t_out, constraints=len(built),
                              matches_saved=same(built, (saved.get(query_id) or {}).get("constructed")))
            except Exception as exc:
                record.update(error_type=type(exc).__name__, error=str(exc)[:1000])
            record["elapsed_sec"] = time.perf_counter() - start
            record["total_tokens"] = record["input_tokens"] + record["generated_tokens"]
            setups[query_id] = record
            log.write(json.dumps(jsonable({"setup": record}), ensure_ascii=False) + "\n")
            log.flush()

        for i, case in enumerate(cases, 1):                 # stage 3 with the saved step-1 constraints
            if case.case_id in done:
                continue
            record = {**case.record_identity(), "metric": METHOD, "status": "error", "score": None,
                      "input_tokens": 0, "generated_tokens": 0}
            start = time.perf_counter()
            try:
                row = saved.get(case.query_id) or {}
                if row.get("status") != "ok" or not row.get("constructed"):
                    record["batch_metadata"] = {"failure_stage": "constraint_setup"}
                    raise ValueError("no constructed constraints saved for this query")
                verify_record(evaluator, case, to_constraints(row["constructed"]), record)
            except Exception as exc:
                record.update(error_type=type(exc).__name__, error=str(exc)[:1000])
            record["elapsed_sec"] = time.perf_counter() - start
            record["total_tokens"] = record["input_tokens"] + record["generated_tokens"]
            done[case.case_id] = record
            log.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
            log.flush()
            print(f"[{i}/{len(cases)}] {case.case_id} R={record['score']}")

    records, setup = [done[c.case_id] for c in cases], [setups[q] for q in questions]
    ok = [s for s in setup if s["status"] == "ok"]
    result = {"metric_name": METHOD, "decision_threshold": 0.5, "records": records, "setup_records": setup,
              "run_config": {"model": JUDGE_MODEL, "ablation": "no_qualification", "score": "R",
                             "constraints": source.name,
                             "construction_match_rate": sum(bool(s.get("matches_saved")) for s in ok) / max(1, len(ok))},
              "coverage": {"requested": len(records), "valid": sum(r["status"] == "ok" for r in records)}}
    save_results(out, name, metadata, {METHOD: result})
    print(f"Saved: {out / (name + '.json')} ({result['coverage']['valid']}/{len(records)} valid; constructed "
          f"constraints equal to the saved ones: {result['run_config']['construction_match_rate']:.0%})")
    release(args.device)


if __name__ == "__main__":
    main()
