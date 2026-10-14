"""Audit the constraints of ExPerT and PREF with ConQual's qualification (Table 2 "-Audit" rows, Table 3).

The constraints were generated and saved by baselines/factscore_expert.py and baselines/pref_perse.py; nothing is
regenerated. For each method:
  1. its saved constraints become ConQual constraints:
       ExPerT  each response's aspects -> answer constraints (one constraint set per response);
       PREF    per question, coverage criteria -> answer constraints, preference criteria -> preference constraints;
  2. ConQualAudit.qualify_constraints audits them (duplicates, necessity, sufficiency), with the PrefEval reference and
     the stated preference exactly as for ConQual-Audit; constraints with the same text are merged first;
  3. the method's OWN scoring step is re-run on every response with the original and with the audited constraints:
       ExPerT  the saved target aspects matched against the response's (original | audited) aspects;
       PREF    the PREF scoring stage on the (original | audited) rubric; an added criterion gets the mean weight.
Methods: "<method>-Original/Qwen" and "<method>-Audited/Qwen" (the audit cost is the Audited setup).

Run:    python -m experiments.audit_baselines --dataset prefeval
Output: results/prefeval/audit_<tag>.json and audit_<tag>_constraints.json (per unit: original and audited
        constraints, the changes with stage and reason, the audit log). Checkpointed per unit (--resume).
"""

import json
import time
from pathlib import Path

from baselines import factscore_expert as m1
from baselines import pref_perse as m2
from benchmarks.base import run_tag
from conqual import ConQualAudit
from utils.llm import JUDGE_MODEL, load_model, release
from utils.runner import (BatchMetricResult, experiment_parser, jsonable, load_benchmark, read_jsonl, run_metric_batches,
                          save_constraints, save_results)

METHODS = ("ExPerT", "PREF")
PER_RESPONSE = ("ExPerT",)            # ExPerT's constraints are extracted from each response
MAX_AUDIT_ROUNDS, COUNTEREXAMPLE_ATTEMPTS = 1, 2
BATCH_SIZE = 16
DUPLICATE_PAIRS_PER_CALL = 100        # a large set's n(n-1) duplicate pairs are judged in chunks of this size


def newest_constraints(out, prefix, method):
    """The newest <prefix>_*_constraints.json with this method's constraints: (value, file name)."""
    for path in sorted(out.glob(f"{prefix}_*_constraints.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        value = json.loads(path.read_text(encoding="utf-8")).get("constraints", {}).get(method)
        if value:
            return value, path.name
    raise SystemExit(f"No {method} constraints in {out}/{prefix}_*_constraints.json: run baselines first.")


# ------------------------------------------------------------------ format conversion
def to_conqual(method, native, case):
    """The method's constraints -> ConQual constraints; each keeps its original under "native"."""
    def constraint(cid, text, item, preference=False):
        c = {"constraint_id": cid, "response_hypothesis": str(text or "").strip(), "native": item}
        return {**c, "source_interest": case.preference, "strength": "strong"} if preference else c
    answer, profile = [], []
    if method == "PREF":
        for item in native:
            (profile if item.get("source") == "preference" else answer).append(
                constraint(str(item["id"]), item.get("description"), item, item.get("source") == "preference"))
    else:
        answer = [constraint(f"ac_{i}", (a.get("description") or a.get("aspect")) if isinstance(a, dict) else a, a)
                  for i, a in enumerate(native, 1)]
    return {"profile_result": ConQualAudit.stated_profile(case.retrieved_profile),
            "profile_constraint": [c for c in profile if c["response_hypothesis"]],
            "answer_constraint": [c for c in answer if c["response_hypothesis"]]}


def from_conqual(method, qualified, native):
    """Audited ConQual constraints -> the method's format (kept ones as they were, added ones converted)."""
    answer, profile = qualified.get("answer_constraint") or [], qualified.get("profile_constraint") or []
    if method == "PREF":
        weights = [float(item["weight"]) for item in native if item.get("weight") is not None]
        default = round(sum(weights) / len(weights), 3) if weights else 1.0
        return [c["native"] if c.get("native") is not None else
                {"id": c["constraint_id"], "source": source, "weight": default, "description": c["response_hypothesis"],
                 "added_by_audit": True}
                for source, rows in (("coverage", answer), ("preference", profile)) for c in rows]
    aspects, taken = [], set()
    for c in answer + profile:
        if c.get("native") is not None:
            aspect = c["native"]
        else:                              # an aspect for a constraint the audit added
            title = " ".join(c["response_hypothesis"].split()[:8]).rstrip(".,;:")
            title = f"{title} ({c['constraint_id']})" if title in taken else title
            evidence = [e for e in c.get("evidence_record") or [] if isinstance(e, str) and e.strip()]
            aspect = {"aspect": title, "description": c["response_hypothesis"],
                      "evidence": evidence or [c["response_hypothesis"]], "added_by_audit": True}
        if aspect.get("aspect") in taken:   # titles are matching keys: keep them unique
            aspect = {**aspect, "aspect": f"{aspect['aspect']} ({c['constraint_id']})"}
        taken.add(aspect.get("aspect"))
        aspects.append(aspect)
    return aspects


def listing(constraints):
    return [{"id": c["constraint_id"], "kind": kind, "hypothesis": c["response_hypothesis"],
             **({"source_interest": c["source_interest"]} if c.get("source_interest") else {}),
             **({"evidence_record": c["evidence_record"]} if c.get("evidence_record") else {}),
             "added_by_audit": c.get("native") is None}
            for key, kind in (("profile_constraint", "preference"), ("answer_constraint", "answer"))
            for c in constraints.get(key) or []]


def changes(original, qualified, audit):
    """Removed constraints (identical text / duplicate / pruned, with the reason) and added ones."""
    kept, why = {c["id"] for c in qualified}, {}
    for e in audit:
        if e.get("stage") in ("duplicate", "prune") and str(e.get("decision", "")).startswith("dropped"):
            why[e["constraint_id"]] = {"stage": e["stage"], "reason": e["decision"],
                                       **{k: e[k] for k in ("duplicate_of", "counterexample", "judge_reason") if e.get(k)}}
    added_by = {cid: {"stage": "close_gaps", "target": e.get("target"), "round": e.get("round"),
                      "counterexample": e.get("counterexample"), "judge_reason": e.get("judge_reason")}
                for e in audit if e.get("stage") == "close_gaps" for cid in e.get("added") or []}
    first = {}
    for c in original:                     # constraints with the same text are merged before qualification
        key = c["hypothesis"].strip().lower()
        if key in first:
            why.setdefault(c["id"], {"stage": "identical", "reason": f"dropped (same text as {first[key]})",
                                     "duplicate_of": first[key]})
        else:
            first[key] = c["id"]
    before = {c["id"] for c in original}
    return {"removed": [{**c, **why.get(c["id"], {"stage": None, "reason": "not in the qualified set"})}
                        for c in original if c["id"] not in kept],
            "added": [{**c, **added_by.get(c["id"], {"stage": None})} for c in qualified if c["id"] not in before],
            "kept": [c["id"] for c in original if c["id"] in kept]}


COUNTS = ("n_original", "n_qualified", "n_removed", "n_added", "n_removed_identical", "n_removed_duplicate",
          "n_removed_prune", "input_tokens", "generated_tokens")


def qualify_set(method, evaluator, case, native):
    cde = to_conqual(method, native, case)
    if not cde["answer_constraint"] and not cde["profile_constraint"]:
        raise ValueError(f"no {method} constraints were saved for this " +
                         ("response" if method in PER_RESPONSE else "question"))
    qualified, t_in, t_out = evaluator.qualify_constraints(case.query, case.reference, cde)
    before, after, log = listing(cde), listing(qualified), qualified.get("audit_log", [])
    change = changes(before, after, log)
    return {"qualified": from_conqual(method, qualified, native), "original_constraints": before,
            "qualified_constraints": after, "changes": change, "audit_log": log,
            "n_original": len(before), "n_qualified": len(after), "n_removed": len(change["removed"]),
            "n_added": len(change["added"]),
            **{f"n_removed_{s}": sum(c["stage"] == s for c in change["removed"]) for s in ("identical", "duplicate",
                                                                                           "prune")},
            "input_tokens": t_in, "generated_tokens": t_out}


def audit(method, evaluator, units, checkpoint, by_case, resume):
    """Qualify every unit's constraints ({query_id: (case, native)}); checkpointed per unit."""
    done = {row["unit_id"]: row for row in read_jsonl(checkpoint)} if resume else {}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        checkpoint.write_text("", encoding="utf-8")
    with checkpoint.open("a", encoding="utf-8") as log:
        for i, (unit_id, (case, native)) in enumerate(units.items(), 1):
            if unit_id in done:
                continue
            row = {"method": f"{method}/Qwen", "unit_id": unit_id, "query_id": case.query_id, "user_id": case.user_id,
                   "query": case.query, "ground_truth": case.reference, "preference": case.preference,
                   "status": "error", "original": native, "qualified": native, **{k: 0 for k in COUNTS}}
            start = time.perf_counter()
            try:
                if method in PER_RESPONSE:
                    per_response = {}
                    for case_id, items in native["responses"].items():
                        response_case = by_case[case_id]
                        per_response[case_id] = {"category": response_case.category,
                                                 "response": response_case.response, "original": items,
                                                 **qualify_set(method, evaluator, response_case, items)}
                    row.update(status="ok", qualified={"responses": {k: r["qualified"] for k, r in per_response.items()}},
                               responses=per_response,
                               audit_log=[e for r in per_response.values() for e in r["audit_log"]],
                               **{k: sum(r[k] for r in per_response.values()) for k in COUNTS})
                else:
                    row.update(status="ok", **qualify_set(method, evaluator, case, native))
            except Exception as exc:              # the original constraints are kept for this unit
                row.update(error_type=type(exc).__name__, error=str(exc)[:1000])
            row["elapsed_sec"] = time.perf_counter() - start
            row["total_tokens"] = row["input_tokens"] + row["generated_tokens"]
            done[unit_id] = row
            log.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
            log.flush()
            print(f"[audit {method} {i}/{len(units)}] {unit_id}: {row['n_original']} -> {row['n_qualified']} "
                  f"constraints ({row['status']})")
    return done


# ------------------------------------------------------------------ each method's own scoring step
def expert_evaluator(model, tokenizer, task, references, response_aspects, cases):
    """ExPerT matching of the saved target aspects (per question) against the given response aspects."""
    expert = m1.ExPerT(task, batch_size=BATCH_SIZE)
    for case in cases:
        sides = references.get(case.query_id) or {}
        for target in m1.ALIGNMENT_TARGETS:
            key = expert.setup_key(case.query, task.alignment_text(case, target))
            if sides.get(target):
                expert.annotation_cache[key] = sides[target]
            else:
                expert.setup_errors[key] = ValueError(f"no {target} aspects for this question")
        if response_aspects.get(case.case_id):
            expert.candidate_cache[(case.query, case.response)] = response_aspects[case.case_id]
    two = m1.TwoTargetEvaluator(expert, m1.expert_score, task)
    return lambda batch: two.evaluate_batch(batch, model, tokenizer)


def pref_evaluator(model, tokenizer, rubrics):
    """The PREF scoring stage with a fixed rubric per question (weighted criterion mean, 0-10)."""
    def evaluate(batch):
        rows, prompts, validators = [], [], []
        for case in batch:
            rubric = rubrics.get(case.query_id) or []
            ids = {str(item["id"]) for item in rubric}
            rows.append(rubric)
            prompts.append(m2.PrefEvaluator.scoring_prompt(
                case.query, [{k: item.get(k) for k in ("id", "weight", "source", "description")} for item in rubric],
                case.response))
            validators.append(lambda value, ids=ids: m2.pref_result_score(value, ids) is not None)
        outcomes = m2.generate_valid_json(prompts, validators, model, tokenizer, "PREF candidate scoring",
                                          m2.SCORE_MAX_NEW_TOKENS, BATCH_SIZE, m2.PrefEvaluator.REPAIR_INSTRUCTION)
        results = []
        for rubric, outcome in zip(rows, outcomes):
            meta = {"generation_attempts": outcome.attempts}
            usage = dict(input_tokens=outcome.input_tokens, generated_tokens=outcome.generated_tokens, metadata=meta)
            if not rubric:
                results.append(BatchMetricResult(error=ValueError("no PREF criteria for this question"), **usage))
            elif outcome.error is not None and not outcome.schema_failure:
                results.append(BatchMetricResult(error=outcome.error, **usage))
            else:
                value = {"score": None, "constraints": rubric, "criterion_scores": None}
                if outcome.error is None:
                    scores = {str(i["id"]): float(i["score"]) for i in outcome.value["criterion_scores"]}
                    weights = {str(i["id"]): float(i["weight"]) for i in rubric}
                    total = sum(weights.values())
                    value.update(score=sum(weights[k] * scores[k] for k in weights) / total if total > 0 else None,
                                 reported_score=float(outcome.value["score"]),
                                 criterion_scores=outcome.value["criterion_scores"])
                else:
                    meta["failure_kind"] = "schema_validation"
                results.append(BatchMetricResult(value=value, **usage))
        return results
    return evaluate


def score_variants(method, model, tokenizer, task, cases, metadata, name, out, audits, references, resume):
    results = {}
    for variant in ("Original", "Audited"):
        constraints = {unit: row["original" if variant == "Original" else "qualified"] for unit, row in audits.items()}
        if method in PER_RESPONSE:                     # per question -> per response
            constraints = {case_id: items for value in constraints.values()
                           for case_id, items in (value.get("responses") or {}).items()}
        if method == "ExPerT":
            evaluator = expert_evaluator(model, tokenizer, task, references, constraints, cases)
            normalizer, score_of, details = (lambda v: v), m1.two_target_score, m1.two_target_details(m1.expert_details)
        else:
            evaluator, normalizer = pref_evaluator(model, tokenizer, constraints), (lambda v: v / 10.0)
            score_of = lambda v: v.get("score") if isinstance(v, dict) else None  # noqa: E731
            details = lambda v: ({k: v.get(k) for k in ("constraints", "criterion_scores", "reported_score")}  # noqa: E731
                                 if isinstance(v, dict) else {})
        metric = f"{method}-{variant}/Qwen"
        results[metric] = run_metric_batches(
            experiment=name, metric_name=metric, cases=cases, dataset_metadata=metadata, output_dir=out,
            batch_evaluator=evaluator, score_extractor=score_of, normalizer=normalizer, detail_extractor=details,
            batch_size=BATCH_SIZE, device=model.device, resume=resume,
            run_config={"judge": JUDGE_MODEL, "method": method, "constraints": variant.lower(),
                        "max_audit_rounds": MAX_AUDIT_ROUNDS, "counterexample_attempts": COUNTEREXAMPLE_ATTEMPTS})
        results[metric]["setup_records"] = [               # the audit belongs to the audited variant only
            {"phase": "constraint_audit", "unit_id": u, "status": r["status"], "elapsed_sec": r["elapsed_sec"],
             "input_tokens": r["input_tokens"], "generated_tokens": r["generated_tokens"],
             "total_tokens": r["total_tokens"]} for u, r in audits.items()] if variant == "Audited" else []
    return results


def load_units(method, cases, out):
    """{query_id: (case, saved constraints)} and the saved ExPerT target aspects, plus the source file name."""
    first, by_case = {}, {c.case_id: c for c in cases}
    for case in cases:
        first.setdefault(case.query_id, case)
    if method == "ExPerT":
        value, source = newest_constraints(out, "factscore_expert", "ExPerT/Qwen")
        references = {r["query_id"]: {"ground_truth": r.get("constraints") or [],
                                      "preference": r.get("preference_constraints") or []}
                      for r in value.get("ground_truth", [])}
        units = {}
        for r in value.get("response", []):
            if r.get("case_id") in by_case:
                case = by_case[r["case_id"]]
                _, native = units.setdefault(case.query_id, (first[case.query_id], {"responses": {}}))
                native["responses"][case.case_id] = [a for a in r.get("constraints") or [] if isinstance(a, dict)]
        return units, references, source
    rows, source = newest_constraints(out, "pref_perse", "PREF/Qwen")
    return ({r["query_id"]: (first[r["query_id"]], r.get("constraints") or []) for r in rows if r.get("query_id") in first},
            {}, source)


def main():
    args = experiment_parser(__doc__.splitlines()[0], datasets=("prefeval",)).parse_args()
    task, cases, metadata, out = load_benchmark(args)
    name = f"audit_{run_tag(metadata)}"
    inputs, sources = {}, {}
    for method in METHODS:                       # fail before loading the model if a file is missing
        units, references, sources[method] = load_units(method, cases, out)
        inputs[method] = (units, references)

    tokenizer, model = load_model(JUDGE_MODEL, args.device)
    evaluator = ConQualAudit(tokenizer, model, task.CONQUAL_PROMPTS, max_audit_rounds=MAX_AUDIT_ROUNDS,
                             counterexample_attempts=COUNTEREXAMPLE_ATTEMPTS)
    single_call = evaluator._entailment_pairs           # duplicate pairs in chunks (same prompt and decision rule)

    def chunked(query, pairs, usage):
        labels, parsed = {}, True
        for start in range(0, len(pairs), DUPLICATE_PAIRS_PER_CALL):
            part, ok = single_call(query, pairs[start:start + DUPLICATE_PAIRS_PER_CALL], usage)
            labels.update(part)
            parsed = parsed and ok
        return labels, parsed
    evaluator._entailment_pairs = chunked

    audits, results = {}, {}
    for method in METHODS:
        units, references = inputs[method]
        checkpoint = Path(out) / "checkpoints" / f"{name}_{method.lower()}_audit.jsonl"
        audits[method] = audit(method, evaluator, units, checkpoint, {c.case_id: c for c in cases}, args.resume)
        save_constraints(out, name, metadata, {f"{m}/Qwen": list(a.values()) for m, a in audits.items()},
                         judge=JUDGE_MODEL, sources=sources)
        results.update(score_variants(method, model, tokenizer, task, cases, metadata, name, out, audits[method],
                                      references, args.resume))
        save_results(out, name, metadata, results)
    print(f"Saved: {out / (name + '.json')}")
    release(args.device)


if __name__ == "__main__":
    main()
