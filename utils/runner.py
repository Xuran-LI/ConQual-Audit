"""Batched evaluation with per-response records, resumable checkpoints and one result file per experiment.

Every evaluator writes the same record per response (the benchmark labels + score, details, time and tokens), so the
table scripts read all methods alike. Record fields: case_id, query_id, category, labels, ..., score, details,
elapsed_sec, input_tokens, generated_tokens, total_tokens, status ("ok" | "invalid_score" | "error").
"""

import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import torch
except ImportError:          # lexical metrics and the tables do not need torch
    torch = None


class MetricEvaluationError(RuntimeError):
    """An evaluator failure that keeps the tokens spent before it."""

    def __init__(self, message, *, input_tokens=0, generated_tokens=0):
        super().__init__(message)
        self.input_tokens, self.generated_tokens = int(input_tokens), int(generated_tokens)


@dataclasses.dataclass
class BatchMetricResult:
    """One response's result from a batched evaluator (elapsed_sec None = an equal share of the batch time)."""

    value: Any = None
    input_tokens: int = 0
    generated_tokens: int = 0
    error: Optional[BaseException] = None
    elapsed_sec: Optional[float] = None
    metadata: Optional[Mapping[str, Any]] = None


def jsonable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    for method in ("item", "tolist"):
        if hasattr(value, method):
            try:
                return jsonable(getattr(value, method)())
            except Exception:
                pass
    return str(value)


def finite_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def synchronize(device):
    if torch is not None and device is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def write_json(path, payload):
    """Atomic JSON write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp_{path.name}")
    temporary.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_jsonl(path):
    """Rows of a JSONL checkpoint; an incomplete last line (interrupted write) is skipped."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines() if Path(path).is_file() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def save_results(output_dir, experiment, dataset_metadata, metrics):
    """<output_dir>/<experiment>.json = {"experiment", "dataset", "metrics": {method: result}}."""
    return write_json(Path(output_dir) / f"{experiment}.json",
                      {"experiment": experiment, "dataset": dict(dataset_metadata), "metrics": dict(metrics)})


def save_constraints(output_dir, experiment, dataset_metadata, constraints, **extra):
    """<output_dir>/<experiment>_constraints.json: the constraints each method generated (and how they were used)."""
    return write_json(Path(output_dir) / f"{experiment}_constraints.json",
                      {"experiment": experiment, **extra, "dataset": dict(dataset_metadata), "constraints": constraints})


def checkpoint_file(output_dir, experiment, name):
    slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return Path(output_dir) / "checkpoints" / f"{experiment}__{slug}.jsonl"


def run_metric_batches(*, experiment, metric_name, cases, dataset_metadata, output_dir, batch_evaluator,
                       score_extractor, batch_size, normalizer=lambda v: v, token_counter=None,
                       detail_extractor=None, run_config=None, threshold=0.5, device=None, resume=False,
                       progress_every=100):
    """Score every case with batch_evaluator(batch) -> [raw value | BatchMetricResult] and return the result dict.

    score = normalizer(score_extractor(raw)); a failed or unparsable evaluation is recorded as "error" /
    "invalid_score" (never as a zero score). Batch wall time is shared equally by the batch's responses unless the
    evaluator reports a per-response time. With resume=True, responses already in the checkpoint are kept.
    """
    config = {**dict(run_config or {}), "batch_size": int(batch_size)}
    fingerprint = hashlib.sha256(json.dumps(
        {"experiment": experiment, "metric": metric_name, "source_sha256": dataset_metadata.get("source_sha256"),
         "run_config": jsonable(config)}, sort_keys=True).encode()).hexdigest()
    checkpoint = checkpoint_file(output_dir, experiment, metric_name)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    done = {r["case_id"]: r for r in read_jsonl(checkpoint) if r.get("run_fingerprint") == fingerprint} \
        if resume else {}
    if not resume:
        checkpoint.write_text("", encoding="utf-8")
    pending = [case for case in cases if case.case_id not in done]
    print(f"[{metric_name}] {len(done)}/{len(cases)} already done; {len(pending)} to evaluate")

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        start, batch_error = None, None
        try:
            synchronize(device)
            start = time.perf_counter()
            raw_values = list(batch_evaluator(batch))
            synchronize(device)
            elapsed = time.perf_counter() - start
            if len(raw_values) != len(batch):
                raise ValueError(f"{metric_name} returned {len(raw_values)} values for {len(batch)} cases.")
        except Exception as exc:
            raw_values, batch_error = [None] * len(batch), exc
            elapsed = time.perf_counter() - start if start is not None else 0.0
        rows = []
        for case, item in zip(batch, raw_values):
            if not isinstance(item, BatchMetricResult):
                item = BatchMetricResult(value=item)
            error = batch_error or item.error
            n_in, n_out = max(0, int(item.input_tokens or 0)), max(0, int(item.generated_tokens or 0))
            record = {**case.record_identity(), "run_fingerprint": fingerprint, "metric": metric_name,
                      "score": None, "raw_score": None,
                      "elapsed_sec": max(0.0, float(item.elapsed_sec)) if item.elapsed_sec is not None
                      else elapsed / max(1, len(batch)),
                      "input_tokens": n_in, "generated_tokens": n_out, "status": "invalid_score"}
            if item.metadata:
                record["batch_metadata"] = jsonable(item.metadata)
            if error is not None:
                record.update(status="error", error_type=type(error).__name__, error=str(error)[:1000])
            else:
                try:
                    raw_score = finite_score(score_extractor(item.value))
                    score = finite_score(normalizer(raw_score)) if raw_score is not None else None
                    record.update(raw_score=raw_score, score=score, status="ok" if score is not None else "invalid_score",
                                  input_tokens=n_in + (int(token_counter(case)) if token_counter else 0))
                    if detail_extractor is not None:
                        record["details"] = jsonable(detail_extractor(item.value))
                except Exception as exc:
                    record.update(status="error", score=None, raw_score=None, error_type=type(exc).__name__,
                                  error=str(exc)[:1000])
            record["total_tokens"] = record["input_tokens"] + record["generated_tokens"]
            rows.append(record)
            done[case.case_id] = record
        with checkpoint.open("a", encoding="utf-8") as stream:
            stream.writelines(json.dumps(jsonable(r), ensure_ascii=False) + "\n" for r in rows)
        processed = offset + len(batch)
        if progress_every and (processed == len(pending) or processed // progress_every != offset // progress_every):
            print(f"[{metric_name}] {processed}/{len(pending)}")

    records = [done[case.case_id] for case in cases]
    valid = sum(r.get("status") == "ok" for r in records)
    return {"metric_name": metric_name, "decision_threshold": threshold, "run_config": config,
            "coverage": {"requested": len(records), "valid": valid}, "records": records, "setup_records": []}


# ------------------------------------------------------------------ command line shared by the experiment scripts
def experiment_parser(description, datasets=("prefeval", "lamp1", "lamp2", "lamp4")):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", required=True, choices=datasets)
    parser.add_argument("--data", type=Path, default=None, help="test set (default: the dataset's DATA path)")
    parser.add_argument("--results", type=Path, default=None, help="results folder (default: results/<dataset>)")
    parser.add_argument("--device", default="cuda:0", help="GPU of the judge model")
    parser.add_argument("--max-users", type=int, default=None, help="evaluate only the first N users (smoke test)")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run from its checkpoints")
    return parser


def load_benchmark(args):
    """(dataset module, cases, metadata, results folder) for the parsed experiment arguments."""
    import benchmarks
    task = benchmarks.get(args.dataset)
    cases, metadata = task.load(args.data or task.DATA, args.max_users)
    out = Path(args.results or Path("results") / args.dataset)
    print(f"{task.NAME}: {len(cases)} responses ({metadata['category_counts']}) from {metadata['source_path']}")
    return task, cases, metadata, out


def loading_record(model_name, seconds):
    """Setup record of loading the judge model (time only), as kept by every LLM evaluator."""
    return {"phase": "judge_model_loading", "status": "ok", "elapsed_sec": seconds, "input_tokens": 0,
            "generated_tokens": 0, "total_tokens": 0, "judge_model": model_name}


class SetupCache:
    """One-off per-query setup work of an evaluator (ExPerT target aspects, PREF rubrics): {key: (value, record)},
    appended to a JSONL file so a resumed run reuses it (and its cost record)."""

    def __init__(self, path, resume=False):
        self.path, self._items = Path(path) if path else None, {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if resume:
                self._items = {row["key"]: (row["value"], row["record"]) for row in read_jsonl(self.path)}
            else:
                self.path.write_text("", encoding="utf-8")

    def items(self):
        return list(self._items.items())

    def record(self, key):
        return self._items.get(key, (None, {}))[1]

    def records(self):
        return [record for _, record in self._items.values()]

    def add(self, key, value, record):
        self._items[key] = (value, record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(jsonable({"key": key, "value": value, "record": record}),
                                        ensure_ascii=False) + "\n")
