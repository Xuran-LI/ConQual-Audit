"""Benchmark cases: one candidate response with its expected label (positive = acceptable, negative = not).

PrefEval: per item, the response generated WITH the stated preference (positive) and WITHOUT it (negative).
LaMP-1/2/4: per user, 10 positive responses (the gold answer in 10 sentence structures) and 20 negatives of two
types - the same structures stating a wrong answer ("wrong_*") or negating the gold one ("negated_gold").
"""

import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class Case:
    case_id: str
    user_id: str
    query_id: str
    category: str                          # positive | negative
    response_id: str
    template_id: str
    sentence_structure: str
    response: str
    reference: str                         # PrefEval: the reference explanation; LaMP: the gold label sentence
    query: str
    retrieved_profile: Tuple[Mapping[str, Any], ...]
    response_strategy: str
    claim_polarity: str
    mentioned_reference: str = ""
    negative_type: Optional[str] = None    # LaMP negatives: wrong_reference / wrong_tag / wrong_title / negated_gold
    preference: str = ""                   # PrefEval: the stated preference
    topic: str = ""

    @property
    def expected_acceptance(self):
        return self.category == "positive"

    def record_identity(self):
        return {"case_id": self.case_id, "user_id": self.user_id, "query_id": self.query_id,
                "category": self.category, "response_id": self.response_id, "template_id": self.template_id,
                "sentence_structure": self.sentence_structure, **({"topic": self.topic} if self.topic else {}),
                "labels": {"expected_acceptance": self.expected_acceptance, "response_strategy": self.response_strategy,
                           "claim_polarity": self.claim_polarity, "mentioned_reference": self.mentioned_reference,
                           "negative_type": self.negative_type},
                "retrieved_profile_record_ids": [str(r.get("record_id") or r.get("id") or r.get("_id"))
                                                 for r in self.retrieved_profile]}


def clip(text, limit=200):
    """Whitespace-normalised text cut at a word boundary to at most `limit` characters (+ "...")."""
    text = " ".join(str(text or "").split())
    return text[:limit].rsplit(" ", 1)[0] + "..." if len(text) > limit else text


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _metadata(name, path, root, cases, users, **extra):
    return {"schema_version": "1.0", "dataset": name, "source_path": str(path),
            "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "selected_users": users,
            "selected_cases": len(cases), "category_counts": dict(Counter(c.category for c in cases)), **extra}


def run_tag(metadata):
    """Suffix of every result file name, e.g. v1_q1only_all_100users."""
    return f"v1_q1only_all_{metadata['selected_users']}users"


def load_prefeval(path, max_users=None):
    """Two cases per PrefEval item: the response with (positive) and without (negative) the stated preference."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run benchmarks/generate_prefeval_responses.py first.")
    root = json.loads(path.read_text(encoding="utf-8"))
    items = root["items"][:max_users] if max_users else root["items"]
    cases = []
    for item in items:
        user_id, preference, question = _text(item.get("user_id")), _text(item.get("preference")), _text(item.get("question"))
        explanation = _text(item.get("explanation"))
        if not (user_id and preference and question and explanation):
            raise ValueError(f"Item {user_id or '?'} is missing user_id / preference / question / explanation.")
        record = {"id": f"{user_id}_pref", "preference": preference}
        for category, key in (("positive", "with_preference"), ("negative", "without_preference")):
            response = _text((item.get("responses") or {}).get(key))
            if not response:
                continue
            cases.append(Case(case_id=f"{user_id}::q1::{key}", user_id=user_id,
                              query_id=_text(item.get("query_id")) or f"{user_id}::q1", category=category,
                              response_id=f"{user_id}::q1::{key}", template_id=key,
                              sentence_structure="generated free text", response=response, reference=explanation,
                              query=question, retrieved_profile=(record,), response_strategy=key,
                              claim_polarity="positive", preference=preference, topic=_text(item.get("topic"))))
    if sum(c.category == "positive" for c in cases) != len(items):
        raise ValueError("Every PrefEval item needs a response generated with the preference.")
    return cases, _metadata("PrefEval", path, root, cases, len(items),
                            responder=(root.get("generation") or {}).get("model"))


def load_lamp(spec, path, max_users=None):
    """The controlled LaMP set of benchmarks/build_lamp*.py: per user one query (query_1) with its BM25-retrieved
    profile records, 10 positive and 20 negative responses (10 per negative type).

    spec: dataset name, answer field of ground_truth / of the candidates, the two negative types and the label
    templates (see benchmarks/lamp*.py)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run benchmarks/build_{spec['task']}.py first.")
    root = json.loads(path.read_text(encoding="utf-8"))
    if root.get("dataset") != spec["name"] or root.get("response_format") != spec["response_format"]:
        raise ValueError(f"{path} is not a {spec['name']} set in format {spec['response_format']}.")
    users = root["users"][:max_users] if max_users else root["users"]
    wrong, negated = spec["negative_types"]
    cases, seen = [], set()
    for user in users:
        user_id, query = str(user["user_id"]), user["query_1"]
        query_id, truth = query["query_id"], query["ground_truth"]
        gold, other = truth[spec["answer_key"]], truth[f"negative_{spec['answer_key']}"]
        if truth["gold_label"] != spec["gold_label"].format(gold) or \
                truth["negated_label"] != spec["negated_label"].format(gold):
            raise ValueError(f"{query_id}: labels inconsistent with the gold answer.")
        profile = {str(r.get("record_id") or r.get("id") or r.get("_id")): r for r in user["user_profile"]["records"]}
        retrieved = tuple(profile[str(i)] for i in query["relevance_information"]["profile_record_ids"])
        for category in ("positive", "negative"):
            candidates = query["response_candidates"][category]
            types = Counter(c.get("violation_type") for c in candidates)
            if types != (Counter({None: 10}) if category == "positive" else Counter({wrong: 10, negated: 10})):
                raise ValueError(f"{query_id}: unexpected {category} responses {dict(types)}.")
            for candidate in candidates:
                negative_type = candidate.get("violation_type")
                expected = other if negative_type == wrong else gold
                if candidate[spec["mention_key"]] != expected:
                    raise ValueError(f"{candidate['candidate_id']} must state {expected!r}.")
                for variant in candidate["equivalent_responses"]:
                    case_id, text = variant["variant_id"], variant["text"].strip()
                    if case_id in seen or not spec["states"](text, expected, gold, other, negative_type == negated):
                        raise ValueError(f"{case_id}: duplicate, or does not state its claim: {text!r}")
                    seen.add(case_id)
                    cases.append(Case(
                        case_id=case_id, user_id=user_id, query_id=query_id, category=category,
                        response_id=candidate["candidate_id"],
                        template_id=variant.get("template_id") or candidate["template_id"],
                        sentence_structure=variant.get("sentence_structure") or candidate["sentence_structure"],
                        response=text, reference=truth["gold_label"], query=query["text"].strip(),
                        retrieved_profile=retrieved,
                        response_strategy="asserted_gold" if category == "positive" else
                        "negated_gold" if negative_type == negated else "asserted_wrong",
                        claim_polarity="negated" if negative_type == negated else "affirmative",
                        mentioned_reference=expected, negative_type=negative_type))
    return cases, _metadata(spec["name"], path, root, cases, len(users), response_format=spec["response_format"])

