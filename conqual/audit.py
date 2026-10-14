"""ConQual-Audit: construct -> qualify (audit) -> verify, with one LLM in the roles G (generator), J (judge), V (verifier).

Stage 1, construction (G): preference constraints from each stated interest that applies to the query, and answer
    constraints from the task reference y*.
Stage 2, qualification (Algorithm 2 of the paper):
    duplicates  one DUPLICATE_CHECK call over all ordered constraint pairs; of two mutually entailing constraints the
                later one is removed (preference constraints first, then answer constraints, in construction order);
    necessity   leave-one-out: ADVERSE(C \\ {c}, {c}) asks G for a response that satisfies every other constraint but
                violates c (checked by V, K = counterexample_attempts tries); if J judges it aligned with both the
                preference and the reference, c is not necessary and is removed;
    sufficiency for each requirement r in (reference, preference): ADVERSE(C, {}, r) asks G for a response that
                satisfies every constraint but breaks r; if J judges it misaligned, UPDATE asks G for at most three
                new constraints of r's kind and keeps those V confirms exclude it (answer constraints must also hold
                for y*); one repair round per requirement (max_audit_rounds).
Stage 3, verification (V): every constraint is labelled Entailment / Contradiction / Neutral against the response,
    then the sentences supporting each entailed constraint are marked. Scores (Eq. 10): R = |C_y^E| / |C|,
    P = word tokens of the supporting sentences / word tokens of the response, F1 = 2PR / (P + R).

Stages 1-2 do not depend on the response: call build_constraints once per query and verify_constraints per response.
The prompts are the dataset's (benchmarks/<dataset>.py, CONQUAL_PROMPTS).
"""

import copy
import json
import re

from utils.llm import generate, parse_json_first


def _render_prompt(template, **values):
    """Insert JSON-encoded data for every {{PLACEHOLDER}}, without letting it close a prompt delimiter."""

    def replace(match):
        key = match.group(1)
        if key not in values:
            raise KeyError(f"No value supplied for prompt placeholder {key}")
        value = json.dumps(values[key], ensure_ascii=False, separators=(",", ":"))
        return value.replace("<", "\\u003c").replace(">", "\\u003e")

    return re.sub(r"\{\{([A-Z_]+)\}\}", replace, template)


class _Usage:
    """Token counter threaded through a stage."""

    def __init__(self):
        self.input_tokens = self.output_tokens = 0


class ConQualAudit:
    LABELS = ("Entailment", "Contradiction", "Neutral")
    VIOLATION_LABELS = frozenset({"Contradiction", "Neutral"})
    # alignment requirement -> (UNDER_CONSTRAINT mode, kind of its repair constraints)
    GAP_TARGETS = (("ground_truth", "BREAK_GROUND_TRUTH", "answer"),
                   ("preference", "BREAK_USER_PREFERENCE", "preference"))

    def __init__(self, tokenizer, model, prompts, max_audit_rounds=1, counterexample_attempts=2,
                 max_new_tokens=512, verify_max_new_tokens=4096, dedup_across_kinds=True):
        """prompts: the dataset's prompt templates. counterexample_attempts = K. verify_max_new_tokens: budget of the
        calls whose JSON has one entry per constraint / pair / sentence (a cut-off JSON cannot be parsed)."""
        self.tokenizer, self.model, self.prompts = tokenizer, model, prompts
        self.max_new_tokens, self.verify_max_new_tokens = int(max_new_tokens), int(verify_max_new_tokens)
        self.max_audit_rounds = max(1, int(max_audit_rounds))
        self.counterexample_attempts = max(1, int(counterexample_attempts))
        self.dedup_across_kinds = bool(dedup_across_kinds)

    # ------------------------------------------------------------------ helpers
    def _call(self, prompt, usage, as_json=True, max_new_tokens=None):
        """One LLM call; returns parsed JSON (an object, else a list, else None) or the plain text."""
        texts, n_in, n_out = generate(self.model, self.tokenizer, [prompt],
                                      max_new_tokens=max_new_tokens or self.max_new_tokens)
        usage.input_tokens += sum(n_in)
        usage.output_tokens += sum(n_out)
        text = texts[0]
        if not as_json:
            return str(text).strip()
        body = str(text).split("</think>")[-1]
        order = [("{", "}"), ("[", "]")]
        if "[" in body and ("{" not in body or body.index("[") < body.index("{")):
            order.reverse()                                     # the output starts with a list
        for start, end in order:
            try:
                return parse_json_first(text, start=start, end=end)
            except ValueError:
                continue
        return None

    @staticmethod
    def split_sentences(text):
        """Response -> sentence units (one per sentence, list item or heading): the numbered premise of V and the
        sentences P counts."""
        units = []
        for line in str(text or "").splitlines():
            line = re.sub(r"^\s*(?:[-*+•]|\d+[.)]|#+)\s+", "", line)      # bullet / number / heading marks
            line = line.replace("**", "").replace("__", "").strip()
            for part in re.split(r"(?<=[.!?])\s+", line):
                part = part.strip(" *_`>|")
                if len(re.findall(r"\w+", part)) >= 2:                     # skip "---", "Absolutely!", emojis
                    units.append(part)
        return units

    @staticmethod
    def _extract_constraint_list(parsed, preferred_key="constraints"):
        """A list of constraint items from a list, {"<key>": [...]} or a single bare constraint object."""
        if isinstance(parsed, list):
            return parsed
        if not isinstance(parsed, dict):
            return []
        for key in (preferred_key, "constraints", "preference_constraints", "answer_constraints", "new_constraints"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        if parsed.get("response_hypothesis") or parsed.get("constraint"):
            return [parsed]
        for value in parsed.values():
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                return value
        return []

    @staticmethod
    def ensure_ids(items, prefix, kind):
        """Dicts with a text, a kind and a unique constraint_id (empty / repeated texts dropped, ids kept)."""
        out, seen = [], set()
        for item in items or []:
            if isinstance(item, str):
                item = {"response_hypothesis": item}
            if not isinstance(item, dict):
                continue
            text = str(item.get("response_hypothesis") or item.get("constraint") or "").strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            out.append({**item, "response_hypothesis": text, "kind": kind})
        used, counter = {c["constraint_id"] for c in out if c.get("constraint_id")}, 0
        for c in out:
            if not c.get("constraint_id"):
                counter += 1
                while f"{prefix}_{counter}" in used:
                    counter += 1
                c["constraint_id"] = f"{prefix}_{counter}"
                used.add(c["constraint_id"])
        return out

    @staticmethod
    def _hyps(constraints):
        return [c["response_hypothesis"] for c in constraints]

    @staticmethod
    def stated_profile(records):
        """The user's stated preference(s), used verbatim: one "strong" interest per record (no LLM summary)."""
        interests = []
        for index, item in enumerate(records, start=1):
            record = copy.deepcopy(dict(item)) if isinstance(item, dict) else {"text": str(item)}
            text = str(record.get("preference") or record.get("text") or "").strip()
            if text and text.lower() not in {i["interest"].lower() for i in interests}:
                interests.append({"interest": text, "strength": "strong", "evidence_record_ids": [f"rec_{index}"]})
        return {"user_description": " ".join(i["interest"] for i in interests), "interests": interests}

    @staticmethod
    def _match_interest(name, interests):
        key = str(name or "").strip().lower()
        if not key:
            return None
        return next((it for it in interests if it["interest"].lower() == key), None) or \
            next((it for it in interests if it["interest"].lower() in key or key in it["interest"].lower()), None)

    def _ground_preferences(self, items, profile_result):
        """Keep only preference constraints that trace to a stated interest; attach its strength."""
        interests = profile_result.get("interests") or []
        grounded = []
        for c in items:
            interest = self._match_interest(c.get("source_interest"), interests) if isinstance(c, dict) else None
            if interest is not None:
                grounded.append({**c, "source_interest": interest["interest"], "strength": interest["strength"]})
        return grounded

    # ------------------------------------------------------------------ stage 1: construction
    def construct_constraints(self, query, reference, user_profile=None):
        """Candidate constraints: preference constraints (from the stated interests) + answer constraints (from y*)."""
        usage, profile_result, profile_constraint = _Usage(), {}, []
        if user_profile:
            profile_result = self.stated_profile(user_profile)
            if profile_result["interests"]:
                raw = self._call(_render_prompt(self.prompts["PREFERENCE"], USER_QUERY=query,
                                                PROFILE_JSON=profile_result), usage)
                grounded = self._ground_preferences(self._extract_constraint_list(raw, "preference_constraints"),
                                                    profile_result)
                profile_constraint = self.ensure_ids(grounded, "pc", "preference")
        raw = self._call(_render_prompt(self.prompts["ANSWER"], USER_QUERY=query, GROUND_TRUTH_ANSWER=reference), usage)
        answer_constraint = self.ensure_ids(self._extract_constraint_list(raw, "constraints"), "ac", "answer")
        return ({"profile_result": profile_result, "profile_constraint": profile_constraint,
                 "answer_constraint": answer_constraint}, usage.input_tokens, usage.output_tokens)

    # ------------------------------------------------------------------ stage 2: qualification
    @staticmethod
    def _evaluation_records(result, expected=None):
        """V's result for every constraint with its evidence (for the audit log); expected: {id: keep | break | ...}."""
        records = []
        for e in result.get("evaluations") or []:
            record = {k: e.get(k) for k in ("constraint_id", "kind", "hypothesis", "label", "evidence_ids",
                                            "evidence", "evidence_sentences", "reasoning")}
            if expected and e.get("constraint_id") in expected:
                want = expected[e["constraint_id"]]
                record["expected"] = want
                record["as_expected"] = (e.get("label") == "Entailment" if want in ("keep", "entailed_by_label")
                                         else e.get("label") in ConQualAudit.VIOLATION_LABELS)
            records.append(record)
        return records

    def _check_counterexample(self, query, response, keep, broken, usage):
        """V must label every constraint in `keep` Entailment and every one in `broken` Contradiction / Neutral."""
        if not str(response or "").strip():
            return {"valid": False, "labels": {}, "problems": ["empty counterexample"], "evaluations": []}
        result = self._verify(response, list(keep) + list(broken), usage, query)
        expected = {**{c["constraint_id"]: "keep" for c in keep}, **{c["constraint_id"]: "break" for c in broken}}
        evaluations = self._evaluation_records(result, expected)
        if result.get("error"):
            return {"valid": False, "labels": {}, "problems": [result["error"]], "evaluations": evaluations}
        labels = {e["constraint_id"]: e["label"] for e in result["evaluations"]}
        problems = [f"{c['constraint_id']} should be kept but is {labels[c['constraint_id']]}"
                    for c in keep if labels[c["constraint_id"]] != "Entailment"]
        problems += [f"{c['constraint_id']} should be broken but is {labels[c['constraint_id']]}"
                     for c in broken if labels[c["constraint_id"]] not in self.VIOLATION_LABELS]
        return {"valid": not problems, "labels": labels, "problems": problems, "evaluations": evaluations}

    def _adverse(self, prompt, query, keep, broken, usage):
        """ADVERSE: generate until V confirms the counterexample (at most K tries). Returns (response | None, check,
        attempts)."""
        check, tries = {"valid": False, "labels": {}, "problems": ["not generated"], "evaluations": []}, []
        for attempt in range(1, self.counterexample_attempts + 1):
            response = self._call(prompt, usage, as_json=False)
            check = self._check_counterexample(query, response, keep, broken, usage)
            tries.append({"attempt": attempt, "response": response, "valid": check["valid"],
                          "problems": check["problems"], "verification": check["evaluations"]})
            check["tries"] = tries
            if check["valid"]:
                return response, check, attempt
        return None, check, self.counterexample_attempts

    def _duplicate_pairs(self, constraints):
        """Every ordered (premise, hypothesis) pair of different constraints (same kind only if not across kinds)."""
        pairs = []
        for a in constraints:
            for b in constraints:
                if a is b or (not self.dedup_across_kinds and a["kind"] != b["kind"]):
                    continue
                pairs.append({"pair_id": f"d{len(pairs) + 1}", "premise_id": a["constraint_id"],
                              "hypothesis_id": b["constraint_id"], "premise": a["response_hypothesis"],
                              "hypothesis": b["response_hypothesis"]})
        return pairs

    def _entailment_pairs(self, query, pairs, usage):
        """ONE DUPLICATE_CHECK call for the pair list -> ({(premise_id, hypothesis_id): label}, parsed). A missing or
        unreadable label is Neutral, so an unreadable check removes nothing."""
        payload = [{"pair_id": p["pair_id"], "premise": p["premise"], "hypothesis": p["hypothesis"]} for p in pairs]
        raw = self._call(_render_prompt(self.prompts["DUPLICATE_CHECK"], USER_QUERY=query, PAIRS_LIST=payload), usage,
                         max_new_tokens=self.verify_max_new_tokens)
        items = raw.get("evaluations") if isinstance(raw, dict) else raw if isinstance(raw, list) else None
        by_pair = {}
        for item in items or []:
            if isinstance(item, dict):
                label = str(item.get("label") or "").strip().lower()
                label = next((l for l in self.LABELS if l.lower() == label), None)
                if label is not None:
                    by_pair.setdefault(str(item.get("pair_id") or "").strip(), label)
        labels = {(p["premise_id"], p["hypothesis_id"]): by_pair.get(p["pair_id"], "Neutral") for p in pairs}
        return labels, isinstance(items, list)

    def _remove_duplicates(self, query, profile_c, answer_c, *, usage, log):
        """Of two mutually entailing constraints, the later one is removed (whatever their kinds)."""
        constraints = list(profile_c) + list(answer_c)
        pairs = self._duplicate_pairs(constraints) if len(constraints) >= 2 else []
        if not pairs:
            return profile_c, answer_c
        labels, parsed_ok = self._entailment_pairs(query, pairs, usage)
        log.append({"stage": "duplicate_check", "pairs": len(pairs), "parsed": parsed_ok,
                    "labels": [{"pair_id": p["pair_id"], "premise_id": p["premise_id"],
                                "hypothesis_id": p["hypothesis_id"],
                                "label": labels[(p["premise_id"], p["hypothesis_id"])]} for p in pairs]})
        removed = {}
        for i, a in enumerate(constraints):
            if a["constraint_id"] in removed:
                continue
            for b in constraints[i + 1:]:
                a_id, b_id = a["constraint_id"], b["constraint_id"]
                if b_id not in removed and labels.get((a_id, b_id)) == "Entailment" \
                        and labels.get((b_id, a_id)) == "Entailment":
                    removed[b_id] = a_id
                    log.append({"stage": "duplicate", "kind": b["kind"], "constraint_id": b_id, "duplicate_of": a_id,
                                "hypothesis": b["response_hypothesis"], "kept_hypothesis": a["response_hypothesis"],
                                "decision": f"dropped (duplicate of {a_id})"})
        return ([c for c in profile_c if c["constraint_id"] not in removed],
                [c for c in answer_c if c["constraint_id"] not in removed])

    @staticmethod
    def _preference_text(profile_result):
        interests = [str(i.get("interest") or "").strip() for i in (profile_result or {}).get("interests") or []]
        return " ".join(i for i in interests if i) or "No stated preference applies."

    def _alignment_judge(self, query, label, profile_result, candidate, usage):
        """J: is the response aligned with the preference AND with the reference? passed: True / False / None."""
        result = self._call(_render_prompt(self.prompts["ALIGNMENT_JUDGE"], USER_QUERY=query,
                                           USER_PREFERENCE=self._preference_text(profile_result),
                                           GROUND_TRUTH=label, CANDIDATE_RESPONSE=candidate), usage)
        if not isinstance(result, dict):
            return {"passed": None, "preference_aligned": None, "ground_truth_aligned": None, "reason": ""}

        def part(key):
            return result.get(key) if isinstance(result.get(key), dict) else None

        def aligned(key):
            value = (part(key) or {}).get("is_aligned")
            if isinstance(value, str):
                value = {"true": True, "false": False}.get(value.strip().lower())
            return value if isinstance(value, bool) else None

        pref, truth = aligned("preference_alignment"), aligned("ground_truth_alignment")
        verdict = str(result.get("final_verdict", "")).strip().upper()
        if pref is not None and truth is not None:
            passed = pref and truth                    # both requirements must hold
        elif pref is False or truth is False or verdict == "FAIL":
            passed = False                             # one requirement is known to fail
        else:
            passed = None                              # unusable: not enough to remove a constraint
        reason = " ".join(str((part(k) or {}).get("reasoning") or "") for k in
                          ("preference_alignment", "ground_truth_alignment") if part(k))
        return {"passed": passed, "preference_aligned": pref, "ground_truth_aligned": truth, "reason": reason.strip(),
                "preference_reason": str((part("preference_alignment") or {}).get("reasoning") or ""),
                "ground_truth_reason": str((part("ground_truth_alignment") or {}).get("reasoning") or "")}

    def _prune(self, query, label, profile_result, profile_c, answer_c, *, usage, log):
        """Necessity (greedy leave-one-out, preference constraints first): c is removed only if a response that
        breaks only c (confirmed by V) is still judged aligned by J. Each constraint is tested against the ones still
        kept; the last remaining constraint is never tested."""
        kept = list(profile_c) + list(answer_c)
        for c in list(kept):
            if len(kept) <= 1:
                break
            others = [x for x in kept if x is not c]
            prompt = _render_prompt(self.prompts["OVER_CONSTRAINT"], USER_QUERY=query,
                                    PRE_CONTEXT_JSON=self._hyps([x for x in others if x["kind"] == "preference"]),
                                    TRUTH_CONTEXT_JSON=self._hyps([x for x in others if x["kind"] != "preference"]),
                                    NEG_CONSTRAINTS_JSON=c["response_hypothesis"])
            violating, check, attempts = self._adverse(prompt, query, keep=others, broken=[c], usage=usage)
            entry = {"stage": "prune", "kind": c["kind"], "constraint_id": c["constraint_id"],
                     "hypothesis": c["response_hypothesis"], "attempts": attempts, "counterexample": violating,
                     "counterexample_valid": violating is not None, "verification_labels": check["labels"],
                     "verification": check["evaluations"], "tries": check.get("tries", []),
                     "problems": check["problems"]}
            if violating is None:
                entry["decision"] = "kept (no response satisfies the others while breaking it)"
            else:
                verdict = self._alignment_judge(query, label, profile_result, violating, usage)
                entry.update(judge_passed=verdict["passed"], preference_aligned=verdict["preference_aligned"],
                             ground_truth_aligned=verdict["ground_truth_aligned"], judge_reason=verdict["reason"])
                if verdict["passed"] is True:
                    kept = others
                    entry["decision"] = "dropped (not necessary: breaking it still gives an aligned response)"
                elif verdict["passed"] is False:
                    entry["decision"] = "kept (necessary: breaking it gives a misaligned response)"
                else:
                    entry["decision"] = "kept (inconclusive: unusable judge verdict)"
            log.append(entry)
        ids = {x["constraint_id"] for x in kept}
        return [x for x in profile_c if x["constraint_id"] in ids], [x for x in answer_c if x["constraint_id"] in ids]

    def _close_gaps(self, query, label, profile_result, profile_c, answer_c, *, usage, log):
        """Sufficiency, per requirement (reference, then the preference if one is stated): if a response that
        satisfies every constraint (confirmed by V) is judged misaligned by J, UPDATE adds the verified new
        constraints that exclude it. Up to max_audit_rounds rounds per requirement."""
        profile_c, answer_c = list(profile_c), list(answer_c)
        has_preference = bool((profile_result or {}).get("interests"))
        for target, mode, _ in self.GAP_TARGETS:
            if target == "preference" and not has_preference:
                continue
            for audit_round in range(1, self.max_audit_rounds + 1):
                current = profile_c + answer_c
                prompt = _render_prompt(self.prompts["UNDER_CONSTRAINT"], USER_QUERY=query, GROUND_TRUTH_LABEL=label,
                                        USER_PREFERENCES=self._preference_text(profile_result),
                                        OPERATIONAL_CONSTRAINTS=self._hyps(current), TARGET_MODE=mode)
                adversary, check, attempts = self._adverse(prompt, query, keep=current, broken=[], usage=usage)
                entry = {"stage": "close_gaps", "target": target, "round": audit_round, "attempts": attempts,
                         "counterexample": adversary, "counterexample_valid": adversary is not None,
                         "verification_labels": check["labels"], "verification": check["evaluations"],
                         "tries": check.get("tries", []), "problems": check["problems"]}
                if adversary is None:
                    entry["decision"] = "no change (no response satisfies every constraint while breaking it)"
                    log.append(entry)
                    break
                verdict = self._alignment_judge(query, label, profile_result, adversary, usage)
                entry.update(judge_passed=verdict["passed"], preference_aligned=verdict["preference_aligned"],
                             ground_truth_aligned=verdict["ground_truth_aligned"], judge_reason=verdict["reason"])
                if verdict["passed"] is not False:
                    entry["decision"] = ("no change (the response is still aligned: no gap)" if verdict["passed"]
                                         else "no change (inconclusive: unusable judge verdict)")
                    log.append(entry)
                    break
                # UPDATE for every requirement the response breaks
                broken = [t for t, _, _ in self.GAP_TARGETS if verdict[f"{t}_aligned"] is False
                          and (t != "preference" or has_preference)] or [target]
                entry["broken_requirements"] = broken
                added_all, proposed_all, rejected_all, repair_checks = [], [], {}, []
                for requirement in broken:
                    kind = next(k for t, _, k in self.GAP_TARGETS if t == requirement)
                    existing = profile_c if kind == "preference" else answer_c
                    raw = self._call(_render_prompt(
                        self.prompts["CONSTRAINT_UPDATE"], USER_QUERY=query, CURRENT_CONSTRAINTS=self._hyps(profile_c + answer_c),
                        REFERENCE_MATERIAL=label if kind == "answer" else profile_result, COUNTEREXAMPLE=adversary,
                        FAILURE_REASON=verdict[f"{requirement}_reason"] or verdict["reason"]), usage)
                    new = [{k: v for k, v in c.items() if k != "constraint_id"}
                           for c in self._extract_constraint_list(raw, "new_constraints") if isinstance(c, dict)]
                    if kind == "preference":
                        new = self._ground_preferences(new, profile_result)
                    merged = self.ensure_ids(existing + new, "acu" if kind == "answer" else "pcu", kind)
                    proposed = merged[len(existing):]                     # "u" = added by the audit
                    added, rejected = self._check_repair(query, label, adversary, proposed, kind=kind, usage=usage,
                                                         record=repair_checks)
                    if kind == "preference":
                        profile_c = profile_c + added
                    else:
                        answer_c = answer_c + added
                    added_all += added
                    proposed_all += proposed
                    rejected_all.update(rejected)
                entry.update(proposed=[c["constraint_id"] for c in proposed_all],
                             added=[c["constraint_id"] for c in added_all], rejected=rejected_all,
                             repair_verification=repair_checks,
                             decision=(f"added {len(added_all)} constraint(s)" if added_all
                                       else "no change (no verified repair)" if proposed_all else "no change (nothing new)"))
                log.append(entry)
                if not added_all:
                    break
        return profile_c, answer_c

    def _check_repair(self, query, label, counterexample, proposed, *, kind, usage, record=None):
        """Keep the proposed constraints that the counterexample violates (V: Contradiction / Neutral) and, for answer
        constraints, that the reference itself entails. Returns (kept, {constraint_id: reason})."""
        if not proposed:
            return [], {}
        rejected = {}
        on_counterexample = self._verify(counterexample, proposed, usage, query)
        if record is not None:
            record.append({"check": "excludes the counterexample", "verification": self._evaluation_records(
                on_counterexample, {c["constraint_id"]: "break" for c in proposed})})
        if on_counterexample.get("error"):
            return [], {c["constraint_id"]: "verifier output unreadable" for c in proposed}
        labels = {e["constraint_id"]: e["label"] for e in on_counterexample["evaluations"]}
        candidates = []
        for c in proposed:
            if labels[c["constraint_id"]] in self.VIOLATION_LABELS:
                candidates.append(c)
            else:
                rejected[c["constraint_id"]] = f"does not exclude the counterexample ({labels[c['constraint_id']]})"
        if kind == "answer" and candidates:
            on_label = self._verify(str(label), candidates, usage, query)
            if record is not None:
                record.append({"check": "holds for the ground truth", "verification": self._evaluation_records(
                    on_label, {c["constraint_id"]: "entailed_by_label" for c in candidates})})
            gold = {e["constraint_id"]: e["label"] for e in on_label["evaluations"]}
            if on_label.get("error"):
                gold = {c["constraint_id"]: "unreadable" for c in candidates}
            for c in list(candidates):
                if gold[c["constraint_id"]] != "Entailment":
                    candidates.remove(c)
                    rejected[c["constraint_id"]] = f"not satisfied by the ground truth ({gold[c['constraint_id']]})"
        return candidates, rejected

    def qualify_constraints(self, query, reference, constraints):
        """Duplicates, then necessity (pruning), then sufficiency (gap repair); decisions in qualified["audit_log"]."""
        usage, log = _Usage(), []
        profile_result = constraints.get("profile_result") or {}
        profile_c = self.ensure_ids(constraints.get("profile_constraint"), "pc", "preference")
        answer_c = self.ensure_ids(constraints.get("answer_constraint"), "ac", "answer")
        profile_c, answer_c = self._remove_duplicates(query, profile_c, answer_c, usage=usage, log=log)
        profile_c, answer_c = self._prune(query, reference, profile_result, profile_c, answer_c, usage=usage, log=log)
        profile_c, answer_c = self._close_gaps(query, reference, profile_result, profile_c, answer_c, usage=usage, log=log)
        return ({"profile_result": profile_result, "profile_constraint": profile_c, "answer_constraint": answer_c,
                 "audit_log": log}, usage.input_tokens, usage.output_tokens)

    # ------------------------------------------------------------------ stage 3: verification
    def _finalize_verification(self, raw, constraints, sentences=None):
        """V's output aligned to the constraint list: label (missing / malformed -> Neutral), evidence ids, quotes."""
        by_id = {}
        if isinstance(raw, dict):
            for e in raw.get("evaluations") or []:
                if isinstance(e, dict) and e.get("constraint_id") is not None:
                    by_id[str(e["constraint_id"])] = e
        evaluations = []
        for c in constraints:
            e = dict(by_id.get(c["constraint_id"], {}))
            label = str(e.get("label", "")).strip().capitalize()
            e.update(constraint_id=c["constraint_id"], hypothesis=c["response_hypothesis"],
                     label=label if label in self.LABELS else "Neutral", kind=c["kind"], strength=c.get("strength"))
            quotes = e.get("evidence")
            quotes = quotes if isinstance(quotes, list) else [quotes] if quotes else []
            e["evidence"] = [str(q).strip() for q in quotes if str(q or "").strip()]
            if sentences is not None:
                ids = e.get("evidence_ids") if isinstance(e.get("evidence_ids"), list) else []
                e["evidence_ids"] = sorted({int(i) for i in ids if str(i).strip().isdigit()
                                            and 1 <= int(i) <= len(sentences)})
                e["evidence_sentences"] = [{"sentence_id": i, "sentence": sentences[i - 1]} for i in e["evidence_ids"]]
            evaluations.append(e)
        result = {"evaluations": evaluations}
        if not (by_id or not constraints):
            result["error"] = "Verifier output could not be parsed."
        return result

    def _verify(self, response, constraints, usage, query="", evidence=True):
        """Step 1 (VERIFICATION): label every constraint against the whole response. Step 2 (EVIDENCE): for the
        entailed constraints, the numbered sentences that support each one."""
        if not constraints:
            return self._finalize_verification({"evaluations": []}, [])
        payload = [{"constraint_id": c["constraint_id"], "hypothesis": c["response_hypothesis"]} for c in constraints]
        sentences = self.split_sentences(response) or [str(response or "").strip()]
        premise = [{"sentence_id": i, "text": text} for i, text in enumerate(sentences, 1)]
        raw = self._call(_render_prompt(self.prompts["VERIFICATION"], USER_QUERY=query, CANDIDATE_RESPONSE=premise,
                                        CONSTRAINTS_LIST=payload), usage, max_new_tokens=self.verify_max_new_tokens)
        result = self._finalize_verification(raw, constraints, sentences=sentences)
        for e in result["evaluations"]:                                # evidence comes only from step 2
            e.update(evidence_ids=[], evidence=[], evidence_sentences=[])
        entailed = [e for e in result["evaluations"] if e["label"] == "Entailment"]
        if evidence and entailed and not result.get("error"):
            self._attach_evidence(query, premise, entailed, usage, result)
        return result

    def _attach_evidence(self, query, premise, entailed, usage, result):
        """Step 2: ONE EVIDENCE call - for every numbered sentence, the entailed constraints it supports."""
        support = {e["constraint_id"]: set() for e in entailed}
        raw = self._call(_render_prompt(
            self.prompts["EVIDENCE"], USER_QUERY=query, CANDIDATE_RESPONSE=premise,
            CONSTRAINTS_LIST=[{"constraint_id": e["constraint_id"], "hypothesis": e["hypothesis"]} for e in entailed]),
            usage, max_new_tokens=self.verify_max_new_tokens)
        rows = raw.get("sentences") if isinstance(raw, dict) else raw if isinstance(raw, list) else None
        judged = set()
        for row in rows or []:
            if not isinstance(row, dict) or not str(row.get("sentence_id", "")).strip().isdigit():
                continue
            sid = int(row["sentence_id"])
            if not 1 <= sid <= len(premise):
                continue
            judged.add(sid)
            supports = row.get("supports")
            for cid in supports if isinstance(supports, list) else [supports] if supports else []:
                if str(cid).strip() in support:
                    support[str(cid).strip()].add(sid)
        for e in entailed:
            sids = sorted(support[e["constraint_id"]])
            e.update(evidence_ids=sids, evidence=[premise[i - 1]["text"] for i in sids],
                     evidence_sentences=[{"sentence_id": i, "sentence": premise[i - 1]["text"]} for i in sids])
        result["evidence_check"] = {
            "parsed": isinstance(rows, list), "sentences_judged": len(judged), "sentences_total": len(premise),
            "entailed_without_evidence": [e["constraint_id"] for e in entailed if not e["evidence_ids"]]}

    def verify_constraints(self, query, response, constraints):
        """Verify a response (premise) against every qualified constraint (hypothesis)."""
        usage = _Usage()
        all_constraints = (self.ensure_ids(constraints.get("profile_constraint"), "pc", "preference")
                           + self.ensure_ids(constraints.get("answer_constraint"), "ac", "answer"))
        result = self._verify(f"{response}", all_constraints, usage, query)
        return result, usage.input_tokens, usage.output_tokens

    # ------------------------------------------------------------------ pipeline
    def build_constraints(self, query, reference, user_profile=None):
        """Stages 1-2. Returns (constructed, qualified, input_tokens, output_tokens)."""
        constructed, in1, out1 = self.construct_constraints(query, reference, user_profile)
        qualified, in2, out2 = self.qualify_constraints(query, reference, constructed)
        return constructed, qualified, in1 + in2, out1 + out2

    def evaluate(self, query, response, reference, user_profile=None):
        """End to end for one response: (R, P, F1 metrics, verification, qualified constraints, constructed ones)."""
        constructed, qualified, _, _ = self.build_constraints(query, reference, user_profile)
        verified, _, _ = self.verify_constraints(query, response, qualified)
        return self.compute_metrics(response, verified), verified, qualified, constructed

    # ------------------------------------------------------------------ metrics (Eq. 10)
    @staticmethod
    def compute_metrics(response, result):
        """R, P and F1 from the two-step verification (no LLM call). None (undefined, not 0): R without constraints,
        P for an empty response. P counts word tokens (lower-case \\w+) of the split_sentences units that support at
        least one entailed constraint."""
        result = result.get("verification_result", result)
        evaluations = result.get("evaluations") or []
        sentences = ConQualAudit.split_sentences(response) or [str(response or "").strip()]
        sentence_tokens = [len(re.findall(r"\w+", str(text or "").lower())) for text in sentences]
        entailed = [e for e in evaluations if str(e.get("label") or "").strip().lower() == "entailment"]
        supported = set()
        for e in entailed:
            supported |= {i for i in e.get("evidence_ids") or [] if isinstance(i, int) and 1 <= i <= len(sentences)}
        r = len(entailed) / len(evaluations) if evaluations else None
        response_tokens = sum(sentence_tokens)
        supported_tokens = sum(sentence_tokens[i - 1] for i in supported)
        p = supported_tokens / response_tokens if response_tokens else None
        f1 = None if p is None or r is None else (2 * p * r / (p + r) if p + r > 0 else 0.0)
        return {"R": r, "P": p, "f1_score": f1, "constraints": len(evaluations), "entailed": len(entailed),
                "P_sentences": len(supported) / len(sentences) if sentences and response_tokens else None,
                "response_tokens": response_tokens, "supported_tokens": supported_tokens,
                "response_sentences": len(sentences), "supported_sentences": sorted(supported),
                "entailed_without_evidence": [e.get("constraint_id") for e in entailed if not e.get("evidence_ids")]}
