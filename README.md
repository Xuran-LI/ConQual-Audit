# ConQual-Audit

Code for *Neither Over- nor Under-Constrained: Evaluating LLM Personalization under a Constraint Qualification Audit*.

ConQual-Audit evaluates a personalized response in three stages:

1. **Construct** candidate constraints from the user's stated preference (preference constraints) and from the
   reference answer y\* (answer constraints).
2. **Qualify** them:
   - remove duplicates (mutual entailment);
   - prune unnecessary constraints: a counterexample that breaks only that constraint is still judged acceptable;
   - close gaps: a counterexample that satisfies every constraint but breaks the preference or the reference is
     judged unacceptable, so at most three new constraints are added that exclude it.
3. **Verify** a response against the qualified constraints and score it by the constraint recall R (accept if R ≥ 0.5).
   The response precision P and F1 are kept in the details.

The generator G, verifier V and judge J are all `Qwen/Qwen3-8B` with greedy decoding. Qualification uses K = 2
counterexample attempts, one gap-repair round per requirement, at most three new constraints per repair, and
counterexamples of at most 200 words.

## Layout

```
conqual/audit.py              ConQual-Audit (construction, qualification, verification, R / P / F1)
benchmarks/                   test sets, and what each evaluator is shown, plus the dataset's prompts (verbatim)
  prefeval.py lamp1.py lamp2.py lamp4.py
  build_prefeval.py           select the 100 PrefEval items and generate the Qwen3-14B responses
  build_lamp.py               build the controlled LaMP-1 / -2 / -4 test sets from the official LaMP files
baselines/                    BLEU-4 / ROUGE-L, BERTScore / MoverScore, GEMBA, FActScore, ExPerT, PREF, PerSE
experiments/
  run_conqual.py              ConQual-Audit on a benchmark
  ablation.py                 ConQual without qualification (the constructed constraints, verified directly)
  audit_baselines.py          ExPerT / PREF constraints audited by ConQual's qualification, then re-scored
tables/                       the paper's tables from the saved results (no GPU needed)
utils/                        judge model + JSON parsing (llm.py), batched runs with checkpoints (runner.py)
data/prefeval/                the PrefEval test responses used in the paper
```

## Setup

```bash
pip install -r requirements.txt        # Python >= 3.10; the paper's runs used one NVIDIA H100 NVL
```

**PrefEval.** `data/prefeval/` contains the 100 selected items (5 per topic) of
[PrefEval](https://huggingface.co/datasets/siyanzhao/prefeval_explicit) and the Qwen3-14B responses used in the paper:
positive = the answer when the preference was stated first, negative = the answer without it.
`python -m benchmarks.build_prefeval` regenerates them. Greedy generation can differ slightly across hardware and
library versions.

**LaMP.** Download the LaMP files ([LaMP benchmark](https://lamp-benchmark.github.io/download)):

- LaMP-1: `dev_questions.json`, `dev_outputs.json` (the LaMP-1 test outputs are not public);
- LaMP-2 and LaMP-4: `test_questions.json`, `test_outputs.json`.

Then build each test set (100 users, 30 responses per query):

```bash
python -m benchmarks.build_lamp --dataset lamp4 --questions <LaMP-4>/test_questions.json --answers <LaMP-4>/test_outputs.json
```

The build is deterministic (seeded user ranking, BM25 profile retrieval, templated responses). It writes
`data/<dataset>/<dataset>_users_test_v1.json`.

## Reproducing the paper

```bash
bash scripts/run_prefeval.sh cuda:0     # all PrefEval runs, then Tables 2-3 and the Figure 3 case
bash scripts/run_lamp.sh cuda:0         # all LaMP-1/2/4 runs, then Table 1
```

Every step writes `results/<dataset>/<step>_<tag>.json` (one record per response with score, details, time and
tokens). Methods that generate constraints also write `..._constraints.json`. Steps are checkpointed, and `--resume`
continues an interrupted run. `--max-users N` runs a quick test.

| Paper | Script | Source runs |
|---|---|---|
| Table 1 (LaMP average) | `python -m tables.lamp` | `baselines.*`, `experiments.run_conqual` on lamp1/2/4 |
| Table 2 (PrefEval) | `python -m tables.prefeval` | `baselines.*`, `experiments.run_conqual`, `experiments.audit_baselines` |
| Table 3, counterfactual "removed only" | `python -m tables.prefeval` | `experiments.audit_baselines` |
| Cost numbers in §5 / Limitations | `python -m tables.prefeval` | `run_conqual`, `ablation`, `audit_baselines` |
| Figure 3 (case study) | `python -m tables.case_study --query prefeval_0594::q1` | `run_conqual`, `ablation` |

How the tables are computed (`tables/common.py`):

- **Signal.** GEMBA (= GEMBA-DA with the reference), PerSE, FActScore and ExPerT score the reference and the
  preference separately. A positive response gets the lower part and a negative response the higher one.
- **Analysis set.** Methods that scored at least 80% of every response kind are compared on the responses all of them
  scored.
- **Decision.** Accept if score ≥ 0.5. FRR = rejected positives, FAR = accepted negatives, False Rate = their mean;
  AUC is tie-aware.
- **Tokens.** End-to-end tokens per response = (online + one-off setup) / responses.
- **#Cons.** The constraints a response is checked against.
- **Audit rows.** They include the method's own constraint generation, the audit, and the re-scoring.

Time depends on the hardware; token counts are comparable.

## Using ConQual-Audit

```python
from benchmarks import prefeval
from conqual import ConQualAudit
from utils.llm import load_model

tokenizer, model = load_model("Qwen/Qwen3-8B", "cuda:0")
audit = ConQualAudit(tokenizer, model, prefeval.CONQUAL_PROMPTS)
metrics, verification, qualified, constructed = audit.evaluate(
    query, response, reference, user_profile=[{"preference": "I dislike team-building activities that involve physical sports."}])
print(metrics["R"], metrics["P"], metrics["f1_score"])
```
