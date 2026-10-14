#!/usr/bin/env bash
# All PrefEval runs of the paper, then Tables 2-3 and the Figure 3 case.  Usage: bash scripts/run_prefeval.sh [cuda:0]
set -euo pipefail
DEVICE=${1:-cuda:0}
for step in baselines.gemba baselines.factscore_expert baselines.pref_perse \
            experiments.run_conqual experiments.ablation experiments.audit_baselines; do
  python -m "$step" --dataset prefeval --device "$DEVICE"
done
python -m tables.prefeval
python -m tables.case_study --query prefeval_0594::q1
