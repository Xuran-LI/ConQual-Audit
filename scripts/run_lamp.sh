#!/usr/bin/env bash
# All LaMP runs of the paper, then Table 1 (build data/lamp*/ first: see README).  Usage: bash scripts/run_lamp.sh [cuda:0]
set -euo pipefail
DEVICE=${1:-cuda:0}
for dataset in lamp1 lamp2 lamp4; do
  python -m baselines.overlap --dataset "$dataset"
  for step in baselines.similarity baselines.gemba baselines.factscore_expert baselines.pref_perse \
              experiments.run_conqual; do
    python -m "$step" --dataset "$dataset" --device "$DEVICE"
  done
done
python -m tables.lamp
