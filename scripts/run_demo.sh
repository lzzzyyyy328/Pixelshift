#!/usr/bin/env bash
set -e

# Run from the project root:
# bash scripts/run_demo.sh

python main.py pipeline \
  --config configs/default.yaml \
  --output runs/demo_pipeline \
  --preset default \
  --height 512 \
  --width 512 \
  --iterations 60 \
  --no-progress
