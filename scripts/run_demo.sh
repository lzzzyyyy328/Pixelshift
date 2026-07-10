#!/usr/bin/env bash
set -e

# Run from the project root:
# bash scripts/run_demo.sh

python main.py pipeline \
  --output runs/demo_pipeline \
  --height 1024 \
  --width 1024 \
  --scale 2 \
  --shifts "0,0;0.5,0;0,0.5;0.5,0.5" \
  --shift-unit lr \
  --blur-kernel 3 \
  --blur-sigma 0 \
  --blur-stage after_downsample \
  --noise-poisson-a 0.0 \
  --noise-gaussian-b 0.0000615 \
  --mode ibp \
  --iterations 40 \
  --step-size 0.20
