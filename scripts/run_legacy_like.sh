#!/usr/bin/env bash
set -e

# This reproduces the shift convention in your uploaded prototype more closely:
# shifts are applied directly in HR-pixel units, not LR-pixel units.

python main.py pipeline \
  --output runs/legacy_like_pipeline \
  --height 512 \
  --width 512 \
  --scale 2 \
  --shifts "0,0;0.5,0;0,0.5;0.5,0.5" \
  --shift-unit hr \
  --blur-kernel 3 \
  --blur-stage after_downsample \
  --noise-poisson-a 0.0 \
  --noise-gaussian-b 0.000984 \
  --mode ibp \
  --iterations 40
