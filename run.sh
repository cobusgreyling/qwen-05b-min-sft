#!/usr/bin/env bash
# Local helper. The intended path is the Colab notebook.
set -euo pipefail
cd "$(dirname "$0")"

cmd="${1:-data}"

case "$cmd" in
  data)
    python3 src/write_data.py
    python3 src/test_pipeline.py -v
    ;;
  mask)
    python3 src/demo_masking.py
    ;;
  train)
    python3 src/write_data.py
    python3 src/train_sft.py --max-steps "${MAX_STEPS:-40}"
    ;;
  eval-base)
    python3 src/evaluate.py --base-only --output outputs/base_eval.json
    ;;
  eval)
    python3 src/evaluate.py --adapter outputs/lora-sft --output outputs/adapter_eval.json
    ;;
  *)
    echo "usage: $0 {data|mask|train|eval-base|eval}"
    exit 2
    ;;
esac
