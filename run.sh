#!/usr/bin/env bash
# Local helper. The intended path is the Colab notebook.
set -euo pipefail
cd "$(dirname "$0")"

cmd="${1:-data}"

usage() {
  cat <<'EOF'
usage: ./run.sh {data|test|mask|train|eval-base|eval-schema|eval-icl|eval|curve|compare|all|publish}

  data         write JSONL splits + unit tests
  test         CPU unit tests only
  mask         tokenizer mask demo
  train        LoRA on the 8-card set (outputs/lora-sft)
  eval-base    stock 0.5B, thin prompt
  eval-schema  stock 0.5B, schema spelled out (no SFT)
  eval-icl     stock 0.5B, 8-shot train cards (no SFT)
  eval         adapter on frozen test
  curve        train+eval LoRA on 2 / 4 / 8 / 16 cards
  compare      print the table; write outputs/RESULTS.md
  all          data + three stock baselines + 8-card train + eval + compare
  publish      upload outputs/lora-sft (needs HF_TOKEN)
EOF
}

case "$cmd" in
  data)
    python3 src/write_data.py
    python3 src/test_pipeline.py -v
    ;;
  test)
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
    python3 src/evaluate.py --base-only --mode thin --output outputs/base_eval.json
    ;;
  eval-schema)
    python3 src/evaluate.py --base-only --mode schema --output outputs/schema_eval.json
    ;;
  eval-icl)
    python3 src/evaluate.py --base-only --mode icl --output outputs/icl_eval.json
    ;;
  eval)
    python3 src/evaluate.py --adapter outputs/lora-sft --output outputs/adapter_eval.json
    ;;
  curve)
    python3 src/write_data.py
    mkdir -p outputs/curve
    for n in 2 4 8 16; do
      echo "==> LoRA-${n}"
      python3 src/train_sft.py --curve "$n" --output-dir "outputs/lora-sft-${n}"
      python3 src/evaluate.py \
        --adapter "outputs/lora-sft-${n}" \
        --output "outputs/curve/lora-${n}.json"
    done
    python3 src/compare.py
    ;;
  compare)
    python3 src/evaluate.py --rescore outputs/base_eval.json || true
    python3 src/evaluate.py --rescore outputs/adapter_eval.json || true
    python3 src/evaluate.py --rescore outputs/schema_eval.json || true
    python3 src/evaluate.py --rescore outputs/icl_eval.json || true
    for n in 2 4 8 16; do
      python3 src/evaluate.py --rescore "outputs/curve/lora-${n}.json" || true
    done
    python3 src/compare.py
    ;;
  all)
    python3 src/write_data.py
    python3 src/test_pipeline.py -v
    python3 src/evaluate.py --base-only --mode thin --output outputs/base_eval.json
    python3 src/evaluate.py --base-only --mode schema --output outputs/schema_eval.json
    python3 src/evaluate.py --base-only --mode icl --output outputs/icl_eval.json
    if [[ ! -f outputs/lora-sft/adapter_model.safetensors ]]; then
      python3 src/train_sft.py --max-steps "${MAX_STEPS:-40}"
    fi
    python3 src/evaluate.py --adapter outputs/lora-sft --output outputs/adapter_eval.json
    python3 src/compare.py
    ;;
  publish)
    python3 scripts/publish_adapter.py
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
