# Scored runs

These JSON files are the last **frozen-test** pictures, not training weights.

| File | What it is |
|------|------------|
| `base_eval.json` | Stock `Qwen/Qwen2.5-0.5B-Instruct` on the 6 test tickets |
| `adapter_eval.json` | Same 6 tickets after 40 LoRA steps on the 8 train cards |

Diff `summary`, then read `per_prompt[].generation`.

Adapters themselves (`outputs/lora-sft/`) are gitignored — a few megabytes, produced by `./run.sh train` or the Colab notebook.
