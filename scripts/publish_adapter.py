#!/usr/bin/env python3
"""Upload outputs/lora-sft to Hugging Face.

Needs HF_TOKEN (or `huggingface-cli login`) and write access to
HF_REPO (default cobusgreyling/qwen-05b-deskcard-lora).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "outputs" / "lora-sft"
CARD = ROOT / "MODEL_CARD.md"
DEFAULT_REPO = "cobusgreyling/qwen-05b-deskcard-lora"

UPLOAD_NAMES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "run_card.json",
)


def main() -> int:
    repo_id = os.environ.get("HF_REPO", DEFAULT_REPO)
    if not (ADAPTER / "adapter_model.safetensors").exists():
        print(f"ERROR: {ADAPTER} has no adapter. Run ./run.sh train first.", file=sys.stderr)
        return 1
    if not CARD.exists():
        print(f"ERROR: {CARD} missing.", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi, login, whoami
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    try:
        me = whoami()
    except Exception as exc:  # noqa: BLE001
        print(
            "ERROR: not logged in to Hugging Face.\n"
            "  export HF_TOKEN=hf_...\n"
            "  python scripts/publish_adapter.py",
            file=sys.stderr,
        )
        print(f"({exc})", file=sys.stderr)
        return 1

    print(f"logged in as {me.get('name') or me.get('fullname') or me}")
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    for name in UPLOAD_NAMES:
        src = ADAPTER / name
        if not src.exists():
            print(f"skip missing {src}")
            continue
        print(f"upload {name}")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
        )
    url = f"https://huggingface.co/{repo_id}"
    print(f"published → {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
