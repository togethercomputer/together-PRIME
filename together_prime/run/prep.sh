#!/bin/bash
# Runs INSIDE the container. Prepares gsm8k parquet + pre-downloads the base model so the
# smoke run can stay offline (avoids HF Hub 429s on repeated runs without an HF_TOKEN).
set -euxo pipefail

export HF_HOME=/data/hf_cache
# prep must reach the HF Hub to download; override the container-level offline flags.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=/workspace/verl:${PYTHONPATH:-}
mkdir -p /data/gsm8k

# gsm8k -> train.parquet / test.parquet (data_source="openai/gsm8k")
python3 /workspace/verl/examples/data_preprocess/gsm8k.py --local_save_dir /data/gsm8k

# pre-fetch policy / implicit-PRM init model (public, no token needed)
python3 - <<'PY'
from huggingface_hub import snapshot_download
print("model at", snapshot_download("PRIME-RL/Eurus-2-7B-SFT"))
PY

ls -la /data/gsm8k
echo "PREP_DONE"
