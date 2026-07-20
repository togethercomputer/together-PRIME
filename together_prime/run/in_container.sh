#!/bin/bash
# Launch the pinned verl container with GPUs + persistent mounts, run the given script
# inside it. Paths derive from this file's location so the harness is relocatable.
#
# Usage: bash run/in_container.sh prep.sh   |   bash run/in_container.sh smoke_run.sh
# Env overrides: VERL_IMAGE, VERL_DIR (the setup.sh checkout), DATA_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOGETHER_PRIME="$(cd "$HERE/.." && pwd)"
# shellcheck disable=SC1091
source "$TOGETHER_PRIME/VERSIONS"

VERL_IMAGE="${VERL_IMAGE:-$IMAGE}"
VERL_DIR="${VERL_DIR:-$TOGETHER_PRIME/verl}"
DATA_DIR="${DATA_DIR:-$TOGETHER_PRIME/data}"
SCRIPT="${1:?usage: in_container.sh <script.sh>}"

if [ ! -d "$VERL_DIR" ]; then
    echo "verl checkout not found at $VERL_DIR — run: bash together_prime/setup.sh" >&2
    exit 1
fi
mkdir -p "$DATA_DIR/hf_cache"

docker run --rm --gpus all \
    --ipc=host --shm-size=32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "$VERL_DIR:/workspace/verl" \
    -v "$DATA_DIR:/data" \
    -v "$HERE:/scripts" \
    -w /workspace/verl \
    "$VERL_IMAGE" \
    bash "/scripts/$SCRIPT"
