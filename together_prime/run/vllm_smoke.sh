#!/bin/bash
# Runs INSIDE the container. Standalone vLLM generation of the base model — NO verl, NO
# Ray. Isolation test: proves vLLM can serve Eurus-2-7B on this B200 independent of the
# RL stack. (During bring-up this is what showed the crashes were verl-colocation, not
# vLLM/cuBLAS — stock vLLM generates fine here.)
set -uxo pipefail
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="PRIME-RL/Eurus-2-7B-SFT", dtype="bfloat16", tensor_parallel_size=1,
          gpu_memory_utilization=0.6, enforce_eager=True, trust_remote_code=True,
          max_model_len=2048)
out = llm.generate(["Which is larger, 9.11 or 9.9?"],
                   SamplingParams(max_tokens=64, temperature=0.0))
print("VLLM_OUTPUT:", out[0].outputs[0].text[:200])
print("VLLM_STANDALONE_OK")
PY
