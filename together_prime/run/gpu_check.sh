#!/bin/bash
# Runs INSIDE the container. Minimal GPU sanity: device info + a bf16/fp16 GEMM on B200.
# Useful to confirm the node/driver/cuBLAS are healthy independent of vLLM/verl.
set -uxo pipefail
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0),
      "count", torch.cuda.device_count())
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); print("bf16 GEMM ok, sum=", float((a @ b).float().sum()))
a16, b16 = a.half(), b.half()
torch.cuda.synchronize(); print("fp16 GEMM ok, sum=", float((a16 @ b16).float().sum()))
print("GPU_CHECK_OK")
PY
