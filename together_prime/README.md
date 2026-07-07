# Together Prime — B200-capable PRIME harness

This directory is the **Blackwell (B200) capable** way to run the PRIME RL workload
(Eurus-2-7B policy + implicit-PRM + vLLM rollout, RLOO). The legacy top-level `training/`
tree vendors an old verl (`vllm<=0.6.3`) that has no `sm_100` kernels and cannot run on
Blackwell — it is left untouched for historical reference. **Use this directory on B200.**

PRIME has since been upstreamed into veRL as `recipe/prime`. Rather than fork or vendor
verl, this harness pins an upstream verl commit, applies **one** small patch to the recipe,
and drives it with launch config. All pins are in [`VERSIONS`](./VERSIONS).

## The fix set (deliberately minimal)

We validated a 3-step smoke test end-to-end on 8xB200, then ablated every workaround to
keep only what is actually required. The result is small:

| # | What | Kind | Why |
|---|------|------|-----|
| 1 | **recipe/prime port** (`patches/recipe-prime-port.patch`) | **code** | The upstream recipe had drifted from current verl. Fixes: `need_reference_policy(config)`; set `reward_fn`/`val_reward_fn` as attributes (dropped from `RayPPOTrainer.__init__`); rewrite `fit()` rollout to `_get_gen_batch` + `async_rollout_manager` (current verl tokenizes server-side from raw messages instead of popping pre-tokenized tensors). |
| 2 | **`rollout.free_cache_engine=False`** | config | **The one Blackwell fix.** vLLM's sleep/wake `cumem` allocator (used to share GPU memory with the colocated FSDP actor) corrupts GPU memory on B200, surfacing as either a FlashInfer worker crash or a cuBLAS `CUBLAS_STATUS_EXECUTION_FAILED` / illegal-memory-access in the model GEMM. Disabling it fixes both. |
| 3 | cu12.9 image `verlai/verl:vllm018.dev1` | config | The cu13.0 images (`vllm020`/`vllm023`) fault in the model cuBLAS GEMM on B200. |
| 4 | `rollout.mode=async`, `rollout.enforce_eager=True` | config | `sync` rollout was removed upstream; eager avoids cudagraph capture during colocation. |
| 5 | `resume_mode=disable`, `default_local_dir=/data/...` | config | Auto-resume trips on PyTorch 2.6 `weights_only=True` un-pickling a leftover dataloader checkpoint; and checkpoints shouldn't land inside the repo. |
| 6 | HF offline (or set `HF_TOKEN`) | config | Repeated runs without a token get HF Hub 429s; `prep.sh` pre-caches the model, the run stays offline. |

**verl core: 0 patches. vLLM: 0 patches.** The only code change is the recipe port (#1);
everything else is a launch flag.

### Workarounds and their removal triggers

- **#2 `free_cache_engine=False`** is a real workaround for a vLLM-on-Blackwell memory bug,
  not a preference — it costs the memory savings of sleep mode. Revisit when vLLM's cumem
  sleep/wake is fixed for `sm_100`; a tracking issue should be filed and linked here.
- **#3 image pin**: move to a newer cu12.x app image once one is validated on B200.
- **#1 recipe port** is a genuine fix against current verl-recipe and is worth upstreaming
  to `verl-project/verl-recipe`; once merged at the pinned commit, drop the patch.

## Run it

```bash
# 0. one-time: build the pinned verl checkout + apply the recipe port
bash together_prime/setup.sh                 # creates together_prime/verl

# 1. prep data + model cache (needs network; downloads ~15GB once)
bash together_prime/run/in_container.sh prep.sh

# 2. smoke test (8xB200, 3 steps) — proves the pipeline runs
bash together_prime/run/in_container.sh smoke_run.sh

# 3. throughput capture (8xB200, 6 steps, realistic batch) — records tokens/sec
bash together_prime/run/in_container.sh throughput_run.sh
```

Expect exit 0, `SMOKE_DONE`, three `step:N` lines with `actor/pg_loss`, `actor/grad_norm`,
`actor/entropy`, `perf/mfu/actor` logged, and zero CUBLAS / illegal-memory / EngineDead
errors. (`acc:0.0` is expected — the smoke config disables the accuracy filter and 3
untrained steps on gsm8k under the reasoning-prompt format won't score; the goal is that
the full pipeline runs.)

Diagnostics: `run/gpu_check.sh` (bare GEMM sanity), `run/vllm_smoke.sh` (standalone vLLM,
no verl) — both via `in_container.sh`.

## Throughput baseline (8xB200, captured 2026-07)

`run/throughput_run.sh` runs PRIME at a realistic shape (64 prompts × `rollout.n=4` =
256 sequences, `max_response_length=3072`, 6 steps) and reaches steady state by step 3.

| Metric | Value |
| --- | --- |
| Step time (steady state) | ~53.9 s |
| ↳ generation (vLLM rollout) | ~36.4 s (68%) |
| ↳ actor update (FSDP) | ~11.6 s (22%) |
| ↳ ref / old_log_prob / verify | ~5.8 s (11%) |
| Generation throughput | ~21k tok/s total (~2.6k/GPU) |
| End-to-end response throughput | ~14.3k tok/s (~1.8k/GPU) |
| Sequences/sec | ~4.8 |
| Actor MFU | ~19% |
| Memory | ~33 GB alloc / ~58 GB reserved of 183 GB |

The profile is healthy and rollout-dominated, as expected for this shape. It is **not
optimized**: `enforce_eager=True` (our Blackwell-safety flag) disables the vLLM CUDA graph
and leaves decode throughput on the table, and there is large memory headroom for a bigger
batch.

## Next

- **Optimization levers:** test `enforce_eager=False` (risks the vLLM cumem/Blackwell
  crash → needs its own validation) and a 2–3× larger batch to use the memory headroom.
- **Reference bar:** no hard reference throughput is available yet; the numbers above are a
  functional baseline, not a target-relative result.
