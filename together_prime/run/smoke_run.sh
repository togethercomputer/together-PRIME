#!/bin/bash
# Runs INSIDE the container. Scaled-down PRIME smoke test on 8xB200: 3 steps, gsm8k only,
# no wandb. Validates the full loop (rollout -> verify -> implicit-PRM update -> RLOO ->
# FSDP policy update) with Eurus-2-7B-SFT.
#
# The ONLY Blackwell-specific fix needed is actor_rollout_ref.rollout.free_cache_engine=False
# (see below). Everything else here is ordinary launch config.
set -euxo pipefail

export HF_HOME=/data/hf_cache
export PYTHONPATH=/workspace/verl:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — vLLM's sleep-mode memory
# pool asserts against it (pytorch/pytorch#147851). We disable sleep mode anyway (below).

cd /workspace/verl

train_path=/data/gsm8k/train.parquet
test_path=/data/gsm8k/test.parquet
model_path=PRIME-RL/Eurus-2-7B-SFT

python3 -m recipe.prime.main_prime \
    data.train_files="['$train_path']" \
    data.val_files="['$test_path']" \
    data.train_batch_size=16 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.filter_accuracy=False \
    data.oversample_factor=1 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.adv_estimator=rloo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.model.path=$model_path \
    reward_model.model.update=before \
    reward_model.model.beta_train=0.05 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.mini_batch_size=16 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name='prime_smoke' \
    trainer.experiment_name='eurus2-7b-b200-smoke' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/data/ckpts/prime_smoke \
    trainer.resume_mode=disable \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    trainer.total_training_steps=3 \
    trainer.total_epochs=1 "$@"

echo "SMOKE_DONE"
