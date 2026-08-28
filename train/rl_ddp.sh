#!/usr/bin/env bash
# Data-parallel Dr.GRPO RL launcher — gloo DDP.
# NCCL DEADLOCKS on some single-node multi-GPU boxes (ranks spin 100% CPU at the first
# collective, 0% GPU); gloo works because the base model is frozen and only the LoRA grads
# all-reduce (same fix as train/pretrain.py). Each rank handles groups_per_step/world WHOLE
# groups; rank 0 logs to wandb + saves checkpoints. groups_per_step must divide by NPROC.
#
# Usage (from the repo root):
#   NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/rl_ddp.sh <rl.py args...>
#   NPROC=2 CUDA_VISIBLE_DEVICES=2,3   bash train/rl_ddp.sh ... (smoke)
set -u
export PYTHONPATH=${PYTHONPATH:-$PWD}
export DDP_BACKEND=gloo            # the whole point — do NOT switch back to nccl
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0          # harmless under gloo; kept for safety on Blackwell
NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-29531}
exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    train/rl.py "$@"
