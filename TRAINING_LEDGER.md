# MAEMM RL training ledger — Qwen3.6-27B universal inverter

Base model Qwen/Qwen3.6-27B, inject L1 / read L42, rsLoRA r64/α16 all-linear, steer-coeff 1.0.

*(Training-box path mapping: `RL/rl_hf.py` = `train/rl.py` in this repo, `bsf27b/rl_ddp.sh` =
`train/rl_ddp.sh`, `SL/pretrain.py` = `train/pretrain.py`.)*

## SFT (shared base for every RL run)
- **`big_sft`** — bank `pool_big_sft` = 1,000,000 examples (500k realact short-ctx ≤512 + 500k probes),
  lr 3e-5, batch 16, **2 epochs (25,000 steps)**, multi-GPU **gloo** DDP on 5×B300 (NCCL deadlocks on this box).
  Checkpoint: `checkpoints/big_sft/final`. Loss 4.15 → ~1.8.

## RL (all runs init from `big_sft/final`, data `pool_rl_mix`)
`pool_rl_mix` = balanced thirds: ⅓ realact + ⅓ probes + ⅓ realact_long (750k). Long-ctx acts (`realact_long`,
uniform pos 1–8192) are in the RL mix only, **not** SFT (per fjiahai: long-ctx features differ; RL-only).
Shared RL hypers: Dr.GRPO (adv = r − group_mean, no /std), lr 1e-5, reward = max-over-token cos × 1000,
KL-coef 0.03 (cap 10, ref = SFT init), **diversity** div-coef 40 (activation-orthogonal, mean-pooled over
tokens ≤ the peak/max-activation token), save-every 25.

### Arm SHORT — exploratory, KILLED 2026-08-27
- max-new-tokens 32, **LP off**, groups 64 × group-size 16 (1024 rollouts/step), micro 12 / score 32.
- eval @25→50: early 0.519→0.527 / mid 0.465→0.473 / long 0.408→0.414 · sae_nam 0.569→0.657 · %unverb 0.393→0.330.

### Arm LONG — exploratory, KILLED 2026-08-27
- max-new-tokens 96, **LP start 16, per-tok 0.5**, groups 32 × group-size 16 (512/step), micro 8 / score 16.
- eval @25→50: early 0.524→0.541 / mid 0.468→0.483 / long 0.400→0.417 · sae_nam 0.545→0.599 · %unverb 0.412→0.395.
- LP @0.5/tok ≈ ~1% of reward variance (very slight); rollouts self-limited to ~42 tok (of 96).

### LONG-HORIZON — the chosen run (2026-08-27)
- = LONG config, but **LP per-tok 1.0** (start 16; ≈3–8% of reward variance — stronger per-feature Pareto length),
  max-new-tokens 96, groups 32 × 16, micro 8 / score 16, div-coef 40, **400 steps**.
- save-dir `checkpoints/big_rl_longhz` (save-every 25). Eval daemon → wandb `uni-inverter/big_rl_longhz_heldout`
  (per ckpt: held-out families + early/mid/long ctx buckets + %unverbalized).

Findings so far: RL lifts all context buckets; early > mid > long (~0.53 vs ~0.41) — long-ctx harder;
%unverbalized SAE dropping ~40% → ~33% with RL; longer rollouts help context buckets, slightly hurt SAE peak.

## Diversity sweep (div_coef Pareto) — 2026-08-27
3 free GPUs, config=main (mnt96, group-size16, groups12, 60 steps). PAIRED per-step cos cost
(seed-deterministic directions => noise-cancelled) vs div_coef=0 baseline (mean cos 0.3977):
  1000: +1.2% cost, div 0.0848, gate 0.94
  1500: +1.9% cost, div 0.0875, gate 0.94
  2000: +2.0% cost, div 0.0893, gate 0.93   <-- PINNED: max diversity <= 2.5% budget
  2500: +2.7% cost, div 0.0904, gate 0.92   (barely more div, over budget)
  4000: +5.4% cost, div 0.1005, gate 0.92
Natural diversity ~0.080 (decays untuned; at div_coef=40 it FELL 0.086->0.067 = bonus too weak).
CHOSEN div_coef=2000: +2.0% cosine, diversity +~11%, gate healthy. Diminishing returns in-budget.

## 4-GPU DATA-PARALLEL (gloo DDP) — 2026-08-27
rl_hf.py -> gloo-DDP: whole-groups-per-rank, TOKEN-WEIGHTED LoRA-grad all-reduce (= EXACT single-GPU
grad over union batch, no LR rescale), rank-0-only log/save, world=1 byte-identical. Original .pre_ddp,
diff RL/rl_hf_ddp.diff, launcher bsf27b/rl_ddp.sh (torchrun --nproc_per_node, DDP_BACKEND=gloo; NCCL deadlocks).
Verified: per-rank LoRA checksums match to 10 decimals/step (exact grad sync), no deadlock.
SPEEDUP 223s->59s/step = 3.8x. RUN big-rl-longhz-dp4 = GPUs0-3 world4 from step_50, div_coef=2000, 400 steps
-> ~6.5h (was ~25h); daemon GPU4 (uni-inverter/big_rl_longhz_dp4_heldout); RL maxact-fast/aodwhowf.
Fallback = run rl_hf.py without torchrun (world=1 identical). NEXT: vLLM +1.6x -> ~6x -> ~4h.

## MODAL 8×B200 SCALE-UP (`modal_rl.py`) — 2026-08-28
Port of the box run to a single Modal 8×B200 container (world=8, gloo — the CPU-tensor grad
all-reduce is gloo-by-design; NCCL raises "No backend type associated with device type cpu").
Data on the `maemm-data` Volume (pool_rl_mix, sft_init adapter, HF cache, ckpts); pool staged to
container-local NVMe (memmap over the FUSE mount untrusted). Lessons baked into the app:
- **Warm-starting RL from a deep-RL ckpt (dp4/step_100) COLLAPSED even at LP 1.0** (fresh optimizer
  + weak KL anchor + policy near the reward-hack cliff). SFT-init is the proven-stable pattern.
- **LP 0.25/tok was the root cause of every box collapse** — LP 1.0/tok is the proven-stable value
  (dp4: 117 steps, gate 0.92–0.97, clip <1.3%). Do not lower it.
- **Un-gate-masked diversity bonus caused a collapse** (reward 390→53, gate 90%→8%): degenerate
  gated-out rollouts still earned the div bonus. Fixed by masking the bonus with the fluency gate
  (`_gmask`, patches/patch_div_gate.py); the app REFUSES to train on a trainer without it.
- B200 = 178GB (box B300 = 288GB): micro-batch 8→4 (pure grad-accum slicing, gradients identical)
  + `PYTORCH_ALLOC_CONF=expandable_segments:True` (variable-length RL batches fragment the allocator).
- Ranks load HF weights with `HF_HUB_OFFLINE=1` after a single-flight `snapshot_download` —
  8 concurrent hub re-resolutions spuriously 404'd shards that exist.

## vLLM ROLLOUT PATH (verified 2026-08-27, `scripts/vllm_smoke.py`)
vllm 0.19.0 + vllm_lens 1.1.0 + transformers 5.15.0. THE FIX: the vllm_lens install was missing its
dist-info, so its vLLM plugin entry point never registered and steering silently no-op'd; restoring
dist-info (+ transformers ≥5 for the vendored Qwen3.6 config) makes injection fire. Verified: worker-
side marker-write counter increments, and the captured Δresidual at the marker == ‖h_clean‖·unit(v)
(cos>0.99, norm ratio ~1.0) — numerically identical to the HF trainer's norm-matched inject.
Requires `attention_backend=TRITON_ATTN` (FLASHINFER lacks the metadata vllm_lens needs → silent
no-op) and `language_model_only=True`. This is the +1.6x rollout speedup path noted above.
