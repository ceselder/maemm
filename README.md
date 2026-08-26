# MAEMM — training a universal activation→text inverter

**MAEMM** (Max-Activating-Example Meta-Model) trains a small LoRA adapter that turns an *activation
direction* inside a language model back into **text** — text whose own activation, on a clean forward
pass, points the same way as the injected direction. In other words: given a direction in a model's
residual stream (an SAE feature, a probe direction, a raw activation, …), it generates a short span
that maximally *evokes* that direction. It's a learned, general-purpose "inverse" of the readout lens.

This repo is the training pipeline for that inverter on **Qwen/Qwen3.6-27B** (read layer 42, inject
layer 1), but the method is model-agnostic — change `mxf/config.py`.

## Method

1. **Inject** a unit direction `v` at `INJECT_LAYER` on a marker token (norm-matched: `h ← h + ‖h‖·v`).
2. **Generate** a short continuation.
3. **Score** it by re-reading the *clean* (adapter-off) activation at `READ_LAYER` and taking the
   max-over-token cosine with `v`. Good inversions produce text that genuinely drives the direction.

Training is two stages:

- **SFT (match-activation):** for each direction, the target text is the *real* corpus span whose
  activation produced it. This matches the objective by construction.
- **RL (Dr. GRPO):** reward = max-over-rollout cosine (× a scale), optimizing the inverter to *actively*
  evoke each direction. RL does most of the heavy lifting; SFT gives it a running start.

**Direction families** (the "data mix" — this is what matters most):

| family | what it is |
|---|---|
| `realact` | raw real activations `unit(h − μ)` at sampled corpus positions |
| `probe` / `cluster` | linear probe / cluster-centroid directions |
| `bsf` | block-sparse subspace-featurizer (SASA) projections |
| `sae` | SAE encoder columns |
| `jlens` | J-lens (pullback) vectors |

Empirically, a **balanced mix of real-activations + probes** generalizes best across held-out families;
adding **long-context activations to the RL mix** (contexts far longer than SFT ever sees) is a cheap way
to cover features that only fire at long range. See `bank/` for every family's builder.

## Layout

```
mxf/                     core library (config, injection hooks, prompts, SAE loader, MFU)
collect/collect_acts.py            collect READ_LAYER residuals from a corpus (fixed-len windows)
collect/collect_acts_longctx.py    long-context activations (uniform position up to 8k) — RL-only source
bank/train_sasa.py                 train the SASA/BSF subspace featurizer
bank/build_universal_bank.py       mine all families -> one (direction, target-span) bank
bank/build_big_sft_bank.py         scaled realact+probes SFT bank (CPU)
bank/build_rl_bank.py              balanced RL mix (realact + probes + long-context)
train/pretrain.py                  match-activation SFT (LoRA; single- or multi-GPU via torchrun)
train/rl.py                        Dr. GRPO RL from the SFT init
eval/eval_universal.py             held-out eval: per-family cosine, SAE norm_act / rank-1 / %unverbalized,
                                   + a random-direction control
```

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD            # so `import mxf...` resolves
export HF_TOKEN=...               # for model + corpus downloads
# optional: export WANDB_API_KEY=... ANTHROPIC_API_KEY=...   (logging / LLM-judge evals)

# 1) collect activations (needs a GPU with the model)
python collect/collect_acts.py --n-seq 20000 --seq-len 512 --out-dir data/acts

# 2) build banks
python bank/build_big_sft_bank.py --n-realact 500000 --n-probe 500000 --out data/pool_sft
python collect/collect_acts_longctx.py --shard 0 --n-shards 5   # (one per GPU) -> data/acts_long
python bank/build_rl_bank.py --n-each 250000 --out data/pool_rl_mix

# 3) SFT  (multi-GPU: use gloo — see note)
DDP_BACKEND=gloo TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD \
  torchrun --standalone --nproc_per_node=5 train/pretrain.py \
    --data-dir data/pool_sft --lr 3e-5 --batch-size 16 --epochs 2 --save-dir ckpts/sft

# 4) RL from the SFT init
python train/rl.py --data-dir data/pool_rl_mix --bank-file vecs.f32 \
  --init-adapter ckpts/sft/final --lr 1e-5 --reward-metric cosine --reward-scale 1000 \
  --max-new-tokens 32 --kl-coef 0.03 --total-steps 100 --save-dir ckpts/rl

# 5) eval (held-out families + %unverbalized SAE)
python eval/eval_universal.py --adapter ckpts/rl/final \
  --sae-path <ae.pt> --maxacts-path <max_acts.pt> --heldout-pool data/pool_heldout
```

> **Multi-GPU note:** use **`DDP_BACKEND=gloo`**, not NCCL. Only the (tiny) LoRA gradients are
> all-reduced, so gloo's CPU-socket comms are plenty — and NCCL can deadlock at DDP init on some
> single-node multi-GPU boxes. Also set `TOKENIZERS_PARALLELISM=false` (forked-rank tokenizer thrash).

## Config

`mxf/config.py` — `MODEL`, `D_MODEL`, `READ_LAYER`, `INJECT_LAYER`, `STEER_COEFF`, corpus, LoRA/RL hparams.
Defaults: rsLoRA r64/α16 all-linear, AdamW lr 3e-5 (SFT) / 1e-5 (RL).

## Notes

Research code — no warranty, APIs may change. No data, weights, checkpoints, or credentials are included;
supply your own model/corpus/keys via env vars.
