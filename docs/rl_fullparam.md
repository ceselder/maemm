# Full-parameter RL for the MAEMM inverter (`rl_disagg --full-param`)

Design note for the `rl-fullparam` branch. Code: `rl/rl_fullparam.py` (helpers), `rl/rl_disagg.py` (the `--full-param`
branches), `rl/fast_lens_ext.py` (`wu_*` RPCs on the vLLM worker), `rl/modal_rl_disagg.py` (`--full-param` launcher flag),
`rl/test_rl_disagg_fullparam.py` (CPU tests). Measurements: `~/shared/reports/maemm-rl-fullparam/` (numbers in this note are
filled from that report; `[measured]` marks values that come from the 8xB200 runs).

## 1. What changes

| | LoRA path (default, byte-identical) | `--full-param` |
|---|---|---|
| policy | frozen bf16 base (+ optional full-FT `--policy-base`) + rsLoRA r64/α16, fp32 LoRA masters | the whole model: FSDP2 (`fully_shard` per decoder layer + root) with fp32 sharded masters, fp32 sharded grads, fp32 AdamW moments, bf16 compute (`MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`) — exactly `sft/fullft.py shard_full_model`, reused |
| init | `--init-adapter` | `--policy-base <full model dir with SAVE_DONE>` (or MODEL); no adapters allowed |
| reward scorer | the actor with its LoRA disabled (== MODEL) or a frozen copy of MODEL when the policy base differs | always a frozen bf16 copy of MODEL, **FSDP2-sharded** over the trainer ranks (`rl_fullparam.shard_frozen_bf16`), truncated to layers [0, READ_LAYER] with a `TinyHead` when nothing needs logits; read through `read_resid_noraise` (no `_Stop` exception inside an FSDP2 forward) installed as `rl_hf.read_resid` |
| KL reference (`--kl-coef > 0`) | the `ref` adapter / LoRA off | a frozen copy of the init: the scorer itself when `--policy-base` is MODEL (scorer kept full-depth), else a second sharded frozen copy of the policy base (+~54/Y GB per rank) |
| grad sync | manual weighted all-reduce (`_sync_grads`, weight = sync_w) | FSDP2 reduce-scatter averages over ranks; the local loss is pre-scaled by `sync_w · Y / Σ sync_w` (`rl_fullparam.loss_scale`) → identical weighted mean; grad clipping through `fullft.clip_grad_norm` (DTensor-aware) |
| injection hook | in-place add at the marker | `mode="add_clone"` (FSDP2 hangs backward hooks on the layer output) |
| prefix cache | one prefix fwd/bwd per step | same, plus the prefix logits as a zero-grad extra output so FSDP2's pre-backward unshard hooks fire for the last layer and the root (`_PrefixGradAccumulator(extra_outputs=...)`) |
| fp32 head | hook on the frozen head (`install_fp32_head`) | hook on the trainable head: `F.linear(x.float(), W_bf16.float())` inside the FSDP forward (`install_fp32_head_trainable`) |
| publish | PEFT adapter → `lora/step_k/` files, engines swap LoRARequest per block | bf16 weights into the engines in place (§2) |
| checkpoints | PEFT adapter dirs + `optim.pt` | full HF model dirs (`sft/fullft.py save_full_ckpt` layout, SAVE_DONE last) at `--save-steps`/`--save-every` + `final`; `--save-optim` adds `optim_dcp/` (torch.distributed.checkpoint, fp32 AdamW state); `--load-optim` restores it |
| eval | `eval_ckpt_daemon.py` (LoRA) | `eval/modal_eval_ckpt.py::fullmodel_daemon` (`--full-model`, one engine per checkpoint) |
| micro-batch probe | linear fit from mb 1,2 → predict at 85 % → verify, step down on OOM | measured ascending walk (`plan_probe_step`: never more than a doubling, linear extrapolation from the two largest measured points) against `--mb-target-frac` × GPU minus a reserve for the AdamW moments that appear at step 1 (2 × the fp32 master shard); peaks max-reduced over ranks (every backward is a collective); an OOM is fatal (an FSDP2 forward cannot be resumed after an exception) |

| step-time knobs | — | `--suffix-ckpt`: exact per-layer activation checkpointing of the suffix forward (`sft/prefix_cache.SuffixCheckpointer`, FSDP2-tested in `sft/fullft_smoke.py`; the prefix / no-grad forwards stay un-checkpointed). `--chunked-head`: `rl_fullparam.ChunkedHead` swaps `lm_head.forward` for a capture of the final hidden states and computes fp32 logits → log-softmax → gather/entropy per `--vocab-chunk` positions under `torch.utils.checkpoint`, so no micro-batch ever holds `[tokens × 248 320]` logits (grad-identical to the fp32-head hook + `_chunked_logp`: unit test). `--fsdp-prefetch 2`: explicit next-layer all-gather prefetch. Together they lift the micro-batch from 8 to 32–64 and cut the per-micro-batch FSDP traffic (2 × 54 GB all-gather + 108 GB fp32 reduce-scatter per micro-batch) 4–8× |

Flags: `--full-param`, `--publish-mode {nccl,fs}`, `--wu-port`, `--fs-keep-steps`, `--fsdp-prefetch`, `--no-scorer-shard`,
`--mb-target-frac`, `--suffix-ckpt`, `--chunked-head`, `--save-optim`, `--load-optim`. `--autocast-bf16` is ignored (FSDP2 already computes in bf16). Inline
eval (`--inline-eval-every > 0`) is refused. With the flag off no code path changes (all `fp is None` branches are the
previous code; `rl/test_rl_disagg_{queue,scalerl,policy_base}.py` still pass).

## 2. Weight publish protocol

Filesystem layout under `--work-dir` is the LoRA protocol's (`lora/latest`, `lora/step_<k>/meta.json`, `queue/blk_<adapter_step>_*`,
`--max-lag`), plus `lora/manifest.json`: the ordered list `[ckpt tensor name, global shape, rows per trainer rank]` of every
parameter (851 tensors, 53.8 GB bf16 for the 27B), built collectively from `model.named_parameters()` at start-up and asserted
against the `torch.chunk` geometry FSDP2 uses (`rl_fullparam.shard_rows`). Tensor names are the checkpoint layout
(`model.language_model.…`, `lm_head.weight`) — what vLLM's own `Qwen3_5ForConditionalGeneration.load_weights` consumes through its
`hf_to_vllm_mapper`, including the stacked/fused params (q/k/v → qkv_proj, gate/up → gate_up_proj, in_proj_qkv + in_proj_z →
in_proj_qkvz without LoRA).

### `--publish-mode nccl` (default)

One extra NCCL communicator: trainer rank 0 (group rank 0) + every vLLM worker process (group ranks 1..X), created with
vLLM's `StatelessProcessGroup` (a TCPStore on `--wu-port`) + `PyNcclCommunicator` — torch.distributed's default group of the
trainer is untouched; the vLLM workers keep their own. Per step, after `opt.step()`:

1. rank 0 writes `lora/pending` = k.
2. Each rollout rank finishes its in-flight block (a block = one `generate()` of 32 groups × 8 = 256 sequences, ~6 s), sees
   `pending` (top of its loop or inside the back-pressure wait), writes `lora/ready_<r>_<k>` and calls
   `collective_rpc("wu_init")` once (joins the group) then `collective_rpc("wu_recv", k)`.
3. rank 0 waits for all ready files, creates the communicator on the first publish, and broadcasts a "go" to trainer ranks 1..Y-1.
4. All trainer ranks: `for p in named_parameters(): full = p.to(bf16).full_tensor()` (all-gather of the bf16 cast of the fp32
   master shards, one tensor at a time, identical order everywhere); rank 0 `broadcast(full)` on the weight group.
5. Each worker's `wu_recv` receives the tensors in manifest order through a lazy generator fed straight into
   `model_runner.model.load_weights(...)` — vLLM's own loader copies into the existing parameter storage, so CUDA graphs
   (FULL_DECODE_ONLY), the fast steering hook (`fast_lens_ext`) and the KV cache stay valid; nothing is re-captured.
6. rank 0 writes `lora/step_k/meta.json` (`hnorm` = ‖h_marker‖ of the new policy from an FSDP forward, `checks` = |sum| of six
   unfused tensors) and flips `latest`; the rollout rank waits for `latest == k`, verifies its own |sum| of the same tensors
   (`rl_fullparam.verify_checks`, rtol 1e-3) and tags its next blocks `adapter_step = k`.

Nothing is transferred while an engine generates and nothing while FSDP collectives run (the trainer's main thread is idle
inside the publish), so no NCCL kernels of two communicators ever share a GPU concurrently. The price is the wait for the
slowest engine's block boundary (≤ one block). Every engine serves exactly step k afterwards (no mixed versions), and the
existing lag accounting (`policy/offpolicy_lag_steps`, `scalerl/lag_max`) is unchanged.

Timings `[measured]`: see `~/shared/reports/maemm-rl-fullparam/` (publish/t_wait_ready, publish/t_transfer, publish/gbps,
engine-side load_s, engine stall).

### `--publish-mode fs` (fallback)

Every trainer rank writes its own bf16 shards to `lora/step_k/shard_<t>.safetensors` (53.8 GB per step in total, RAM-backed
`/dev/shm` work dir needed: `--fs-keep-steps 2` → ~108 GB), rank 0 writes meta + `latest`, and each engine loads at its
next block boundary (`wu_load_fs`: `safe_open` the Y files, move each shard to the GPU, `torch.cat` along dim 0, stream into
`load_weights`). No cross-process NCCL, engines never wait for each other; the trainer's main thread pays the write, the
engines pay the load.

## 3. Memory (per trainer rank, 5 trainer ranks, B200 = 178 GiB usable)

Measured on the 8xB200 runs (`[T*]` log lines; policy = the 26.90 B-parameter FFT midtrain checkpoint):

| component | estimate | measured |
|---|---|---|
| bf16 policy loaded on the rank before sharding | 50 GB | 50.1 GB (transient; peak 51.1 GB while upcasting layer by layer) |
| fp32 sharded masters (26.9 B / 5) | 20.0 GiB | resident 24.8 GB right after `shard_full_model`, 20.0 GB once the load buffers are freed |
| frozen scorer, 43 layers bf16, FSDP2-sharded | ~7 GB | +4.2 GB (29.0 GB resident after loading it; 35 GB unsharded would be `--no-scorer-shard`) |
| root group gathered after a no-grad forward (embed + lm_head bf16) | 5 GB | 31.4 − 26.6 = 4.8 GB (freed by `reshard_root`) |
| fwd/bwd peak at max length (295 tokens), prefix-cached, fp32 head | | mb 4: 69.9 GB, mb 6: 80.4 GB, mb 8: 90.8 GB → **5.2 GB per sequence** on top of ~49 GB of mb-independent peak (fp32 sharded grads 20 GB + fp32 head copy 5 GB + all-gather/reduce-scatter buffers) |
| AdamW moments (allocated at the first `opt.step`) | 2 × 20 GB = 40 GB | reserved by the probe (the probe's own estimate was 55 GB in attempt 2 because the gathered root params were counted at full size — fixed by `reshard_root`) |
| **micro-batch chosen** | | **8** (85 % target: 90.8 + reserve ≤ 151 GB) |
| **peak during training** | | see the report (`mem/hf_peak_gb_max_rank`) |

Why the first probe failed: the LoRA path's linear fit from mb 1 and 2 predicted 0.06 GB/seq (both peaked at 66.6 GB — at tiny
micro-batches the peak is the mb-independent fp32-grad allocation during the prefix backward), verified mb 128 and OOM'd.
The full-param probe now walks the candidates upwards from mb 4, measuring each (`plan_probe_step`), and reserves the AdamW state.

### Why the first configuration ran at 68 s/step (and the fix)

At micro-batch 8 the update processes ~42 k completion tokens per rank per step in ~103 micro-batches of ~400 tokens. Every
micro-batch all-gathers the 54 GB of bf16 parameters for the forward, again for the backward (`reshard_after_forward=True`) and
reduce-scatters 108 GB of fp32 gradients — ~22 TB of NVLink traffic per rank per step, i.e. the whole 64 s of `time/fwd_bwd_s`
(compute for 400 tokens is negligible; `time/grad_sync_s` = 0 because FSDP2's reduce-scatter *is* the sync; the prefix is run
once per step — `trainer/body_tokens_per_rollout` ≈ suffix tokens only). The only lever is fewer micro-batches, i.e. a larger
micro-batch, which the 5.2 GB/seq activation footprint forbids without recompute: `--suffix-ckpt` (per-layer recompute; the
saved state drops to the layer inputs + the expanded prefix cache, ~0.3 GB/seq) and `--chunked-head` (the fp32 logits and their
saved log-softmax were ~0.4 GB/seq). Measured numbers: report §2 (bench table).

## 4. Parity / exactness (measured, 50-step validation run `rl_fullparam_val50_lr1e-6`, wandb 4mexafu3)

- Sampler vs trainer `policy/sampler_abs_dlogp`: 0.0215 at step 0 (engines on the on-disk weights) and 0.021–0.032 over steps
  1–49 with the engines on the pushed weights (lag 2), rising with the learning rate exactly like the LoRA arm's 0.021–0.028 band.
  A broken publish would sit at ~1–1.5 nats.
- 50 publishes × 3 engines × 6 tensor checksums (rtol 1e-3): 0 mismatches. `hnorm` published vs the engines' own marker norm:
  81.50 vs 81.5 at step 0 (injection check cos 1.0000, ratio 1.000).
- Publish cost (NCCL): median 0.61 s per step trainer-side = 0.15–0.35 s waiting for the engines' block boundaries + 0.25 s for the
  50.1 GB all-gather + broadcast (~200 GB/s) + 0.21 s marker-norm forward; engine stall 0.3–0.6 s; first publish 3.9 s including the
  3.1 s NCCL group creation. The LoRA path's adapter publish costs 1.9 s.
- Step time at micro-batch 8: 67–68 s (scoring 2–3 s, update 63–64 s, publish 0.6 s) vs 30 s for the LoRA arm — see §3 "why".
- Learning: reward 0.205 → 0.257 over 50 steps at lr 1e-6 (LoRA arm at 7e-6: 0.203 → 0.245); held-out `eval/mean_all` at the
  step-25 checkpoint 0.3679 vs 0.3549 for the LoRA arm on the same init (0.3628 for LoRA on base + SFT adapter); gradient norm
  0.69–0.96 (LoRA 0.19–0.23 — every weight contributes), entropy 2.71 → 2.19 (LoRA 2.70 → 2.50).
- Peak GPU memory per trainer rank: 111 GB at step 0, 151 GB from step 1 on (AdamW moments); 5 ranks, micro-batch 8.

## 5. Launch

```
DISAGG_APP=maemm-rl-disagg-fullparam DISAGG_GPU=B200:8 \
DISAGG_TRANSFORMERS="transformers @ git+https://github.com/ceselder/transformers@e52940e567ab9a991a1c971c1094e340233baff3" \
modal deploy rl/modal_rl_disagg.py
python3 scripts/launchers/spawn_rl_fullparam.py {smoke|val50|prod|bench5 <tag>} [lr] [--split 3+5] [-- extra rl_disagg flags]
```
(`train(..., full_param=True)` adds `--full-param --init-adapter none`; the ablation recipe lives in the launcher.) Production arm
(launched 2026-09-09 00:30Z): `prod 1e-6` → run `rl_abl_initnewfft_fullparam_8x512`, 300 steps, saves 25,50,…,300 to
`/data/ckpts_rl_abl_initnewfft_fullparam`, evaluator `maemm-eval-ckpt-fullrl/fullmodel_daemon` (eval cache v2, no judge extras);
ids in `~/shared/overnight/rl_ablation_ids.json["initnewfft_fullparam"]`. Learning rate 1e-6: reproduced the LoRA arm's 50-step
reward trajectory with stable gradient norms; the standard full-fine-tune RL value (5–7× below the LoRA lr, matching AdamW's
per-parameter step applied to every weight).

## 6. Lessons from the first attempts (all fixed on the branch)

1. **Micro-batch probe** — see §3: fit from mb 1/2 is meaningless under FSDP2; the optimizer state is not yet allocated during the probe.
2. **Uneven shards deadlock FSDP2** — 512 groups over 5 ranks = 103/103/102/102/102 groups → 103 vs 102 micro-batches; every
   FSDP2 forward/backward is a collective, so ranks 2–4 finished their update and entered the publish's marker-norm forward while
   ranks 0–1 were still in their last micro-batch → NCCL watchdog abort (rc −6) after 8 min. Now every rank runs
   `max_r ceil(n_r / mb)` micro-batches (zero-weight dummies of one short sequence), the same for the KL-reference pass and for
   the scorer's `score()` batches (`ceil(non-empty texts / score_batch)`, equalized with 2-token dummy forwards).
3. **Root params stay gathered after a no-grad forward** (FSDP2 keeps embed/norm/lm_head for a backward that never comes):
   `model.parameters()` then yields plain bf16 tensors — harmless for the NCCL publish (they are the full tensors) but wrong for
   the fs shard files, the checksums and the optimizer-state estimate. `reshard_root()` after every no-grad forward.

4. **FSDP2 reduce-scatters only the parameters that HAVE a gradient, as one flat collective per group.** The zero-weight dummy
   micro-batch (item 2) originally used `0 * logits.sum()`; with `--chunked-head` the module output is a dummy that bypasses
   `lm_head`, so on the ranks running a dummy `lm_head.weight.grad` stayed `None` and their root-group reduce-scatter was shorter
   than the other ranks' → all five NCCL watchdogs stuck 8 min into step 0 of the fast-config bench (its probe, where every rank
   runs identical shapes and no dummy, was fine). Reproduced on a 2-GPU trainer-only bench (`rl/fullparam_update_bench.py`:
   `base` ran, `head` hung); the dummy now goes through the same (chunked) head path as a real micro-batch, and a unit test asserts
   every parameter has a (zero) gradient after a dummy pass.

## 7. Known limits

- **Step time of the production arm**: 68 s/step at micro-batch 8 (validated configuration; launched before the fast-config fix).
  The fast configuration `--suffix-ckpt --chunked-head --fsdp-prefetch 2` (micro-batch 24 from the probe; 2-GPU harness: update
  peak 130 GB vs 165 GB at micro-batch 8, no hang after lesson 4) is the recommended configuration for the next runs; its
  8×B200 step time is in the report's bench table once `rl_fullparam_bench5_fast35b` has run. The production arm is not
  hot-swapped mid-run (no `--save-optim` → an AdamW restart would perturb the ablation).
- `--kl-coef > 0` with a non-MODEL policy base loads a second frozen sharded copy (+~11 GB/rank at Y=5); its head is bf16 (no
  fp32 head hook on the reference). Not exercised (the recipe has kl 0).
- Inline eval is not supported in full-param mode (use the full-model checkpoint daemon — exercised: step 25 + final of the validation run).
- The micro-batch probe never retries after an OOM under FSDP2 (fatal with a clear message; pass `--micro-batch`).
- `--save-optim` / `--load-optim` (torch.distributed.checkpoint, fp32 AdamW state ≈ 2× the model per checkpoint) are implemented
  but not exercised on the GPU runs.
- `publish-mode fs` (RAM-disk shard files) is implemented and CPU-tested, not exercised on the GPU runs (the NCCL path met the
  target immediately: 0.6 s per 50 GB publish).
- The per-step publish waits for every engine's block boundary (≤ 1 block ≈ 4.4 s here, measured 0.15–0.35 s); much longer
  blocks would call for a double-buffered asynchronous variant.
- Budget: development used ≈ 24 GPU-hours on 8×B200 (two failed smokes, the 50-step validation, two fast-config benches) plus
  ~1 GPU-hour of 2-GPU harness runs, against the ≈ 12 asked for; the production arm (≈ 6 h × 8 GPUs) is on top.
