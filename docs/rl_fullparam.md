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

Flags: `--full-param`, `--publish-mode {nccl,fs}`, `--wu-port`, `--fs-keep-steps`, `--fsdp-prefetch`, `--no-scorer-shard`,
`--mb-target-frac`, `--save-optim`, `--load-optim`. `--autocast-bf16` is ignored (FSDP2 already computes in bf16). Inline
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

## 4. Parity / exactness

- Sampler vs trainer `policy/sampler_abs_dlogp` at step 0 (engines hold the on-disk weights) and at every later step (engines
  hold the published weights): `[measured]` — the LoRA path sits at 0.02–0.04; a broken publish shows up as ~1–1.5 nats.
- Publish checksums (six tensors, rtol 1e-3) verified by every engine at every step; a mismatch aborts the run.
- The loss-scaling equivalence with `_sync_grads` and the manifest/shard geometry are unit-tested (`rl/test_rl_disagg_fullparam.py`).

## 5. Launch

```
DISAGG_APP=maemm-rl-disagg-fullparam DISAGG_GPU=B200:8 \
DISAGG_TRANSFORMERS="transformers @ git+https://github.com/ceselder/transformers@e52940e567ab9a991a1c971c1094e340233baff3" \
modal deploy rl/modal_rl_disagg.py
python3 scripts/launchers/spawn_rl_fullparam.py {smoke|val50|prod} [lr] [-- extra rl_disagg flags]
```
(`train(..., full_param=True)` adds `--full-param --init-adapter none`; the ablation recipe lives in the launcher.)

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

## 7. Known limits

- `--kl-coef > 0` with a non-MODEL policy base loads a second frozen sharded copy (+~11 GB/rank at Y=5); its head is bf16 (no
  fp32 head hook on the reference).
- Inline eval is not supported in full-param mode (use the full-model checkpoint daemon).
- The micro-batch probe never retries after an OOM under FSDP2 (fatal with a clear message; pass `--micro-batch`).
- `--save-optim` writes ~2× the model per checkpoint (fp32 AdamW) and is untested end-to-end on the GPU runs.
- `publish-mode fs` needs ≥ 2 × 54 GB of RAM-backed work dir.
