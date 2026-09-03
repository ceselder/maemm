# MAEMMBench

The consolidated evaluation suite for the **MAEMM universal activation→text inverter**
(see the [repo README](../README.md) for the method). Everything that measures "how well does
the inverter turn a residual-stream direction back into text" lives here — the held-out
direction-family evals, the SAE-feature evals, the in-training evals, the async per-checkpoint
eval daemon, an LLM-judge autointerp eval, and the analysis suite — each documented below with
*what it measures, the input, the metric and how to read it, how to run it, and the numbers we
have observed*.

Model under test: **Qwen/Qwen3.6-27B** (D_MODEL 5120, read layer 42, inject layer 1) + a LoRA
inverter adapter. Published artifacts: best adapter
[`ceselder/qwen36-27b-maemm-inverter`](https://huggingface.co/ceselder/qwen36-27b-maemm-inverter),
SAE [`ceselder/qwen36-27b-sae-l42`](https://huggingface.co/ceselder/qwen36-27b-sae-l42)
(BatchTopK, F=131072, k=64, EV 0.6925). Live curves log to wandb project `uni-inverter`
(evals) / `maxact-fast` (training).

---

## Summary table

| # | Eval | Module | Input | Key metrics | Status |
|---|------|--------|-------|-------------|--------|
| 1 | Held-out family cosine | [`eval_universal.py`](eval_universal.py) (`run_eval`) | frozen held-out directions, n=512/family | `eval/<fam>/cos` per family, `eval/mean_all` (control-excluded) | **live** |
| 2 | SAE inversion | [`eval_universal.py`](eval_universal.py) (sae family) | held-out SAE encoder columns + corpus peaks | `sae/norm_act`, `fired`, `beat_corpus`, `rank1_frac`, `mean_rank`, `mrr`, `unverbalized_frac`, `unverbalized_p10` | **live** |
| 3 | In-training greedy eval | [`training_evals.py`](training_evals.py) → `rl/rl.py` `greedy_eval` | reserved bank directions (never trained on) | `eval/greedy_act_{mean,max}`, `greedy_beat_frac` | **live** (in RL loop) |
| 4 | In-training SAE eval (greedy + best-of-N) | [`training_evals.py`](training_evals.py) → `rl/rl.py` `sae_eval`/`sae_score` | held-out SAE feature split | `eval/sae_norm_act`, `sae_beat_frac`, `sae_bo{N}_*` | **live** (in RL loop) |
| 5 | Context-bucket families | [`build_ctx_eval.py`](build_ctx_eval.py) → eval 1 | held-out long-ctx acts, bucketed by token position | `eval/realact_{early,mid,long}/cos` | **live** |
| 6 | In-distribution held-out families | [`build_indist_eval.py`](build_indist_eval.py) → eval 1 | tail slice of the actual RL training pool | `eval/indist_{realact,probe,long}/cos` | **live** |
| 7 | Per-checkpoint eval daemon | [`modal_eval.py`](modal_eval.py) | every saved RL checkpoint | all of 1+2+5+6 → wandb, keyed by `ckpt_step` | **live** |
| 8 | Autointerp detection (Opus-5 judge) | [`../eval/autointerp_detection.py`](../eval/autointerp_detection.py) | rollouts as feature *descriptions* vs corpus spans | detection AUC per (arm, N-rollouts), diversity curve; marginal-value + hard-negative variants | **done** — first results in §8 |
| 9 | WorkspaceBench | (external: `camilablank/workspace-bench`) | frozen 700-item baseline banks, 7 tasks | concept-naming score vs J-lens, Opus-5 judge | **planned** (researched, see below) |
| 10 | Snippet-locality | [`snippet_locality.py`](snippet_locality.py) | autointerp testbed rollouts + real max-act examples | per-token profile locality (`peak_share`, `win3/5_share`, `gini`, `spread_half`), paired maemm-vs-real | **done** — see §10 |
| 11 | Datamix-ablation analysis | [`analysis/`](analysis/) | eval-1/2 outputs across training mixes | plots + JSON | **analysis** (done) |

---

## The shared protocol (applies to evals 1–7)

Every eval uses the **exact training inject recipe** and the **exact clean-base scoring
protocol**, so eval numbers are directly comparable to training reward:

- **Inject:** unit direction `v`, norm-matched (`h ← h + ‖h‖·v`) at `INJECT_LAYER` on the
  trailing `' ?'` marker token; sample `bo` rollouts (best-of-N) or greedy-decode.
- **Score:** re-tokenize each rollout STANDALONE, forward through the **clean base**
  (`actor.disable_adapter()` — the policy cannot inflate its own score), read residuals at
  `READ_LAYER=42`, prepend a BOS sink and skip position 0 (attention-sink guard), drop tokens
  with norm > 10× the batch median (norm-filter), then take the **max over remaining tokens** of
  `cos(h_t, v)` (or the SAE feature activation, for SAE metrics).
- **Frozen eval sets:** every direction family is sampled ONCE with a fixed seed and cached
  (`eval_sets_heldout.pt`); generation RNG is forked and fixed (`GEN_SEED`) — so per-checkpoint
  evals are deterministic, low-variance, and comparable across the whole run.

**Anti-gaming design** (each guard exists because the failure it catches actually happened):
the **random control family** (§1) catches scorer-level gaming; the norm filter + sink skip
catch high-norm outlier tokens; the adapter-disabled re-encode stops self-scoring; the
"all rollouts identical" assert in the greedy evals catches steering silently not firing
(the FLASHINFER / missing-dist-info failure mode); and `mean_all` excludes the control so a
degrading control can never *improve* the headline number.

---

## 1. Held-out family-cosine eval

**What it measures.** Can the inverter invert directions it has *never trained on*, across
qualitatively different direction families? This is the generalization headline.

**Input.** Frozen held-out direction sets, n=512 per family (fp32 unit rows in the cache):

| family | directions | held-out how |
|---|---|---|
| `realact` | real L42 token activations `unit(h−μ)` | train-disjoint pool split |
| `bsf` | SASA/BSF subspace projections | train-disjoint pool split |
| `cluster` | cluster-probe directions | train-disjoint pool split |
| `jlens` | J-lens pullbacks `unit(W_U[t]·J_42)` | train-disjoint pool split |
| **`random`** | **isotropic Gaussian unit dirs — off-manifold CONTROL** | in no pool, uninvertible by construction |
| `realact_early/mid/long` | long-ctx acts bucketed by position (§5) | complement of the RL bank sample |
| `indist_realact/probe/long` | tail rows of the actual RL pool (§6) | reserved tail slice |

**Metric.** `eval/<family>/cos` = mean over directions of best-of-`bo` max-over-token cosine
(protocol above). Plus **`eval/mean_all`** = unweighted mean of the *higher-is-better* families
— **`random` is EXCLUDED** (it is a lower-is-better control; folding it in dragged the mean down
and would move `mean_all` the *wrong* way if the control ever degraded). Range 0–1; the random
floor tells you what "no signal" looks like.

**How to read it.**
- `random/cos` must stay ≈ **0.03** (theoretical isotropic floor 1/√5120 ≈ 0.014; the max-over-
  tokens/best-of-N takes it to ~0.03). **If it rises, the metric is being gamed — distrust every
  family.**
- On-manifold families land far above the floor when inversion works: `realact` ~0.42–0.52,
  `bsf` ~0.32–0.34, `cluster` ~0.25–0.27, `jlens` ~0.10–0.12 (J-lens is the hardest family).

**How to run.**
```bash
PYTHONPATH=$PWD python eval/eval_universal.py \
  --adapter ckpts/rl/final --sae-path ae.pt --maxacts-path max_acts.pt \
  --heldout-pool data/pool_heldout --n 512 --bo 4 \
  --out eval_universal.json --wandb uni-inverter
# or from code: build_eval_sets(...) once, then run_eval(...) per checkpoint
```

**Observed numbers.** `mean_all` ≈ **0.36** at RL step 25 rising to a **~0.41–0.42** plateau on
the 8×B200 paper runs; `realact/cos` 0.52; `random/cos` 0.028–0.034 across every healthy run.
A *flat* `mean_all` with drifting training reward was the reward-hack prelude in the v2 paper
run — treat flat-while-reward-moves as an alarm, not a plateau.

---

## 2. SAE eval (held-out feature inversion)

**What it measures.** Given a held-out SAE feature's encoder direction, can the inverter
generate text that actually *fires that feature* — and how strongly vs the best the pretraining
corpus ever managed? This is the most interpretable absolute measure ("did we verbalize the
feature?").

**Input.** Held-out SAE features (never in any training bank): unit encoder columns
`unit(W_enc[:,f])` as inject directions, paired with each feature's **corpus peak** activation
(max over the max-acts scan of the pretraining corpus).

**Metrics** (all from the best of `bo` rollouts per feature, scored on the TRUE feature
activation at the rollout's peak token):
- **`sae/norm_act`** = generated activation ÷ corpus peak (mean over features). 1.0 = our 64-token
  rollout drives the feature as hard as the single best span in the whole scanned corpus.
- `sae/fired` = fraction with raw act > 1.0; **`sae/beat_corpus`** = fraction beating their corpus peak.
- **`sae/rank1_frac`** / `mean_rank` / `mrr` = full-SAE rank of the target feature among all 131k
  features at the rollout's peak token (rank 1 = the rollout is *most* about the target feature).
  (Dropped from the fast daemon path for speed; available in the standalone eval.)
- **`sae/unverbalized_frac`** = fraction of held-out features NO rollout can fire (act ≤ 1.0) —
  the "can't-verbalize-it-at-all" mass; `unverbalized_p10` = fraction reaching <10% of corpus peak.

**How to read it.** `norm_act` ≥ ~0.6 means rollouts typically reach a healthy fraction of the
corpus-best activation on features never seen in training. `unverbalized_frac` falling under RL
is the "RL rescues features SFT can't verbalize" effect.

**How to run.** Same entry point as §1 (the sae family runs automatically inside
`eval_universal.py run_eval`).

**Observed numbers.**
- Best-ever checkpoint: **`norm_act` 0.875, `rank1_frac` 0.34** (`uni_rl` step_225 — the
  HF-published adapter; its bank included SAE-basis directions). Run ranking by `norm_act`:
  uni_rl **0.875** ≫ longhz 0.634 > divshort 0.569 ≈ dp4 0.567 > divlong 0.545.
- Paper run (realact+probes+long mix, no SAE family in training): 0.635 @ step 25, peaking
  **0.776** @ steps 200–250 — then eroding to 0.65 as the run reward-hacked (the SAE metric
  degraded *before* the fluency gate broke: it is the early-warning canary).
- `unverbalized_frac`: ~**0.40 after SFT → ~0.33 with RL**.

---

## 3. In-training greedy eval

**What it measures.** A cheap held-out signal *inside the RL loop*: is the policy improving on
directions it never trains on, without waiting for the full daemon pass?

**Input.** The first `--n-eval-dirs` UNIQUE directions of the training bank are **reserved**
(never sampled for training); when the pool's `records.jsonl` carries a `corpus_max_proj`
baseline, each direction also gets the best candidate-span projection from the corpus.

**Metric.** One greedy (T=0) rollout per direction through the same vLLM+steering path, scored
by the training scorer (clean base, max-over-kept-positions `x_t · unit(v)` — raw projection
units, not cosine): `eval/greedy_act_mean`, `eval/greedy_act_max`, and `eval/greedy_beat_frac`
(fraction of directions whose rollout beats the corpus baseline projection).

**How to read it.** The raw-projection units depend on residual norms — read the *trend* and
`beat_frac` (absolute, 0–1) rather than the raw mean. Includes the identical-texts assert:
identical rollouts across distinct directions = steering is not firing → crash loudly.

**How to run.** Flags on `rl/rl.py`: `--eval-every 25 --n-eval-dirs 64` (defaults on).
Import for reuse: `from MAEMMBench.training_evals import greedy_eval`.

---

## 4. In-training SAE eval (greedy + best-of-N)

**What it measures.** §2's headline metrics, live in the RL loop, on a fixed held-out feature
split — including the greedy-vs-sampling gap.

**Input.** `--sae-split split.json` (a `{"eval": [feature ids]}` held-out split),
`--n-sae-eval-feats` of them; each feature's dataset-max activation as the baseline.

**Metrics.** Greedy arm: `eval/sae_act_mean`, **`eval/sae_norm_act`** (canonical greedy
cross-uplift number), `eval/sae_beat_frac`. Best-of-N arm (`--sae-eval-bo N --sae-eval-temp t`):
`eval/sae_bo{N}_{act_mean,norm_act,beat_frac}` — sampling is far more productive than greedy
(the Bo16 curve sits well above greedy), so quote which arm a number comes from.

**How to run.** `rl/rl.py --sae-eval-every 50 --sae-split split.json --sae-eval-bo 16`.
Import for reuse: `from MAEMMBench.training_evals import sae_eval, sae_score`.

**Observed numbers.** On the 250k ProbeMaxxer precursor, held-out SAE features fired (greedy):
SFT 54% → RL 58%. Greedy `norm_act` runs well below the best-of-N numbers in §2 — greedy is a
floor, not the headline.

---

## 5. Context-bucket eval (`realact_early` / `realact_mid` / `realact_long`)

**What it measures.** Does inversion quality depend on the *context position* the target
activation came from? Long-context activations (position 2k–8k) carry information about far-back
context that a 64-token rollout must somehow evoke — hypothesized (and confirmed) harder.

**Input.** [`build_ctx_eval.py`](build_ctx_eval.py) augments the frozen cache once: it
reproduces the RL bank builder's seed-0 split of the long-context activation dump, takes the
**train-disjoint complement**, buckets by token position (early ≤512 / mid 512–2048 / long
2048–8192), and mints 512 `unit(h−μ_long)` directions per bucket (same centering as training).

**Metric.** `eval/realact_{early,mid,long}/cos` — same cosine protocol as §1; the buckets also
enter `mean_all`.

**How to read it.** Compare the three curves: a widening early-vs-long gap means the model
inverts local content but misses long-range context; long-context RL data was added to the mix
specifically to close it.

**How to run** (one-shot, before starting the daemon):
```bash
MAEMM_ACTS_LONG=data/acts_long MAEMM_EVAL_CACHE=data/eval_universal_ho/eval_sets_heldout.pt \
  PYTHONPATH=$PWD python eval/build_ctx_eval.py
```

**Observed numbers.** early **0.556** / mid **0.502** / long **0.439** (paper run, step 25) —
monotone in context depth, long-ctx confirmed harder; longer rollouts (96 tokens) helped the
ctx buckets while slightly hurting the SAE peak.

---

## 6. In-distribution held-out eval (`indist_realact` / `indist_probe` / `indist_long`)

**What it measures.** The §1 families come from the *old universal bank* — not the actual RL
training pool. These families measure held-out inversion **on the training distribution
itself** (`pool_rl_mix`), separating "generalizes off-distribution" from "works on its own
distribution".

**Input.** [`build_indist_eval.py`](build_indist_eval.py) reserves the LAST 2000 rows of each
family block of `pool_rl_mix` (realact / cluster / realact_long) and seed-samples 512 of each
into the cache. **Caveat (stated in the script):** the trainer samples uniformly over all 750k
rows, so the tail is not *strictly* train-disjoint — each tail row has ~1.7% probability of
ever being drawn in a 400-step run (expected ~34 of 2000 per family). Negligible but nonzero.

**Metric.** `eval/indist_{realact,probe,long}/cos` — same protocol as §1; enters `mean_all`.

**How to read it.** `indist_*` should sit at or above the corresponding off-distribution family;
if `indist_*` climbs while the §1 families stall, the model is fitting its pool rather than
learning general inversion.

**How to run.**
```bash
MAEMM_POOL=data/pool_rl_mix MAEMM_EVAL_CACHE=data/eval_universal_ho/eval_sets_heldout.pt \
  PYTHONPATH=$PWD python eval/build_indist_eval.py     # idempotent
```

**Observed numbers.** `indist_long/cos` **0.507** @ step 25 (vs off-dist `realact_long` 0.439 —
in-distribution long-ctx is easier, as expected).

---

## 7. Per-checkpoint eval daemon (`modal_eval.py`)

**What it measures.** Nothing new — it *runs* evals 1+2+5+6 asynchronously on **every saved RL
checkpoint** and logs to wandb, so training never blocks on eval.

**How it works.** One B200 on Modal, mounting the same `maemm-data` volume as training. Loop:
`vol.reload()` → pick the **highest un-evaled** `step_*` checkpoint (latest-first; naturally
backfills the backlog newest-first once caught up) → load base(+adapter) → fast eval (all cos
families + the rank-free SAE family; the 131k-rank metric is dropped for speed, ~12–15 min/pass)
→ `wandb.log(..., commit=True)` with **`ckpt_step` as the x-axis metric** (`define_metric`), so
out-of-order backfill points land correctly. Evaled-checkpoint state persists on the volume;
a `min_ckpt_mtime` cutoff ignores stale checkpoint dirs from cancelled/rolled-back legs. This
replaces the older on-box `eval_daemon_universal.py` (same pattern, GPU-adjacent to training).

**`eval/mean_all`** is computed here exactly as in §1: mean of the higher-is-better cos
families, `random` (the control) excluded.

**How to run.**
```bash
MODAL_PROFILE=<profile> modal deploy eval/modal_eval.py    # repo-root modal_eval.py shim also works
MODAL_PROFILE=<profile> python -c "import modal; modal.Function.from_name('maemm-eval-heldout','daemon').spawn()"
```
Deploy + spawn, **not** `modal run --detach` — killing an ephemeral app's client cancels the app.

---

## 8. Autointerp-detection eval (Opus-5 judge) — NEW

**Code:** [`../eval/autointerp_detection.py`](../eval/autointerp_detection.py) (canonical —
referenced here, not duplicated) + the GPU-stage launcher
[`../modal_autointerp_detection.py`](../modal_autointerp_detection.py).

**What it measures.** Are MAEMM rollouts a good *explanation* of an SAE feature — good enough
that a third party can predict where the feature fires? Standard autointerp **detection**
scoring (Bills et al. / EleutherAI), with the feature description built from inverter rollouts
instead of a natural-language summary. This is the eval that connects MAEMM to the
interpretability literature.

**Input.** Held-out SAE features. Per feature: 1 greedy + 8 temperature rollouts (exact train
inject recipe); a test set of 10 POSITIVE corpus windows (real max-act windows, ranked after the
ones reserved for the baseline description, peak act > 1.0) + 10 NEGATIVE windows (random
windows from a *disjoint* corpus slice, **verified** near-zero for the feature via clean-base
SAE encode). Ground truth comes from the SAE itself.

**Metric.** The judge (`claude-opus-5`, Anthropic Batches API) sees N description snippets + one
test snippet and rates 0–100 how likely the feature fires on it. Score = **detection AUC**
(Mann-Whitney, ties 0.5) per feature, averaged; balanced accuracy @50 secondary. Two arms ×
diversity curve:
- **`maemm` arm**, N ∈ {1,2,4,8} rollouts (N=1 greedy, N>1 nested temperature samples) — a
  *rising* AUC-vs-N curve is direct evidence the rollout diversity carries real autointerp value;
- **`examples` baseline arm**: the feature's top-N max-act corpus examples as the description
  (matched N, disjoint from test positives by construction). MAEMM ≥ baseline means the
  *generated* descriptions match the informativeness of the corpus's own best evidence.

**How to read it.** AUC 0.5 = chance, 1.0 = perfect. Compare `maemm` vs `examples` at each N,
and the slope over N.

**How to run** (three stages; judge/score need only `ANTHROPIC_API_KEY_BATCH`):
```bash
MODAL_PROFILE=<profile> modal run modal_autointerp_detection.py         # GPU stage -> testbed.json
python eval/autointerp_detection.py judge --testbed testbed.json --state batch_state.json
python eval/autointerp_detection.py score --state batch_state.json --out results.json
```

**Observed numbers** (2026-08-29, 64 held-out features, best adapter; full report:
`reports/view/autointerp-detection-eval`): maemm AUC 0.870→0.894 (N=1→8; paired diversity gain
+0.024 ±0.010), examples baseline 0.886→0.957. Failure tail = inversion failure (r=0.77 between
rollout fire-rate and AUC; AUC 0.976 on the 45/64 features where ≥half the rollouts fire).
Two follow-up variants (same testbed): **marginal** — appended to a fixed top-4-real-example
description, rollouts add ~nothing vs random negatives (+0.003 ±0.006; 40/64 features at
ceiling); **hardneg** — against mined embedding-NN negatives (topically close, verified
non-firing) the marginal lift emerges at +0.022 ±0.009, and both description types degrade
similarly (random→embnn: −0.059 rollouts / −0.062 examples), i.e. the judge isn't just
topic-matching.

---

## 9. WorkspaceBench — planned (researched)

External suite: `camilablank/workspace-bench` (MATS-cohort, built on **Qwen3.6-27B — our exact
model**), from the "Verbalizable Representations Form a Global Workspace" line of work. It
benchmarks an **"oracle lens"** — a free-text readout of an activation — against the J-lens,
scored on whether the readout *names the latent concept* (strict Opus-5 judge + a deterministic
matcher, vs permutation-chance baselines). An oracle lens is literally what the MAEMM inverter
is, so this is a purpose-built external benchmark for it.

**Tier-1 plan (ready to build):** the **7 frozen baseline banks** (~700 items: readout,
association, multihop, multilingual, poetry, typo, directed-modulation). Capture L42 activations
once (~10 min on 1×B200), generate readouts per log-spaced checkpoint {12, 25, 50, 100, 200,
400, 800}, score with the deterministic matcher (free) + an Opus-5 batch (~4200 calls, ~$12–15);
~1.5–2 h GPU per full sweep.

**Eval hygiene (non-negotiable):** the baseline banks are *never tuned against* — no
hyperparameter selection on them; select on the hillclimbing families and reserve the banks for
final headline numbers. Status: awaiting the hygiene sign-off + coordination with the suite's
author for apples-to-apples numbers.

---

## 10. Snippet-locality eval

**Code:** [`snippet_locality.py`](snippet_locality.py) + the GPU-stage launcher
[`../modal_snippet_locality.py`](../modal_snippet_locality.py).

**What it measures.** Within a MAEMM rollout, does the target SAE feature fire on a LOCALIZED
short snippet, or is the activation smeared across the whole text — and are rollouts as
localized as the feature's own max-activating corpus examples? A good max-act example fires on
a crisp interpretable snippet; this eval checks the inverter reproduces that property, i.e.
that the peak-token scoring used across MAEMMBench reflects genuine localized evocation.

**Input.** The autointerp testbed (`testbed_v2.json`): per held-out feature, the 9 MAEMM
rollouts (greedy + 8 temp) and the 8 real top max-act corpus examples. Per text: clean-base
L42 per-token ReLU activation profile of the target feature under the exact shared read path
(BOS-sink skip, right-pad mask, 10×-median norm filter).

**Metrics** (per text; aggregated over FIRING texts only, peak > fire threshold, with firing
fraction per arm reported): `peak_share` (peak / total positive mass), `win3_share` /
`win5_share` (best contiguous 3/5-token window's mass fraction — the most length-robust,
read these first), `gini` (profile concentration), `spread_half` (#tokens ≥ 50% of peak).
Headline = **paired per-feature maemm-vs-real diff with 95% CI**, plus cross-links of
per-feature rollout locality to the §8 autointerp AUC and rollout fire-rate. Length caveat:
rollouts (≤64 new tokens) are ~2× longer than the 32-token max-act windows; `peak_share` and
`gini` are length-sensitive, the fixed-k window shares much less so.

**How to run:**
```bash
MODAL_PROFILE=<profile> modal run modal_snippet_locality.py      # GPU -> locality.json
python eval/snippet_locality.py score --locality locality.json \
    --autointerp-results results.json --testbed testbed_v2.json --out locality_results.json
```

**Observed numbers** (2026-08-29, 64 held-out features, best adapter; full report:
`reports/view/snippet-locality`): rollouts are MUCH less localized than real examples on every
metric — win5 mass 0.31 vs 0.69 (paired −0.38 ±0.03; crop-32 length control −0.31 ±0.02),
peak_share 0.16 vs 0.38, spread 11.7 vs 3.3 tokens — while firing at *higher* peaks (19.3 vs
15.9 mean peak act). And the hypothesis INVERTS: more-localized rollouts predict *worse*
inversion and detection (r(win5, fire-rate) = −0.90, r(win5, autointerp AUC) = −0.77) — a
smeared profile means the whole rollout is on-concept; a localized rollout profile is a
symptom of marginal firing, not interpretability.

---

## 11. Datamix-ablation analysis suite

[`analysis/`](analysis/) — the plotting scripts for the training-data-mix ablation (which
direction families should the SFT/RL banks contain?). These consume eval-1/2 outputs across
runs; the JSON data files live in the (private) research workspace, so the scripts are kept
here as provenance for the published plots. Findings (mean held-out over 5 families, n=512
each, example-matched mixes of 179,919 spans, RL step 100):

| training mix | mean held-out (final) | SAE norm_act |
|---|---|---|
| 50/50 realact+probes | **0.391** | — |
| 33/33/33 BSF+probe+realact | 0.388 | **0.750** |
| realact only | 0.369 | 0.676 |
| probe only | 0.359 | 0.679 |
| BSF only | 0.352 | 0.652 |

- **Mixing beats any single source, and the lead grows with RL** (`plot_mix_and_sftrl.py`,
  `plot_mix_meanall.py`, `plot_mix_perfamily.py`); the balanced mix dominates on held-out SAE.
- **SFT alone contributes little; RL adds a large ~constant lift at every SFT level**
  (SFT-only means ~0.16–0.29 → +~0.10–0.15 after 50 RL steps).
- **Probe-only training generalizes worse than the universal mix at plateau**
  (`plot_probemix_vs_bsf.py`: SAE norm_act 0.658 vs 0.856, realact 0.413 vs 0.496 —
  confounded by LR, treated as suggestive; the clean test is the mix ablation above).

---

## Layout & compatibility

```
eval/
  README.md               this file
  eval_universal.py       evals 1+2 (canonical; moved from eval/eval_universal.py)
  training_evals.py       evals 3+4 (re-exports greedy_eval/sae_eval/sae_score from rl/rl.py)
  build_ctx_eval.py       eval 5 cache builder (context buckets)
  build_indist_eval.py    eval 6 cache builder (in-distribution held-out)
  modal_eval.py           eval 7, the per-checkpoint daemon (moved from repo root)
  analysis/               eval 11, the datamix-ablation plot suite
eval/eval_universal.py    backwards-compat shim -> MAEMMBench.eval_universal
eval/autointerp_detection.py   eval 8 (canonical, lives in eval/)
modal_eval.py             backwards-compat shim -> MAEMMBench.modal_eval (modal deploy still works)
```

Old import paths and CLI invocations keep working via the shims (repo root on `PYTHONPATH`,
as the repo already requires for `mxf`). The frozen eval cache (`eval_sets_heldout.pt`) is
mode/seed/SAE-validated on load — a cache built for a different pool, n, seed, or SAE errors
out instead of silently producing incomparable curves.
