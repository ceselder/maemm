"""Universal held-out eval suite for the Qwen3.6-27B direction->text inverter (wandb-facing).

Evaluates SIX frozen held-out direction families with the EXACT SFT/RL inject recipe (unit dir
injected norm-matched at INJECT_LAYER on the trailing ' ?' marker, best-of-bo sampling) and the
EXACT eval_dirs scoring protocol (clean base via actor.disable_adapter(), re-encode @READ_LAYER,
sink-prepended + 10x-median norm-filtered, max over content tokens):

  1. probe    held-out rows of the RL probe bank (probes.f32)      -> eval/probe/cos
  2. sae      held-out SAE features (unit encoder columns)         -> eval/sae/{norm_act,fired,
              beat_corpus,mean_rank,rank1_frac,mrr}
  3. jlens    J-lens steering vectors unit(W_U[t] @ J_42) for
              random vocab tokens t (workspace-paper direction)    -> eval/jlens/cos
  4. cluster  see NOTE below — probe-bank rows DISJOINT from (1)   -> eval/cluster/cos
  5. random   isotropic Gaussian unit dirs — OFF-manifold CONTROL  -> eval/random/cos
  6. realact  real L42 token activations from a HELD-OUT slice
              (last 5% of sequences) of the bsf27b acts dump       -> eval/realact/cos

  CONTROL: eval/random/cos must stay ~= 1/sqrt(D_MODEL) = 1/sqrt(5120) ~= 0.014. Random dirs are
  off-manifold and uninvertible; if this number rises, the metric is being gamed (e.g. the scorer
  is picking up a shared high-norm direction, or the norm filter broke) — distrust every family.

  NOTE on the cluster family: the clustering stage's centroids.npy is in ZCA-whitened +
  L2-normalized space (meta cluster_space="zca_whiten_l2norm"), NOT raw residual space, so it
  CANNOT be injected as-is (same caveat as SL/build_dom_probes.py). There is no raw-space
  cluster-centroid direction file under /root/pmx/data, so the cluster family reuses the RL
  probe bank (cluster-probe directions ARE the per-cluster raw-space directions) with a sample
  DISJOINT from family (1) — i.e. it is a second independent probe sample, kept as a separate
  curve so the wandb panel layout survives if a true centroid file lands later.

HELD-OUT MODE (--heldout-pool, PREFERRED for the universal inverter): the families above are
FRESH-SAMPLED from the same sources the training bank draws from, so they OVERLAP pool_train
(the sae family is ~100% train-contaminated). Passing --heldout-pool <dir> (e.g.
data/pool_universal/pool_heldout, built train-disjoint per family by SL/build_universal_bank.py)
instead evals the pool's own families {bsf, realact, sae, jlens, cluster} — directions are the
pool's vecs.f32 rows, sae feature ids come from the records — plus the synthetic random control.
Cosine families become [bsf, realact, jlens, cluster, random] (probe exists only in the legacy
fresh-sample mode) and the cache lives at <cache>/eval_sets_heldout.pt so the two modes never
share a cache. Scoring/metrics are IDENTICAL in both modes.

Every family is sampled ONCE with a fixed seed and cached to <cache>/eval_sets.pt; all later
evals reuse the identical directions (and identical generation RNG via a forked, fixed seed), so
the wandb curves are low-variance and checkpoint-comparable. run_eval() is a clean function with
no wandb calls — import it from the trainer and log the returned flat dict:

    from eval_universal import build_eval_sets, run_eval
    es = build_eval_sets(cache_path, sae, wu, j42, probe_bank, acts_dir, n, dev)   # cached
    wandb.log(run_eval(actor, tok, prompt_ids, marker, sub, es, sae,
                       bo=4, temp=1.0, max_new=64, min_new=16, dev=dev), step=step)

Standalone:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python eval/eval_universal.py \
        --adapter ckpts/rl/final --sae-path ... --maxacts-path ... \
        --heldout-pool data/pool_heldout --out eval_universal.json --wandb uni-inverter
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from mxf.config import MODEL, D_MODEL, INJECT_LAYER, READ_LAYER, STEER_COEFF
from mxf.inject import get_layer, make_inject_hook, hooked, read_resid
from mxf.prompts import build_prompt_ids
from mxf.sae import load_sae, load_max_acts

NORM_FILTER_MULT = 10.0   # same as eval_dirs: drop re-encoded tokens with norm > 10x batch median
SAE_FIRE = 1.0            # eval_dirs --sae-fire default: raw act > 1.0 counts as "fired"
GEN_SEED = 1234           # fixed sampling noise per eval (forked RNG — does not touch trainer RNG)
HELDOUT_FRAC = 0.05       # realact: last 5% of acts.f16 sequences are eval-only (never trained on)

COS_FAMILIES = ["probe", "jlens", "cluster", "random", "realact"]   # legacy fresh-sample mode
COS_FAMILIES_HELDOUT = ["bsf", "realact", "jlens", "cluster", "random"]   # held-out-pool mode
HELDOUT_POOL_FAMILIES = ["bsf", "realact", "sae", "jlens", "cluster"]     # record families in pool_heldout
CONTROL_FAMS = {"random"}   # lower-is-better control(s): logged per-family, EXCLUDED from mean_all
# EXTRA families (eval cache v2, data/mlp42_bank_worker.py): meta["extra_families"] lists cosine families scored IN
# ADDITION to meta["cos_families"] and NEVER folded into eval/mean_all, so old and new runs stay comparable. Each
# extra family <fam> ships <fam>_dirs [n, d] plus <fam>_neuron / <fam>_polarity / <fam>_corpus_max [n, k] (k members
# per direction: 1 for "mlp" single neurons, 2 for "mlp_pair" composites) for the layer-42 MLP fire-back metric.
EXTRA_FAMS_KEY = "extra_families"
MLP_LAST_K = 5                       # fire-back: max over the LAST 5 kept tokens of polarity * a_i / corpus_max_i
MLP_FIRE_LEVELS = (0.10, 0.25, 0.50)  # -> eval/<fam>/fired10 / fired25 / fired50
ENV_EVAL_CACHE = "MAEMM_EVAL_CACHE"  # env override of the eval-cache path (rl.py / rl_disagg.py / eval_ckpt_daemon.py --eval-cache default)


def extra_families(eval_sets):
    """Extra (non-mean_all) cosine families present in an eval-set cache — [] for v1 caches."""
    return list(eval_sets["meta"].get(EXTRA_FAMS_KEY, []))


# ---------------------------------------------------------------------------------------------
# scoring (protocol identical to SL/eval_dirs.py; duplicated so this module imports standalone)
# ---------------------------------------------------------------------------------------------

def _reencode(gen_texts, actor, tok, device, sbatch=32):
    """Yield (s, h [B,T,d] fp32, keep [B,T] bool) on the CLEAN base @READ_LAYER, sink-prepended +
    norm-filtered, in sub-batches. EXACT copy of eval_dirs._reencode."""
    prev = tok.padding_side; tok.padding_side = "right"
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    try:
        for s in range(0, len(gen_texts), sbatch):
            batch = [t if t.strip() else " " for t in gen_texts[s:s + sbatch]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=95, add_special_tokens=False).to(device)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), sink, device=device, dtype=enc["input_ids"].dtype), enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=device, dtype=enc["attention_mask"].dtype), enc["attention_mask"]], 1)
            with actor.disable_adapter():
                h, mask = read_resid(actor, READ_LAYER, {"input_ids": ids, "attention_mask": am}, pool="all")
            keep = mask.clone(); keep[:, 0] = False
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            keep = keep & (nrm <= NORM_FILTER_MULT * med)
            yield s, h, keep
    finally:
        tok.padding_side = prev


@torch.no_grad()
def score_probe_cos(gen_texts, dirs, actor, tok, device):
    """max-over-content-token cosine(h_t, dir_i) for gen i; dirs [N,d] unit. == eval_dirs."""
    out = torch.zeros(len(gen_texts))
    for s, h, keep in _reencode(gen_texts, actor, tok, device):
        d = dirs[s:s + h.shape[0]].to(device).float()                       # [b,d] unit
        hn = F.normalize(h.float(), dim=-1)                                 # [b,T,d]
        cos = torch.einsum("btd,bd->bt", hn, d)                             # [b,T]
        out[s:s + h.shape[0]] = cos.masked_fill(~keep, -1.0).max(1).values.float().cpu()
    return out


class _Stop(Exception):
    pass


@torch.no_grad()
def _reencode_mlp(gen_texts, neuron_ids, actor, tok, device, sbatch=32):
    """_reencode that ALSO returns the layer-42 MLP neuron values (down_proj input) of each row's k paired neurons.
    neuron_ids: LongTensor / list [N, k]. Yields (s, h [b,T,d] fp32, keep [b,T], a_sel [b,k,T] fp32, lastk [b,T]) with
    lastk = keep restricted to the last MLP_LAST_K positions of the row (sink counted in the length, as in the
    verbalization scorer of data/mlp42_neurons_worker.py). Same tokenization / sink / 10x-median norm filter as _reencode."""
    layer = get_layer(actor, READ_LAYER)
    down = layer.mlp.down_proj
    nid_all = torch.as_tensor(neuron_ids, dtype=torch.long)
    if nid_all.dim() == 1:
        nid_all = nid_all[:, None]
    prev = tok.padding_side; tok.padding_side = "right"
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    try:
        for s in range(0, len(gen_texts), sbatch):
            batch = [t if t.strip() else " " for t in gen_texts[s:s + sbatch]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=95, add_special_tokens=False).to(device)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), sink, device=device, dtype=enc["input_ids"].dtype), enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=device, dtype=enc["attention_mask"].dtype), enc["attention_mask"]], 1)
            cap = {}

            def pre(_m, inp):
                cap["a"] = inp[0]

            def post(_m, _i, out):
                cap["h"] = (out[0] if isinstance(out, tuple) else out).float()
                raise _Stop

            h1 = down.register_forward_pre_hook(pre); h2 = layer.register_forward_hook(post)
            try:
                with actor.disable_adapter():
                    actor(input_ids=ids, attention_mask=am)
            except _Stop:
                pass
            finally:
                h1.remove(); h2.remove()
            h = cap["h"]
            nid = nid_all[s:s + B].to(device)                                       # [b, k]
            a_sel = cap["a"].float()[torch.arange(B, device=device)[:, None], :, nid]  # [b, k, T]
            keep = am.bool().clone(); keep[:, 0] = False
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            keep = keep & (nrm <= NORM_FILTER_MULT * med)
            L = am.sum(1)                                                            # incl. sink
            pos = torch.arange(h.shape[1], device=device)[None, :]
            lastk = keep & (pos >= (L - MLP_LAST_K)[:, None])
            yield s, h, keep, a_sel, lastk
    finally:
        tok.padding_side = prev


@torch.no_grad()
def score_mlp_fireback(gen_texts, neuron_ids, polarity, corpus_max, actor, tok, device):
    """Layer-42 MLP fire-back of the generated texts on the CLEAN base: per text and paired neuron, max over the last
    MLP_LAST_K kept tokens of polarity * a_i / corpus_max_i. Returns (na_min [N], na_max [N]) cpu fp32 = the WEAKEST and
    the STRONGEST member's normalized fire-back (k=1 singles: identical). neuron_ids / polarity / corpus_max: [N, k]."""
    pol_all = torch.as_tensor(polarity, dtype=torch.float32); cm_all = torch.as_tensor(corpus_max, dtype=torch.float32)
    if pol_all.dim() == 1:
        pol_all = pol_all[:, None]; cm_all = cm_all[:, None]
    na_min = torch.zeros(len(gen_texts)); na_max = torch.zeros(len(gen_texts))
    for s, h, keep, a_sel, lastk in _reencode_mlp(gen_texts, neuron_ids, actor, tok, device):
        b = h.shape[0]
        pol = pol_all[s:s + b].to(device); cm = cm_all[s:s + b].to(device).clamp_min(1e-6)
        av = (a_sel * pol[:, :, None]).masked_fill(~lastk[:, None, :], -float("inf")).max(2).values   # [b, k]
        av = torch.where(torch.isfinite(av), av, torch.zeros_like(av))                                  # no kept token -> 0
        na = av / cm
        na_min[s:s + b] = na.min(1).values.float().cpu(); na_max[s:s + b] = na.max(1).values.float().cpu()
    return na_min, na_max


def mlp_metrics(fam, best_na, best_na_any=None):
    """eval/<fam>/{norm_act, fired10, fired25, fired50} from best-of-bo normalized fire-back per direction (np [n]);
    for k>1 families also any_* from the strongest member."""
    best_na = np.asarray(best_na, np.float64)
    out = {f"eval/{fam}/norm_act": float(best_na.mean())}
    for lv in MLP_FIRE_LEVELS:
        out[f"eval/{fam}/fired{int(round(lv * 100))}"] = float(np.mean(best_na >= lv))
    if best_na_any is not None:
        best_na_any = np.asarray(best_na_any, np.float64)
        out[f"eval/{fam}/any_norm_act"] = float(best_na_any.mean())
        for lv in MLP_FIRE_LEVELS:
            out[f"eval/{fam}/any_fired{int(round(lv * 100))}"] = float(np.mean(best_na_any >= lv))
    return out


@torch.no_grad()
def score_sae_peaks(gen_texts, feat_ids, sae, actor, tok, device):
    """Per-text max-over-content-token activation of ITS paired feature (same numbers as
    eval_dirs.score_sae) PLUS the clean-base L42 hidden state at that peak token, for the
    full-SAE rank metric. Returns (acts [N] cpu, peak_h [N,d] fp32 cpu)."""
    acts = torch.zeros(len(gen_texts))
    peaks = torch.zeros(len(gen_texts), D_MODEL)
    for s, h, keep in _reencode(gen_texts, actor, tok, device):
        b = h.shape[0]
        per = sae.encode_features(h, feat_ids[s:s + b])                     # [b,T,b]
        bi = torch.arange(b, device=per.device)
        a = per[bi, :, bi].masked_fill(~keep, -1.0)                         # -1 so argmax skips masked
        best, tpos = a.max(1)
        acts[s:s + b] = best.clamp(min=0.0).float().cpu()                   # == masked_fill(0).max
        peaks[s:s + b] = h[bi, tpos].float().cpu()
    return acts, peaks


@torch.no_grad()
def sae_rank_at_peaks(sae, peak_h, feats, chunk=256):
    """Rank of each paired feature among ALL features at its peak token. BatchTopKSAE exposes no
    full-encode method, so this is sae.encode_features generalized to every column:
    relu((x - b_dec) @ W_enc + b_enc) — pre-topk post-ReLU, same convention as scoring.
    rank n = 1 + #features strictly more active; n == 1 => the paired feature is the single
    most-active feature at its own peak token. Returns int64 np array [N]."""
    feats_t = torch.as_tensor(feats, dtype=torch.long)
    ranks = torch.zeros(len(feats), dtype=torch.long)
    for s in range(0, len(feats), chunk):
        x = peak_h[s:s + chunk].to(sae.W_enc.device, sae.W_enc.dtype)       # [c,d]
        full = torch.relu((x - sae.b_dec) @ sae.W_enc + sae.b_enc)          # [c,F]
        af = full.gather(1, feats_t[s:s + chunk].to(full.device).unsqueeze(1))   # [c,1]
        ranks[s:s + chunk] = ((full > af).sum(1) + 1).cpu()
    return ranks.numpy()


# ---------------------------------------------------------------------------------------------
# generation (recipe identical to eval_dirs.run_set)
# ---------------------------------------------------------------------------------------------

def _gen_batches(tag, dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                 bo, temp, max_new, min_new, gen_chunk):
    """Yield (rows, texts): each dir injected bo times, fb = gen_chunk//bo dirs per GPU batch."""
    n = len(dirs_unit)
    fb = max(1, gen_chunk // bo)
    for s in range(0, n, fb):
        rows = [i for i in range(s, min(s + fb, n)) for _ in range(bo)]
        vecs = [dirs_unit[i:i + 1] for i in rows]
        hook = make_inject_hook(vecs, [[marker]] * len(rows), STEER_COEFF, dev, torch.bfloat16, mode="add")
        ids = torch.tensor([list(prompt_ids)] * len(rows), device=dev)
        with hooked(sub, hook):
            gen = actor.generate(ids, do_sample=True, temperature=temp, top_p=1.0, top_k=0, min_p=0.0,
                                 max_new_tokens=max_new, min_new_tokens=min_new,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, len(prompt_ids):], skip_special_tokens=True)
        if s % (fb * 10) == 0:
            print(f"  [{tag}] {min(s + fb, n)}/{len(dirs_unit)}", flush=True)
        yield rows, texts


@torch.no_grad()
def eval_cos_family(tag, dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                    bo, temp, max_new, min_new, gen_chunk):
    """Best-of-bo max-token cosine per direction. Returns np [N]."""
    best = np.full(len(dirs_unit), -1e9)
    for rows, texts in _gen_batches(tag, dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                                    bo, temp, max_new, min_new, gen_chunk):
        rdirs = F.normalize(torch.stack([dirs_unit[i] for i in rows]), dim=-1)
        cos = score_probe_cos(texts, rdirs, actor, tok, dev)
        np.maximum.at(best, rows, cos.numpy().astype(np.float64))
    return best


@torch.no_grad()
def eval_sae_family(dirs_unit, feats, sae, actor, tok, prompt_ids, marker, sub, dev,
                    bo, temp, max_new, min_new, gen_chunk):
    """Best-of-bo max-token target-feature act per feature + full-SAE rank at the best gen's peak
    token. Returns (best_act np [N], ranks np int64 [N])."""
    n = len(feats)
    best = np.full(n, -1e9)
    peak_h = torch.zeros(n, D_MODEL)
    for rows, texts in _gen_batches("sae", dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                                    bo, temp, max_new, min_new, gen_chunk):
        acts, peaks = score_sae_peaks(texts, [feats[i] for i in rows], sae, actor, tok, dev)
        for j, i in enumerate(rows):
            if acts[j].item() > best[i]:
                best[i] = acts[j].item()
                peak_h[i] = peaks[j]
    ranks = sae_rank_at_peaks(sae, peak_h, feats)
    return best, ranks


@torch.no_grad()
def eval_mlp_family(tag, dirs_unit, neuron, polarity, corpus_max, actor, tok, prompt_ids, marker, sub, dev,
                    bo, temp, max_new, min_new, gen_chunk):
    """Extra (MLP neuron) family: best-of-bo max-token cosine (the standard metric) AND best-of-bo normalized fire-back
    (weakest member / strongest member). Returns (best_cos [N], best_na [N], best_na_any [N]) np arrays."""
    n = len(dirs_unit)
    best_cos = np.full(n, -1e9); best_na = np.full(n, -1e9); best_any = np.full(n, -1e9)
    for rows, texts in _gen_batches(tag, dirs_unit, actor, tok, prompt_ids, marker, sub, dev,
                                    bo, temp, max_new, min_new, gen_chunk):
        rdirs = F.normalize(torch.stack([dirs_unit[i] for i in rows]), dim=-1)
        cos = score_probe_cos(texts, rdirs, actor, tok, dev)
        na_min, na_max = score_mlp_fireback(texts, neuron[rows], polarity[rows], corpus_max[rows], actor, tok, dev)
        np.maximum.at(best_cos, rows, cos.numpy().astype(np.float64))
        np.maximum.at(best_na, rows, na_min.numpy().astype(np.float64))
        np.maximum.at(best_any, rows, na_max.numpy().astype(np.float64))
    return best_cos, best_na, best_any


# ---------------------------------------------------------------------------------------------
# frozen eval sets
# ---------------------------------------------------------------------------------------------

def _load_heldout_pool(pool_dir):
    """Load one SL/build_universal_bank.py pool: records.jsonl rows (list of dicts, file order ==
    vec_idx order) + vecs.f32 memmap [N, D_MODEL] fp32 unit rows (row i == vec_idx i)."""
    vp, rp = os.path.join(pool_dir, "vecs.f32"), os.path.join(pool_dir, "records.jsonl")
    if not (os.path.exists(vp) and os.path.exists(rp)):
        raise FileNotFoundError(f"held-out pool {pool_dir} is missing vecs.f32 / records.jsonl")
    sz = os.path.getsize(vp)
    assert sz % (4 * D_MODEL) == 0, f"{vp} size {sz} is not a multiple of 4*{D_MODEL}"
    vecs = np.memmap(vp, dtype=np.float32, mode="r", shape=(sz // (4 * D_MODEL), D_MODEL))
    recs = [json.loads(l) for l in open(rp)]
    assert len(recs) == vecs.shape[0], (f"{rp} has {len(recs)} records but vecs.f32 has "
                                        f"{vecs.shape[0]} rows — pool is corrupt/truncated")
    return recs, vecs


def _build_heldout_eval_sets(cache_path, sae, heldout_pool, n, seed, maxacts_path):
    """HELD-OUT mode: eval families come from a build_universal_bank.py pool_heldout (train-
    disjoint per family) instead of fresh-sampling the sources the training bank draws from.

    Selection is a SEEDED SAMPLE per family (np.random.default_rng(seed), families drawn in fixed
    HELDOUT_POOL_FAMILIES order): min(n, avail) record rows without replacement, then sorted by
    vec_idx — frozen across runs and memmap-friendly. (First-n-by-vec_idx would over-concentrate
    bsf/sae in a few blocks/features because the pool writer emits those families grouped.)
    Directions are the pool's own vecs.f32 rows (unit on disk; re-normalized defensively). The sae
    family also keeps the records' "feature" ids ALIGNED to the selected rows (one index array
    selects both) and computes corpus_peak for exactly those ids. The random control stays
    synthetic Gaussian (seeded) — in no pool, never trained on."""
    pool_key = os.path.abspath(heldout_pool)
    if os.path.exists(cache_path):
        es = torch.load(cache_path, map_location="cpu", weights_only=False)
        mt = es["meta"]
        if mt.get("heldout_pool") != pool_key or mt["n"] != n or mt["seed"] != seed:
            raise ValueError(
                f"eval-set cache {cache_path} was built with heldout_pool={mt.get('heldout_pool')} "
                f"n={mt.get('n')} seed={mt.get('seed')}, requested heldout_pool={pool_key} n={n} "
                f"seed={seed}; delete the cache to resample (WARNING: breaks curve comparability)")
        if mt["d_sae"] != sae.d_sae:
            raise ValueError(f"eval-set cache {cache_path} was built for d_sae={mt['d_sae']} but "
                             f"the loaded SAE has d_sae={sae.d_sae} — wrong SAE or stale cache")
        print(f"[eval-sets] loaded HELD-OUT cache {cache_path} (pool={pool_key}, n={n}, "
              f"seed={seed})", flush=True)
        return es

    recs, vecs = _load_heldout_pool(heldout_pool)
    by_fam = {}
    for i, r in enumerate(recs):
        by_fam.setdefault(r["family"], []).append(i)
    rng = np.random.default_rng(seed)   # families drawn in FIXED order — do not reorder
    es, rows_meta = {}, {}
    for fam in HELDOUT_POOL_FAMILIES:
        rows = by_fam.get(fam, [])
        if not rows:
            raise ValueError(f"held-out pool {heldout_pool} has no '{fam}' records "
                             f"(families present: {sorted(by_fam)})")
        take = min(n, len(rows))
        if take < n:
            print(f"[eval-sets] WARN held-out family '{fam}' has only {take} rows < n={n} — "
                  "using all of them", flush=True)
        sel = sorted(int(rows[j]) for j in rng.choice(len(rows), take, replace=False))
        vidx = np.array([recs[i]["vec_idx"] for i in sel], np.int64)
        es[f"{fam}_dirs"] = F.normalize(
            torch.from_numpy(np.asarray(vecs[vidx]).astype(np.float32)), dim=-1)
        rows_meta[fam] = vidx.tolist()
        if fam == "sae":
            feats = [int(recs[i]["feature"]) for i in sel]    # SAME sel as the dirs rows — aligned
            assert max(feats) < sae.d_sae, (f"held-out sae feature id {max(feats)} >= "
                                            f"d_sae {sae.d_sae} — pool built for a different SAE")
            es["sae_feats"] = feats
            ma = load_max_acts(path=maxacts_path)["max_acts"]
            es["corpus_peak"] = ma.reshape(ma.shape[0], -1).max(1).values.float()[
                torch.tensor(feats)].cpu()

    # random control: synthetic Gaussian unit dirs (seeded) — off-manifold, in no pool.
    g = torch.Generator().manual_seed(seed)
    es["random_dirs"] = F.normalize(torch.randn(n, D_MODEL, generator=g), dim=-1)

    es["meta"] = {"mode": "heldout_pool", "heldout_pool": pool_key, "n": n, "seed": seed,
                  "d_sae": sae.d_sae, "cos_families": list(COS_FAMILIES_HELDOUT),
                  "pool_rows": len(recs), "rows": rows_meta}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(es, cache_path)
    print("[eval-sets] built + cached HELD-OUT " + cache_path + ": "
          + " ".join(f"{f}={len(rows_meta[f])}" for f in HELDOUT_POOL_FAMILIES)
          + f" random={n} (pool {pool_key}, {len(recs)} rows, seed={seed})", flush=True)
    return es


def build_eval_sets(cache_path, sae, wu, j42, probe_bank_path, acts_dir, n, dev,
                    seed=0, maxacts_path=None, heldout_pool=None):
    """Sample all six frozen direction families ONCE (fixed seed) and cache to cache_path; later
    calls load the cache so every eval scores the IDENTICAL directions (low-variance curves).

    wu: base.lm_head.weight.detach().float() [vocab, d].  j42: lens.pt["J"][READ_LAYER] [d, d].
    maxacts_path: load_max_acts path for the SAE corpus peaks (None -> HF hub default).
    heldout_pool: path to a build_universal_bank.py pool_heldout dir — switches to HELD-OUT mode
    (families {bsf, realact, sae, jlens, cluster} from the pool + the synthetic random control;
    wu/j42/probe_bank_path/acts_dir/dev are IGNORED and may be None). Use a heldout-specific
    cache_path (e.g. eval_sets_heldout.pt) so the two modes never share a cache.
    Returns {probe_dirs, sae_feats, sae_dirs, corpus_peak, jlens_dirs, cluster_dirs, random_dirs,
    realact_dirs, meta} — all dirs [n, d] fp32 UNIT rows on cpu ({fam}_dirs keyed by
    meta["cos_families"] + sae in held-out mode)."""
    if heldout_pool is not None:
        return _build_heldout_eval_sets(cache_path, sae, heldout_pool, n, seed, maxacts_path)
    if os.path.exists(cache_path):
        es = torch.load(cache_path, map_location="cpu", weights_only=False)
        if es["meta"].get("heldout_pool") is not None:
            raise ValueError(f"eval-set cache {cache_path} is a HELD-OUT cache (pool="
                             f"{es['meta']['heldout_pool']}) but heldout_pool was not requested — "
                             "wrong cache path")
        if es["meta"]["n"] != n or es["meta"]["seed"] != seed:
            raise ValueError(f"eval-set cache {cache_path} was built with n={es['meta']['n']} "
                             f"seed={es['meta']['seed']}, requested n={n} seed={seed}; delete the "
                             "cache to resample (WARNING: breaks curve comparability)")
        if es["meta"]["d_sae"] != sae.d_sae:
            raise ValueError(f"eval-set cache {cache_path} was built for d_sae={es['meta']['d_sae']} "
                             f"but the loaded SAE has d_sae={sae.d_sae} — wrong SAE or stale cache")
        print(f"[eval-sets] loaded cache {cache_path} (n={n}, seed={seed})", flush=True)
        return es
    rng = np.random.default_rng(seed)   # families drawn in FIXED order — do not reorder

    # (1)+(4) probe + cluster: two DISJOINT samples from the RL probe bank (see module docstring
    # for why the cluster family falls back to the probe bank: centroids.npy is whitened-space).
    nrows = os.path.getsize(probe_bank_path) // (4 * D_MODEL)
    assert nrows >= 2 * n, f"probe bank has {nrows} rows < 2*n={2 * n} (probe+cluster disjoint split)"
    bank = np.memmap(probe_bank_path, dtype=np.float32, mode="r", shape=(nrows, D_MODEL))
    perm = rng.choice(nrows, 2 * n, replace=False)
    pidx, cidx = np.sort(perm[:n]), np.sort(perm[n:])
    probe_dirs = F.normalize(torch.from_numpy(np.asarray(bank[pidx]).astype(np.float32)), dim=-1)
    cluster_dirs = F.normalize(torch.from_numpy(np.asarray(bank[cidx]).astype(np.float32)), dim=-1)

    # (2) SAE features: unit encoder columns + corpus peak act per feature (eval_dirs recipe).
    feats = np.sort(rng.choice(sae.d_sae, min(n, sae.d_sae), replace=False)).tolist()
    sae_dirs = sae.enc_dirs(feats).float().cpu()
    ma = load_max_acts(path=maxacts_path)["max_acts"]
    corpus_peak = ma.reshape(ma.shape[0], -1).max(1).values.float()[torch.tensor(feats)].cpu()

    # (3) J-lens: unit(W_U[t] @ J_42) for random vocab tokens (serve_playground direction).
    vocab = wu.shape[0]
    tids = np.sort(rng.choice(vocab, min(n, vocab), replace=False))
    tv = torch.as_tensor(tids, dtype=torch.long, device=wu.device)
    jlens_dirs = F.normalize(wu[tv].float() @ j42.to(wu.device).float(), dim=-1).cpu()

    # (5) random: isotropic Gaussian unit dirs — off-manifold control (expect cos ~ 1/sqrt(d)).
    g = torch.Generator().manual_seed(seed)
    random_dirs = F.normalize(torch.randn(n, D_MODEL, generator=g), dim=-1)

    # (6) realact: real L42 token acts from the HELD-OUT tail (last 5% of sequences) of the
    # bsf27b acts dump — on-manifold targets never seen in training. Skip token 0 (attention-sink
    # norms) and norm-filter 10x-median, mirroring the scoring protocol; sample with margin.
    meta_a = json.load(open(os.path.join(acts_dir, "meta.json")))
    n_seq, seq_len, d = meta_a["n_seq"], meta_a["seq_len"], meta_a["d_model"]
    assert d == D_MODEL, f"acts d_model {d} != D_MODEL {D_MODEL}"
    acts = np.memmap(os.path.join(acts_dir, "acts.f16"), dtype=np.float16, mode="r",
                     shape=(n_seq, seq_len, d))
    lo = int(np.ceil((1.0 - HELDOUT_FRAC) * n_seq))
    sidx = rng.integers(lo, n_seq, size=4 * n)
    tpos = rng.integers(1, seq_len, size=4 * n)
    X = torch.from_numpy(np.stack([np.asarray(acts[si, ti]) for si, ti in zip(sidx, tpos)])
                         .astype(np.float32))
    nrm = X.norm(dim=-1)
    ok = (nrm > 0) & (nrm <= NORM_FILTER_MULT * nrm.median())
    assert int(ok.sum()) >= n, f"only {int(ok.sum())} realact rows survive the norm filter (< n={n})"
    realact_dirs = F.normalize(X[ok][:n], dim=-1)

    es = {"probe_dirs": probe_dirs, "cluster_dirs": cluster_dirs,
          "sae_feats": feats, "sae_dirs": sae_dirs, "corpus_peak": corpus_peak,
          "jlens_dirs": jlens_dirs, "random_dirs": random_dirs, "realact_dirs": realact_dirs,
          "meta": {"mode": "fresh_sample", "heldout_pool": None,
                   "cos_families": list(COS_FAMILIES),
                   "n": n, "seed": seed, "probe_bank": probe_bank_path, "acts_dir": acts_dir,
                   "probe_rows": pidx.tolist(), "cluster_rows": cidx.tolist(),
                   "jlens_tids": tids.tolist(), "d_sae": sae.d_sae, "vocab": int(vocab),
                   "realact_lo": lo, "heldout_frac": HELDOUT_FRAC}}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(es, cache_path)
    print(f"[eval-sets] built + cached {cache_path}: 6 families x n={n} (seed={seed}); "
          f"probe bank {nrows} rows, d_sae {sae.d_sae}, vocab {vocab}, realact seqs [{lo},{n_seq})",
          flush=True)
    return es


# ---------------------------------------------------------------------------------------------
# eval entry point (imported by the trainer — no wandb in here, caller logs the returned dict)
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def run_eval(actor, tok, prompt_ids, marker, sub, eval_sets, sae, bo, temp, max_new, min_new, dev,
             gen_chunk=64, sae_fire=SAE_FIRE):
    """Run every family in eval_sets (meta["cos_families"] cosine families + the sae metric
    family); return a FLAT {wandb scalar name: float} dict. Generation RNG is forked + fixed
    (GEN_SEED) so repeat evals of the same checkpoint are deterministic and the trainer's RNG
    stream is untouched."""
    was_training = actor.training
    actor.eval()
    fork_devs = [dev] if str(dev).startswith("cuda") else []
    # legacy fresh-sample caches predate meta["cos_families"] -> default to the legacy list
    fams = list(eval_sets["meta"].get("cos_families", COS_FAMILIES))
    xfams = extra_families(eval_sets)
    out = {}
    try:
        with torch.random.fork_rng(devices=fork_devs):
            torch.manual_seed(GEN_SEED)
            gen_args = (actor, tok, prompt_ids, marker, sub, dev, bo, temp, max_new, min_new, gen_chunk)
            for fam in fams:
                best = eval_cos_family(fam, eval_sets[f"{fam}_dirs"], *gen_args)
                out[f"eval/{fam}/cos"] = float(best.mean())
            best_act, ranks = eval_sae_family(eval_sets["sae_dirs"], eval_sets["sae_feats"], sae, *gen_args)
            cp = eval_sets["corpus_peak"].numpy().astype(np.float64)
            r = ranks.astype(np.float64)
            out["eval/sae/norm_act"] = float(np.mean(best_act / np.maximum(cp, 1e-6)))
            out["eval/sae/fired"] = float(np.mean(best_act > sae_fire))
            out["eval/sae/beat_corpus"] = float(np.mean(best_act > cp))
            out["eval/sae/mean_rank"] = float(r.mean())
            out["eval/sae/rank1_frac"] = float(np.mean(ranks == 1))
            out["eval/sae/mrr"] = float(np.mean(1.0 / r))
            na = best_act / np.maximum(cp, 1e-6)
            out["eval/sae/unverbalized_frac"] = float(np.mean(best_act <= sae_fire))  # cannot be made to fire at all
            out["eval/sae/unverbalized_p10"] = float(np.mean(na < 0.10))  # inversion reached <10pct of corpus peak
            # extra families (cache v2: layer-42 MLP neurons / co-firing pairs): cosine + fire-back, NOT in mean_all
            for fam in xfams:
                k = eval_sets[f"{fam}_neuron"].shape[1]
                bc, bn, ba = eval_mlp_family(fam, eval_sets[f"{fam}_dirs"], eval_sets[f"{fam}_neuron"],
                                             eval_sets[f"{fam}_polarity"], eval_sets[f"{fam}_corpus_max"], *gen_args)
                out[f"eval/{fam}/cos"] = float(bc.mean())
                out.update(mlp_metrics(fam, bn, ba if k > 1 else None))
    finally:
        if was_training:
            actor.train()
    # mean_all = mean over the HIGHER-IS-BETTER cos families only. `random` is the control
    # (should stay ~0.03, LOWER is better) — folding it in would drag the mean down and move
    # mean_all the WRONG way if the control ever degraded. It stays logged separately. Extra
    # families (xfams) are NOT included either, so mean_all is comparable across cache versions.
    out["eval/mean_all"] = float(np.mean([out[f"eval/{f}/cos"] for f in fams
                                          if f not in CONTROL_FAMS]))
    # headline mirror group — one wandb panel with every family side by side
    # legacy mode: probe/jlens/cluster/random/realact _cos; held-out mode: bsf replaces probe
    for fam in fams + xfams:
        out[f"eval/all/{fam}_cos"] = out[f"eval/{fam}/cos"]
    out["eval/all/sae_norm_act"] = out["eval/sae/norm_act"]
    out["eval/all/sae_unverbalized"] = out["eval/sae/unverbalized_frac"]
    out["eval/all/sae_mean_rank"] = out["eval/sae/mean_rank"]
    for fam in xfams:
        out[f"eval/all/{fam}_norm_act"] = out[f"eval/{fam}/norm_act"]
        out[f"eval/all/{fam}_fired10"] = out[f"eval/{fam}/fired10"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--sae-path", required=True)
    ap.add_argument("--maxacts-path", required=True)
    ap.add_argument("--lens-path", default="/root/pmx/lenses/qwen3.6-27b/j-lens/lens.pt")
    ap.add_argument("--acts-dir", default="/root/pmx/bsf27b/acts")
    ap.add_argument("--probe-bank", default="/root/pmx/data/pool_rl_1M/probes.f32")
    ap.add_argument("--heldout-pool", default=None,
                    help="path to a build_universal_bank.py pool_heldout dir (records.jsonl + "
                         "vecs.f32). When set, eval families come from this train-DISJOINT pool "
                         "{bsf,realact,sae,jlens,cluster} + the synthetic random control, instead "
                         "of fresh-sampling sources that overlap pool_train")
    ap.add_argument("--cache", default="/root/pmx/data/eval_universal",
                    help="cache DIR; frozen eval sets live at <cache>/eval_sets.pt "
                         "(<cache>/eval_sets_heldout.pt with --heldout-pool)")
    ap.add_argument("--cache-file", default=os.environ.get(ENV_EVAL_CACHE),
                    help=f"FULL path of an existing eval-set cache (overrides --cache's file name; env {ENV_EVAL_CACHE}), "
                         "e.g. .../eval_sets_heldout_v2.pt with the extra mlp / mlp_pair families")
    ap.add_argument("--n", type=int, default=1024, help="directions per family")
    ap.add_argument("--bo", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--min-new-tokens", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--sae-fire", type=float, default=SAE_FIRE)
    ap.add_argument("--out", default=None, help="optional JSON dump of the metric dict")
    ap.add_argument("--wandb", default=None, help="optional wandb project; init + log one step")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    dev = "cuda:0"

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok); marker = mpos[0]
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    actor = PeftModel.from_pretrained(base, a.adapter, is_trainable=False); actor.eval()
    sub = get_layer(actor, INJECT_LAYER)
    wu = base.lm_head.weight.detach().float()                                # [vocab, d] on dev
    j42 = None if a.heldout_pool else \
        torch.load(a.lens_path, map_location="cpu", weights_only=False)["J"][READ_LAYER].float().to(dev)
    sae = load_sae(path=a.sae_path, device=dev, dtype=torch.float32)
    print(f"[eval-universal] adapter={a.adapter} | n={a.n} bo={a.bo} | WU {tuple(wu.shape)} | "
          + (f"heldout pool {a.heldout_pool}" if a.heldout_pool
             else f"J{READ_LAYER} {tuple(j42.shape)}"), flush=True)

    cache_path = a.cache_file or os.path.join(a.cache,
                                             "eval_sets_heldout.pt" if a.heldout_pool else "eval_sets.pt")
    if a.cache_file:
        assert os.path.exists(a.cache_file), f"--cache-file {a.cache_file} does not exist"
    es = build_eval_sets(cache_path, sae, wu, j42, a.probe_bank, a.acts_dir, a.n, dev,
                         seed=0, maxacts_path=a.maxacts_path, heldout_pool=a.heldout_pool)
    if extra_families(es):
        print(f"[eval-universal] extra families {extra_families(es)} (cosine + MLP fire-back; not in mean_all)", flush=True)
    m = run_eval(actor, tok, prompt_ids, marker, sub, es, sae, a.bo, a.temp,
                 a.max_new_tokens, a.min_new_tokens, dev, gen_chunk=a.gen_chunk, sae_fire=a.sae_fire)
    print("=== EVAL-UNIVERSAL ===\n" + json.dumps(m, indent=1), flush=True)
    if a.out:
        json.dump({"adapter": a.adapter, "n": a.n, "bo": a.bo, "metrics": m}, open(a.out, "w"), indent=1)
        print(f"EVAL_UNIVERSAL_DONE {a.out}", flush=True)
    if a.wandb:
        import wandb
        wandb.init(project=a.wandb, name=a.run_name or os.path.basename(a.adapter.rstrip("/")),
                   config={"adapter": a.adapter, "n": a.n, "bo": a.bo, "temp": a.temp,
                           "max_new_tokens": a.max_new_tokens, "min_new_tokens": a.min_new_tokens,
                           "cache": cache_path, "heldout_pool": a.heldout_pool})
        wandb.log(m)
        wandb.finish()


if __name__ == "__main__":
    main()
