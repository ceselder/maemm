"""Add IN-DISTRIBUTION held-out families from the ACTUAL RL training pool (pool_rl_mix) to the
frozen eval cache (eval_sets_heldout.pt), so the eval daemon reports inversion on the same
distribution the live run trains on (the base eval families come from the OLD universal bank
pool_heldout, not pool_rl_mix).

pool_rl_mix: vecs.f32 fp32 [750000, D_MODEL] (directions ALREADY minted — reused as-is, only
defensively re-unit-normalized) + records.jsonl (family + target_text; families are contiguous
blocks: realact_long [0,250k), realact [250k,500k), cluster [500k,750k)). New eval families:
    indist_realact <- realact       (real L42 token activations, short ctx)
    indist_probe   <- cluster       (cluster-probe directions)
    indist_long    <- realact_long  (real L42 token activations, long ctx)

HELD-OUT CAVEAT (not strictly train-disjoint): the RL run (--eval-every 0) samples training rows
uniformly at random over ALL 750k rows — there is NO reserved slice on disk. We reserve the LAST
N_TAIL=2000 rows of each family block; with 32 groups/step x 400 steps = 12.8k training draws
over 750k rows, any given tail row has ~1.7% probability of ever being trained on (expected ~34
of the 2000 per family) — negligible but nonzero contamination.

Idempotent: families already present in the cache are skipped; re-running is a no-op.

Run (paths overridable via env):
    MAEMM_POOL=data/pool_rl_mix MAEMM_EVAL_CACHE=data/eval_universal_ho/eval_sets_heldout.pt \
        PYTHONPATH=$PWD python eval/build_indist_eval.py
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mxf.config import D_MODEL

POOL = os.environ.get("MAEMM_POOL", "/root/pmx/data/pool_rl_mix")
CACHE = os.environ.get("MAEMM_EVAL_CACHE", "/root/pmx/data/eval_universal_ho/eval_sets_heldout.pt")
N_TAIL = 2000     # reserved held-out tail rows per family (see caveat above)
N_PER = 512       # eval dirs per family (matches every other family in the cache, n=512)
SEED = 11
FAM_MAP = {"indist_realact": "realact", "indist_probe": "cluster", "indist_long": "realact_long"}

cache = torch.load(CACHE, map_location="cpu", weights_only=False)
fams_now = list(cache["meta"]["cos_families"])
todo = {k: v for k, v in FAM_MAP.items() if k not in fams_now or f"{k}_dirs" not in cache}
if not todo:
    print(f"all indist families already present ({sorted(FAM_MAP)}) — nothing to do")
    raise SystemExit(0)

# --- pool: records give the family layout; vecs.f32 row i == records line i (vec_idx) ---
sz = os.path.getsize(f"{POOL}/vecs.f32")
assert sz % (4 * D_MODEL) == 0
vecs = np.memmap(f"{POOL}/vecs.f32", dtype=np.float32, mode="r", shape=(sz // (4 * D_MODEL), D_MODEL))
rows_by_fam, texts = {}, []
for i, l in enumerate(open(f"{POOL}/records.jsonl")):
    r = json.loads(l)
    assert r["vec_idx"] == i, f"records.jsonl line {i} has vec_idx {r['vec_idx']} — order broken"
    rows_by_fam.setdefault(r["family"], []).append(i)
    texts.append(r.get("target_text", ""))
assert len(texts) == vecs.shape[0], f"{len(texts)} records != {vecs.shape[0]} vec rows"
print(f"pool {POOL}: {len(texts)} rows, families " +
      " ".join(f"{f}={len(v)}" for f, v in rows_by_fam.items()))

# --- reserve the LAST N_TAIL rows per source family, seed-sample N_PER of them ---
rng = np.random.default_rng(SEED)   # families drawn in FIXED (sorted) order — do not reorder
meta_indist = cache["meta"].setdefault("indist", {})
added = []
for fam in sorted(todo):
    src = todo[fam]
    rows = rows_by_fam[src]
    tail = rows[-N_TAIL:]
    sel = sorted(int(tail[j]) for j in rng.choice(len(tail), min(N_PER, len(tail)), replace=False))
    X = torch.from_numpy(np.asarray(vecs[np.array(sel, np.int64)]).astype(np.float32))
    nrm = X.norm(dim=-1)
    assert (nrm > 1e-6).all(), f"{fam}: {int((nrm <= 1e-6).sum())} zero-norm vecs in tail sample"
    cache[f"{fam}_dirs"] = F.normalize(X, dim=-1)          # dirs already minted; unit defensively
    meta_indist[fam] = {"pool": POOL, "source_family": src, "n_tail": N_TAIL, "seed": SEED,
                        "rows": sel, "target_texts": [texts[i] for i in sel]}
    added.append(fam)
    print(f"{fam}: source={src} tail=[{tail[0]},{tail[-1]}] took={len(sel)} "
          f"norm med {nrm.median():.3f}")

cache["meta"]["cos_families"] = fams_now + [f for f in added if f not in fams_now]
torch.save(cache, CACHE)
print("cos_families now:", cache["meta"]["cos_families"])
