"""Add held-out CONTEXT-LENGTH BUCKET families (realact_early <=512 / realact_mid 512-2048 /
realact_long 2048-8192, by token position of the activation in its document) as eval families to
the frozen eval cache (eval_sets_heldout.pt), so the eval daemon reports early-vs-long-context
inversion per RL checkpoint.

Held-out = the COMPLEMENT of what bank/build_rl_bank.py sampled from the long-context activation
dump: build_rl_bank takes default_rng(0).permutation(flat)[:N_TRAIN] for training, so
[N_TRAIN:] is train-disjoint by construction. Directions = unit(h - mu_long), matching the
realact_long training centering (mu_long = mean over ALL collected long-ctx acts, reproduced
here exactly as the bank builder computes it).

One-shot, idempotent-ish (re-running re-mints the same families and re-appends nothing new if
they are already in cos_families — but simplest is to run it exactly once per cache).

Run (paths overridable via env):
    MAEMM_ACTS_LONG=data/acts_long MAEMM_EVAL_CACHE=data/eval_universal_ho/eval_sets_heldout.pt \
        PYTHONPATH=$PWD python MAEMMBench/build_ctx_eval.py
"""
import glob
import json
import os

import numpy as np
import torch

from mxf.config import D_MODEL

ACTS_LONG = os.environ.get("MAEMM_ACTS_LONG", "/root/pmx/bsf27b/acts_long")
CACHE = os.environ.get("MAEMM_EVAL_CACHE", "/root/pmx/data/eval_universal_ho/eval_sets_heldout.pt")
N_TRAIN = int(os.environ.get("MAEMM_N_TRAIN", 250000))   # rows build_rl_bank took for training
N_PER = 512                                              # eval dirs per bucket (matches n=512/family)
BUCKETS = {"realact_early": (1, 512), "realact_mid": (512, 2048), "realact_long": (2048, 8192)}

# --- reproduce build_rl_bank's flat list + seed-0 split ---
shards = []
for rf in sorted(glob.glob(f"{ACTS_LONG}/records_s*.jsonl")):
    s = rf.split("_s")[-1].split(".")[0]
    recs = [json.loads(l) for l in open(rf)]
    H = np.memmap(f"{ACTS_LONG}/acts_long_s{s}.f16", dtype=np.float16, mode="r").reshape(-1, D_MODEL)
    shards.append((s, recs, H))
Hmap = {s: H for s, _, H in shards}
flat = [(s, r) for s, rs, _ in shards for r in rs]
perm = np.random.default_rng(0).permutation(len(flat))
heldout = perm[N_TRAIN:]                                  # disjoint from training
print(f"flat={len(flat)} train={N_TRAIN} heldout={len(heldout)}")

# --- mu_long (same as build_rl_bank: mean over ALL collected long-ctx acts) ---
acc = np.zeros(D_MODEL, np.float64)
cnt = 0
for _, recs, H in shards:
    idx = np.array([r["idx"] for r in recs])
    acc += H[idx].astype(np.float64).sum(0)
    cnt += len(idx)
mu_long = (acc / cnt).astype(np.float32)

# --- bucket the held-out complement by position, mint N_PER dirs each ---
rng = np.random.default_rng(7)
cache = torch.load(CACHE, map_location="cpu", weights_only=False)
added = []
for fam, (lo, hi) in BUCKETS.items():
    pool = [i for i in heldout if lo <= flat[i][1]["pos"] < hi]
    rng.shuffle(pool)
    take = pool[:N_PER]
    dirs = np.zeros((len(take), D_MODEL), np.float32)
    for k, i in enumerate(take):
        s, r = flat[i]
        h = Hmap[s][r["idx"]].astype(np.float32) - mu_long
        dirs[k] = h / max(np.linalg.norm(h), 1e-6)
    cache[f"{fam}_dirs"] = torch.from_numpy(dirs)
    added.append(fam)
    print(f"{fam}: pool={len(pool)} took={len(take)}")
cache["meta"]["cos_families"] = list(cache["meta"]["cos_families"]) + [
    f for f in added if f not in cache["meta"]["cos_families"]]
torch.save(cache, CACHE)
print("cos_families now:", cache["meta"]["cos_families"])
