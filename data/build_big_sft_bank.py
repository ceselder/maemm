"""Build a BIG realact+probes SFT bank (CPU-only) for the scaled universal-inverter run.

realact: unit(act - mu) at sampled (s < 95%-of-seqs [eval holds the last 5%], p in [16, seq_len-1]),
  10x-median raw-norm filter (eval_universal's realact hygiene), target = decode(toks[s, p-L+1:p+1]),
  L ~ U[16,64]. SHORT context (<= seq_len=512) -- SFT stays short per fjiahai; long ctx is RL-only.
probes: sample --n-probe records from pool_sft_1M, copy + re-unit-normalize their vecs.

Writes {out}/vecs.f32 [N,5120] f32 (row i == vec_idx i) + records.jsonl {vec_idx,target_text,family}.
Usage: PYTHONPATH=/root/pmx/helpers python bsf/build_big_sft_bank.py --n-realact 500000 --n-probe 500000
"""
import argparse, os, json, numpy as np, random
from transformers import AutoTokenizer
from mxf.config import MODEL, D_MODEL

NORM_FILTER_MULT = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="/root/pmx/bsf27b/acts")
    ap.add_argument("--cluster-pool", default="/root/pmx/data/pool_sft_1M")
    ap.add_argument("--out", default="/root/pmx/data/pool_big_sft")
    ap.add_argument("--n-realact", type=int, default=500000)
    ap.add_argument("--n-probe", type=int, default=500000)
    ap.add_argument("--train-seq-frac", type=float, default=0.95)   # first 95% seqs; eval holds last 5%
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    meta = json.load(open(f"{a.acts_dir}/meta.json"))
    NS, T = int(meta["n_seq"]), int(meta["seq_len"])
    acts = np.memmap(f"{a.acts_dir}/acts.f16", dtype=np.float16, mode="r", shape=(NS, T, D_MODEL))
    toks = np.memmap(f"{a.acts_dir}/toks.i32", dtype=np.int32, mode="r", shape=(NS, T))
    mu = np.load(f"{a.acts_dir}/whiten_mu.npy").astype(np.float32)
    n_train = int(NS * a.train_seq_frac)
    print(f"[bigbank] acts {NS}x{T}, train seqs 0..{n_train-1}; targets realact={a.n_realact} probe={a.n_probe}", flush=True)

    # --- realact raw-norm filter threshold (10x median of a presample) ---
    ps = 8000; sp_s = nrng.integers(0, n_train, ps); sp_p = nrng.integers(16, T, ps)
    pnorm = np.linalg.norm(acts[sp_s, sp_p].astype(np.float32) - mu, axis=1)
    thr = NORM_FILTER_MULT * float(np.median(pnorm))
    print(f"[bigbank] realact norm filter: median {np.median(pnorm):.1f}, thr {thr:.1f}", flush=True)

    N = a.n_realact + a.n_probe
    vecs = np.memmap(f"{a.out}/vecs.f32", dtype=np.float32, mode="w+", shape=(N, D_MODEL))
    recs = open(f"{a.out}/records.jsonl", "w")
    row = 0
    # --- family 1: realact (one pass over train seqs, ~pos_per_seq sampled positions each) ---
    per = int(np.ceil(a.n_realact / n_train)) + 4
    for s in range(n_train):
        if row >= a.n_realact: break
        seq = acts[s].astype(np.float32)                      # [T, d]
        d = seq - mu; nrm = np.linalg.norm(d, axis=1)
        ok = np.flatnonzero((nrm <= thr) & (np.arange(T) >= 16))
        if len(ok) == 0: continue
        nrng.shuffle(ok)
        for p in ok[:per]:
            if row >= a.n_realact: break
            L = rng.randint(16, 64)
            txt = tok.decode(toks[s, max(0, p - L + 1): p + 1].tolist())
            if len(txt.strip()) < 3: continue
            vecs[row] = d[p] / nrm[p]
            recs.write(json.dumps({"vec_idx": row, "target_text": txt, "family": "realact",
                                   "seq": int(s), "pos": int(p)}) + "\n")
            row += 1
        if s % 500 == 0: print(f"[bigbank] realact {row}/{a.n_realact} (seq {s})", flush=True)
    n_realact_done = row
    # --- family 2: probes (sample from pool_sft_1M) ---
    prec = [json.loads(l) for l in open(f"{a.cluster_pool}/records.jsonl")]
    npv = len(prec)
    pv = np.memmap(f"{a.cluster_pool}/vecs.f32", dtype=np.float32, mode="r", shape=(npv, D_MODEL))
    idx = nrng.permutation(npv)[:a.n_probe]
    for j in idx:
        v = pv[j].astype(np.float32); nn = np.linalg.norm(v)
        if nn < 1e-8: continue
        vecs[row] = v / nn
        recs.write(json.dumps({"vec_idx": row, "target_text": prec[j]["target_text"], "family": "cluster",
                               "src_cluster": prec[j].get("cluster")}) + "\n")
        row += 1
    vecs.flush(); recs.close()
    json.dump({"n_examples": row, "n_realact": n_realact_done, "n_probe": row - n_realact_done,
               "d_model": D_MODEL, "families": {"realact": n_realact_done, "cluster": row - n_realact_done},
               "acts_dir": a.acts_dir, "cluster_pool": a.cluster_pool, "train_seq_frac": a.train_seq_frac,
               "norm_thr": thr, "seed": a.seed, "kind": "big-sft realact+probes short-ctx"},
              open(f"{a.out}/build_stats.json", "w"), indent=1)
    print(f"[bigbank] DONE {row} examples (realact {n_realact_done}, probe {row-n_realact_done}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
