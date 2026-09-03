"""Build the balanced RL data-mix bank (CPU): realact(short) + probes + realact_long, ~equal thirds.
long-context is ONE component of the mix, not the only thing (per user). realact_long = unit(h - mu_long)
from the [1,8192] collection; realact(short) + probes sampled from pool_big_sft (rows 0-499999 realact,
500000-999999 cluster). Writes {out}/vecs.f32 [N,5120] + records.jsonl {vec_idx,target_text,family}.
Usage: PYTHONPATH=/root/pmx/helpers python bsf/build_rl_bank.py --n-each 250000
"""
import argparse, os, json, glob, numpy as np
from mxf.config import D_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-long", default="/root/pmx/bsf27b/acts_long")
    ap.add_argument("--big-sft", default="/root/pmx/data/pool_big_sft")
    ap.add_argument("--out", default="/root/pmx/data/pool_rl_mix")
    ap.add_argument("--n-each", type=int, default=250000)   # per family: realact, probe, realact_long
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    nrng = np.random.default_rng(a.seed); os.makedirs(a.out, exist_ok=True)

    # ---- gather realact_long shards (records + per-shard acts memmap) ----
    shards = []
    for rf in sorted(glob.glob(f"{a.acts_long}/records_s*.jsonl")):
        s = rf.split("_s")[-1].split(".")[0]
        meta = json.load(open(f"{a.acts_long}/meta_s{s}.json"))
        n = int(meta["n_acts"]); cap = int(meta["n_docs"]) * 4 if False else None
        recs = [json.loads(l) for l in open(rf)]
        H = np.memmap(f"{a.acts_long}/acts_long_s{s}.f16", dtype=np.float16, mode="r").reshape(-1, D_MODEL)
        shards.append((s, recs, H))
    total_long = sum(len(r) for _, r, _ in shards)
    print(f"[rlbank] realact_long: {total_long} dirs across {len(shards)} shards", flush=True)
    # mu_long: streaming mean over all collected long-ctx acts (natural centering for this family)
    acc = np.zeros(D_MODEL, np.float64); cnt = 0
    for _, recs, H in shards:
        idx = np.array([r["idx"] for r in recs])
        acc += H[idx].astype(np.float64).sum(0); cnt += len(idx)
    mu_long = (acc / cnt).astype(np.float32)
    print(f"[rlbank] mu_long over {cnt} acts (||mu||={np.linalg.norm(mu_long):.1f})", flush=True)

    n_long = min(a.n_each, total_long)
    N = a.n_each * 2 + n_long                                    # realact + probe + realact_long
    vecs = np.memmap(f"{a.out}/vecs.f32", dtype=np.float32, mode="w+", shape=(N, D_MODEL))
    recs_out = open(f"{a.out}/records.jsonl", "w")
    row = 0
    # ---- family: realact_long (sample n_long across shards) ----
    flat = [(s, r) for s, rs, _ in shards for r in rs]
    sel = nrng.permutation(len(flat))[:n_long]
    Hmap = {s: H for s, _, H in shards}
    for j in sel:
        s, r = flat[j]; h = Hmap[s][r["idx"]].astype(np.float32) - mu_long
        nn = np.linalg.norm(h)
        if nn < 1e-6: continue
        vecs[row] = h / nn
        recs_out.write(json.dumps({"vec_idx": row, "target_text": r["target_text"],
                                   "family": "realact_long", "pos": r.get("pos")}) + "\n")
        row += 1
    n_long_done = row
    # ---- families: realact(short) + probes from pool_big_sft ----
    bprec = [json.loads(l) for l in open(f"{a.big_sft}/records.jsonl")]
    nbp = len(bprec)
    bvec = np.memmap(f"{a.big_sft}/vecs.f32", dtype=np.float32, mode="r", shape=(nbp, D_MODEL))
    ra_idx = [i for i, r in enumerate(bprec) if r["family"] == "realact"]
    cl_idx = [i for i, r in enumerate(bprec) if r["family"] == "cluster"]
    for fam, pool in [("realact", ra_idx), ("cluster", cl_idx)]:
        take = nrng.permutation(len(pool))[:a.n_each]
        for t in take:
            i = pool[int(t)]; v = bvec[i].astype(np.float32); nn = np.linalg.norm(v)
            if nn < 1e-6: continue
            vecs[row] = v / nn
            recs_out.write(json.dumps({"vec_idx": row, "target_text": bprec[i]["target_text"],
                                       "family": fam}) + "\n")
            row += 1
    vecs.flush(); recs_out.close()
    fam_counts = {}
    for l in open(f"{a.out}/records.jsonl"):
        f = json.loads(l)["family"]; fam_counts[f] = fam_counts.get(f, 0) + 1
    json.dump({"n_examples": row, "families": fam_counts, "n_each_target": a.n_each,
               "acts_long": a.acts_long, "big_sft": a.big_sft, "kind": "rl-mix realact+probes+realact_long"},
              open(f"{a.out}/build_stats.json", "w"), indent=1)
    print(f"[rlbank] DONE {row} examples {fam_counts} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
