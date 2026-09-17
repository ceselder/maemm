"""Modal app (CPU): LONG-context real-activation rows from a 512-token layer-42 store -> a single-family bank in the exact
sft/pretrain.py + rl/rl.py + data/modal_mix_5m_bank.py format.

    /data/banks/<out_name>/vecs.f32        [N, 5120] float32 UNIT rows   = unit(act[s, p] - mu)        (10x-median norm filter)
    /data/banks/<out_name>/records.jsonl   line i == vec_idx i: {vec_idx, family, target_text, seq, pos, start, n_tok, fire_from_end}
    /data/banks/<out_name>/build_stats.json + meta.json

Recipe == the "realact_long" family of data/modal_bank_everything.py / the WINDOW harvest of data/modal_big_bank.py: a random
sequence s of the store and a DEEP position p ~ U[p_lo, p_hi] (default [256, 511] of the 512-token window: the activation saw
p+1 tokens of context), direction = unit(act[s, p] - mu) with mu = the suite's realact convention (/data/acts27b/whiten_mu.npy;
pass mu_path to use the store's own mean), target_text = decode(toks[s, p-W+1 : p+1]) with W ~ U[w_lo, w_hi] (the firing token is
the LAST target token), <= max_per_doc rows per sequence. Every row of the store is eligible (the store itself is the split:
build RL stores and eval stores from disjoint document ranges).

    MODAL_PROFILE=safety-sahan UFW_LONG_APP=maemm-ufw-long-bank-s2m modal deploy data/modal_ufw_long_bank.py
    python -c "import modal; print(modal.Function.from_name('maemm-ufw-long-bank-s2m','build').spawn(acts_dir='/data/acts_ufw_rl', out_name='ufw_long_rl_500k', n_rows=500000).object_id)"
"""
import os

import modal

APP_NAME = os.environ.get("UFW_LONG_APP", "maemm-ufw-long-bank")
app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy<2.3", "transformers==5.15.0", "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet")
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

MODEL, D_MODEL, READ_LAYER = "Qwen/Qwen3.6-27B", 5120, 42     # == mxf.config (kept import-free: no torch here)
MU_DEFAULT = "/data/acts27b/whiten_mu.npy"
NORM_FILTER_MULT = 10.0
NORM_PRESAMPLE = 20_000


def _pread_full(fd, n, off):
    buf = bytearray(n); mv = memoryview(buf); got = 0
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise IOError(f"short read at {off + got}")
        got += k
    return buf


@app.function(image=image, cpu=32, memory=98304, ephemeral_disk=512 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=6 * 3600)
def build(acts_dir: str, out_name: str, n_rows: int, p_lo: int = 256, p_hi: int = 511, w_lo: int = 16, w_hi: int = 64,
          max_per_doc: int = 8, seed: int = 11, threads: int = 48, chunk: int = 100_000, mu_path: str = MU_DEFAULT,
          family: str = "realact_long", overwrite: bool = False):
    import json, random, shutil, time
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np

    vol.reload()
    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json") and not overwrite:
        raise RuntimeError(f"{out} already built (overwrite=False)")
    for p in (f"{acts_dir}/acts.f16", f"{acts_dir}/toks.i32", f"{acts_dir}/meta.json", mu_path):
        assert os.path.exists(p), f"missing input {p}"
    meta = json.load(open(f"{acts_dir}/meta.json"))
    assert meta.get("acts_complete", True), f"{acts_dir}/meta.json says acts_complete=false — run finalize first"
    NS, T = int(meta["n_seq"]), int(meta["seq_len"])
    assert 0 <= p_lo <= p_hi < T and 1 <= w_lo <= w_hi <= T, (p_lo, p_hi, w_lo, w_hi, T)
    asize = os.path.getsize(f"{acts_dir}/acts.f16"); assert asize == NS * T * D_MODEL * 2, (asize, NS, T)
    os.environ["HF_HOME"] = "/data/hf_cache"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    mu = np.load(mu_path).astype(np.float32); assert mu.shape == (D_MODEL,)
    toks = np.fromfile(f"{acts_dir}/toks.i32", dtype=np.int32).reshape(NS, T)
    afd = os.open(f"{acts_dir}/acts.f16", os.O_RDONLY)
    nrng = np.random.default_rng(seed); wrng = random.Random(seed)
    T0 = time.time()

    def log(m):
        print(f"[ufw-long +{time.time() - T0:5.0f}s] {m}", flush=True)

    def read_act(s, p, dst):
        dst[:] = np.frombuffer(_pread_full(afd, D_MODEL * 2, ((s * T) + p) * D_MODEL * 2), np.float16)

    log(f"store {acts_dir}: {NS} x {T} ({meta.get('dataset')}, mode {meta.get('mode')}, hf_stream {meta.get('hf_stream')}) -> {out}: "
        f"{n_rows} rows, p in [{p_lo},{p_hi}], W in [{w_lo},{w_hi}], <= {max_per_doc}/seq, mu {mu_path}")
    # ---- norm-filter threshold: 10x median ||act - mu|| over a presample (suite hygiene) ----
    ps_s = nrng.integers(0, NS, NORM_PRESAMPLE); ps_p = nrng.integers(p_lo, p_hi + 1, NORM_PRESAMPLE)
    ps_raw = np.empty((NORM_PRESAMPLE, D_MODEL), np.float16)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda k: read_act(int(ps_s[k]), int(ps_p[k]), ps_raw[k]), range(NORM_PRESAMPLE), chunksize=64))
    med = float(np.median(np.linalg.norm(ps_raw.astype(np.float32) - mu, axis=1))); thr = NORM_FILTER_MULT * med
    log(f"norm filter: median {med:.1f} thr {thr:.1f}")

    # ---- candidates: seeded sample WITHOUT replacement over (seq, pos), streamed in chunks ----
    n_pos = p_hi - p_lo + 1
    n_cand = min(int(n_rows * 1.6) + 4096, NS * n_pos)
    assert NS * max_per_doc >= n_rows, f"store too small: {NS} seqs x {max_per_doc}/seq < {n_rows}"
    flat = nrng.choice(NS * n_pos, size=n_cand, replace=False)
    cand_s = (flat // n_pos).astype(np.int64); cand_p = (p_lo + flat % n_pos).astype(np.int64); del flat
    os.makedirs("/root/bank", exist_ok=True)
    kept_vecs = np.memmap("/root/bank/kept.f32", np.float32, "w+", shape=(n_rows, D_MODEL))
    recs, per_doc = [], np.zeros(NS, np.int32)
    drop_norm = drop_txt = drop_cap = 0; kept = 0
    raw = np.empty((chunk, D_MODEL), np.float16)
    for c0 in range(0, n_cand, chunk):
        if kept >= n_rows:
            break
        c1 = min(c0 + chunk, n_cand); m = c1 - c0
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(lambda k: read_act(int(cand_s[c0 + k]), int(cand_p[c0 + k]), raw[k]), range(m), chunksize=256))
        d = raw[:m].astype(np.float32) - mu
        norms = np.linalg.norm(d, axis=1)
        for k in range(m):
            if kept >= n_rows:
                break
            if not (1e-6 < norms[k] <= thr):
                drop_norm += 1; continue
            s, p = int(cand_s[c0 + k]), int(cand_p[c0 + k])
            if per_doc[s] >= max_per_doc:
                drop_cap += 1; continue
            W = wrng.randint(w_lo, w_hi)
            start_tok, end = max(0, p - W + 1), p + 1                     # window ENDS at the firing token
            ids = toks[s, start_tok:end].tolist()
            txt = tok.decode(ids)
            if len(txt.strip()) < 3:
                drop_txt += 1; continue
            kept_vecs[kept] = d[k] / norms[k]
            recs.append({"family": family, "target_text": txt, "seq": s, "pos": p, "start": start_tok, "n_tok": len(ids),
                         "W": W, "ctx_len": p + 1, "fire_from_end": 0, "src_store": acts_dir})
            per_doc[s] += 1; kept += 1
        log(f"{kept}/{n_rows} (cands {c1}/{n_cand}; drops norm={drop_norm} txt={drop_txt} cap={drop_cap})")
    assert kept == n_rows, f"quota missed: {kept}/{n_rows} (drops norm={drop_norm} txt={drop_txt} cap={drop_cap})"

    # ---- seeded shuffle -> vecs.f32 in final order, records line i == vec_idx i ----
    order = np.random.default_rng(seed + 1).permutation(n_rows)
    vecs = np.memmap("/root/bank/vecs.f32", np.float32, "w+", shape=(n_rows, D_MODEL))
    for i0 in range(0, n_rows, 50_000):
        idx = order[i0:i0 + 50_000]
        vecs[i0:i0 + len(idx)] = kept_vecs[idx]
    vecs.flush(); del vecs
    with open("/root/bank/records.jsonl", "w") as rf:
        for i, j in enumerate(order.tolist()):
            r = dict(recs[j]); r["vec_idx"] = i
            rf.write(json.dumps(r, ensure_ascii=False) + "\n")
    mm = np.memmap("/root/bank/vecs.f32", np.float32, "r", shape=(n_rows, D_MODEL))
    chk = np.linalg.norm(mm[np.sort(np.random.default_rng(0).choice(n_rows, min(4096, n_rows), replace=False))], axis=1)
    assert 0.99 < chk.min() and chk.max() < 1.01, (chk.min(), chk.max())
    n_tok = np.array([r["n_tok"] for r in recs])
    stats = {"kind": f"{family}: DEEP-position window harvest of a 512-token layer-42 store (p~U[{p_lo},{p_hi}], target = W~U[{w_lo},{w_hi}]-token "
                     "window ENDING at p, direction unit(act[s,p]-mu), 10x-median norm filter)",
             "model": MODEL, "layer": READ_LAYER, "d": D_MODEL, "n_examples": n_rows, "n_vecs": n_rows, "families": {family: n_rows},
             "acts": acts_dir, "store_meta": {k: meta.get(k) for k in ("n_seq", "seq_len", "dataset", "seed", "mode", "hf_stream", "n_tokens")},
             "dataset": meta.get("dataset"), "mu_path": mu_path, "recipe": {"p_lo": p_lo, "p_hi": p_hi, "w_lo": w_lo, "w_hi": w_hi, "max_per_doc": max_per_doc,
                                                                            "seqs_used": int((per_doc > 0).sum()), "n_seq": NS},
             "norm_filter": {"median": med, "thr": thr, "dropped": drop_norm}, "drop_txt": drop_txt, "drop_cap": drop_cap,
             "n_tok": {"min": int(n_tok.min()), "mean": float(n_tok.mean()), "max": int(n_tok.max())}, "seed": seed,
             "layout": "seeded shuffle; records.jsonl line i == vec_idx i", "files": {"vecs.f32": f"float32 [{n_rows},{D_MODEL}]"},
             "created": time.time(), "wall_s": time.time() - T0}
    os.makedirs(out, exist_ok=True)
    for fn in ("vecs.f32", "records.jsonl"):
        shutil.copy(f"/root/bank/{fn}", f"{out}/{fn}")
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1); json.dump(stats, open(f"{out}/meta.json", "w"), indent=1)
    vol.commit()
    assert os.path.getsize(f"{out}/vecs.f32") == n_rows * D_MODEL * 4
    log(f"DONE -> {out}: {n_rows} rows ({n_rows * D_MODEL * 4 / 2**30:.1f} GB)")
    return {k: stats[k] for k in ("n_examples", "families", "norm_filter", "drop_cap", "n_tok", "wall_s")}


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def peek(out_name: str, n: int = 3):
    import json
    vol.reload()
    out = f"/data/banks/{out_name}"
    st = json.load(open(f"{out}/build_stats.json"))
    with open(f"{out}/records.jsonl") as f:
        recs = [json.loads(next(f)) for _ in range(n)]
    return {"n_examples": st["n_examples"], "families": st["families"], "sample": recs}
