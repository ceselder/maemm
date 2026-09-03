"""Modal app: build the BIG realact+probes SFT bank at /data/banks/big_rp (CPU only, no GPU).

The scaled SFT-only run of the ARB doc's winning 50/50 real-acts+probes recipe. Two families:
  realact  HARVESTED AS IN THE ARB DOC: pick a random FineFineWeb document (train = first 95% of
           sequences; the last 5% are the eval hold-out) and an index p in [14, 91] of its prefill;
           direction = unit(act[s,p] - whiten_mu) (10x-median norm filter); target = the ACTUAL text
           of that document from its start through token p plus 1-4 tokens after (for diversity),
           i.e. decode(toks[s, 0 : p+1+extra]), lengths 16-96. Standalone re-encoding of the target
           reproduces the exact prefix context that produced the activation, and the firing token
           sits inside the last 5 tokens (the last-5 reward window). <= --max-per-doc examples per
           document so the bank isn't dominated by near-identical prefixes of the same documents.
           Streamed in chunks into a disk memmap.
  probes   the 249,296 cluster/probe rows of /data/banks/last5_rp (already peak-re-anchored into
           the last 5 tokens by modal_last5_bank.py's GPU stage) — vectors copied once, records
           written --probe-dup times (= that many probe epochs; only 250k probes exist on the volume).
Deliverables under /data/banks/big_rp/ (the exact modal_sft.py bank format):
    records.jsonl   one JSON/line: vec_idx (row into vecs.f32) + target_text + family (+meta), SHUFFLED
    vecs.f32        raw LE float32 row-major [N, 5120], unit rows
    build_stats.json

Run (profile safety-sahan):
    modal run modal_big_bank.py::build                      # ~1 h CPU
    modal run modal_big_bank.py::peek                       # verify
Then:  modal run modal_sft.py::launch --run-name big_rp --data-dir /data/banks/big_rp --n-ckpts 20
"""
import modal

app = modal.App("maemm-big-bank")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy<2.3", "transformers==5.15.0", "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet")
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

MODEL, D_MODEL = "Qwen/Qwen3.6-27B", 5120          # == mxf.config (kept import-free: no torch here)
ACTS = "/data/acts27b"
SRC_BANK = "/data/banks/last5_rp"                    # probes source (re-anchored cluster rows)
OUT_DEFAULT = "banks/big_rp"
NORM_FILTER_MULT = 10.0
NORM_PRESAMPLE = 20_000


def _pread_full(fd, n, off):
    import os
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise IOError(f"short read at {off + got}")
        got += k
    return buf


@app.function(image=image, cpu=32, memory=98304, ephemeral_disk=512 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=8 * 3600)
def build(out_name: str = OUT_DEFAULT, n_realact: int = 600_000, probe_dup: int = 2, seed: int = 1,
          threads: int = 48, chunk: int = 100_000, p_lo: int = 14, p_hi: int = 91, max_per_doc: int = 14,
          n_window: int = 0, win_p_lo: int = 16, win_p_hi: int = 511, w_lo: int = 16, w_hi: int = 64):
    # n_window > 0: that many of the n_realact real-acts use the eval-matched WINDOW harvest (p~U[win_p_lo,win_p_hi],
    # target = a W~U[w_lo,w_hi]-token window ENDING at p, the last5_rp recipe); the rest use the ARB-doc PREFIX harvest.
    import json, os, random, shutil, time
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np

    vol.reload()
    out = f"/data/{out_name}"
    assert not os.path.exists(f"{out}/build_stats.json"), f"{out} already built"
    for p in (f"{ACTS}/acts.f16", f"{ACTS}/toks.i32", f"{ACTS}/whiten_mu.npy", f"{ACTS}/meta.json",
              f"{SRC_BANK}/records.jsonl", f"{SRC_BANK}/vecs.f32", f"{SRC_BANK}/build_stats.json"):
        assert os.path.exists(p), f"missing input {p}"
    os.environ["HF_HOME"] = "/data/hf_cache"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    meta = json.load(open(f"{ACTS}/meta.json"))
    NS, T = int(meta["n_seq"]), int(meta["seq_len"])
    n_train = int(np.ceil(NS * 0.95))               # rows 0..n_train-1 train; last 5% held out (== eval)
    mu = np.load(f"{ACTS}/whiten_mu.npy").astype(np.float32)
    toks = np.fromfile(f"{ACTS}/toks.i32", dtype=np.int32).reshape(NS, T)
    afd = os.open(f"{ACTS}/acts.f16", os.O_RDONLY)
    nrng = np.random.default_rng(seed)
    wrng = random.Random(seed)

    def read_act(s, p, dst):
        dst[:] = np.frombuffer(_pread_full(afd, D_MODEL * 2, ((s * T) + p) * D_MODEL * 2), np.float16)

    # ---- probes: which rows of last5_rp are cluster/probe ----
    precs = []
    with open(f"{SRC_BANK}/records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("family") == "cluster":
                precs.append(r)
    n_probe = len(precs)
    print(f"[bank] probes available in {SRC_BANK}: {n_probe}", flush=True)

    # ---- output memmap on local NVMe: realact rows first, then the probe rows ----
    n_total = n_realact + n_probe
    os.makedirs("/root/bank", exist_ok=True)
    vecs = np.memmap("/root/bank/vecs.f32", np.float32, "w+", shape=(n_total, D_MODEL))

    # ---- norm-filter threshold: 10x median ||act - mu|| over a presample (suite hygiene) ----
    t0 = time.time()
    ps_s = nrng.integers(0, n_train, NORM_PRESAMPLE); ps_p = nrng.integers(16, T, NORM_PRESAMPLE)
    ps_raw = np.empty((NORM_PRESAMPLE, D_MODEL), np.float16)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda k: read_act(int(ps_s[k]), int(ps_p[k]), ps_raw[k]), range(NORM_PRESAMPLE), chunksize=64))
    med = float(np.median(np.linalg.norm(ps_raw.astype(np.float32) - mu, axis=1)))
    thr = NORM_FILTER_MULT * med
    print(f"[bank] norm filter: median {med:.1f} thr {thr:.1f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- realact: up to two harvests, each streamed in chunks; a shared <= max_per_doc cap ----
    #   prefix: p~U[p_lo,p_hi], target = the doc's text from its start through p + 1-4 tokens after (ARB doc)
    #   window: p~U[win_p_lo,win_p_hi], target = W~U[w_lo,w_hi]-token window ENDING at p (eval positions are
    #           uniform over the document, so this half keeps mid/late-position coverage)
    n_prefix = n_realact - n_window
    assert n_prefix >= 0
    recs = []
    per_doc = np.zeros(n_train, np.int32)
    drop_norm = drop_txt = drop_cap = 0
    kept_total = 0
    raw = np.empty((chunk, D_MODEL), np.float16)
    for mode, n_target, lo, hi in (("prefix", n_prefix, p_lo, p_hi), ("window", n_window, win_p_lo, win_p_hi)):
        if n_target <= 0:
            continue
        n_pos = hi - lo + 1
        n_cand = min(int(n_target * 1.6) + 4096, n_train * n_pos)
        flat = nrng.choice(n_train * n_pos, size=n_cand, replace=False)
        cand_s = (flat // n_pos).astype(np.int64); cand_p = (lo + flat % n_pos).astype(np.int64)
        del flat
        kept = 0
        t0 = time.time()
        for c0 in range(0, n_cand, chunk):
            if kept >= n_target:
                break
            c1 = min(c0 + chunk, n_cand)
            m = c1 - c0
            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(lambda k: read_act(int(cand_s[c0 + k]), int(cand_p[c0 + k]), raw[k]), range(m), chunksize=256))
            d = raw[:m].astype(np.float32) - mu
            norms = np.linalg.norm(d, axis=1)
            for k in range(m):
                if kept >= n_target:
                    break
                if not (1e-6 < norms[k] <= thr):
                    drop_norm += 1
                    continue
                s, p = int(cand_s[c0 + k]), int(cand_p[c0 + k])
                if per_doc[s] >= max_per_doc:
                    drop_cap += 1
                    continue
                if mode == "prefix":
                    extra = wrng.randint(1, 4)                   # 1-4 tokens AFTER the firing token
                    start_tok, end = 0, min(p + 1 + extra, T)
                else:
                    extra = 0
                    W = wrng.randint(w_lo, w_hi)
                    start_tok, end = max(0, p - W + 1), p + 1     # window ends AT the firing token
                ids = toks[s, start_tok:end].tolist()
                txt = tok.decode(ids)
                if len(txt.strip()) < 3:
                    drop_txt += 1
                    continue
                vecs[kept_total] = d[k] / norms[k]
                recs.append({"vec_idx": kept_total, "target_text": txt, "family": "realact", "harvest": mode,
                             "seq": s, "pos": p, "start": start_tok, "extra": extra, "n_tok": len(ids),
                             "fire_from_end": end - 1 - p})
                per_doc[s] += 1
                kept += 1
                kept_total += 1
            print(f"[bank] realact/{mode} {kept}/{n_target} (cands {c1}/{n_cand}, {kept / max(time.time() - t0, 1):.0f}/s, "
                  f"drops norm={drop_norm} txt={drop_txt} cap={drop_cap})", flush=True)
        assert kept == n_target, f"realact/{mode} quota missed: {kept} (drops norm={drop_norm} txt={drop_txt} cap={drop_cap})"
    assert kept_total == n_realact
    n_realact_tok = sum(r["n_tok"] for r in recs)

    # ---- probes: copy vectors once (rows n_realact..), records x probe_dup ----
    t0 = time.time()
    pfd = os.open(f"{SRC_BANK}/vecs.f32", os.O_RDONLY)
    row_b = D_MODEL * 4

    def copy_probe(j):
        vecs[n_realact + j] = np.frombuffer(_pread_full(pfd, row_b, precs[j]["vec_idx"] * row_b), np.float32)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(copy_probe, range(n_probe), chunksize=256))
    pn = np.linalg.norm(vecs[n_realact:n_realact + 2048], axis=1)
    assert np.all(np.abs(pn - 1) < 1e-2), f"probe rows not unit-norm: {pn[:5]}"
    for dup in range(probe_dup):
        for j, r in enumerate(precs):
            recs.append({"vec_idx": n_realact + j, "target_text": r["target_text"], "family": "cluster",
                         "src_vec_idx": r["vec_idx"], "dup": dup})
    n_probe_tok = sum(len(tok.encode(r["target_text"], add_special_tokens=False)) for r in precs[:5000]) / 5000
    print(f"[bank] probes copied: {n_probe} vecs, {n_probe * probe_dup} records ({time.time() - t0:.0f}s)", flush=True)

    # ---- shuffle records, write, stats, publish ----
    random.Random(seed).shuffle(recs)
    with open("/root/bank/records.jsonl", "w") as rf:
        for r in recs:
            rf.write(json.dumps(r) + "\n")
    vecs.flush(); del vecs
    stats = {"kind": "big_rp: ARB-doc harvest — realact = doc prefix through index p in [p_lo,p_hi] + 1-4 tokens after (lengths 16-96), <= max_per_doc per document; probes = last5_rp re-anchored cluster rows x dup",
             "harvest": {"prefix": n_prefix, "window": n_window, "p_lo": p_lo, "p_hi": p_hi, "extra_after": [1, 4],
                         "win_p_lo": win_p_lo, "win_p_hi": win_p_hi, "w_range": [w_lo, w_hi], "max_per_doc": max_per_doc, "drop_cap": drop_cap,
                         "docs_used": int((per_doc > 0).sum()), "n_train_docs": int(n_train)},
             "n_examples": len(recs), "n_vecs": n_total,
             "families": {"realact": n_realact, "cluster": n_probe * probe_dup},
             "probe_dup": probe_dup, "probe_src": SRC_BANK, "probe_unique": n_probe,
             "tokens_est": {"realact": n_realact_tok, "cluster": int(n_probe_tok * n_probe * probe_dup)},
             "realact": {"train_rows": [0, n_train - 1], "held_out_rows": [n_train, NS - 1], "p_range": [p_lo, p_hi],
                         "window": "doc prefix [0, p+extra], extra~U[1,4]", "seed": seed,
                         "norm_filter": {"median": med, "thr": thr, "dropped": drop_norm}, "drop_txt": drop_txt},
             "acts": ACTS, "d_model": D_MODEL, "model": MODEL}
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for fn in ("vecs.f32", "records.jsonl"):
        shutil.copy(f"/root/bank/{fn}", f"{out}/{fn}")
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
    vol.commit()
    vsize = os.path.getsize(f"{out}/vecs.f32")
    assert vsize == n_total * row_b, f"vecs.f32 {vsize} B != {n_total} x {row_b}"
    print(f"[bank] DONE -> {out}: {len(recs)} records, {n_total} vecs ({vsize / 2**30:.1f} GB), "
          f"published in {time.time() - t0:.0f}s", flush=True)
    return stats


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def peek(out_name: str = OUT_DEFAULT, n: int = 3):
    import json, os
    import numpy as np
    vol.reload()
    out = f"/data/{out_name}"
    st = json.load(open(f"{out}/build_stats.json"))
    print(json.dumps({k: v for k, v in st.items() if k in ("n_examples", "n_vecs", "families", "tokens_est")}, indent=1))
    recs = [json.loads(l) for _, l in zip(range(n * 40), open(f"{out}/records.jsonl"))]
    vecs = np.memmap(f"{out}/vecs.f32", np.float32, "r", shape=(st["n_vecs"], D_MODEL))
    for fam in ("realact", "cluster"):
        for r in [r for r in recs if r["family"] == fam][:n]:
            print(f"[{fam}] |v|={np.linalg.norm(vecs[r['vec_idx']]):.3f} :: {r['target_text'][:110]!r}")
    return st["n_examples"]
