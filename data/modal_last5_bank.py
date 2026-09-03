"""Modal app: build the peak-in-last-5 SFT bank at /data/banks/last5_rp on `maemm-data`.

Two TOKEN-MATCHED families, 250k examples each (targets end where the direction fires):

  realact  sampled (s, p) from /data/acts27b (train = first 95% of sequences, rows
           0..ceil(0.95*n_seq)-1; the last 5% stay held out; p in [16, 511]), the suite's
           10x-median norm filter on ||act[s,p] - whiten_mu|| (build_big_sft_bank hygiene),
           direction = unit(act[s,p] - whiten_mu), target = decode(toks[s, p-W+1 : p+1]),
           W ~ U[16,64] clipped at the window start — the anchor p is the LAST token.

  probes   METHOD A: reuse the ~250k cluster/probe rows of /data/pool_rl_mix (vec +
           target_text = the probe's top span). One batched 27B forward per span
           (bank/last5_worker.py, mounted at /pmx/last5_worker.py) finds where the
           direction's per-token cosine peaks; causal prefix-truncation then re-anchors
           the peak into the last 5 tokens EXACTLY (activations of a prefix are
           unchanged), dropping only degenerate rows. No 250k-dirs x 25M-token cross
           product — 250k short forwards on 4xB200.

Deliverables under /data/banks/last5_rp/ (the exact modal_sft.py bank format):
    records.jsonl      one JSON/line: vec_idx (row into vecs.f32) + target_text (+family/meta)
    vecs.f32           raw LE float32 row-major [N, 5120]
    build_stats.json   counts + method notes
    verify.json        re-scored peak-in-last-5 hit rates per family (from final artifacts)

Launch (MODAL_PROFILE=safety-sahan):
    modal run modal_last5_bank.py::run_smoke              # 1xB200 tiny end-to-end validation
    modal run --detach modal_last5_bank.py::run_build     # the real 250k+250k build
    modal run modal_last5_bank.py::run_peek               # CPU inspection of the result

Needs Modal secret `maemm-hf` (HF_TOKEN). Model loads offline from /data/hf_cache.
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent   # repo root (this launcher lives one level down)

APP_NAME = "maemm-last5-bank"
app = modal.App(APP_NAME)

# identical pins to modal_acts27b.py / modal_sft.py (one environment across the suite)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.15.0",
        "accelerate==1.14.0",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    .add_local_file(REPO / "data" / "last5_worker.py", "/pmx/last5_worker.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

ACTS = "/data/acts27b"
POOL = "/data/pool_rl_mix"
NORM_FILTER_MULT = 10.0
NORM_PRESAMPLE = 8000
READ_THREADS = 32


def _pread_full(fd, n, off):
    buf = b""
    while len(buf) < n:
        chunk = __import__("os").pread(fd, n - len(buf), off + len(buf))
        assert chunk, f"short read at offset {off}"
        buf += chunk
    return buf


def _build(out_name: str, n_realact: int, n_probe: int, world: int, seed: int,
           verify_n: int, overwrite_smoke: bool = False):
    import json
    import os
    import random
    import shutil
    import subprocess
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL

    out = f"/data/{out_name}"
    if os.path.exists(f"{out}/build_stats.json"):
        if overwrite_smoke and "smoke" in out_name:
            shutil.rmtree(out)
        else:
            raise RuntimeError(f"{out}/build_stats.json exists — bank already built")
    for p in (f"{ACTS}/acts.f16", f"{ACTS}/toks.i32", f"{ACTS}/whiten_mu.npy",
              f"{ACTS}/meta.json", f"{POOL}/records.jsonl", f"{POOL}/vecs.f32"):
        assert os.path.exists(p), f"missing input {p}"

    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL)
    vol.commit()
    print(f"[modal] base model in cache ({time.time() - t0:.0f}s)", flush=True)

    meta = json.load(open(f"{ACTS}/meta.json"))
    NS, T, BOS = int(meta["n_seq"]), int(meta["seq_len"]), int(meta["bos_id"])
    n_train = int(np.ceil(NS * 0.95))          # rows 0..n_train-1; last 5% held out
    row_b = D_MODEL * 4

    # ---- stage B first (cheap): extract the cluster/probe rows of pool_rl_mix so the GPU
    # workers can start (model load ~10 min) while the driver does the realact CPU stage ----
    t0 = time.time()
    crecs = []
    with open(f"{POOL}/records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("family") == "cluster":
                crecs.append((int(r["vec_idx"]), r["target_text"]))
    assert len(crecs) >= n_probe, f"only {len(crecs)} cluster rows < n_probe {n_probe}"
    crecs = crecs[:n_probe]
    idxs = [v for v, _ in crecs]
    contig = idxs == list(range(idxs[0], idxs[0] + len(idxs)))
    with open(f"{POOL}/vecs.f32", "rb") as fin, open("/root/probe_dirs.f32", "wb") as fout:
        if contig:
            fin.seek(idxs[0] * row_b)
            left = len(idxs) * row_b
            while left:
                buf = fin.read(min(64 << 20, left))
                assert buf, "unexpected EOF in pool vecs"
                fout.write(buf)
                left -= len(buf)
        else:
            for v in idxs:
                fin.seek(v * row_b)
                fout.write(fin.read(row_b))
    with open("/root/probe_in.jsonl", "w") as fout:
        for i, (v, txt) in enumerate(crecs):
            fout.write(json.dumps({"i": i, "src_vec_idx": v, "text": txt}) + "\n")
    print(f"[modal] probe inputs staged: {n_probe} cluster rows "
          f"(contiguous={contig}, {time.time() - t0:.0f}s)", flush=True)

    # ---- stage C: probe-rescoring workers, one per GPU ----
    procs = []
    for r in range(world):
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(r), "PYTHONPATH": "/pmx/helpers",
                    "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1"})
        cmd = [sys.executable, "/pmx/last5_worker.py", "--mode", "probes",
               "--rank", str(r), "--world", str(world),
               "--in-jsonl", "/root/probe_in.jsonl", "--dirs", "/root/probe_dirs.f32",
               "--out", f"/root/probe_out_r{r}.jsonl", "--bos-id", str(BOS)]
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(2)

    # ---- stage A (concurrent with C): realact sampling over the acts memmap via pread ----
    mu = np.load(f"{ACTS}/whiten_mu.npy").astype(np.float32)
    toks = np.fromfile(f"{ACTS}/toks.i32", dtype=np.int32).reshape(NS, T)
    afd = os.open(f"{ACTS}/acts.f16", os.O_RDONLY)
    nrng = np.random.default_rng(seed)
    wrng = random.Random(seed)

    def read_act(s, p, dst):
        dst[:] = np.frombuffer(
            _pread_full(afd, D_MODEL * 2, ((s * T) + p) * D_MODEL * 2), np.float16)

    # norm-filter threshold: 10x median of ||act - mu|| over a presample (suite hygiene)
    t0 = time.time()
    ps_s = nrng.integers(0, n_train, NORM_PRESAMPLE)
    ps_p = nrng.integers(16, T, NORM_PRESAMPLE)
    ps_raw = np.empty((NORM_PRESAMPLE, D_MODEL), np.float16)
    with ThreadPoolExecutor(READ_THREADS) as ex:
        list(ex.map(lambda k: read_act(int(ps_s[k]), int(ps_p[k]), ps_raw[k]),
                    range(NORM_PRESAMPLE), chunksize=64))
    pnorm = np.linalg.norm(ps_raw.astype(np.float32) - mu, axis=1)
    med = float(np.median(pnorm))
    thr = NORM_FILTER_MULT * med
    print(f"[modal] realact norm filter: median {med:.1f} thr {thr:.1f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    n_cand = int(n_realact * 1.2) + 2048
    flat = nrng.choice(n_train * (T - 16), size=n_cand, replace=False)
    cand_s = (flat // (T - 16)).astype(np.int64)
    cand_p = (16 + flat % (T - 16)).astype(np.int64)
    t0 = time.time()
    raw = np.empty((n_cand, D_MODEL), np.float16)
    done = [0]

    def fill(k):
        read_act(int(cand_s[k]), int(cand_p[k]), raw[k])
        done[0] += 1
        if done[0] % 50000 == 0:
            print(f"[modal] realact reads {done[0]}/{n_cand} "
                  f"({done[0] / max(time.time() - t0, 1):.0f}/s)", flush=True)

    with ThreadPoolExecutor(READ_THREADS) as ex:
        list(ex.map(fill, range(n_cand), chunksize=256))
    print(f"[modal] realact candidate acts read ({time.time() - t0:.0f}s)", flush=True)

    norms = np.empty(n_cand, np.float32)
    for b in range(0, n_cand, 32768):
        d = raw[b:b + 32768].astype(np.float32) - mu
        norms[b:b + 32768] = np.linalg.norm(d, axis=1)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    realact_vecs = np.empty((n_realact, D_MODEL), np.float32)
    rrecs = []
    kept = drop_norm = drop_txt = 0
    t0 = time.time()
    for k in range(n_cand):
        if kept >= n_realact:
            break
        if not (1e-6 < norms[k] <= thr):
            drop_norm += 1
            continue
        s, p = int(cand_s[k]), int(cand_p[k])
        W = wrng.randint(16, 64)
        ids = toks[s, max(0, p - W + 1): p + 1].tolist()
        txt = tok.decode(ids)
        if len(txt.strip()) < 3:
            drop_txt += 1
            continue
        d = raw[k].astype(np.float32) - mu
        realact_vecs[kept] = d / norms[k]
        rrecs.append({"vec_idx": kept, "target_text": txt, "family": "realact",
                      "seq": s, "pos": p, "n_tok": len(ids)})
        kept += 1
        if kept % 50000 == 0:
            print(f"[modal] realact {kept}/{n_realact}", flush=True)
    assert kept == n_realact, f"realact quota missed: {kept} (drops norm={drop_norm} txt={drop_txt})"
    print(f"[modal] realact family done: {kept} (dropped norm={drop_norm} txt={drop_txt}, "
          f"{time.time() - t0:.0f}s)", flush=True)
    del raw

    # ---- join workers, merge probe outputs ----
    fails = [r for r, p in enumerate(procs) if p.wait() != 0]
    if fails:
        raise RuntimeError(f"probe worker rank(s) {fails} failed")
    pouts = {}
    for r in range(world):
        with open(f"/root/probe_out_r{r}.jsonl") as f:
            for line in f:
                o = json.loads(line)
                pouts[o["i"]] = o
    assert len(pouts) == n_probe, f"probe outputs {len(pouts)} != {n_probe}"

    # ---- stage D: assemble the bank locally, then copy onto the volume ----
    t0 = time.time()
    os.makedirs("/root/bank", exist_ok=True)
    pdirs = np.memmap("/root/probe_dirs.f32", np.float32, "r", shape=(n_probe, D_MODEL))
    pdrops, plens, n_trunc, kept_p = {}, [], 0, 0
    row = n_realact
    with open("/root/bank/vecs.f32", "wb") as vf, \
            open("/root/bank/records.jsonl", "w") as rf:
        realact_vecs.tofile(vf)
        for rr in rrecs:
            rf.write(json.dumps(rr) + "\n")
        for i in range(n_probe):
            o = pouts[i]
            if not o["keep"]:
                pdrops[o["reason"]] = pdrops.get(o["reason"], 0) + 1
                continue
            v = pdirs[i].astype(np.float32)
            nn = float(np.linalg.norm(v))
            if nn < 1e-6:
                pdrops["zero_dir"] = pdrops.get("zero_dir", 0) + 1
                continue
            (v / nn).tofile(vf)
            rf.write(json.dumps({
                "vec_idx": row, "target_text": o["target_text"], "family": "cluster",
                "src_vec_idx": o["src_vec_idx"], "peak_idx": o["peak_idx"],
                "n_tok": o["n_tok_final"], "truncated": o["truncated"],
                "cos_peak": o["cos_peak"]}) + "\n")
            plens.append(o["n_tok_final"])
            n_trunc += bool(o["truncated"])
            row += 1
            kept_p += 1
    n_total = row
    vsize = os.path.getsize("/root/bank/vecs.f32")
    assert vsize == n_total * row_b, f"vecs.f32 {vsize} B != {n_total} x {row_b}"

    plens_a = np.array(plens)
    stats = {
        "kind": "last5_rp: peak-in-last-5 SFT bank (realact + probes/cluster)",
        "method": "probes = METHOD A: re-score each pool_rl_mix cluster span standalone "
                  "([sink]+span fwd, uncentered per-token cos like train/rl.py reward), "
                  "causal prefix-truncate to [0..t*+2] when the peak isn't already in the "
                  "last 5 (prefix activations unchanged -> re-anchored peak exact); "
                  "realact = unit(act[s,p]-mu) with the span ending at anchor p",
        "n_examples": n_total,
        "families": {"realact": n_realact, "cluster": kept_p},
        "tokens": {"realact": int(sum(r["n_tok"] for r in rrecs)),
                   "cluster": int(plens_a.sum())},
        "probe_drops": pdrops,
        "probe_truncated": n_trunc,
        "probe_len_final_pct": {q: int(np.percentile(plens_a, q))
                                for q in (1, 10, 50, 90, 99)},
        "realact": {"train_rows": [0, n_train - 1], "held_out_rows": [n_train, NS - 1],
                    "p_range": [16, T - 1],
                    "window": "W~U[16,64] ending at p, clipped at seq start",
                    "norm_filter": {"median": med, "thr": thr, "dropped": drop_norm},
                    "text_dropped": drop_txt,
                    "dir": "unit(act[s,p] - whiten_mu)"},
        "scoring": "cos(h_t, v) UNCENTERED on [sink=bos]+span forward, pos 0 dropped "
                   "(matches train/rl.py reward metric)",
        "sources": {"acts": ACTS, "pool": POOL, "model": MODEL},
        "seed": seed, "world": world, "d_model": D_MODEL, "created": time.time(),
    }
    os.makedirs(out, exist_ok=True)
    for fn in ("vecs.f32", "records.jsonl"):
        shutil.copyfile(f"/root/bank/{fn}", f"{out}/{fn}")
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
    vol.commit()
    print(f"[modal] bank written -> {out}: {n_total} rows "
          f"({vsize / 1e9:.1f} GB vecs, {time.time() - t0:.0f}s)", flush=True)
    print(json.dumps(stats, indent=1), flush=True)

    # ---- stage E: verify from the FINAL artifacts (re-tokenize like the SFT harness) ----
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/pmx/helpers",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1"})
    rc = subprocess.call([sys.executable, "/pmx/last5_worker.py", "--mode", "verify",
                          "--bank", out, "--n", str(verify_n), "--seed", "0",
                          "--mu", f"{ACTS}/whiten_mu.npy", "--bos-id", str(BOS)], env=env)
    if rc != 0:
        raise RuntimeError(f"verify exited rc={rc} (bank itself is written at {out})")
    vol.commit()
    verify = json.load(open(f"{out}/verify.json"))
    print("[modal] VERIFY:", json.dumps(verify["families"], indent=1), flush=True)
    print(f"[modal] COMPLETE -> {out} ({n_total} rows: {n_realact} realact + "
          f"{kept_p} cluster; probe drops {pdrops})", flush=True)


@app.function(
    image=image,
    gpu="B200:4",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=86400,
    cpu=16,
    memory=98304,
)
def build(out_name: str = "banks/last5_rp", n_realact: int = 250000,
          n_probe: int = 250000, seed: int = 0, verify_n: int = 1024):
    _build(out_name, n_realact, n_probe, world=4, seed=seed, verify_n=verify_n)


@app.function(
    image=image,
    gpu="B200:1",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=14400,
    cpu=8,
    memory=32768,
)
def smoke(out_name: str = "banks/last5_rp_smoke", n_realact: int = 512,
          n_probe: int = 512, seed: int = 0, verify_n: int = 128):
    """1xB200 end-to-end validation of the exact build+verify path on a tiny target."""
    _build(out_name, n_realact, n_probe, world=1, seed=seed, verify_n=verify_n,
           overwrite_smoke=True)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=1800,
    cpu=4,
)
def peek(out_name: str = "banks/last5_rp"):
    """CPU-only sanity read of a finished bank: stats + verify + sample records + sizes."""
    import json
    import os
    import sys

    import numpy as np

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL

    out = f"/data/{out_name}"
    stats = json.load(open(f"{out}/build_stats.json"))
    print(json.dumps(stats, indent=1), flush=True)
    if os.path.exists(f"{out}/verify.json"):
        print(json.dumps(json.load(open(f"{out}/verify.json")), indent=1), flush=True)
    n_lines = sum(1 for _ in open(f"{out}/records.jsonl"))
    vsize = os.path.getsize(f"{out}/vecs.f32")
    print(f"records lines={n_lines}  vecs.f32={vsize} B = {vsize / (D_MODEL * 4):.0f} rows",
          flush=True)
    recs = [json.loads(l) for l in open(f"{out}/records.jsonl")]
    fd = os.open(f"{out}/vecs.f32", os.O_RDONLY)
    for j in (0, len(recs) // 2, len(recs) - 1):
        r = recs[j]
        v = np.frombuffer(_pread_full(fd, D_MODEL * 4, r["vec_idx"] * D_MODEL * 4),
                          np.float32)
        print(f"--- row {j} fam={r['family']} |v|={np.linalg.norm(v):.4f} "
              f"text={r['target_text'][:100]!r}", flush=True)


@app.local_entrypoint()
def run_build(out_name: str = "banks/last5_rp", n_realact: int = 250000,
              n_probe: int = 250000, seed: int = 0, verify_n: int = 1024):
    build.remote(out_name=out_name, n_realact=n_realact, n_probe=n_probe, seed=seed,
                 verify_n=verify_n)


@app.local_entrypoint()
def run_smoke(out_name: str = "banks/last5_rp_smoke", n_realact: int = 512,
              n_probe: int = 512):
    smoke.remote(out_name=out_name, n_realact=n_realact, n_probe=n_probe)


@app.local_entrypoint()
def run_peek(out_name: str = "banks/last5_rp"):
    peek.remote(out_name=out_name)
