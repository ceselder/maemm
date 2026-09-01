"""Modal app (CPU-only): build the last-5 RL direction pool at /data/pool_rl_last5 on `maemm-data`.

100k directions, 4 families ~25k each, for the peak-in-last-5 RL stage (modal_rl_last5.py):

  realact / cluster / realact_long   sampled BY FAMILY from /data/pool_rl_mix (750k x 5120,
                                     records.jsonl {vec_idx, target_text, family[, pos]});
                                     records carried over (vec_idx re-based, src_vec_idx kept).
  sae                                25k features sampled from /data/sae/ae.pt: unit ENCODER
                                     columns unit(W_enc[:,f]) (mxf.sae.enc_dirs — the suite's
                                     conditioning/reward-dir convention; the feature FIRES on
                                     the encoder projection, so this is the direction to invert.
                                     build() first wrote decoder rows; fix_sae() rewrote the
                                     block in place — same feature ids). records carry the id
                                     as "feature" (the build_universal_bank.py convention).

Deliverables under /data/pool_rl_last5/ (exact pool_rl_mix / train/rl.py format):
    vecs.f32           raw LE float32 row-major [100000, 5120]
    records.jsonl      one JSON/line: vec_idx (row into vecs.f32) + family (+ per-family meta)
    build_stats.json   counts + sources; carries "n_examples" (train/rl.py reads this key)

Launch (MODAL_PROFILE=safety-sahan):
    modal run modal_pool_last5.py::run_inspect    # CPU peek at the inputs (schema + ae.pt keys)
    modal run modal_pool_last5.py::run_build      # the real 100k build (~minutes, CPU)
    modal run modal_pool_last5.py::run_peek       # verify the finished pool from final artifacts
"""

from pathlib import Path

import modal

REPO = Path(__file__).parent

APP_NAME = "maemm-pool-last5"
app = modal.App(APP_NAME)

# CPU-only: torch cpu wheel (ae.pt load) + numpy. mxf mounted for the canonical SAE loader.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("numpy==2.4.6", "huggingface_hub==1.27.0")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

POOL = "/data/pool_rl_mix"
SAE_PT = "/data/sae/ae.pt"
OUT = "/data/pool_rl_last5"
POOL_FAMILIES = ("realact", "cluster", "realact_long")
N_PER_FAMILY = 25_000
READ_THREADS = 32


def _pread_full(fd, n, off):
    import os

    buf = b""
    while len(buf) < n:
        chunk = os.pread(fd, n - len(buf), off + len(buf))
        assert chunk, f"short read at offset {off}"
        buf += chunk
    return buf


@app.function(image=image, volumes={"/data": vol}, timeout=3600, cpu=4, memory=16384)
def inspect_inputs():
    """CPU peek: pool_rl_mix schema + family counts, ae.pt keys/shapes. No writes."""
    import json
    import os
    import sys

    import torch

    sys.path.insert(0, "/pmx/helpers")

    print("== pool_rl_mix build_stats.json ==", flush=True)
    print(json.dumps(json.load(open(f"{POOL}/build_stats.json")), indent=1), flush=True)
    print(f"vecs.f32 size = {os.path.getsize(f'{POOL}/vecs.f32')} B", flush=True)

    fam_counts, first_by_fam = {}, {}
    with open(f"{POOL}/records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            fam = r.get("family")
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            if fam not in first_by_fam:
                first_by_fam[fam] = r
    print(f"records families: {fam_counts}", flush=True)
    for fam, r in first_by_fam.items():
        r = dict(r)
        if "target_text" in r:
            r["target_text"] = r["target_text"][:80]
        print(f"  first {fam}: {json.dumps(r)}", flush=True)

    print("== sae/ae.pt ==", flush=True)
    params = torch.load(SAE_PT, map_location="cpu", weights_only=False)
    if hasattr(params, "items"):
        for k, v in params.items():
            print(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else type(v)} "
                  f"{getattr(v, 'dtype', '')}", flush=True)
    else:
        print(f"  (non-dict payload: {type(params)})", flush=True)
    from mxf.sae import load_sae
    sae = load_sae(path=SAE_PT, device="cpu", dtype=torch.float32)
    print(f"  load_sae OK: d_in={sae.d_in} d_sae={sae.d_sae} "
          f"W_dec {tuple(sae.W_dec.shape)} row-norm mean "
          f"{sae.W_dec.norm(dim=1).mean().item():.6f}", flush=True)


@app.function(image=image, volumes={"/data": vol}, timeout=14400, cpu=16, memory=49152)
def build(seed: int = 0):
    """Assemble /data/pool_rl_last5: 25k realact + 25k cluster + 25k realact_long (sampled by
    family from pool_rl_mix) + 25k unit SAE decoder directions. Verifies from the written files."""
    import json
    import os
    import shutil
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL
    from mxf.sae import load_sae

    if os.path.exists(f"{OUT}/build_stats.json"):
        raise RuntimeError(f"{OUT}/build_stats.json exists — pool already built")
    for p in (f"{POOL}/records.jsonl", f"{POOL}/vecs.f32", SAE_PT):
        assert os.path.exists(p), f"missing input {p}"
    row_b = D_MODEL * 4
    rng = np.random.default_rng(seed)

    # ---- pass 1: line number -> family over pool_rl_mix records ----
    t0 = time.time()
    fam_lines = {f: [] for f in POOL_FAMILIES}
    n_lines = 0
    with open(f"{POOL}/records.jsonl") as f:
        for i, line in enumerate(f):
            fam = json.loads(line).get("family")
            if fam in fam_lines:
                fam_lines[fam].append(i)
            n_lines += 1
    print(f"[pool] scanned {n_lines} records: "
          f"{ {f: len(v) for f, v in fam_lines.items()} } ({time.time() - t0:.0f}s)", flush=True)

    # ---- sample N_PER_FAMILY line numbers per family (sorted: keeps identical-direction
    # duplicate rows adjacent + memmap/pread-friendly source reads) ----
    picked = {}
    for fam in POOL_FAMILIES:
        avail = fam_lines[fam]
        assert len(avail) >= N_PER_FAMILY, f"{fam}: only {len(avail)} rows < {N_PER_FAMILY}"
        sel = rng.choice(len(avail), size=N_PER_FAMILY, replace=False)
        picked[fam] = set(int(avail[j]) for j in sel)
    del fam_lines

    # ---- pass 2: collect the sampled records, family-ordered ----
    t0 = time.time()
    sampled = {f: [] for f in POOL_FAMILIES}
    with open(f"{POOL}/records.jsonl") as f:
        for i, line in enumerate(f):
            for fam in POOL_FAMILIES:
                if i in picked[fam]:
                    sampled[fam].append(json.loads(line))
                    break
    for fam in POOL_FAMILIES:
        assert len(sampled[fam]) == N_PER_FAMILY, (fam, len(sampled[fam]))
        assert all(r["family"] == fam for r in sampled[fam])
    print(f"[pool] sampled records collected ({time.time() - t0:.0f}s)", flush=True)

    # ---- gather the source vectors (pread over the volume file, sorted src order) ----
    t0 = time.time()
    order = [r for fam in POOL_FAMILIES for r in sampled[fam]]   # new row i = order[i]
    src_idx = [int(r["vec_idx"]) for r in order]
    pool_vecs = np.empty((len(order), D_MODEL), np.float32)
    pfd = os.open(f"{POOL}/vecs.f32", os.O_RDONLY)

    def read_row(k):
        pool_vecs[k] = np.frombuffer(_pread_full(pfd, row_b, src_idx[k] * row_b), np.float32)

    with ThreadPoolExecutor(READ_THREADS) as ex:
        list(ex.map(read_row, range(len(order)), chunksize=256))
    os.close(pfd)
    nrm = np.linalg.norm(pool_vecs, axis=1)
    assert np.isfinite(pool_vecs).all() and (nrm > 1e-6).all(), "bad source vectors"
    print(f"[pool] {len(order)} source rows read; |v| mean {nrm.mean():.4f} "
          f"min {nrm.min():.4f} max {nrm.max():.4f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- SAE family: unit-normalized DECODER rows (load_sae orientation: W_dec [F, d]) ----
    t0 = time.time()
    sae = load_sae(path=SAE_PT, device="cpu", dtype=torch.float32)
    F_total = sae.d_sae
    feats = np.sort(rng.choice(F_total, size=N_PER_FAMILY, replace=False))
    sae_vecs = torch.nn.functional.normalize(
        sae.W_dec[torch.as_tensor(feats.copy(), dtype=torch.long)], dim=-1).numpy().astype(np.float32)
    snrm = np.linalg.norm(sae_vecs, axis=1)
    assert np.abs(snrm - 1.0).max() < 1e-4, f"sae dirs not unit: max dev {np.abs(snrm - 1.0).max()}"
    print(f"[pool] {N_PER_FAMILY} SAE decoder dirs of F={F_total} sampled+normalized "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- write locally, then copy onto the volume ----
    t0 = time.time()
    os.makedirs("/root/pool", exist_ok=True)
    n_total = len(order) + N_PER_FAMILY
    with open("/root/pool/vecs.f32", "wb") as vf, open("/root/pool/records.jsonl", "w") as rf:
        pool_vecs.tofile(vf)
        for row, r in enumerate(order):
            out_r = dict(r)
            out_r["src_vec_idx"] = int(r["vec_idx"])   # provenance: row in pool_rl_mix
            out_r["vec_idx"] = row                     # row in THIS pool's vecs.f32
            rf.write(json.dumps(out_r) + "\n")
        sae_vecs.tofile(vf)
        for j, fid in enumerate(feats):
            rf.write(json.dumps({"vec_idx": len(order) + j, "family": "sae",
                                 "feature": int(fid)}) + "\n")
    vsize = os.path.getsize("/root/pool/vecs.f32")
    assert vsize == n_total * row_b, f"vecs.f32 {vsize} B != {n_total} x {row_b}"

    stats = {
        "kind": "pool_rl_last5: RL direction pool for the peak-in-last-5 run (modal_rl_last5.py)",
        "n_examples": n_total,   # train/rl.py reads THIS key for the bank row count
        "families": {**{f: N_PER_FAMILY for f in POOL_FAMILIES}, "sae": N_PER_FAMILY},
        "layout": "rows [0,25k) realact | [25k,50k) cluster | [50k,75k) realact_long | [75k,100k) sae",
        "sae": {"source": SAE_PT, "n_features_total": int(F_total),
                "dir": "unit(W_dec[f]) — L2-normalized decoder row per sampled feature",
                "records_key": "feature"},
        "pool_sampling": "uniform without replacement per family from pool_rl_mix records; "
                         "records carried over with vec_idx re-based (src_vec_idx = source row)",
        "sources": {"pool": POOL, "sae": SAE_PT},
        "seed": seed, "d_model": D_MODEL, "created": time.time(),
    }
    os.makedirs(OUT, exist_ok=True)
    for fn in ("vecs.f32", "records.jsonl"):
        shutil.copyfile(f"/root/pool/{fn}", f"{OUT}/{fn}")
    json.dump(stats, open(f"{OUT}/build_stats.json", "w"), indent=1)
    vol.commit()
    print(f"[pool] written -> {OUT}: {n_total} rows ({vsize / 1e9:.2f} GB vecs, "
          f"{time.time() - t0:.0f}s)", flush=True)
    print(json.dumps(stats, indent=1), flush=True)
    _verify()


def _verify():
    """Re-verify from the FINAL on-volume artifacts (counts, sizes, norms, spot-check vs source)."""
    import json
    import os
    import sys

    import numpy as np

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL

    row_b = D_MODEL * 4
    stats = json.load(open(f"{OUT}/build_stats.json"))
    recs = [json.loads(l) for l in open(f"{OUT}/records.jsonl")]
    vsize = os.path.getsize(f"{OUT}/vecs.f32")
    n = len(recs)
    assert n == stats["n_examples"] == 100_000, (n, stats["n_examples"])
    assert vsize == n * row_b, f"vecs.f32 {vsize} B != {n} x {row_b}"
    assert [r["vec_idx"] for r in recs] == list(range(n)), "vec_idx must be 0..N-1 in order"
    fam_counts = {}
    for r in recs:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    assert fam_counts == stats["families"], (fam_counts, stats["families"])
    assert all("feature" in r for r in recs if r["family"] == "sae"), "sae rows need feature ids"

    fd = os.open(f"{OUT}/vecs.f32", os.O_RDONLY)
    # sae rows unit-norm
    sae_rows = [r["vec_idx"] for r in recs if r["family"] == "sae"]
    chk = sae_rows[:: max(1, len(sae_rows) // 512)]
    devs = []
    for i in chk:
        v = np.frombuffer(_pread_full(fd, row_b, i * row_b), np.float32)
        devs.append(abs(float(np.linalg.norm(v)) - 1.0))
    assert max(devs) < 1e-3, f"sae row norm dev {max(devs)}"
    # pool rows byte-identical to their pool_rl_mix source rows
    sfd = os.open(f"{POOL}/vecs.f32", os.O_RDONLY)
    pool_rows = [r for r in recs if r["family"] != "sae"]
    for r in pool_rows[:: max(1, len(pool_rows) // 64)]:
        a = _pread_full(fd, row_b, r["vec_idx"] * row_b)
        b = _pread_full(sfd, row_b, r["src_vec_idx"] * row_b)
        assert a == b, f"row {r['vec_idx']} != pool_rl_mix row {r['src_vec_idx']}"
    os.close(fd)
    os.close(sfd)
    print(f"[verify] OK: {n} rows, vecs.f32 {vsize} B, families {fam_counts}, "
          f"sae unit-norm max dev {max(devs):.2e}, pool rows byte-match source", flush=True)


@app.function(image=image, volumes={"/data": vol}, timeout=7200, cpu=8, memory=49152)
def fix_sae():
    """Rewrite ONLY the sae family (rows 75000..99999) of the built pool: unit ENCODER columns
    unit(W_enc[:,f]) via mxf.sae.enc_dirs — the suite's conditioning/reward-dir convention
    (what sae_score/eval_universal measure: the feature FIRES on the encoder projection).
    The original build wrote decoder rows (the wrong thing to reward toward). Same 25k feature
    ids (read back from records.jsonl), rows 0..74999 byte-identical, records untouched."""
    import hashlib
    import json
    import os
    import shutil
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL
    from mxf.sae import load_sae

    row_b = D_MODEL * 4
    stats = json.load(open(f"{OUT}/build_stats.json"))
    assert stats["n_examples"] == 100_000
    recs = [json.loads(l) for l in open(f"{OUT}/records.jsonl")]
    sae_recs = [r for r in recs if r["family"] == "sae"]
    n_head = len(recs) - len(sae_recs)
    assert len(sae_recs) == N_PER_FAMILY and n_head == 3 * N_PER_FAMILY
    assert [r["vec_idx"] for r in sae_recs] == list(range(n_head, len(recs))), \
        "sae rows must be the contiguous tail block"
    feats = [int(r["feature"]) for r in sae_recs]   # SAME features as the original build

    sae = load_sae(path=SAE_PT, device="cpu", dtype=torch.float32)
    enc = sae.enc_dirs(feats).numpy().astype(np.float32)   # [25k, d] unit(W_enc[:,f]) — canon
    assert enc.shape == (N_PER_FAMILY, D_MODEL)
    dev = np.abs(np.linalg.norm(enc, axis=1) - 1.0).max()
    assert dev < 1e-4, f"enc dirs not unit: {dev}"

    # rebuild locally: byte-copy rows [0, n_head) from the on-volume file, append encoder block
    t0 = time.time()
    h_head = hashlib.sha256()
    old_dec = np.empty((N_PER_FAMILY, D_MODEL), np.float32)
    with open(f"{OUT}/vecs.f32", "rb") as fin, open("/root/vecs_fixed.f32", "wb") as fout:
        left = n_head * row_b
        while left:
            buf = fin.read(min(64 << 20, left))
            assert buf, "unexpected EOF in head block"
            h_head.update(buf)
            fout.write(buf)   # head is byte-identical BY CONSTRUCTION (literal copy)
            left -= len(buf)
        old_dec[:] = np.frombuffer(fin.read(N_PER_FAMILY * row_b), np.float32).reshape(-1, D_MODEL)
        assert not fin.read(1), "trailing bytes after sae block"
        enc.tofile(fout)
    head_sha = h_head.hexdigest()
    assert os.path.getsize("/root/vecs_fixed.f32") == 100_000 * row_b
    # sanity: the block actually changed (decoder != encoder dirs), same features
    cos = (old_dec * enc).sum(1)
    print(f"[fix-sae] old(decoder)·new(encoder) cos: mean {cos.mean():.4f} "
          f"min {cos.min():.4f} max {cos.max():.4f} (should be << 1)", flush=True)
    assert (np.abs(cos) < 0.999).any(), "new sae block identical to old — nothing changed?"

    shutil.copyfile("/root/vecs_fixed.f32", f"{OUT}/vecs.f32")
    stats["sae"]["dir"] = ("unit(W_enc[:,f]) — L2-unit ENCODER column per sampled feature "
                           "(mxf.sae.enc_dirs, the suite's conditioning/reward-dir convention)")
    stats["sae"]["fixed"] = {"when": time.time(),
                             "note": "original build wrote unit decoder rows (W_dec); replaced "
                                     "with unit encoder columns, same 25k feature ids, rows "
                                     "0..74999 byte-identical (sha256-checked head copy)"}
    json.dump(stats, open(f"{OUT}/build_stats.json", "w"), indent=1)
    vol.commit()
    print(f"[fix-sae] sae block rewritten with encoder columns ({time.time() - t0:.0f}s)", flush=True)

    # re-verify from the final on-volume artifacts: head sha unchanged + full _verify +
    # sae rows == mxf.sae.enc_dirs(feature ids) exactly
    h_chk = hashlib.sha256()
    with open(f"{OUT}/vecs.f32", "rb") as fin:
        left = n_head * row_b
        while left:
            buf = fin.read(min(64 << 20, left))
            h_chk.update(buf)
            left -= len(buf)
    assert h_chk.hexdigest() == head_sha, "head block changed on volume!"
    _verify()
    fd = os.open(f"{OUT}/vecs.f32", os.O_RDONLY)
    for j in (0, N_PER_FAMILY // 2, N_PER_FAMILY - 1):
        v = np.frombuffer(_pread_full(fd, row_b, (n_head + j) * row_b), np.float32)
        assert np.array_equal(v, enc[j]), f"sae row {j} != enc_dirs(feature {feats[j]})"
    os.close(fd)
    print(f"[fix-sae] OK: head sha256 {head_sha[:16]}… unchanged; on-volume sae rows == "
          "mxf.sae.enc_dirs(feature ids) exactly", flush=True)


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=4, memory=8192)
def peek():
    """CPU-only verify of the finished pool from the final artifacts."""
    import json

    print(json.dumps(json.load(open(f"{OUT}/build_stats.json")), indent=1), flush=True)
    _verify()


@app.local_entrypoint()
def run_inspect():
    inspect_inputs.remote()


@app.local_entrypoint()
def run_build(seed: int = 0):
    build.remote(seed=seed)


@app.local_entrypoint()
def run_fix_sae():
    fix_sae.remote()


@app.local_entrypoint()
def run_peek():
    peek.remote()
