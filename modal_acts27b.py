"""Modal app: regenerate per-token layer-42 activations + token ids for Qwen/Qwen3.6-27B
over m-a-p/FineFineWeb onto the `maemm-data` volume (replaces the lost activation data;
SFT banks are built from these downstream).

One 8xB200 container runs 8 single-GPU worker processes (collect/collect_acts27b_worker.py,
mounted at /pmx/collect_acts27b_worker.py). Each rank owns a DISJOINT slice of FineFineWeb
files chosen round-robin across all ~67 domain folders — naive load_dataset streaming would
read the 66k jsonl files alphabetically (all-aerospace first) and take forever to resolve;
instead the driver lists the repo once and hands each rank ~2 files per domain slot.

Deliverables under /data/acts27b/ (finalize assembles these, then deletes the shards):
    acts.f16        [n_seq, 512, 5120] fp16  RAW layer-42 resid_post per content token
    toks.i32        [n_seq, 512] i32         content-token ids (banks decode spans from these)
    whiten_mu.npy   [5120] f32               streaming mean over all stored tokens
    meta.json       n_seq/seq_len/d/layer/model/dataset/seed + provenance

Conventions (match the rest of the suite): forward = [BOS] + 512 content tokens; position 0
(BOS/sink) is DROPPED before storing, so row t of acts/toks is content token t. Acts stored
raw — the 10x-median norm filter and any normalization happen downstream at bank build.

Launch (MODAL_PROFILE=safety-sahan):
    modal run modal_acts27b.py::run_smoke                # 1xB200 end-to-end validation
    modal run --detach modal_acts27b.py::run_collect     # the real 50k-seq (25.6M tok) run
    modal run modal_acts27b.py::run_peek                 # inspect finished outputs (CPU)
Crash/24h-cap safe: workers shard-and-commit at chunk boundaries with resume offsets in
per-rank manifests (heartbeat committer publishes every 60s); rerunning run_collect with the
same --out-name resumes every rank where it left off and re-finalizes.

Needs Modal secret `maemm-hf` (HF_TOKEN). Model loads offline from /data/hf_cache after a
single-flight snapshot_download; the corpus is fetched over the network (per-file
hf_hub_download to container-local disk — never into the volume's hf_cache).
"""

from pathlib import Path

import modal

REPO = Path(__file__).parent

APP_NAME = "maemm-acts27b"
app = modal.App(APP_NAME)

# identical pins to modal_rl.py / modal_sft.py (one environment across the suite);
# + datasets, used only by the fineweb fallback reader.
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
        "datasets",
    )
    .add_local_file(REPO / "collect" / "collect_acts27b_worker.py",
                    "/pmx/collect_acts27b_worker.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

SEQ_LEN = 512
DATASET = "m-a-p/FineFineWeb"
FALLBACK = ("HuggingFaceFW/fineweb", "sample-10BT")


def _build_assignment(world: int, seed: int):
    """List FineFineWeb's jsonl files, pick 2 per domain (seeded), deal them round-robin to
    ranks so every rank spans ~n_domains/world domains and ranks never share a file. Falls
    back to fineweb streaming (modulo doc sharding) if FineFineWeb can't be listed."""
    import random

    from huggingface_hub import HfApi

    try:
        files = [f for f in HfApi().list_repo_files(DATASET, repo_type="dataset")
                 if f.endswith(".jsonl")]
        assert files, "no jsonl files listed"
    except Exception as e:
        print(f"[modal] {DATASET} unavailable ({type(e).__name__}: {e}) — FALLING BACK to "
              f"{FALLBACK[0]}:{FALLBACK[1]}", flush=True)
        return {"mode": "hfstream", "dataset": FALLBACK[0], "config": FALLBACK[1],
                "split": "train", "note": f"fallback: FineFineWeb failed with "
                                          f"{type(e).__name__}: {e}"}
    by_dom = {}
    for f in files:
        by_dom.setdefault(f.split("/")[0], []).append(f)
    rng = random.Random(seed)
    doms = sorted(by_dom)
    for d in doms:
        by_dom[d].sort()
        rng.shuffle(by_dom[d])
    ordered = []
    for tier in range(2):                     # tier 0 = one file per domain, tier 1 = spare
        tier_files = [by_dom[d][tier] for d in doms if len(by_dom[d]) > tier]
        rng.shuffle(tier_files)
        ordered += tier_files
    return {"mode": "fffw", "repo": DATASET, "dataset": DATASET, "n_domains": len(doms),
            "n_files": len(ordered), "ranks": [ordered[r::world] for r in range(world)]}


def _run(out_name: str, n_seq: int, world: int, batch: int, chunk_seqs: int,
         max_wins: int, seed: int):
    import json
    import os
    import subprocess
    import sys
    import threading
    import time

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import MODEL

    out = f"/data/{out_name}"
    if os.path.exists(f"{out}/meta.json"):
        raise RuntimeError(f"{out}/meta.json exists — run already complete; new --out-name?")
    shards = f"{out}/shards"
    os.makedirs(shards, exist_ok=True)

    # single-flight base-model download into the persistent volume cache; workers then load
    # with local_files_only=True (8 concurrent hub re-resolutions caused spurious
    # missing-shard errors on the RL app). HF_HUB_OFFLINE stays UNSET: the corpus needs net.
    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL)
    vol.commit()
    print(f"[modal] base model in cache ({time.time() - t0:.0f}s)", flush=True)

    assign = _build_assignment(world, seed)
    assign_path = "/tmp/acts27b_assign.json"
    json.dump(assign, open(assign_path, "w"))
    print(f"[modal] corpus mode={assign['mode']} dataset={assign['dataset']}"
          + (f" domains={assign['n_domains']} files={assign['n_files']}"
             if assign["mode"] == "fffw" else ""), flush=True)

    def _committer():                # publish shards/manifests every 60s -> crash-durable
        while True:
            try:
                with open(f"{out}/heartbeat", "w") as f:
                    f.write(str(time.time()))
                vol.commit()
            except Exception as e:
                print(f"[modal] heartbeat/commit failed: {e}", flush=True)
            time.sleep(60)

    threading.Thread(target=_committer, daemon=True).start()

    per = [n_seq // world + (1 if r < n_seq % world else 0) for r in range(world)]
    procs = []
    for r in range(world):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(r)
        env["PYTHONPATH"] = "/pmx/helpers"
        env["TOKENIZERS_PARALLELISM"] = "false"
        cmd = [sys.executable, "/pmx/collect_acts27b_worker.py",
               "--rank", str(r), "--world", str(world), "--n-seq", str(per[r]),
               "--seq-len", str(SEQ_LEN), "--batch", str(batch),
               "--chunk-seqs", str(chunk_seqs), "--max-wins", str(max_wins),
               "--seed", str(seed), "--out", shards, "--assignment", assign_path]
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(2)
    fails = [r for r, p in enumerate(procs) if p.wait() != 0]
    vol.commit()
    if fails:
        raise RuntimeError(f"worker rank(s) {fails} failed — shards persisted; rerun "
                           f"run_collect (same out-name) to resume")
    print("[modal] all workers done — finalizing", flush=True)
    _finalize(out, world, n_seq, seed, assign)


def _finalize(out: str, world: int, n_seq: int, seed: int, assign: dict):
    """Assemble shards -> acts.f16 / toks.i32 / whiten_mu.npy / meta.json, verify, clean up."""
    import json
    import os
    import shutil
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL, READ_LAYER

    shards = f"{out}/shards"
    mans = []
    for r in range(world):
        m = json.load(open(f"{shards}/manifest_r{r}.json"))
        assert m["done"], f"rank {r} manifest not done — resume the collect first"
        mans.append(m)
    total = sum(m["kept"] for m in mans)
    # ranks may overshoot the target by <batch at the last chunk boundary — keep everything
    assert total >= n_seq, f"collected {total} < target {n_seq}"
    n_seq = total

    order = [(m["rank"], ch["c"], ch["n"]) for m in mans for ch in m["chunks"]]
    rows = {"acts.f16": SEQ_LEN * D_MODEL * 2, "toks.i32": SEQ_LEN * 4}
    for ext, row in rows.items():
        t0 = time.time()
        tmp = f"{out}/{ext}.tmp"
        with open(tmp, "wb") as fout:
            for r, c, n in order:
                p = f"{shards}/r{r}_c{c:04d}.{ext}"
                sz = os.path.getsize(p)
                assert sz == n * row, f"{p}: {sz} B != {n} rows x {row} B"
                with open(p, "rb") as fin:
                    shutil.copyfileobj(fin, fout, 64 * 1024 * 1024)
        assert os.path.getsize(tmp) == n_seq * row
        os.replace(tmp, f"{out}/{ext}")
        print(f"[modal] {ext} assembled ({n_seq * row / 1e9:.1f} GB, "
              f"{time.time() - t0:.0f}s)", flush=True)

    musum = np.zeros(D_MODEL, np.float64)
    count = 0
    for m in mans:
        musum += np.load(f"{shards}/musum_r{m['rank']}.npy")
        count += m["mu_count"]
    assert count == n_seq * SEQ_LEN, f"mu_count {count} != {n_seq * SEQ_LEN}"
    mu = (musum / count).astype(np.float32)
    np.save(f"{out}/whiten_mu.npy", mu)

    # spot-verify the assembled memmaps against first + last shard before deleting shards
    mm = np.memmap(f"{out}/acts.f16", np.float16, "r", shape=(n_seq, SEQ_LEN, D_MODEL))
    fr, fc, _ = order[0]
    first = np.fromfile(f"{shards}/r{fr}_c{fc:04d}.acts.f16", np.float16,
                        count=SEQ_LEN * D_MODEL).reshape(SEQ_LEN, D_MODEL)
    assert np.array_equal(mm[0], first), "row-0 mismatch vs first shard"
    lr, lc, ln = order[-1]
    last = np.fromfile(f"{shards}/r{lr}_c{lc:04d}.acts.f16",
                       np.float16).reshape(ln, SEQ_LEN, D_MODEL)
    assert np.array_equal(mm[n_seq - 1], last[-1]), "last-row mismatch vs last shard"
    tt = np.memmap(f"{out}/toks.i32", np.int32, "r", shape=(n_seq, SEQ_LEN))
    assert int(tt.min()) >= 0, "negative token ids"
    norms = np.linalg.norm(mm[0].astype(np.float32), axis=-1)
    print(f"[modal] verify OK; row0 token-norm median {np.median(norms):.1f} "
          f"mu-norm {np.linalg.norm(mu):.2f}", flush=True)

    meta = {"n_seq": n_seq, "n_seq_target": sum(m["n_seq_target"] for m in mans),
            "seq_len": SEQ_LEN, "d": D_MODEL, "layer": READ_LAYER,
            "model": MODEL, "dataset": assign["dataset"], "seed": seed,
            "n_tokens": n_seq * SEQ_LEN, "world": world, "mode": assign["mode"],
            "note": assign.get("note", ""), "n_domains": assign.get("n_domains"),
            "bos_id": mans[0]["bos_id"],
            "convention": "forward=[BOS]+512 content toks; pos 0 (BOS/sink) dropped; row t "
                          "= content token t; acts RAW layer-42 resid_post (no norm filter; "
                          "whitening/filtering downstream: unit(act - whiten_mu))",
            "files": {"acts.f16": f"float16 [{n_seq},{SEQ_LEN},{D_MODEL}]",
                      "toks.i32": f"int32 [{n_seq},{SEQ_LEN}]",
                      "whiten_mu.npy": f"float32 [{D_MODEL}] mean over all stored tokens"},
            "order": "rank-major (rank asc, chunk asc)", "created": time.time()}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=2)
    vol.commit()
    shutil.rmtree(shards)
    vol.commit()
    print(f"[modal] COMPLETE -> {out} (n_seq={n_seq}, {n_seq * SEQ_LEN:,} tokens)", flush=True)


@app.function(
    image=image,
    gpu="B200:8",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=86400,
)
def collect(n_seq: int = 50000, out_name: str = "acts27b", batch: int = 64,
            chunk_seqs: int = 512, max_wins: int = 8, seed: int = 0):
    _run(out_name, n_seq, world=8, batch=batch, chunk_seqs=chunk_seqs,
         max_wins=max_wins, seed=seed)


@app.function(
    image=image,
    gpu="B200:1",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=14400,
)
def smoke(n_seq: int = 128, out_name: str = "acts27b_smoke", batch: int = 16,
          chunk_seqs: int = 64, max_wins: int = 4, seed: int = 0):
    """1xB200 end-to-end validation of the exact collect+finalize path on a tiny target."""
    _run(out_name, n_seq, world=1, batch=batch, chunk_seqs=chunk_seqs,
         max_wins=max_wins, seed=seed)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=1800,
    cpu=4,
)
def peek(out_name: str = "acts27b"):
    """CPU-only sanity read of a finished output dir: meta + decoded tokens + norm stats."""
    import json
    import os
    import sys

    import numpy as np

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL

    out = f"/data/{out_name}"
    meta = json.load(open(f"{out}/meta.json"))
    print(json.dumps(meta, indent=2), flush=True)
    n, L = meta["n_seq"], meta["seq_len"]
    mm = np.memmap(f"{out}/acts.f16", np.float16, "r", shape=(n, L, D_MODEL))
    tt = np.memmap(f"{out}/toks.i32", np.int32, "r", shape=(n, L))
    mu = np.load(f"{out}/whiten_mu.npy")
    os.environ["HF_HOME"] = "/data/hf_cache"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for r in sorted({0, n // 2, n - 1}):
        norms = np.linalg.norm(mm[r].astype(np.float32), axis=-1)
        print(f"--- row {r}: token-norm med {np.median(norms):.1f} max {norms.max():.1f}\n"
              f"    {tok.decode(tt[r][:48].tolist())!r}", flush=True)
    print(f"mu: shape {mu.shape} dtype {mu.dtype} norm {np.linalg.norm(mu):.2f}", flush=True)


@app.local_entrypoint()
def run_collect(n_seq: int = 50000, out_name: str = "acts27b", batch: int = 64,
                chunk_seqs: int = 512, max_wins: int = 8, seed: int = 0):
    collect.remote(n_seq=n_seq, out_name=out_name, batch=batch, chunk_seqs=chunk_seqs,
                   max_wins=max_wins, seed=seed)


@app.local_entrypoint()
def run_smoke(n_seq: int = 128, out_name: str = "acts27b_smoke"):
    smoke.remote(n_seq=n_seq, out_name=out_name)


@app.local_entrypoint()
def run_peek(out_name: str = "acts27b"):
    peek.remote(out_name=out_name)
