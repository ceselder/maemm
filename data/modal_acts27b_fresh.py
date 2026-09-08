"""Modal app `maemm-acts27b-fresh`: a FRESH 512-token layer-42 activation store at /data/acts27b_fresh on `maemm-data`.

Same deliverables / conventions as data/modal_acts27b.py (acts.f16 [n_seq,512,5120] RAW resid_post fp16, toks.i32 [n_seq,512],
whiten_mu.npy, meta.json; forward = [BOS]+512 content tokens, BOS dropped), same worker (data/collect_acts27b_worker.py), but
built ONLY from FineFineWeb documents that no existing store or training bank has touched:
  * file-level exclusion: every jsonl file used by /data/acts27b (its seed-0 assignment, recomputed from the same repo
    listing) and by every /data/banks/realact_short_20m* part (their assignment.json; 24 parts, mutually disjoint) is
    removed from the file pool before the seeded 2-files-per-domain assignment is drawn (new seed);
  * document-level exclusion: the collectors' rule — sha1[:16] of the first 64 token ids of EVERY row of /data/acts27b
    (the realact eval hold-out's source store); a doc whose 512-aligned window starts hash into that set is skipped.
Finalize (shards -> single files) runs in a SEPARATE CPU function so the 8 GPUs never idle on 400 GB of copying:
toks.i32 + whiten_mu.npy + meta.json (acts_complete=false) land first — token-only consumers (MLP scan, SAE max-acts)
can start — then acts.f16 is assembled and meta.json is rewritten with acts_complete=true.

    MODAL_PROFILE=safety-sahan modal deploy data/modal_acts27b_fresh.py
    python -c "import modal; print(modal.Function.from_name('maemm-acts27b-fresh','collect').spawn(n_seq=80000).object_id)"
    python -c "import modal; print(modal.Function.from_name('maemm-acts27b-fresh','finalize').spawn().object_id)"
    python -c "import modal; print(modal.Function.from_name('maemm-acts27b-fresh','peek').remote())"
Needs Modal secret `maemm-hf` (HF_TOKEN).
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = os.environ.get("MAEMM_ACTS_FRESH_APP", "maemm-acts27b-fresh")
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet", "datasets")
    .pip_install("flash-linear-attention==0.5.2")   # GDN forward via fla's Triton chunk kernel (== data/modal_collect_bank.py)
    .add_local_file(REPO / "data" / "collect_acts27b_worker.py", "/pmx/collect_acts27b_worker.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

SEQ_LEN = 512
DATASET = "m-a-p/FineFineWeb"
OUT_DEFAULT = "acts27b_fresh"
ACTS27B = "/data/acts27b"
ACTS27B_SEED, ACTS27B_WORLD = 0, 8              # data/modal_acts27b.py defaults that built /data/acts27b
EXCLUDE_STORES = ["/data/acts27b", "/data/acts27b_long", "/data/acts_longctx"]   # hash every row of any that exists
COLLECT_GPU = os.environ.get("ACTS_FRESH_GPU", "H200:8")


def _list_files():
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files(DATASET, repo_type="dataset") if f.endswith(".jsonl")]
    assert files, "no FineFineWeb jsonl files listed"
    return files


def _assignment(files, world, seed):
    """== data/modal_acts27b._build_assignment on a given file list: 2 files per domain (seeded), dealt round-robin."""
    import random
    by_dom = {}
    for f in files:
        by_dom.setdefault(f.split("/")[0], []).append(f)
    rng = random.Random(seed)
    doms = sorted(by_dom)
    for d in doms:
        by_dom[d].sort()
        rng.shuffle(by_dom[d])
    ordered = []
    for tier in range(2):
        tier_files = [by_dom[d][tier] for d in doms if len(by_dom[d]) > tier]
        rng.shuffle(tier_files)
        ordered += tier_files
    return {"mode": "fffw", "repo": DATASET, "dataset": DATASET, "n_domains": len(doms), "n_files": len(ordered),
            "seed": seed, "ranks": [ordered[r::world] for r in range(world)]}


def _used_files():
    """Every FineFineWeb file already consumed by an existing store / training bank."""
    import glob
    import json
    files = _list_files()
    acts27b_files = [f for r in _assignment(files, ACTS27B_WORLD, ACTS27B_SEED)["ranks"] for f in r]
    parts = {}
    for p in sorted(glob.glob("/data/banks/realact_short_20m*")):
        for cand in (f"{p}/assignment.json", f"{p}/shards/assignment.json"):
            if os.path.exists(cand):
                a = json.load(open(cand))
                parts[os.path.basename(p)] = [f for r in a.get("ranks", []) for f in r]
                break
    used = set(acts27b_files) | {f for fs in parts.values() for f in fs}
    return files, used, {"acts27b(seed0,recomputed)": len(acts27b_files), **{k: len(v) for k, v in parts.items()}}


def _exclusion_hashes():
    """== data/modal_collect_bank._exclusion_hashes: sha1[:16] of the first 64 tokens of every row of every existing store."""
    import hashlib
    import json
    import numpy as np
    out, per_store = set(), {}
    for store in EXCLUDE_STORES:
        if not (os.path.exists(f"{store}/meta.json") and os.path.exists(f"{store}/toks.i32")):
            continue
        meta = json.load(open(f"{store}/meta.json"))
        n, L = int(meta["n_seq"]), int(meta["seq_len"])
        tt = np.memmap(f"{store}/toks.i32", np.int32, "r", shape=(n, L))
        hs = {hashlib.sha1(np.ascontiguousarray(tt[i, :64]).astype(np.int32).tobytes()).hexdigest()[:16] for i in range(n)}
        per_store[store] = len(hs)
        out |= hs
    return sorted(out), per_store


@app.function(image=image, gpu=COLLECT_GPU, cpu=32, memory=192 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=86400)
def collect(n_seq: int = 80_000, out_name: str = OUT_DEFAULT, batch: int = 64, chunk_seqs: int = 512, max_wins: int = 4,
            seed: int = 5000):
    """8 single-GPU workers over DISJOINT fresh files -> shards under /data/<out_name>/shards (crash-resume: rerun = resume).
    Does NOT finalize (see `finalize`, CPU)."""
    import json
    import subprocess
    import sys
    import threading
    import time
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import MODEL
    vol.reload()
    out = f"/data/{out_name}"
    if os.path.exists(f"{out}/meta.json"):
        raise RuntimeError(f"{out}/meta.json exists — store already finalized; new out_name?")
    shards = f"{out}/shards"
    os.makedirs(shards, exist_ok=True)
    world = len([ln for ln in subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines() if ln.strip()])
    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    vol.commit()
    print(f"[fresh] base model in cache ({time.time() - t0:.0f}s); world={world}", flush=True)

    assign_path = f"{shards}/assignment.json"
    if os.path.exists(assign_path):
        assign = json.load(open(assign_path))
        assert assign["world"] == world, "resume with a different world size is not supported"
        print(f"[fresh] RESUME with existing assignment ({assign['n_files']} files)", flush=True)
    else:
        files, used, per_src = _used_files()
        fresh_files = [f for f in files if f not in used]
        assign = _assignment(fresh_files, world, seed)
        assert not (set(f for r in assign["ranks"] for f in r) & used)
        assign.update({"world": world, "n_repo_files": len(files), "n_excluded_files": len(used), "excluded_files_by_source": per_src,
                       "excluded_files": sorted(used)})
        json.dump(assign, open(assign_path, "w"))
        print(f"[fresh] assignment: {assign['n_files']} fresh files over {world} ranks from {len(fresh_files)} unused of "
              f"{len(files)} repo files (excluded {len(used)}: {per_src})", flush=True)
    excl_path = f"{shards}/exclude_hashes.json"
    if not os.path.exists(excl_path):
        hashes, per_store = _exclusion_hashes()
        json.dump(hashes, open(excl_path, "w"))
        json.dump(per_store, open(f"{shards}/exclude_sources.json", "w"))
        print(f"[fresh] exclusion hashes: {len(hashes)} from {per_store}", flush=True)
    vol.commit()

    stop = threading.Event()

    def committer():
        while not stop.wait(60):
            try:
                with open(f"{out}/heartbeat", "w") as f:
                    f.write(str(time.time()))
                vol.commit()
            except Exception as e:  # noqa
                print(f"[fresh] heartbeat/commit failed: {e}", flush=True)
    threading.Thread(target=committer, daemon=True).start()

    per = [n_seq // world + (1 if r < n_seq % world else 0) for r in range(world)]
    procs = []
    for r in range(world):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(r)
        env["PYTHONPATH"] = "/pmx/helpers"
        env["TOKENIZERS_PARALLELISM"] = "false"
        cmd = [sys.executable, "/pmx/collect_acts27b_worker.py", "--rank", str(r), "--world", str(world), "--n-seq", str(per[r]),
               "--seq-len", str(SEQ_LEN), "--batch", str(batch), "--chunk-seqs", str(chunk_seqs), "--max-wins", str(max_wins),
               "--seed", str(seed), "--out", shards, "--assignment", assign_path, "--exclude-hashes", excl_path]
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(2)
    fails = [r for r, p in enumerate(procs) if p.wait() != 0]
    stop.set()
    vol.commit()
    if fails:
        raise RuntimeError(f"worker rank(s) {fails} failed — shards persisted; rerun collect (same out_name) to resume")
    wall = time.time() - t0
    json.dump({"collect_wall_s": wall, "world": world, "gpu": COLLECT_GPU, "n_seq_target": n_seq, "batch": batch,
               "max_wins": max_wins, "seed": seed}, open(f"{shards}/collect_done.json", "w"))
    vol.commit()
    print(f"[fresh] all {world} workers done in {wall / 60:.1f} min -> run `finalize` (CPU)", flush=True)
    return {"out": out, "world": world, "wall_s": wall}


@app.function(image=image, cpu=16, memory=64 * 1024, volumes={"/data": vol}, timeout=8 * 3600)
def finalize(out_name: str = OUT_DEFAULT, keep_shards: bool = False):
    """CPU: shards -> toks.i32 + whiten_mu.npy + meta.json (acts_complete=false) FIRST, then acts.f16, then meta acts_complete=true."""
    import json
    import shutil
    import sys
    import time
    import numpy as np
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL, READ_LAYER
    vol.reload()
    out = f"/data/{out_name}"
    shards = f"{out}/shards"
    assign = json.load(open(f"{shards}/assignment.json"))
    world = int(assign["world"])
    mans = []
    for r in range(world):
        m = json.load(open(f"{shards}/manifest_r{r}.json"))
        assert m["done"], f"rank {r} manifest not done — resume the collect first"
        mans.append(m)
    n_seq = sum(m["kept"] for m in mans)
    order = [(m["rank"], ch["c"], ch["n"]) for m in mans for ch in m["chunks"]]
    done_info = json.load(open(f"{shards}/collect_done.json")) if os.path.exists(f"{shards}/collect_done.json") else {}
    excl_src = json.load(open(f"{shards}/exclude_sources.json")) if os.path.exists(f"{shards}/exclude_sources.json") else {}
    T0 = time.time()

    def assemble(ext, row_b):
        t0 = time.time()
        tmp = f"{out}/{ext}.tmp"
        with open(tmp, "wb") as fout:
            for r, c, n in order:
                p = f"{shards}/r{r}_c{c:04d}.{ext}"
                sz = os.path.getsize(p)
                assert sz == n * row_b, f"{p}: {sz} B != {n} rows x {row_b} B"
                with open(p, "rb") as fin:
                    shutil.copyfileobj(fin, fout, 64 * 1024 * 1024)
        assert os.path.getsize(tmp) == n_seq * row_b
        os.replace(tmp, f"{out}/{ext}")
        print(f"[fresh] {ext} assembled ({n_seq * row_b / 1e9:.1f} GB, {time.time() - t0:.0f}s)", flush=True)

    assemble("toks.i32", SEQ_LEN * 4)
    musum = np.zeros(D_MODEL, np.float64); count = 0
    for m in mans:
        musum += np.load(f"{shards}/musum_r{m['rank']}.npy"); count += m["mu_count"]
    assert count == n_seq * SEQ_LEN, f"mu_count {count} != {n_seq * SEQ_LEN}"
    mu = (musum / count).astype(np.float32)
    np.save(f"{out}/whiten_mu.npy", mu)
    tt = np.memmap(f"{out}/toks.i32", np.int32, "r", shape=(n_seq, SEQ_LEN))
    assert int(tt.min()) >= 0
    meta = {"n_seq": n_seq, "n_seq_target": sum(m["n_seq_target"] for m in mans), "seq_len": SEQ_LEN, "d": D_MODEL,
            "layer": READ_LAYER, "model": MODEL, "dataset": assign["dataset"], "seed": assign["seed"], "n_tokens": n_seq * SEQ_LEN,
            "world": world, "mode": assign["mode"], "n_domains": assign.get("n_domains"), "n_files": assign.get("n_files"),
            "bos_id": mans[0]["bos_id"], "docs_seen": int(sum(m.get("docs", 0) for m in mans)),
            "docs_excluded_hash": int(sum(m.get("skipped_docs", 0) for m in mans)),
            "fresh": {"kind": "documents disjoint from every existing store / training bank",
                      "file_exclusion": {"n_excluded_files": assign.get("n_excluded_files"), "by_source": assign.get("excluded_files_by_source"),
                                         "n_repo_files": assign.get("n_repo_files")},
                      "hash_exclusion": {"rule": "sha1[:16] of the first 64 token ids at every 512-aligned doc offset vs every row of",
                                         "stores": excl_src}},
            "convention": "forward=[BOS]+512 content toks; pos 0 (BOS/sink) dropped; row t = content token t; acts RAW layer-42 "
                          "resid_post (no norm filter; whitening/filtering downstream: unit(act - whiten_mu))",
            "whiten_mu_note": "this store's own mean; the suite's realact convention centers with /data/acts27b/whiten_mu.npy — "
                              "bank builders pass mu_path explicitly",
            "files": {"acts.f16": f"float16 [{n_seq},{SEQ_LEN},{D_MODEL}]", "toks.i32": f"int32 [{n_seq},{SEQ_LEN}]",
                      "whiten_mu.npy": f"float32 [{D_MODEL}] mean over all stored tokens"},
            "order": "rank-major (rank asc, chunk asc)", "collect": done_info, "acts_complete": False, "created": time.time()}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=2)
    vol.commit()
    print(f"[fresh] toks/mu/meta published (acts_complete=false): n_seq={n_seq} docs={meta['docs_seen']} "
          f"excluded={meta['docs_excluded_hash']} ({time.time() - T0:.0f}s)", flush=True)

    assemble("acts.f16", SEQ_LEN * D_MODEL * 2)
    mm = np.memmap(f"{out}/acts.f16", np.float16, "r", shape=(n_seq, SEQ_LEN, D_MODEL))
    fr, fc, _ = order[0]
    first = np.fromfile(f"{shards}/r{fr}_c{fc:04d}.acts.f16", np.float16, count=SEQ_LEN * D_MODEL).reshape(SEQ_LEN, D_MODEL)
    assert np.array_equal(mm[0], first), "row-0 mismatch vs first shard"
    lr, lc, ln = order[-1]
    last = np.fromfile(f"{shards}/r{lr}_c{lc:04d}.acts.f16", np.float16).reshape(ln, SEQ_LEN, D_MODEL)
    assert np.array_equal(mm[n_seq - 1], last[-1]), "last-row mismatch vs last shard"
    norms = np.linalg.norm(mm[0].astype(np.float32), axis=-1)
    meta["acts_complete"] = True
    meta["verify"] = {"row0_token_norm_median": float(np.median(norms)), "mu_norm": float(np.linalg.norm(mu))}
    meta["finalize_s"] = time.time() - T0
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=2)
    vol.commit()
    if not keep_shards:
        shutil.rmtree(shards)
        vol.commit()
    print(f"[fresh] COMPLETE -> {out} (n_seq={n_seq}, {n_seq * SEQ_LEN:,} tokens) in {(time.time() - T0) / 60:.1f} min", flush=True)
    return {"out": out, "n_seq": n_seq, "docs": meta["docs_seen"], "excluded": meta["docs_excluded_hash"], "minutes": (time.time() - T0) / 60}


@app.function(image=image, cpu=4, memory=16384, volumes={"/data": vol}, timeout=1800)
def status(out_name: str = OUT_DEFAULT):
    """Progress of a running collect (manifests) — CPU, no GPU."""
    import glob
    import json
    vol.reload()
    out = f"/data/{out_name}"
    res = {"meta": os.path.exists(f"{out}/meta.json")}
    if res["meta"]:
        m = json.load(open(f"{out}/meta.json")); res["acts_complete"] = m.get("acts_complete"); res["n_seq"] = m["n_seq"]
        res["acts_bytes"] = os.path.getsize(f"{out}/acts.f16") if os.path.exists(f"{out}/acts.f16") else 0
    ranks = {}
    for p in sorted(glob.glob(f"{out}/shards/manifest_r*.json")):
        m = json.load(open(p)); ranks[m["rank"]] = {"kept": m["kept"], "target": m["n_seq_target"], "done": m["done"], "docs": m.get("docs"), "skipped": m.get("skipped_docs")}
    res["ranks"] = ranks
    res["kept_total"] = sum(v["kept"] for v in ranks.values())
    return res


@app.function(image=image, cpu=4, memory=16384, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=1800)
def peek(out_name: str = OUT_DEFAULT):
    import json
    import sys
    import numpy as np
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL
    vol.reload()
    out = f"/data/{out_name}"
    meta = json.load(open(f"{out}/meta.json"))
    print(json.dumps({k: v for k, v in meta.items() if k != "fresh"}, indent=1), flush=True)
    n, L = meta["n_seq"], meta["seq_len"]
    tt = np.memmap(f"{out}/toks.i32", np.int32, "r", shape=(n, L))
    os.environ["HF_HOME"] = "/data/hf_cache"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for r in sorted({0, n // 2, n - 1}):
        print(f"--- row {r}: {tok.decode(tt[r][:48].tolist())!r}", flush=True)
    if meta.get("acts_complete"):
        mm = np.memmap(f"{out}/acts.f16", np.float16, "r", shape=(n, L, D_MODEL))
        for r in (0, n - 1):
            norms = np.linalg.norm(mm[r].astype(np.float32), axis=-1)
            print(f"row {r}: token-norm median {np.median(norms):.1f} max {norms.max():.1f}", flush=True)
    return meta["n_seq"]
