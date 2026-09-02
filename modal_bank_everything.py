"""Modal app: build the "EVERYTHING" RL direction bank at /data/banks/everything on `maemm-data`.

Five ON-MANIFOLD direction families, --n-per-family (default 100,000) rows EACH, in the EXACT bank format
train/rl.py consumes (== /data/pool_rl_last5, /data/pool_rl_mix):
    vecs.f32            raw LE float32 row-major [N, 5120], UNIT rows (layer-42 residual directions of Qwen/Qwen3.6-27B)
    records.jsonl       one JSON/line, line i == vec_idx i: {vec_idx, family, target_text, ...per-family source fields}
    build_stats.json    "n_examples" == N == rows of vecs.f32 (train/rl.py reads n_examples as the memmap row count)
    meta.json           split summary, recipes, exclusion summary, verification numbers
    exclusions.json     the FULL exclusion id lists actually applied
Row order is a seeded SHUFFLE of all families (so train/rl.py's --n-eval-dirs front reservation is a family mix).

Families (train/rl.py samples rows uniformly, so equal row counts == an even mix):
  realact       SHORT-context real L42 activations — the modal_big_bank.py PREFIX harvest: random TRAIN document of
                /data/acts27b (seq rows [0, ceil(0.95*n_seq)); the last 5% are the eval hold-out), index p ~ U[14, 91];
                direction = unit(act[s,p] - whiten_mu), 10x-median norm filter; target_text = decode(toks[s, 0:p+1+extra]),
                extra ~ U[1,4] (re-encoding the target standalone reproduces the exact prefix context).
  realact_long  LONG-context real L42 activations — the modal_big_bank.py WINDOW harvest restricted DEEP into the
                512-token sequences: p ~ U[256, 511]; direction = unit(act[s,p] - whiten_mu); target_text = a
                W ~ U[16,64]-token window ENDING at p (the activation itself saw the full p+1-token context).
  sae           unit ENCODER columns unit(W_enc[:, f]) of OUR SAE (/data/sae/ae.pt == HF ceselder/qwen36-27b-sae-l42
                trainer_0 ae.pt, F=131072 k=64; mxf.sae.BatchTopKSAE.enc_dirs — the convention of the eval's "held-out
                SAE features (unit encoder columns)" and of pool_rl_last5). ALIVE features only (corpus peak > 0 in
                /data/sae/maxacts.pt), EXCLUDING every feature id of the eval (all 13,107 pool_heldout sae features, a
                superset of the eval cache's 512 sae_feats, which the autointerp testbed also draws from).
                target_text = the feature's top max-activating 32-token corpus window (peak token index recorded).
  bsf           block-sparse featurizer (SASA, HF ceselder/qwen36-27b-bsf-l42-1b: G=32768 blocks x b=8 dims, k=32)
                subspace projections of REAL L42 activations: for a real act x at (s, p), p ~ U[16, 511], whiten
                y = (x - mu_bsf) @ zca; block activations gn_g = ||(normalize(y) @ E).view(G, b)[g]||; take the TOP
                active block (rank 1; when that block already holds --bsf-cap rows, the best-ranked block within the
                token's top --bsf-ranks that still has capacity); y_b = Q[b]^T Q[b] y with Q = blocks_Q.pt (per-block
                orthonormal basis of the block's decoder columns — it lives in WHITENED space); map the component back
                through the INVERSE whitening x_b = y_b @ zca^-1 (zca is symmetric) so x_b is a genuine additive
                component of (x - mu) in residual space; direction = unit(x_b). cos(direction, x - mu) is recorded per
                row ("cos_x") and summarized in meta.json. <= --bsf-cap rows per block, --doc-cap rows per document.
                EXCLUDES (a) the literal block ids of the eval's bsf family (pool_heldout bsf "block" — NB those ids
                index the ORIGINAL Aug-21 BSF, a different training run, so this is a formality) and (b) the top-1
                block, under THIS BSF, of every one of the 512 eval bsf directions (the meaningful exclusion).
  cluster       PROBE directions: the cluster/probe rows of /data/banks/last5_rp (peak re-anchored into the last 5
                tokens), EXCLUDING the eval's indist_probe reservation (pool_rl_mix cluster tail rows
                [748000, 750000) via src_vec_idx) and any row within cos > 0.999 of a pool_heldout cluster direction
                or an eval cluster / indist_probe direction. Family name kept as "cluster" (bank/eval convention).
J-lens directions are NOT in the bank (fully held-out generalization family). A final direction-level leakage check
asserts max cos < 0.999 between EVERY bank row and EVERY eval-cache direction (all 11 cos families + sae_dirs) and
every pool_heldout row.

Run (MODAL_PROFILE=safety-sahan):
    modal run modal_bank_everything.py::run_smoke                 # 1k/family -> banks/everything_smoke (~10 min)
    modal deploy modal_bank_everything.py && python -c "import modal; print(modal.Function.from_name(
        'maemm-bank-everything', 'build').spawn(n_per_family=100000, bsf_scan_seqs=20000, seed=7).object_id)"
                                                                  # 100k/family -> banks/everything (~1 h, 1 GPU);
                                                                  # deploy+spawn survives the launching client
    modal run modal_bank_everything.py::run_verify                # re-verify the FINAL artifacts on the volume
    modal run modal_bank_everything.py::run_peek                  # CPU: stats + sample rows
Trainer: --data-dir /data/banks/everything --bank-file vecs.f32 --direction-source cluster
Needs Modal secret `maemm-hf` (HF_TOKEN) for the one-time BSF download (cached to /data/bsf27b_1b).
"""
from pathlib import Path

import modal

REPO = Path(__file__).parent
APP_NAME = "maemm-bank-everything"
app = modal.App(APP_NAME)

# same pins as modal_last5_bank.py / modal_acts27b.py (one environment across the suite)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

ACTS = "/data/acts27b"
PROBE_BANK = "/data/banks/last5_rp"                 # probe rows (family "cluster")
POOL_RL_MIX = "/data/pool_rl_mix"                   # indist_* eval reservation = last 2000 rows per family block
POOL_HELDOUT = "/data/pool_heldout"                 # the eval's held-out pool (families bsf/realact/sae/jlens/cluster)
EVAL_CACHE = "/data/eval_universal_ho/eval_sets_heldout.pt"
SAE_PT = "/data/sae/ae.pt"
SAE_HF = ("ceselder/qwen36-27b-sae-l42", "saes_Qwen_Qwen3.6-27B_batch_top_k/resid_post_layer_42/trainer_0/ae.pt")
SAE_HF_SIZE = 5369256453                            # byte size of that HF file (identity check of /data/sae/ae.pt)
MAXACTS_PT = "/data/sae/maxacts.pt"
BSF_HF = "ceselder/qwen36-27b-bsf-l42-1b"
BSF_DIR = "/data/bsf27b_1b"                         # volume cache of the HF BSF files
BSF_FILES = ("sasa.pt", "blocks_Q.pt", "whiten_mu.npy", "whiten_zca.npy", "meta.json")
OUT_DEFAULT = "banks/everything"
FAMILIES = ("realact", "realact_long", "sae", "bsf", "cluster")
N_TAIL_INDIST = 2000                                # MAEMMBench/build_indist_eval.py N_TAIL
NORM_FILTER_MULT = 10.0
NORM_PRESAMPLE = 20_000
LEAK_COS = 0.999                                    # direction-level leakage threshold (exact dups are ~1.0)


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


def _max_cos_vs(bank_rows_iter, ref, dev):
    """Yield per-chunk max cosine of bank rows against ref [M, d] (unit, on dev). bank_rows_iter yields
    np.float32 [c, d] chunks (unit or not — normalized here)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    for chunk in bank_rows_iter:
        x = F.normalize(torch.from_numpy(np.ascontiguousarray(chunk)).to(dev), dim=-1)
        yield (x @ ref.T).max(1).values.float().cpu().numpy()


# ----------------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------------
# GPU: any of these is plenty (peak ~14 GB: SASA encoder 5.4 GB fp32 + a 4096x262144 block-code chunk); a fallback
# list schedules far faster than a single type. CPU/RAM kept modest for the same reason (peak RSS ~25 GB).
BUILD_GPUS = ["H100", "A100-80GB", "L40S", "A100-40GB"]


@app.function(image=image, gpu=BUILD_GPUS, cpu=8, memory=65536, ephemeral_disk=512 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=8 * 3600)
def build(out_name: str = OUT_DEFAULT, n_per_family: int = 100_000, seed: int = 7, threads: int = 48,
          chunk: int = 50_000, doc_cap: int = 8, bsf_scan_seqs: int = 20_000, bsf_cap: int = 4, bsf_ranks: int = 8,
          short_p_lo: int = 14, short_p_hi: int = 91, long_p_lo: int = 256, long_p_hi: int = 511,
          w_lo: int = 16, w_hi: int = 64, overwrite_smoke: bool = False):
    import json, os, random, shutil, time
    from concurrent.futures import ThreadPoolExecutor
    import sys
    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL
    from mxf.sae import load_sae

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda:0"
    T0 = time.time()

    def log(msg):
        print(f"[bank +{time.time() - T0:6.0f}s] {msg}", flush=True)

    vol.reload()
    out = f"/data/{out_name}"
    if os.path.exists(f"{out}/build_stats.json"):
        if overwrite_smoke and "smoke" in out_name:
            shutil.rmtree(out)
        else:
            raise RuntimeError(f"{out}/build_stats.json exists — bank already built")
    for p in (f"{ACTS}/acts.f16", f"{ACTS}/toks.i32", f"{ACTS}/whiten_mu.npy", f"{ACTS}/meta.json",
              f"{PROBE_BANK}/records.jsonl", f"{PROBE_BANK}/vecs.f32", f"{POOL_RL_MIX}/records.jsonl",
              f"{POOL_HELDOUT}/records.jsonl", f"{POOL_HELDOUT}/vecs.f32", EVAL_CACHE, SAE_PT, MAXACTS_PT):
        assert os.path.exists(p), f"missing input {p}"
    sae_size = os.path.getsize(SAE_PT)
    assert sae_size == SAE_HF_SIZE, (f"{SAE_PT} is {sae_size} B, expected {SAE_HF_SIZE} B (= HF {SAE_HF[0]} "
                                     f"{SAE_HF[1]}) — different SAE on the volume; refusing to guess")
    os.environ["HF_HOME"] = "/data/hf_cache"
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception as e:  # noqa
        log(f"tokenizer not in /data/hf_cache ({type(e).__name__}) — fetching")
        tok = AutoTokenizer.from_pretrained(MODEL)
    row_b = D_MODEL * 4

    # ---- BSF files: one-time HF download cached on the volume ----
    if not all(os.path.exists(f"{BSF_DIR}/{f}") for f in BSF_FILES):
        from huggingface_hub import hf_hub_download
        os.makedirs(BSF_DIR, exist_ok=True)
        for f in BSF_FILES:
            if os.path.exists(f"{BSF_DIR}/{f}"):
                continue
            t0 = time.time()
            p = hf_hub_download(BSF_HF, f, local_dir="/root/bsf_dl", token=os.environ.get("HF_TOKEN"))
            shutil.copyfile(p, f"{BSF_DIR}/{f}.part")
            os.replace(f"{BSF_DIR}/{f}.part", f"{BSF_DIR}/{f}")
            log(f"BSF {f}: downloaded + cached ({os.path.getsize(p) / 2**30:.2f} GB, {time.time() - t0:.0f}s)")
        vol.commit()
    bsf_meta = json.load(open(f"{BSF_DIR}/meta.json"))
    log(f"BSF meta: {bsf_meta}")

    # ---- acts27b ----
    meta = json.load(open(f"{ACTS}/meta.json"))
    NS, T = int(meta["n_seq"]), int(meta["seq_len"])
    n_train = int(np.ceil(NS * 0.95))               # rows 0..n_train-1 train; last 5% held out (== eval convention)
    mu_acts = np.load(f"{ACTS}/whiten_mu.npy").astype(np.float32)
    toks = np.fromfile(f"{ACTS}/toks.i32", dtype=np.int32).reshape(NS, T)
    afd = os.open(f"{ACTS}/acts.f16", os.O_RDONLY)
    nrng = np.random.default_rng(seed)
    wrng = random.Random(seed)
    per_doc = np.zeros(n_train, np.int32)            # shared document cap across realact / realact_long / bsf

    def read_act(s, p, dst):
        dst[:] = np.frombuffer(_pread_full(afd, D_MODEL * 2, ((s * T) + p) * D_MODEL * 2), np.float16)

    def read_seq(s):
        return np.frombuffer(_pread_full(afd, T * D_MODEL * 2, s * T * D_MODEL * 2), np.float16).reshape(T, D_MODEL)

    def window_text(s, p):
        W = wrng.randint(w_lo, w_hi)
        start = max(0, p - W + 1)
        ids = toks[s, start:p + 1].tolist()
        return start, ids, tok.decode(ids)

    # ---- eval sets + exclusion lists ----
    es = torch.load(EVAL_CACHE, map_location="cpu", weights_only=False)
    eval_dirs = {k[:-5]: F.normalize(es[k].float(), dim=-1) for k in es if k.endswith("_dirs")}
    eval_sae_feats = sorted(int(f) for f in es["sae_feats"])
    log(f"eval cache: dirs families {sorted(eval_dirs)} | sae_feats {len(eval_sae_feats)} | "
        f"cos_families {es['meta'].get('cos_families')}")
    recs_ho = [json.loads(l) for l in open(f"{POOL_HELDOUT}/records.jsonl")]
    ho_sz = os.path.getsize(f"{POOL_HELDOUT}/vecs.f32")
    ho_vecs = np.memmap(f"{POOL_HELDOUT}/vecs.f32", np.float32, "r", shape=(ho_sz // row_b, D_MODEL))
    assert len(recs_ho) == ho_vecs.shape[0]
    ho_rows = {}
    for r in recs_ho:
        ho_rows.setdefault(r["family"], []).append(int(r["vec_idx"]))
    excl_sae = sorted({int(r["feature"]) for r in recs_ho if r["family"] == "sae"} | set(eval_sae_feats))
    excl_bsf_old = sorted({int(r["block"]) for r in recs_ho if r["family"] == "bsf"})
    eval_bsf_old = sorted({int(recs_ho[i]["block"]) for i in es["meta"]["rows"]["bsf"]})
    assert set(eval_bsf_old) <= set(excl_bsf_old) and set(eval_sae_feats) <= set(excl_sae)
    # pool_rl_mix: indist_* reservation = the LAST N_TAIL rows of each contiguous family block
    mix_rows = {}
    for i, l in enumerate(open(f"{POOL_RL_MIX}/records.jsonl")):
        r = json.loads(l)
        assert int(r["vec_idx"]) == i
        mix_rows.setdefault(r["family"], []).append(i)
    mix_tail = {f: rows[-N_TAIL_INDIST:] for f, rows in mix_rows.items()}
    for fam_i, src in (("indist_probe", "cluster"), ("indist_realact", "realact"), ("indist_long", "realact_long")):
        used = es["meta"].get("indist", {}).get(fam_i, {}).get("rows", [])
        assert set(used) <= set(mix_tail[src]), f"{fam_i} rows not inside the pool_rl_mix {src} tail"
    excl_probe_src = sorted(mix_tail["cluster"])   # pool_rl_mix vec_idx of probe rows reserved for indist_probe
    log(f"exclusions: sae feats {len(excl_sae)} (eval cache 512 ⊆) | bsf OLD block ids {len(excl_bsf_old)} "
        f"(eval uses {len(eval_bsf_old)}) | probe src rows {excl_probe_src[0]}..{excl_probe_src[-1]} "
        f"({len(excl_probe_src)}) | pool_heldout rows {ho_vecs.shape[0]} for direction-level checks")

    # ---- output staging (local NVMe) ----
    os.makedirs("/root/bank", exist_ok=True)
    stage = {}      # fam -> (memmap [n_fam, d], records list)
    fam_stats = {}

    # ============================================================================================================
    # family: cluster (probe directions from banks/last5_rp)
    # ============================================================================================================
    t0 = time.time()
    precs = []
    with open(f"{PROBE_BANK}/records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("family") == "cluster":
                precs.append(r)
    n_probe_all = len(precs)
    tail_set = set(excl_probe_src)
    keep_mask = np.array([int(r["src_vec_idx"]) not in tail_set for r in precs])
    n_drop_tail = int((~keep_mask).sum())
    pidx = np.array([int(r["vec_idx"]) for r in precs], np.int64)
    contig = bool(np.all(np.diff(pidx) == 1))
    pfd = os.open(f"{PROBE_BANK}/vecs.f32", os.O_RDONLY)
    P = np.empty((n_probe_all, D_MODEL), np.float32)
    if contig:
        for c0 in range(0, n_probe_all, 8192):
            c1 = min(c0 + 8192, n_probe_all)
            P[c0:c1] = np.frombuffer(_pread_full(pfd, (c1 - c0) * row_b, int(pidx[c0]) * row_b),
                                     np.float32).reshape(c1 - c0, D_MODEL)
    else:
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(lambda j: P.__setitem__(j, np.frombuffer(_pread_full(pfd, row_b, int(pidx[j]) * row_b),
                                                                  np.float32)), range(n_probe_all), chunksize=256))
    os.close(pfd)
    pn = np.linalg.norm(P, axis=1)
    assert np.all(np.abs(pn - 1) < 1e-2), "last5_rp probe rows are not unit-norm"
    # direction-level exclusion vs ALL pool_heldout cluster dirs + eval cluster_dirs + indist_probe_dirs
    ref_cl = torch.cat([torch.from_numpy(np.asarray(ho_vecs[np.array(ho_rows["cluster"], np.int64)])),
                        eval_dirs["cluster"], eval_dirs["indist_probe"]]).float()
    ref_cl = F.normalize(ref_cl, dim=-1).to(dev)
    maxcos = np.concatenate(list(_max_cos_vs((P[c0:c0 + 16384] for c0 in range(0, n_probe_all, 16384)), ref_cl, dev)))
    dir_leak = maxcos > LEAK_COS
    # how many EVAL cluster / indist_probe directions (the 512+512 actually scored) have an exact dup in the probe bank?
    n_ho_cl = len(ho_rows["cluster"])
    ref_max = torch.full((ref_cl.shape[0],), -1.0, device=dev)
    with torch.no_grad():
        for c0 in range(0, n_probe_all, 16384):
            x = torch.from_numpy(np.ascontiguousarray(P[c0:c0 + 16384])).to(dev)
            ref_max = torch.maximum(ref_max, (ref_cl @ x.T).max(1).values)
    ref_max = ref_max.cpu().numpy()
    eval_dups = {"pool_heldout_cluster": int((ref_max[:n_ho_cl] > LEAK_COS).sum()),
                 "eval_cluster_dirs": int((ref_max[n_ho_cl:n_ho_cl + 512] > LEAK_COS).sum()),
                 "indist_probe_dirs": int((ref_max[n_ho_cl + 512:] > LEAK_COS).sum())}
    log(f"probe bank {PROBE_BANK} contains exact dups (cos>{LEAK_COS}) of eval dirs: {eval_dups} "
        f"(of {n_ho_cl} / 512 / 512) — those bank rows are EXCLUDED here")
    n_drop_dir = int((dir_leak & keep_mask).sum())
    keep_mask &= ~dir_leak
    cand = np.flatnonzero(keep_mask)
    n_probe = min(n_per_family, len(cand))
    sel = np.sort(nrng.choice(cand, n_probe, replace=False))
    vec_cl = np.memmap("/root/bank/stage_cluster.f32", np.float32, "w+", shape=(n_probe, D_MODEL))
    vec_cl[:] = P[sel]
    rec_cl = []
    for j, i in enumerate(sel.tolist()):
        r = precs[i]
        rec_cl.append({"family": "cluster", "target_text": r["target_text"], "src_vec_idx": int(r["src_vec_idx"]),
                       "last5_vec_idx": int(r["vec_idx"]), "peak_idx": r.get("peak_idx"), "n_tok": r.get("n_tok"),
                       "truncated": r.get("truncated"), "cos_peak": r.get("cos_peak")})
    stage["cluster"] = (vec_cl, rec_cl)
    fam_stats["cluster"] = {"source": PROBE_BANK, "available": n_probe_all, "dropped_indist_tail": n_drop_tail,
                            "dropped_dir_leak": n_drop_dir, "candidates": int(len(cand)), "taken": n_probe,
                            "shortfall": n_per_family - n_probe, "max_cos_vs_eval_cluster_after": float(maxcos[sel].max()),
                            "eval_dirs_with_exact_dup_in_source_bank": eval_dups, "probe_dup": 1}
    del P, ref_cl
    log(f"cluster: {n_probe}/{n_per_family} (avail {n_probe_all}, drop tail {n_drop_tail}, drop dir-leak {n_drop_dir}, "
        f"{time.time() - t0:.0f}s)")

    # ============================================================================================================
    # family: sae (unit encoder columns, alive, eval-excluded)
    # ============================================================================================================
    t0 = time.time()
    sae = load_sae(path=SAE_PT, device=dev, dtype=torch.float32)
    ma = torch.load(MAXACTS_PT, map_location="cpu", weights_only=False)
    Fd = sae.d_sae
    assert Fd == es["meta"]["d_sae"] == ma["max_acts"].shape[0] == 131072
    peak = ma["max_acts"].reshape(Fd, -1).max(1).values.float().numpy()
    alive = peak > 0
    excl_mask = np.zeros(Fd, bool)
    excl_mask[np.array(excl_sae, np.int64)] = True
    cand = np.flatnonzero(alive & ~excl_mask)
    n_sae = min(n_per_family, len(cand))
    feats = np.sort(nrng.choice(cand, n_sae, replace=False))
    vec_sae = np.memmap("/root/bank/stage_sae.f32", np.float32, "w+", shape=(n_sae, D_MODEL))
    for c0 in range(0, n_sae, 8192):
        vec_sae[c0:c0 + 8192] = sae.enc_dirs(feats[c0:c0 + 8192].tolist()).float().cpu().numpy()
    rec_sae = []
    win_tok, win_act = ma["max_tokens"][:, 0], ma["max_acts"][:, 0]          # top-1 window per feature [F, 32]
    for f in feats.tolist():
        ids = win_tok[f].tolist()
        acts = win_act[f]
        rec_sae.append({"family": "sae", "feature": f, "target_text": tok.decode(ids), "n_tok": len(ids),
                        "peak_idx": int(acts.argmax()), "corpus_peak": float(peak[f]), "window_peak": float(acts.max())})
    stage["sae"] = (vec_sae, rec_sae)
    fam_stats["sae"] = {"source": SAE_PT, "hf": SAE_HF, "d_sae": Fd, "dead": int((~alive).sum()),
                        "excluded": len(excl_sae), "candidates": int(len(cand)), "taken": n_sae,
                        "shortfall": n_per_family - n_sae, "dir": "unit(W_enc[:, f]) (mxf.sae.enc_dirs)",
                        "target": "top-1 max-activating 32-token corpus window from /data/sae/maxacts.pt"}
    del sae, ma
    torch.cuda.empty_cache()
    log(f"sae: {n_sae}/{n_per_family} (alive {int(alive.sum())}, excluded {len(excl_sae)}, cand {len(cand)}, "
        f"{time.time() - t0:.0f}s)")

    # ============================================================================================================
    # norm-filter threshold (10x median ||act - mu|| over a presample) — shared by every real-activation family
    # ============================================================================================================
    t0 = time.time()
    ps_s = nrng.integers(0, n_train, NORM_PRESAMPLE); ps_p = nrng.integers(16, T, NORM_PRESAMPLE)
    ps_raw = np.empty((NORM_PRESAMPLE, D_MODEL), np.float16)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda k: read_act(int(ps_s[k]), int(ps_p[k]), ps_raw[k]), range(NORM_PRESAMPLE), chunksize=64))
    med = float(np.median(np.linalg.norm(ps_raw.astype(np.float32) - mu_acts, axis=1)))
    thr = NORM_FILTER_MULT * med
    del ps_raw
    log(f"norm filter: median {med:.1f} thr {thr:.1f} ({NORM_PRESAMPLE} presample, "
        f"{NORM_PRESAMPLE / max(time.time() - t0, 1):.0f} reads/s)")

    # ============================================================================================================
    # families: realact (short, PREFIX harvest) + realact_long (WINDOW harvest deep in the sequence)
    # ============================================================================================================
    raw = np.empty((chunk, D_MODEL), np.float16)
    for fam, mode, lo, hi in (("realact", "prefix", short_p_lo, short_p_hi), ("realact_long", "window", long_p_lo, long_p_hi)):
        t0 = time.time()
        n_pos = hi - lo + 1
        n_cand = min(int(n_per_family * 1.6) + 4096, n_train * n_pos)
        flat = nrng.choice(n_train * n_pos, size=n_cand, replace=False)
        cand_s = (flat // n_pos).astype(np.int64); cand_p = (lo + flat % n_pos).astype(np.int64)
        del flat
        vec = np.memmap(f"/root/bank/stage_{fam}.f32", np.float32, "w+", shape=(n_per_family, D_MODEL))
        recs = []
        kept = drop_norm = drop_txt = drop_cap = 0
        for c0 in range(0, n_cand, chunk):
            if kept >= n_per_family:
                break
            c1 = min(c0 + chunk, n_cand); m = c1 - c0
            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(lambda k: read_act(int(cand_s[c0 + k]), int(cand_p[c0 + k]), raw[k]), range(m), chunksize=256))
            d = raw[:m].astype(np.float32) - mu_acts
            norms = np.linalg.norm(d, axis=1)
            for k in range(m):
                if kept >= n_per_family:
                    break
                if not (1e-6 < norms[k] <= thr):
                    drop_norm += 1; continue
                s, p = int(cand_s[c0 + k]), int(cand_p[c0 + k])
                if per_doc[s] >= doc_cap:
                    drop_cap += 1; continue
                if mode == "prefix":
                    extra = wrng.randint(1, 4)
                    start, end = 0, min(p + 1 + extra, T)
                    ids = toks[s, start:end].tolist(); txt = tok.decode(ids)
                else:
                    extra = 0
                    start, ids, txt = window_text(s, p); end = p + 1
                if len(txt.strip()) < 3:
                    drop_txt += 1; continue
                vec[kept] = d[k] / norms[k]
                recs.append({"family": fam, "target_text": txt, "harvest": mode, "seq": s, "pos": p, "start": start,
                             "extra": extra, "n_tok": len(ids), "fire_from_end": end - 1 - p, "ctx_tokens": p + 1,
                             "act_norm": round(float(norms[k]), 2)})
                per_doc[s] += 1; kept += 1
            log(f"{fam}/{mode} {kept}/{n_per_family} (cands {c1}/{n_cand}, {kept / max(time.time() - t0, 1):.0f}/s, "
                f"drops norm={drop_norm} txt={drop_txt} cap={drop_cap})")
        assert kept == n_per_family, f"{fam} quota missed: {kept} (drops norm={drop_norm} txt={drop_txt} cap={drop_cap})"
        stage[fam] = (vec, recs)
        fam_stats[fam] = {"source": ACTS, "harvest": mode, "p_range": [lo, hi], "taken": kept, "shortfall": 0,
                          "dir": "unit(act[s,p] - whiten_mu)", "train_rows": [0, n_train - 1],
                          "held_out_rows": [n_train, NS - 1], "drops": {"norm": drop_norm, "txt": drop_txt, "doc_cap": drop_cap},
                          "target": ("doc prefix [0, p+extra], extra~U[1,4]" if mode == "prefix"
                                     else f"W~U[{w_lo},{w_hi}]-token window ending at p"),
                          "docs_used": int(len({r['seq'] for r in recs}))}
    del raw

    # ============================================================================================================
    # family: bsf (SASA top-block subspace component of real activations, mapped back to residual space)
    # ============================================================================================================
    t0 = time.time()
    sasa = torch.load(f"{BSF_DIR}/sasa.pt", map_location="cpu", weights_only=False)
    G, b, k_bsf = int(sasa["G"]), int(sasa["b"]), int(sasa.get("k", bsf_meta.get("k", 32)))
    assert int(sasa["d"]) == D_MODEL and G * b == sasa["E"].shape[1]
    E = sasa["E"].float().to(dev)                                # [d, G*b]
    del sasa
    Q = torch.load(f"{BSF_DIR}/blocks_Q.pt", map_location="cpu", weights_only=False)["Q"].float()   # [G, b, d]
    assert tuple(Q.shape) == (G, b, D_MODEL)
    mu_bsf = torch.from_numpy(np.load(f"{BSF_DIR}/whiten_mu.npy").astype(np.float32)).to(dev)
    zca_np = np.load(f"{BSF_DIR}/whiten_zca.npy").astype(np.float64)
    zca_asym = float(np.abs(zca_np - zca_np.T).max())
    zca_inv_np = np.linalg.inv(zca_np)
    inv_err = float(np.abs(zca_np @ zca_inv_np - np.eye(D_MODEL)).max())
    zca = torch.from_numpy(zca_np.astype(np.float32)).to(dev)
    zca_inv = torch.from_numpy(zca_inv_np.astype(np.float32)).to(dev)
    log(f"BSF loaded: G={G} b={b} k={k_bsf} | zca asym {zca_asym:.2e} | ||zca zca^-1 - I||max {inv_err:.2e} "
        f"({time.time() - t0:.0f}s)")

    @torch.no_grad()
    def block_topk(x_dev, n_ranks):
        """x_dev [n, d] RAW acts (fp32, on dev) -> (top block ids [n, n_ranks] int64, gn values [n, n_ranks])."""
        y = F.normalize((x_dev - mu_bsf) @ zca, dim=-1)
        outs_i, outs_v = [], []
        for c0 in range(0, y.shape[0], 4096):
            z = (y[c0:c0 + 4096] @ E).view(-1, G, b)
            gn = z.norm(dim=-1)
            v, i = gn.topk(n_ranks, dim=-1)
            outs_i.append(i); outs_v.append(v)
        return torch.cat(outs_i), torch.cat(outs_v)

    # (b) eval-derived exclusion: top-1 block under THIS BSF of each eval bsf direction (unit, already centered)
    with torch.no_grad():
        ev = eval_dirs["bsf"].to(dev)
        yv = F.normalize(ev @ zca, dim=-1)
        gnv = (yv @ E).view(-1, G, b).norm(dim=-1)
        eval_bsf_new = sorted(set(gnv.argmax(1).cpu().tolist()))
    excl_blk = np.zeros(G, bool)
    excl_blk[np.array(excl_bsf_old, np.int64)] = True
    excl_blk[np.array(eval_bsf_new, np.int64)] = True
    excl_bsf_all = sorted(np.flatnonzero(excl_blk).tolist())
    log(f"bsf exclusions: {len(excl_bsf_old)} literal old ids + {len(eval_bsf_new)} eval-dir top blocks under this BSF "
        f"-> {len(excl_bsf_all)} blocks excluded, {G - len(excl_bsf_all)} usable")

    # scan: whole TRAIN sequences (contiguous reads), positions >= 16, top-`bsf_ranks` blocks per token
    n_scan = min(bsf_scan_seqs, n_train)
    scan_seqs = np.sort(nrng.choice(n_train, n_scan, replace=False))
    P0 = 16
    n_pos = T - P0
    top_i = np.empty((n_scan, n_pos, bsf_ranks), np.int32)
    ok = np.zeros((n_scan, n_pos), bool)
    mu_acts_t = torch.from_numpy(mu_acts).to(dev)
    SC = 32
    t1 = time.time()
    with ThreadPoolExecutor(16) as ex:
        for c0 in range(0, n_scan, SC):
            ss = scan_seqs[c0:c0 + SC]
            blk = np.stack(list(ex.map(read_seq, ss.tolist())))[:, P0:, :]          # [c, n_pos, d] f16
            x = torch.from_numpy(blk.astype(np.float32)).to(dev).view(-1, D_MODEL)
            nrm = (x - mu_acts_t).norm(dim=-1)
            ti, _ = block_topk(x, bsf_ranks)
            c = len(ss)
            top_i[c0:c0 + c] = ti.view(c, n_pos, bsf_ranks).cpu().numpy()
            ok[c0:c0 + c] = ((nrm > 1e-6) & (nrm <= thr)).view(c, n_pos).cpu().numpy()
            if (c0 // SC) % 20 == 0:
                el = time.time() - t1
                log(f"bsf scan {c0 + c}/{n_scan} seqs ({(c0 + c) * T * D_MODEL * 2 / 2**30 / max(el, 1):.2f} GB/s, "
                    f"{(c0 + c) / max(el, 1):.1f} seq/s)")
    top1_blocks = np.unique(top_i[..., 0][ok])
    log(f"bsf scan done: {n_scan} seqs x {n_pos} pos = {n_scan * n_pos} tokens, {len(top1_blocks)} distinct top-1 blocks "
        f"({time.time() - t1:.0f}s)")

    # selection: level-by-level (each block gets <= 1 new row per level), within a level prefer lower rank
    n_tok = n_scan * n_pos
    ti_flat = top_i.reshape(n_tok, bsf_ranks)
    seq_of = np.repeat(scan_seqs, n_pos)
    ok_flat = ok.reshape(n_tok)
    used = np.zeros(n_tok, bool)
    block_cnt = np.zeros(G, np.int32)
    sel_tok, sel_rank = [], []
    total = 0
    for level in range(bsf_cap):
        for r in range(bsf_ranks):
            if total >= n_per_family:
                break
            bl = ti_flat[:, r]
            cand = ok_flat & ~used & (block_cnt[bl] == level) & ~excl_blk[bl] & (per_doc[seq_of] < doc_cap)
            idx = np.flatnonzero(cand)
            if len(idx) == 0:
                continue
            nrng.shuffle(idx)
            _, first = np.unique(bl[idx], return_index=True)            # one random token per block
            picks = idx[np.sort(first)]
            # per-document cap within this pass: keep the first (doc_cap - per_doc[s]) picks of each document
            order = nrng.permutation(len(picks)); picks = picks[order]
            s_p = seq_of[picks]
            so = np.argsort(s_p, kind="stable"); s_sorted = s_p[so]
            starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
            grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(s_sorted)]))
            within = np.arange(len(s_sorted)) - starts[grp]
            keep = within < (doc_cap - per_doc[s_sorted])
            picks = picks[so[keep]]
            if total + len(picks) > n_per_family:
                picks = picks[: n_per_family - total]
            used[picks] = True
            np.add.at(block_cnt, bl[picks], 1)
            np.add.at(per_doc, seq_of[picks], 1)
            sel_tok.append(picks); sel_rank.append(np.full(len(picks), r + 1, np.int8))
            total += len(picks)
            log(f"bsf select level {level + 1}/{bsf_cap} rank {r + 1}: +{len(picks)} -> {total}/{n_per_family} "
                f"(blocks with rows: {int((block_cnt > 0).sum())})")
        if total >= n_per_family:
            break
    sel_tok = np.concatenate(sel_tok) if sel_tok else np.zeros(0, np.int64)
    sel_rank = np.concatenate(sel_rank) if sel_rank else np.zeros(0, np.int8)
    n_bsf = int(len(sel_tok))
    sel_blk = ti_flat[sel_tok, sel_rank.astype(np.int64) - 1].astype(np.int64)
    sel_s = seq_of[sel_tok].astype(np.int64); sel_p = (P0 + sel_tok % n_pos).astype(np.int64)
    assert not excl_blk[sel_blk].any() and (block_cnt.max() <= bsf_cap)
    del top_i, ok, ti_flat, seq_of, ok_flat, used
    log(f"bsf selected {n_bsf}/{n_per_family}: {int((block_cnt > 0).sum())} blocks, rows/block max {int(block_cnt.max())}, "
        f"rank hist {np.bincount(sel_rank, minlength=bsf_ranks + 1)[1:].tolist()}")

    # re-read the selected activations (random preads), mint the directions, verify
    vec_bsf = np.memmap("/root/bank/stage_bsf.f32", np.float32, "w+", shape=(n_bsf, D_MODEL))
    rec_bsf = [None] * n_bsf
    raw_b = np.empty((n_bsf, D_MODEL), np.float16)
    t1 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda j: read_act(int(sel_s[j]), int(sel_p[j]), raw_b[j]), range(n_bsf), chunksize=256))
    log(f"bsf re-read {n_bsf} acts ({n_bsf / max(time.time() - t1, 1):.0f}/s)")
    cos_x_all = np.empty(n_bsf, np.float32); gn_all = np.empty(n_bsf, np.float32)
    rank_ok = 0; whiten_frac = np.empty(n_bsf, np.float32)
    with torch.no_grad():
        for c0 in range(0, n_bsf, 2048):
            c1 = min(c0 + 2048, n_bsf)
            x = torch.from_numpy(raw_b[c0:c1].astype(np.float32)).to(dev)
            xc = x - mu_bsf                                                  # centered raw act
            y = xc @ zca                                                     # whitened (unnormalized)
            bl = torch.from_numpy(sel_blk[c0:c1]).to(dev)
            Qb = Q[bl.cpu()].to(dev)                                         # [c, b, d] orthonormal rows
            coords = torch.einsum("cd,cbd->cb", y, Qb)
            yb = torch.einsum("cb,cbd->cd", coords, Qb)                      # component of y in block subspace
            xb = yb @ zca_inv                                                # back to residual space
            dirs = F.normalize(xb, dim=-1)
            cos_x_all[c0:c1] = F.cosine_similarity(xb, xc, dim=-1).cpu().numpy()
            whiten_frac[c0:c1] = (yb.norm(dim=-1) / y.norm(dim=-1)).cpu().numpy()
            ti, tv = block_topk(x, bsf_ranks)                                # consistency with the scan
            hit = (ti == bl[:, None])
            rank_ok += int(hit.any(1).sum())
            gn_all[c0:c1] = torch.where(hit, tv, torch.zeros_like(tv)).sum(1).cpu().numpy()
            vec_bsf[c0:c1] = dirs.cpu().numpy()
    for j in range(n_bsf):
        s, p = int(sel_s[j]), int(sel_p[j])
        start, ids, txt = window_text(s, p)
        rec_bsf[j] = {"family": "bsf", "target_text": txt, "block": int(sel_blk[j]), "rank": int(sel_rank[j]),
                      "gnorm": round(float(gn_all[j]), 4), "cos_x": round(float(cos_x_all[j]), 4),
                      "whiten_frac": round(float(whiten_frac[j]), 4), "seq": s, "pos": p, "start": start,
                      "n_tok": len(ids), "fire_from_end": 0, "ctx_tokens": p + 1}
    stage["bsf"] = (vec_bsf, rec_bsf)
    rows_per_block = np.bincount(block_cnt[block_cnt > 0]) if n_bsf else np.zeros(1)
    fam_stats["bsf"] = {"source": f"{BSF_HF} (cached {BSF_DIR}) on {ACTS} train sequences", "G": G, "b": b, "k": k_bsf,
                        "scan_seqs": int(n_scan), "scan_tokens": int(n_tok), "pos_range": [P0, T - 1],
                        "distinct_top1_blocks_in_scan": int(len(top1_blocks)), "taken": n_bsf,
                        "shortfall": n_per_family - n_bsf, "blocks_used": int((block_cnt > 0).sum()),
                        "cap_per_block": bsf_cap, "rows_per_block_hist": rows_per_block.tolist(),
                        "rank_hist": np.bincount(sel_rank, minlength=bsf_ranks + 1)[1:].tolist(),
                        "excluded_blocks": {"literal_old_ids": len(excl_bsf_old), "eval_dir_top1_under_this_bsf": len(eval_bsf_new),
                                            "total": len(excl_bsf_all)},
                        "block_consistency_rescan": rank_ok / max(n_bsf, 1),
                        "cos_x": {q: float(np.percentile(cos_x_all, q)) for q in (5, 10, 25, 50, 75, 90, 95)} | {"mean": float(cos_x_all.mean())},
                        "whiten_frac": {"median": float(np.median(whiten_frac)), "mean": float(whiten_frac.mean())},
                        "zca_asym_max": zca_asym, "zca_inv_err_max": inv_err,
                        "dir": "unit((Q[b]^T Q[b] ((x - mu_bsf) @ zca)) @ zca^-1), b = top active block (rank <= bsf_ranks)",
                        "target": f"W~U[{w_lo},{w_hi}]-token window ending at p", "docs_used": int(len(set(sel_s.tolist())))}
    del E, Q, raw_b, zca, zca_inv
    torch.cuda.empty_cache()
    log(f"bsf: {n_bsf}/{n_per_family} | cos(dir, x-mu) median {np.median(cos_x_all):.3f} mean {cos_x_all.mean():.3f} "
        f"p10 {np.percentile(cos_x_all, 10):.3f} | whitened-norm fraction median {np.median(whiten_frac):.3f} | "
        f"block re-scan consistency {rank_ok}/{n_bsf} ({time.time() - t0:.0f}s)")

    # ============================================================================================================
    # assemble: seeded shuffle of ALL rows -> vecs.f32 + records.jsonl (line i == vec_idx i)
    # ============================================================================================================
    t0 = time.time()
    counts = {f: len(stage[f][1]) for f in FAMILIES}
    N = sum(counts.values())
    perm = nrng.permutation(N)                        # staged global row g -> final row perm[g]
    inv = np.empty(N, np.int64); inv[perm] = np.arange(N)
    fam_of = np.concatenate([np.full(counts[f], i, np.int8) for i, f in enumerate(FAMILIES)])
    off = np.cumsum([0] + [counts[f] for f in FAMILIES])
    vecs = np.memmap("/root/bank/vecs.f32", np.float32, "w+", shape=(N, D_MODEL))
    recs_final = [None] * N
    for i in range(N):
        g = int(inv[i]); fi = int(fam_of[g]); j = g - int(off[fi])
        r = dict(stage[FAMILIES[fi]][1][j]); r = {"vec_idx": i, **r}
        recs_final[i] = r
    for fi, f in enumerate(FAMILIES):
        v, _ = stage[f]
        rows = perm[off[fi]:off[fi + 1]]
        for c0 in range(0, counts[f], 8192):
            vecs[rows[c0:c0 + 8192]] = v[c0:c0 + 8192]
        del v
    vecs.flush()
    with open("/root/bank/records.jsonl", "w") as rf:
        for r in recs_final:
            rf.write(json.dumps(r) + "\n")
    log(f"assembled {N} rows ({time.time() - t0:.0f}s): " + " ".join(f"{f}={counts[f]}" for f in FAMILIES))

    # ============================================================================================================
    # verification from the assembled files: unit norms, alignment, direction-level leakage vs eval + pool_heldout
    # ============================================================================================================
    t0 = time.time()
    vecs = np.memmap("/root/bank/vecs.f32", np.float32, "r", shape=(N, D_MODEL))
    nrm_min, nrm_max = 1.0, 1.0
    for c0 in range(0, N, 32768):
        nn = np.linalg.norm(vecs[c0:c0 + 32768], axis=1)
        nrm_min, nrm_max = min(nrm_min, float(nn.min())), max(nrm_max, float(nn.max()))
    assert abs(nrm_min - 1) < 1e-3 and abs(nrm_max - 1) < 1e-3, f"non-unit rows: [{nrm_min}, {nrm_max}]"
    ref_names = sorted(eval_dirs) + [f"pool_heldout/{f}" for f in sorted(ho_rows)]
    ref = torch.cat([eval_dirs[f] for f in sorted(eval_dirs)] +
                    [F.normalize(torch.from_numpy(np.asarray(ho_vecs[np.array(ho_rows[f], np.int64)])).float(), dim=-1)
                     for f in sorted(ho_rows)]).to(dev)
    ref_off = np.cumsum([0] + [len(eval_dirs[f]) for f in sorted(eval_dirs)] + [len(ho_rows[f]) for f in sorted(ho_rows)])
    fam_arr = np.array([FAMILIES.index(r["family"]) for r in recs_final], np.int8)
    leak = np.zeros((len(FAMILIES), len(ref_names)), np.float32) - 1
    worst = []
    with torch.no_grad():
        for c0 in range(0, N, 8192):
            x = torch.from_numpy(np.ascontiguousarray(vecs[c0:c0 + 8192])).to(dev)
            cos = x @ ref.T                                                      # [c, M]
            for ri in range(len(ref_names)):
                cm = cos[:, ref_off[ri]:ref_off[ri + 1]].max(1).values.cpu().numpy()
                for fi in range(len(FAMILIES)):
                    m = fam_arr[c0:c0 + 8192] == fi
                    if m.any():
                        leak[fi, ri] = max(leak[fi, ri], float(cm[m].max()))
            mx, am = cos.max(1)
            bad = (mx > LEAK_COS).nonzero().flatten().tolist()
            for j in bad:
                worst.append((c0 + j, int(am[j]), float(mx[j])))
    leak_tbl = {FAMILIES[fi]: {ref_names[ri]: round(float(leak[fi, ri]), 4) for ri in range(len(ref_names))}
                for fi in range(len(FAMILIES))}
    log("leakage max-cos table (bank family x eval/held-out family):")
    for f in FAMILIES:
        log(f"  {f:13s} " + " ".join(f"{n.split('/')[-1][:12]}={leak_tbl[f][n]:.3f}" for n in ref_names))
    assert not worst, f"{len(worst)} bank rows within cos>{LEAK_COS} of an eval/held-out direction: {worst[:5]}"
    # id-level asserts
    for r in recs_final:
        if r["family"] == "sae":
            assert not excl_mask[r["feature"]]
        elif r["family"] == "bsf":
            assert not excl_blk[r["block"]]
        elif r["family"] == "cluster":
            assert r["src_vec_idx"] not in tail_set
        else:
            assert 0 <= r["seq"] < n_train
    assert all(recs_final[i]["vec_idx"] == i for i in range(N)) and len(recs_final) == N
    assert not any(r["family"] == "jlens" for r in recs_final)
    log(f"verification passed ({time.time() - t0:.0f}s): unit norms [{nrm_min:.5f}, {nrm_max:.5f}], no id leakage, "
        f"max cos vs any eval/held-out dir < {LEAK_COS}")

    # ============================================================================================================
    # publish
    # ============================================================================================================
    t0 = time.time()
    stats = {"kind": "everything: even mix of realact (short-ctx prefix harvest) + realact_long (deep-position window "
                     "harvest) + sae (unit encoder cols) + bsf (top-block subspace component of real acts) + cluster (probes)",
             "n_examples": N, "n_vecs": N, "families": counts, "layout": "seeded shuffle of all families (records.jsonl line i == vec_idx i)",
             "n_per_family_target": n_per_family, "seed": seed, "d_model": D_MODEL, "model": MODEL, "acts": ACTS,
             "created": time.time()}
    meta_out = {"bank": out, "n_rows": N, "families": counts, "n_per_family_target": n_per_family,
                "shortfalls": {f: fam_stats[f]["shortfall"] for f in FAMILIES},
                "row_order": "np.random.default_rng(seed).permutation over all staged rows (families interleaved)",
                "seed": seed, "d_model": D_MODEL, "model": MODEL, "layer": 42,
                "trainer_args": {"--data-dir": out, "--bank-file": "vecs.f32", "--direction-source": "cluster"},
                "family_recipes": fam_stats,
                "whitening_mu": {"realact/realact_long": f"{ACTS}/whiten_mu.npy", "bsf": f"{BSF_DIR}/whiten_mu.npy (+ whiten_zca.npy)"},
                "norm_filter": {"mult": NORM_FILTER_MULT, "median": med, "thr": thr, "presample": NORM_PRESAMPLE},
                "doc_cap": doc_cap, "acts_train_rows": [0, n_train - 1], "acts_held_out_rows": [n_train, NS - 1],
                "exclusions_summary": {
                    "sae_features": {"n": len(excl_sae), "sources": [f"{POOL_HELDOUT}/records.jsonl family=sae 'feature' (13107)",
                                                                    f"{EVAL_CACHE} sae_feats (512, subset)",
                                                                    "dead features (corpus peak <= 0 in /data/sae/maxacts.pt)"],
                                     "dead": int((~alive).sum())},
                    "bsf_blocks": {"literal_old_ids": len(excl_bsf_old), "eval_dir_top1_under_this_bsf": len(eval_bsf_new),
                                   "total": len(excl_bsf_all),
                                   "sources": [f"{POOL_HELDOUT}/records.jsonl family=bsf 'block' (ids of the ORIGINAL Aug-21 BSF)",
                                               f"{EVAL_CACHE} bsf_dirs -> argmax block under {BSF_HF}"]},
                    "probe_rows": {"n": len(excl_probe_src), "range": [excl_probe_src[0], excl_probe_src[-1]],
                                   "sources": [f"{POOL_RL_MIX} cluster tail (last {N_TAIL_INDIST} rows) = indist_probe reservation "
                                               "(MAEMMBench/build_indist_eval.py N_TAIL)",
                                               f"direction-level: cos > {LEAK_COS} vs pool_heldout cluster dirs + eval cluster_dirs + indist_probe_dirs"],
                                   "dropped_indist_tail": n_drop_tail, "dropped_dir_leak": n_drop_dir},
                    "realact": {"rule": f"acts27b seq rows < {n_train} only (last 5% held out); eval realact/indist/ctx families come from other acts dumps"},
                    "jlens": "no J-lens rows in the bank (fully held-out family)"},
                "leakage_check": {"threshold": LEAK_COS, "max_cos_table": leak_tbl, "refs": ref_names},
                "unit_norm_range": [nrm_min, nrm_max], "eval_cache": EVAL_CACHE,
                "eval_cache_cos_families": es["meta"].get("cos_families")}
    excl_out = {"sae_features": excl_sae, "bsf_blocks_literal_old_ids": excl_bsf_old,
                "bsf_blocks_eval_dir_top1_under_this_bsf": eval_bsf_new, "bsf_blocks_all": excl_bsf_all,
                "probe_src_vec_idx_pool_rl_mix": excl_probe_src, "sae_dead_features": np.flatnonzero(~alive).tolist(),
                "eval_cache_sae_feats": eval_sae_feats, "eval_bsf_blocks_old_ids_used_by_cache": eval_bsf_old}
    os.makedirs(out, exist_ok=True)
    for fn in ("vecs.f32", "records.jsonl"):
        shutil.copyfile(f"/root/bank/{fn}", f"{out}/{fn}")
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
    json.dump(meta_out, open(f"{out}/meta.json", "w"), indent=1)
    json.dump(excl_out, open(f"{out}/exclusions.json", "w"))
    vol.commit()
    vsize = os.path.getsize(f"{out}/vecs.f32")
    assert vsize == N * row_b, f"vecs.f32 {vsize} B != {N} x {row_b}"
    log(f"DONE -> {out}: {N} rows ({vsize / 2**30:.1f} GB) " + " ".join(f"{f}={counts[f]}" for f in FAMILIES)
        + f" | published in {time.time() - t0:.0f}s | total {(time.time() - T0) / 60:.1f} min")
    return {"out": out, "n": N, "families": counts, "shortfalls": meta_out["shortfalls"], "bsf_cos_x": fam_stats["bsf"]["cos_x"]}


# ----------------------------------------------------------------------------------------------------------------
# verify (from the FINAL artifacts on the volume)
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=BUILD_GPUS, cpu=8, memory=49152, volumes={"/data": vol}, timeout=3600)
def verify(out_name: str = OUT_DEFAULT):
    import json, os, sys, time
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL
    dev = "cuda:0"
    vol.reload()
    out = f"/data/{out_name}"
    st = json.load(open(f"{out}/build_stats.json"))
    mt = json.load(open(f"{out}/meta.json"))
    ex = json.load(open(f"{out}/exclusions.json"))
    N = st["n_examples"]
    row_b = D_MODEL * 4
    assert os.path.getsize(f"{out}/vecs.f32") == N * row_b, "vecs.f32 size != n_examples rows"
    vecs = np.memmap(f"{out}/vecs.f32", np.float32, "r", shape=(N, D_MODEL))
    recs = [json.loads(l) for l in open(f"{out}/records.jsonl")]
    assert len(recs) == N and all(r["vec_idx"] == i for i, r in enumerate(recs)), "records.jsonl misaligned"
    counts = {}
    for r in recs:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
        assert r.get("target_text") is not None and "family" in r
    assert counts == st["families"], (counts, st["families"])
    assert "jlens" not in counts
    excl_sae = set(ex["sae_features"]) | set(ex["sae_dead_features"]); excl_blk = set(ex["bsf_blocks_all"])
    excl_probe = set(ex["probe_src_vec_idx_pool_rl_mix"])
    n_train = mt["acts_train_rows"][1] + 1
    for r in recs:
        f = r["family"]
        if f == "sae":
            assert r["feature"] not in excl_sae
        elif f == "bsf":
            assert r["block"] not in excl_blk
        elif f == "cluster":
            assert r["src_vec_idx"] not in excl_probe
        else:
            assert 0 <= r["seq"] < n_train
    # unit norms + direction-level leakage vs eval cache + pool_heldout
    es = torch.load(EVAL_CACHE, map_location="cpu", weights_only=False)
    names = sorted(k[:-5] for k in es if k.endswith("_dirs"))
    ho_sz = os.path.getsize(f"{POOL_HELDOUT}/vecs.f32")
    ho = np.memmap(f"{POOL_HELDOUT}/vecs.f32", np.float32, "r", shape=(ho_sz // row_b, D_MODEL))
    ref = F.normalize(torch.cat([es[f"{n}_dirs"].float() for n in names] + [torch.from_numpy(np.asarray(ho)).float()]), dim=-1).to(dev)
    names_all = names + ["pool_heldout(all)"]
    offs = np.cumsum([0] + [len(es[f"{n}_dirs"]) for n in names] + [ho.shape[0]])
    fams = sorted(counts)
    fam_arr = np.array([fams.index(r["family"]) for r in recs])
    leak = np.full((len(fams), len(names_all)), -1.0)
    nmin, nmax, n_bad = 1.0, 1.0, 0
    t0 = time.time()
    with torch.no_grad():
        for c0 in range(0, N, 8192):
            x = torch.from_numpy(np.ascontiguousarray(vecs[c0:c0 + 8192])).to(dev)
            nn = x.norm(dim=-1); nmin, nmax = min(nmin, float(nn.min())), max(nmax, float(nn.max()))
            cos = x @ ref.T
            n_bad += int((cos.max(1).values > LEAK_COS).sum())
            for ri in range(len(names_all)):
                cm = cos[:, offs[ri]:offs[ri + 1]].max(1).values.cpu().numpy()
                for fi in range(len(fams)):
                    m = fam_arr[c0:c0 + 8192] == fi
                    if m.any():
                        leak[fi, ri] = max(leak[fi, ri], float(cm[m].max()))
    assert abs(nmin - 1) < 1e-3 and abs(nmax - 1) < 1e-3, (nmin, nmax)
    assert n_bad == 0, f"{n_bad} rows within cos>{LEAK_COS} of an eval/held-out direction"
    print(f"[verify] {out}: {N} rows, families {counts}, unit norms [{nmin:.5f},{nmax:.5f}], "
          f"no id leakage, no direction leakage (>{LEAK_COS}) vs {len(names_all)} ref sets ({time.time() - t0:.0f}s)")
    print("[verify] max-cos table (rows: bank family; cols: eval family):")
    print("               " + " ".join(f"{n[:11]:>11s}" for n in names_all))
    for fi, f in enumerate(fams):
        print(f"  {f:13s}" + " ".join(f"{leak[fi, ri]:11.3f}" for ri in range(len(names_all))))
    print("[verify] shortfalls:", mt["shortfalls"], "| bsf cos_x:", mt["family_recipes"]["bsf"]["cos_x"])
    for f in fams:
        r = next(r for r in recs if r["family"] == f)
        print(f"  sample [{f}] |v|={np.linalg.norm(vecs[r['vec_idx']]):.4f} {json.dumps({k: (v[:90] if k == 'target_text' else v) for k, v in r.items()})}")
    return {"n": N, "families": counts, "leak": {f: {n: float(leak[fi, ri]) for ri, n in enumerate(names_all)} for fi, f in enumerate(fams)}}


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=4, memory=16384)
def peek(out_name: str = OUT_DEFAULT, n: int = 3):
    import json, os
    import numpy as np
    vol.reload()
    out = f"/data/{out_name}"
    st = json.load(open(f"{out}/build_stats.json"))
    mt = json.load(open(f"{out}/meta.json"))
    print(json.dumps({k: st[k] for k in ("n_examples", "n_vecs", "families", "layout")}, indent=1))
    print("shortfalls:", mt["shortfalls"]); print("exclusions:", json.dumps(mt["exclusions_summary"], indent=1))
    print("bsf recipe:", json.dumps({k: v for k, v in mt["family_recipes"]["bsf"].items() if k != "rows_per_block_hist"}, indent=1))
    fd = os.open(f"{out}/vecs.f32", os.O_RDONLY)
    seen = {}
    with open(f"{out}/records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if seen.get(r["family"], 0) >= n:
                continue
            seen[r["family"]] = seen.get(r["family"], 0) + 1
            v = np.frombuffer(_pread_full(fd, 5120 * 4, r["vec_idx"] * 5120 * 4), np.float32)
            print(f"[{r['family']}] row {r['vec_idx']} |v|={np.linalg.norm(v):.4f} :: {r['target_text'][:110]!r}")
            if all(c >= n for c in seen.values()) and len(seen) == len(st["families"]):
                break
    return st["n_examples"]


@app.local_entrypoint()
def run_smoke(out_name: str = "banks/everything_smoke", n: int = 1000, bsf_scan_seqs: int = 256):
    print(build.remote(out_name=out_name, n_per_family=n, bsf_scan_seqs=bsf_scan_seqs, overwrite_smoke=True))
    print(verify.remote(out_name=out_name))


@app.local_entrypoint()
def run_build(out_name: str = OUT_DEFAULT, n: int = 100_000, bsf_scan_seqs: int = 20_000, seed: int = 7):
    print(build.remote(out_name=out_name, n_per_family=n, bsf_scan_seqs=bsf_scan_seqs, seed=seed))
    print(verify.remote(out_name=out_name))


@app.local_entrypoint()
def run_verify(out_name: str = OUT_DEFAULT):
    print(verify.remote(out_name=out_name))


@app.local_entrypoint()
def run_peek(out_name: str = OUT_DEFAULT):
    peek.remote(out_name=out_name)
