"""Modal app `maemm-sae2m-bank`: the SFT MIDTRAIN BANK of the 2,097,152-feature layer-42 SAE (/data/sae2m/trainer_0/ae.pt) from its
max-activating windows (/data/sae2m/maxacts_top5.pt) -> /data/banks/<out_name>/{vecs.f32, records.jsonl, build_stats.json, meta.json}
in exactly the schema of the "sae"/"sae_dec" families of data/modal_bank_everything.py, so data/modal_mix_5m_bank.build takes it as
    {"name": "sae2m", "banks": ["/data/banks/sae2m_bank"], "dtype": "f32", "families": ["sae2m", "sae2m_dec"], "n": {...}}

Per window TWO rows with the same target_text: family "sae2m" = unit(encoder row f) and "sae2m_dec" = unit(decoder column f), both
in RAW layer-42 space (the merge folded the norm factor). target_text = the stored 32-token window decoded so that it ENDS at the peak
token (left pads stripped by `lengths`), kept only if it re-tokenises to exactly the window's ids.

    source ~/modal_venv/bin/activate; export MODAL_PROFILE=safety-sahan
    cd ~/maemm-pub && modal deploy sae2m_bank/modal_sae2m_bank.py
    S='import modal, sys; f=modal.Function.from_name("maemm-sae2m-bank", sys.argv[1]); print(f.spawn(**eval(sys.argv[2] if len(sys.argv)>2 else "{}")).object_id)'
    python -c "$S" inputs_status                                   # CPU: what is on the volume
    python -c "$S" build '{"out_name": "sae2m_bank_smoke", "maxacts": "/data/sae2m/maxacts_top5_smoke.pt"}'      # smoke
    python -c "$S" build                                           # FULL: out_name sae2m_bank from /data/sae2m/maxacts_top5.pt
    python -c "$S" verify '{"out_name": "sae2m_bank"}'             # re-check the final files (unit norms, alignment, leak table)

Pipeline of build() (B200:8, one container):
  1. selection (CPU): live features (fire_count > 0, top act > threshold); candidates = top-5 windows with >= min_tok true tokens,
     no pad/BOS id inside, decode -> re-encode roundtrip exact, text-distinct within the feature; breadth-first over ranks
     (every feature gets its rank-0 window before any gets rank-1, ...) up to windows_per_feature and max_rows_per_family windows
  2. END-ANCHOR check (8 GPUs, torchrun sae2m_bank/anchor_worker.py): every selected window re-forwarded standalone through the
     truncated 27B + the feature's encoder row; kept iff the peak is the LAST token (rule "last")
  3. LEAK GUARD (GPU 0): per direction max cos vs every `*_dirs` family of the v2 eval cache and every /data/pool_heldout row;
     rows > 0.999 dropped, final files re-verified; plus the (report-only) max-cos histogram of the 2M directions vs the 131k eval
     SAE's decoder / encoder directions (all 131,072 and the eval feature subset) -- flagged when > 1% exceed 0.95
  4. publish: seeded global shuffle of all rows, vecs.f32 streamed to the volume, records line i == vec_idx i, stats + meta
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = os.environ.get("SAE2M_BANK_APP", "maemm-sae2m-bank")
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet", "datasets==4.5.0")
    .pip_install("flash-linear-attention==0.5.2")   # Qwen3.5 GatedDeltaNet -> fla Triton chunk kernel (== sae2m/modal_sae2m.py)
    .add_local_dir(REPO / "sae2m_bank", "/pmx/sae2m_bank", ignore=["__pycache__", "tests", "*.md"])
    .add_local_dir(REPO / "sae2m", "/pmx/sae2m", ignore=["__pycache__", "tests", "*.md"])
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

ROOT = "/data/sae2m"
AE_DEFAULT = f"{ROOT}/trainer_0/ae.pt"
MAXACTS_DEFAULT = f"{ROOT}/maxacts_top5.pt"
EVAL_CACHE_V2 = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
POOL_HELDOUT = "/data/pool_heldout"
SAE131K = "/data/sae/ae.pt"
SAE131K_SIZE = 5369256453                 # identity check (== data/modal_bank_everything.py SAE_HF_SIZE)
MODEL = "Qwen/Qwen3.6-27B"
LAYER = 42
D = 5120
BOS_FALLBACK = 248044
LEAK_COS = 0.999
FAM_ENC, FAM_DEC = "sae2m", "sae2m_dec"
BUILD_GPU = os.environ.get("SAE2M_BANK_GPU", "B200:8")
SMALL_GPUS = ["H100", "A100-80GB", "L40S", "B200", "H200"]


def _env():
    env = os.environ.copy()
    env.update({"HF_HOME": "/data/hf_cache", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTHONPATH": "/pmx/sae2m_bank:/pmx/sae2m:/pmx/helpers", "OMP_NUM_THREADS": "6", "PYTHONUNBUFFERED": "1"})
    return env


def _n_gpus():
    import subprocess
    try:
        return len([ln for ln in subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines() if ln.strip()])
    except Exception:  # noqa
        return 0


def _run(cmd, env, grace_s=600):
    """Run a torchrun command as its own process group, mirroring stdout; SIGTERM/SIGKILL the group on cancel/error."""
    import signal, subprocess, time
    print("[sae2m-bank] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx/sae2m_bank", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
        return p.wait()
    finally:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                t0 = time.time()
                while p.poll() is None and time.time() - t0 < grace_s:
                    time.sleep(5)
            except Exception:  # noqa
                pass
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:  # noqa
                    pass


def _load_refs(eval_cache, pool_heldout, dev):
    """Leak-guard reference directions exactly as data/modal_mix_5m_bank.build: every `*_dirs` tensor of the eval cache + every
    pool_heldout row grouped by family. Returns (ref [M, d] unit fp32 on dev, names, offs, es)."""
    import json, numpy as np, torch, torch.nn.functional as F
    es = torch.load(eval_cache, map_location="cpu", weights_only=False)
    names = sorted(k[:-5] for k in es if k.endswith("_dirs") and torch.is_tensor(es[k]))
    refs = [F.normalize(es[f"{n}_dirs"].float().reshape(-1, D), dim=-1) for n in names]
    ho_n = os.path.getsize(f"{pool_heldout}/vecs.f32") // (4 * D)
    ho = np.memmap(f"{pool_heldout}/vecs.f32", np.float32, "r", shape=(ho_n, D))
    ho_recs = [json.loads(l) for l in open(f"{pool_heldout}/records.jsonl")]
    ho_fam = np.array([r["family"] for r in ho_recs])
    for hf in sorted(set(ho_fam.tolist())):
        refs.append(F.normalize(torch.from_numpy(np.asarray(ho[np.flatnonzero(ho_fam == hf)])).float(), dim=-1)); names.append(f"pool_heldout/{hf}")
    ref = torch.cat(refs).to(dev)
    offs = np.cumsum([0] + [len(r) for r in refs])
    ho_sae_feats = sorted({int(r["feature"]) for r in ho_recs if r["family"] == "sae" and "feature" in r})
    return ref, names, offs, es, ho_sae_feats


# ----------------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=BUILD_GPU, cpu=48, memory=384 * 1024, ephemeral_disk=1024 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=12 * 3600)
def build(out_name: str = "sae2m_bank", ae: str = AE_DEFAULT, maxacts: str = MAXACTS_DEFAULT, windows_per_feature: int = 3, min_tok: int = 8,
          max_rows_per_family: int = 3_000_000, seed: int = 2026, overwrite: bool = False, anchor_batch: int = 512, anchor_rule: str = "last",
          skip_anchor: bool = False, eval_cache: str = EVAL_CACHE_V2, pool_heldout: str = POOL_HELDOUT, sae131k: str = SAE131K,
          overlap_flag_frac: float = 0.01, max_features: int = 0, feature_split: str = "", split_key: str = ""):
    """max_rows_per_family caps the number of WINDOWS (= rows per family; total rows = 2x). max_features > 0: only the first
    max_features feature ids are considered (debug). skip_anchor=True keeps every roundtrip-valid window (debug only).
    feature_split + split_key: an .npz of disjoint sorted int feature-id arrays (e.g. /data/sae2m/feature_split.npz with keys
    eval / rl / sft); ONLY the features under split_key are eligible (the others are treated as dead), and the arrays are
    asserted disjoint so a bank can never contain a held-out feature."""
    import glob, hashlib, json, shutil, sys, time
    import numpy as np
    import torch
    import torch.nn.functional as F
    for k, v in _env().items():
        os.environ[k] = v
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    sys.path.insert(0, "/pmx/sae2m_bank"); sys.path.insert(0, "/pmx/sae2m"); sys.path.insert(0, "/pmx/helpers")
    from bank_lib import (live_mask, candidate_mask, window_ids, special_in_span, roundtrip_texts, dedupe_texts, breadth_first_select, coverage_hist,
                          anchor_summary, max_cos_table, cos_hist, make_record, check_bank_files, assemble_rows)
    torch.backends.cuda.matmul.allow_tf32 = True
    T0 = time.time()
    timings = {}

    def log(m):
        print(f"[sae2m-bank +{time.time() - T0:6.0f}s] {m}", flush=True)

    vol.reload()
    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json") and not overwrite:
        raise RuntimeError(f"{out} already finalized (overwrite=False)")
    for p in (ae, maxacts, eval_cache, f"{pool_heldout}/vecs.f32", f"{pool_heldout}/records.jsonl", sae131k):
        assert os.path.exists(p), f"missing input {p}"
    assert os.path.getsize(sae131k) == SAE131K_SIZE, f"{sae131k} is not the 131k eval SAE (size {os.path.getsize(sae131k)})"
    world = _n_gpus()
    dev = "cuda:0"
    stage = "/root/stage"
    shutil.rmtree(stage, ignore_errors=True); os.makedirs(f"{stage}/anchor_in", exist_ok=True); os.makedirs(f"{stage}/anchor_out", exist_ok=True)
    K = int(windows_per_feature)
    log(f"out={out} ae={ae} maxacts={maxacts} K={K} min_tok={min_tok} cap/family={max_rows_per_family} seed={seed} gpus={world} rule={anchor_rule}")

    # ---------------------------------------------------------------- 1. inputs
    t0 = time.time()
    sd = torch.load(ae, map_location="cpu", mmap=True, weights_only=False)
    Fd, d = sd["encoder.weight"].shape
    assert d == D and tuple(sd["decoder.weight"].shape) == (D, Fd), (sd["encoder.weight"].shape, sd["decoder.weight"].shape)
    thr = float(sd["threshold"].item()); k_sae = int(sd["k"].item()) if "k" in sd else -1
    log(f"SAE: F={Fd} d={d} k={k_sae} thr={thr:.4f} enc {sd['encoder.weight'].dtype} dec {sd['decoder.weight'].dtype} "
        f"b_enc {sd['encoder.bias'].dtype} b_dec {sd['b_dec'].dtype} ({os.path.getsize(ae) / 2**30:.1f} GB)")
    ma = torch.load(maxacts, map_location="cpu", weights_only=False)
    F_ma, N_st, L_win = ma["max_tokens"].shape
    assert F_ma == Fd, (F_ma, Fd)
    pad_id = int(ma.get("pad_id", 0))
    thr_ma = float(ma.get("threshold", thr))
    if abs(thr_ma - thr) > 1e-3 * max(1.0, abs(thr)):
        log(f"WARNING maxacts threshold {thr_ma:.4f} != ae.pt threshold {thr:.4f} (maxacts was scanned with the former)")
    max_acts = ma["max_acts"].float().numpy(); lengths = ma["lengths"].numpy().astype(np.int64); fire = ma["fire_counts"].numpy().astype(np.int64)
    doc_ids = ma["doc_ids"].numpy().astype(np.int64); positions = ma["positions"].numpy().astype(np.int64)
    max_tokens = ma["max_tokens"].numpy()
    ma_meta = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in ma.items() if not torch.is_tensor(v)}
    if max_features and max_features > 0:
        keep_f = np.zeros(Fd, bool); keep_f[:max_features] = True
        fire = np.where(keep_f, fire, 0)
    split_info = None
    if feature_split:
        assert split_key, "feature_split needs split_key (a key of the npz, e.g. eval | rl | sft)"
        sp = np.load(feature_split)
        assert split_key in sp.files, (split_key, sp.files)
        allowed = np.asarray(sp[split_key], np.int64)
        others = {k: np.asarray(sp[k], np.int64) for k in sp.files if k != split_key}
        assert allowed.size and allowed.min() >= 0 and allowed.max() < Fd and len(np.unique(allowed)) == len(allowed), "bad split ids"
        for k, o in others.items():
            assert np.intersect1d(allowed, o).size == 0, f"feature split {split_key} overlaps {k}"
        n_live_before = int(live_mask(max_acts, fire, thr_ma).sum())
        keep_f = np.zeros(Fd, bool); keep_f[allowed] = True
        fire = np.where(keep_f, fire, 0)
        split_info = {"path": feature_split, "key": split_key, "n_allowed": int(len(allowed)), "n_live_before_split": n_live_before,
                      "other_keys": {k: int(len(o)) for k, o in others.items()},
                      "npz_sha256_16": hashlib.sha256(open(feature_split, "rb").read()).hexdigest()[:16]}
        log(f"feature split {feature_split}[{split_key}]: {len(allowed)} eligible features (others {split_info['other_keys']} excluded; "
            f"live before split {n_live_before})")
    live = live_mask(max_acts, fire, thr_ma)
    cand = candidate_mask(max_acts, lengths, thr_ma, min_tok) & live[:, None]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else BOS_FALLBACK
    special = special_in_span(max_tokens, lengths, {pad_id, bos})
    n_special = int((cand & special).sum())
    cand &= ~special
    cf, cr = np.nonzero(cand)                                       # sorted by (feature, rank)
    n_live = int(live.sum()); n_cand = len(cf)
    corpus_peak = max_acts.max(1)
    log(f"maxacts: F={F_ma} N={N_st} L={L_win} pad_id={pad_id} tokens_seen={ma_meta.get('tokens_seen')} | live {n_live}/{Fd} "
        f"({100 * n_live / Fd:.2f}%) | candidate windows {n_cand} (slots firing {(max_acts > thr_ma).sum()}, of which >= {min_tok} tok "
        f"{((max_acts > thr_ma) & (lengths >= min_tok)).sum()}, minus {n_special} with pad/BOS inside) ({time.time() - t0:.0f}s)")
    timings["inputs_s"] = time.time() - t0

    # ---------------------------------------------------------------- 2. decode + roundtrip + dedupe
    t0 = time.time()
    ids_lists = [window_ids(max_tokens[f, r], lengths[f, r]) for f, r in zip(cf.tolist(), cr.tolist())]
    log(f"{len(ids_lists)} candidate windows materialised ({time.time() - t0:.0f}s); decoding + re-tokenising ...")
    texts, ok, reason = roundtrip_texts(tok, ids_lists, bad_ids=(), batch=50_000)
    dup = dedupe_texts(cf, texts, ok)
    valid = ok & ~dup
    rt_stats = {r: int((reason == r).sum()) for r in ("ok", "roundtrip_fail", "blank")}
    rt_stats["special_id_in_window"] = n_special
    rt_stats["duplicate_text_within_feature"] = int(dup.sum()); rt_stats["valid"] = int(valid.sum())
    log(f"roundtrip: {rt_stats} ({time.time() - t0:.0f}s)")
    timings["roundtrip_s"] = time.time() - t0

    # ---------------------------------------------------------------- 3. breadth-first selection
    sel, sel_info = breadth_first_select(cf, cr, valid, K, max_rows_per_family, seed=seed)
    S = np.flatnonzero(sel)                                          # candidate indices, sorted by (feature, rank)
    cov_sel = coverage_hist(cf[S], Fd, K)
    live_cov_sel = [c if i > 0 else c - (Fd - n_live) for i, c in enumerate(cov_sel)]     # bin 0 among LIVE features only
    log(f"selected {len(S)} windows from {len(np.unique(cf[S]))} features; per-depth {sel_info['per_depth']}; cap_hit={sel_info['cap_hit']}; "
        f"coverage hist (windows/feature over live features, 0..{K}) {live_cov_sel}")

    # ---------------------------------------------------------------- 4. end-anchor check (torchrun, all GPUs)
    t0 = time.time()
    n_sel = len(S)
    if skip_anchor or n_sel == 0:
        argpos = np.array([len(ids_lists[j]) - 1 for j in S], np.int64); act_last = np.full(n_sel, np.nan, np.float32); act_max = act_last.copy()
        anchor_info = {"skipped": True}
    else:
        bounds = np.linspace(0, n_sel, world + 1).astype(np.int64)      # contiguous (feature-sorted) equal-row shards
        for r in range(world):
            js = S[bounds[r]:bounds[r + 1]]
            feats_r = cf[js]
            lens_r = np.array([len(ids_lists[j]) for j in js], np.int64)
            offs_r = np.concatenate([[0], np.cumsum(lens_r)])
            flat = np.concatenate([np.asarray(ids_lists[j], np.int32) for j in js]) if len(js) else np.zeros(0, np.int32)
            torch.save({"feat": torch.from_numpy(feats_r.astype(np.int64)), "ids_flat": torch.from_numpy(flat), "offs": torch.from_numpy(offs_r),
                        "f_lo": int(feats_r.min()) if len(js) else 0, "f_hi": int(feats_r.max()) if len(js) else 0}, f"{stage}/anchor_in/anchor_in_r{r}.pt")
        log(f"anchor inputs staged for {world} ranks: rows/rank {np.diff(bounds).tolist()}")
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={world}", "/pmx/sae2m_bank/anchor_worker.py", "--ae", ae, "--in-dir", f"{stage}/anchor_in",
               "--out-dir", f"{stage}/anchor_out", "--model", MODEL, "--layer", str(LAYER), "--bos", str(bos), "--batch", str(anchor_batch), "--d-model", str(D)]
        rc = _run(cmd, _env())
        outs = [f"{stage}/anchor_out/anchor_out_r{r}.npz" for r in range(world)]
        assert all(os.path.exists(p) for p in outs), f"anchor worker rc={rc}; missing outputs {[p for p in outs if not os.path.exists(p)]}"
        parts = [np.load(p) for p in outs]
        argpos = np.concatenate([p["argpos"] for p in parts]); act_last = np.concatenate([p["act_last"] for p in parts]); act_max = np.concatenate([p["act_max"] for p in parts])
        n_tok_chk = np.concatenate([p["n_tok"] for p in parts])
        assert len(argpos) == n_sel and np.array_equal(n_tok_chk, np.array([len(ids_lists[j]) for j in S]))
        anchor_info = {"rc": rc, "ranks": [json.load(open(f"{stage}/anchor_out/anchor_out_r{r}.json")) for r in range(world)]}
        log(f"anchor workers done rc={rc}: " + " ".join(f"r{i['rank']}={i['n']}rows/{i.get('score_s', 0):.0f}s/{i.get('peak_mem_gb', 0):.0f}GB" for i in anchor_info["ranks"]))
    n_tok_sel = np.array([len(ids_lists[j]) for j in S], np.int64)
    keep_anchor, anchor_summ = anchor_summary(argpos, n_tok_sel, act_last, act_max, thr, rule=anchor_rule)
    Wn = S[keep_anchor]                                              # final windows (candidate indices)
    cov_final = coverage_hist(cf[Wn], Fd, K)
    live_cov_final = [c if i > 0 else c - (Fd - n_live) for i, c in enumerate(cov_final)]
    log(f"end-anchor ({anchor_rule}): pass_last {anchor_summ['pass_last']:.4f} pass_last2 {anchor_summ['pass_last2']:.4f} fire_last {anchor_summ['fire_last_rate(>thr)']:.4f} "
        f"-> keep {len(Wn)}/{n_sel} windows; coverage hist (live features) {live_cov_final} ({time.time() - t0:.0f}s)")
    timings["anchor_s"] = time.time() - t0

    # ---------------------------------------------------------------- 5. directions (GPU 0): tables over the unique final features
    t0 = time.time()
    U = np.unique(cf[Wn]).astype(np.int64)                            # sorted unique features
    nU = len(U)
    pos_of = np.full(Fd, -1, np.int64); pos_of[U] = np.arange(nU)
    enc_tab = torch.empty(nU, D, dtype=torch.bfloat16, device=dev)
    CH = 131072
    for s in range(0, Fd, CH):
        e = min(Fd, s + CH)
        m = (U >= s) & (U < e)
        if m.any():
            blk = sd["encoder.weight"][s:e].to(dev, dtype=torch.bfloat16)
            enc_tab[torch.from_numpy(np.flatnonzero(m)).to(dev)] = blk[torch.from_numpy(U[m] - s).to(dev)]
            del blk
    log(f"encoder rows gathered for {nU} features ({time.time() - t0:.0f}s)")
    t1 = time.time()
    dec_full = torch.empty(D, Fd, dtype=torch.bfloat16, device=dev)
    RCH = 256
    for r0 in range(0, D, RCH):
        dec_full[r0:r0 + RCH] = sd["decoder.weight"][r0:r0 + RCH].to(dev, dtype=torch.bfloat16)
    dec_tab = dec_full[:, torch.from_numpy(U).to(dev)].T.contiguous()
    del dec_full; torch.cuda.empty_cache()
    log(f"decoder columns gathered ({time.time() - t1:.0f}s); GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    del sd

    def unit_rows(tab, idx):
        return F.normalize(tab[idx].float(), dim=-1)

    enc_dec_cos = np.concatenate([(unit_rows(enc_tab, slice(c0, c0 + 65536)) * unit_rows(dec_tab, slice(c0, c0 + 65536))).sum(-1).cpu().numpy()
                                  for c0 in range(0, nU, 65536)]) if nU else np.zeros(0, np.float32)
    dec_norm = np.concatenate([dec_tab[c0:c0 + 65536].float().norm(dim=-1).cpu().numpy() for c0 in range(0, nU, 65536)]) if nU else np.zeros(0)
    log(f"cos(enc, dec) median {np.median(enc_dec_cos):.3f} p5 {np.percentile(enc_dec_cos, 5):.3f} p95 {np.percentile(enc_dec_cos, 95):.3f} | "
        f"stored decoder col norms [{dec_norm.min():.4f}, {dec_norm.max():.4f}]")
    timings["directions_s"] = time.time() - t0

    # ---------------------------------------------------------------- 6. leak guard vs eval cache + pool_heldout (per unique direction)
    t0 = time.time()
    ref, ref_names, ref_offs, es, ho_sae_feats = _load_refs(eval_cache, pool_heldout, dev)
    log(f"leak refs: {ref.shape[0]} directions in {len(ref_names)} sets")

    def tab_iter(tab, ch=16384):
        for c0 in range(0, nU, ch):
            yield unit_rows(tab, slice(c0, c0 + ch)).cpu().numpy()

    leak_tbl, maxcos, leak_arg = {}, {}, {}
    for fam, tab in ((FAM_ENC, enc_tab), (FAM_DEC, dec_tab)):
        mc, per_set, am = max_cos_table(tab_iter(tab), ref, ref_offs, len(ref_names), dev)
        maxcos[fam] = mc; leak_arg[fam] = am
        leak_tbl[fam] = {n: round(float(per_set[i]), 4) for i, n in enumerate(ref_names)}
        log(f"  max-cos {fam:10s} " + " ".join(f"{n.split('/')[-1][:11]}={leak_tbl[fam][n]:.3f}" for n in ref_names))
    leak_feat = {fam: np.flatnonzero(maxcos[fam] > LEAK_COS) for fam in (FAM_ENC, FAM_DEC)}    # positions into U
    leak_examples = {fam: [{"feature": int(U[p]), "max_cos": float(maxcos[fam][p]), "ref_set": ref_names[int(np.searchsorted(ref_offs, leak_arg[fam][p], side="right") - 1)]}
                           for p in leak_feat[fam][:20]] for fam in (FAM_ENC, FAM_DEC)}
    log(f"leak guard (> {LEAK_COS}): features flagged enc {len(leak_feat[FAM_ENC])} dec {len(leak_feat[FAM_DEC])} ({time.time() - t0:.0f}s) {leak_examples}")
    del ref; torch.cuda.empty_cache()
    timings["leak_s"] = time.time() - t0

    # ---------------------------------------------------------------- 7. overlap with the 131k eval SAE (report only)
    t0 = time.time()
    p131 = torch.load(sae131k, map_location="cpu", weights_only=False)
    W_enc131 = F.normalize(p131["encoder.weight"].float(), dim=-1).to(dev)                        # [131072, d] rows = encoder dirs
    W_dec131 = F.normalize(p131["decoder.weight"].float().T.contiguous(), dim=-1).to(dev)         # [131072, d] rows = decoder dirs
    del p131
    F131 = W_dec131.shape[0]
    eval_feats = sorted(set(int(f) for f in es["sae_feats"]) | set(ho_sae_feats))
    ev_idx = torch.tensor(eval_feats, device=dev)
    overlap = {"n_eval_sae_features": len(eval_feats), "eval_cache_sae_feats": int(len(es["sae_feats"])), "pool_heldout_sae_feats": len(ho_sae_feats),
               "n_2m_features": int(nU), "d_sae_131k": int(F131), "hists": {}}
    win_feat_pos = pos_of[cf[Wn]]                                    # per final window -> position in U (row weighting)
    for fam, tab in ((FAM_ENC, enc_tab), (FAM_DEC, dec_tab)):
        for ref_name, R in (("131k_dec_eval_feats", W_dec131[ev_idx]), ("131k_dec_all", W_dec131), ("131k_enc_all", W_enc131)):
            offs1 = np.array([0, R.shape[0]])
            mc, _, _ = max_cos_table(tab_iter(tab), R, offs1, 1, dev)
            h = cos_hist(mc); h_rows = cos_hist(mc[win_feat_pos])
            overlap["hists"][f"{fam}_vs_{ref_name}"] = {"per_feature": h, "per_row": h_rows}
            log(f"  overlap {fam:10s} vs {ref_name:20s}: per-feature max-cos p50 {h['percentiles'].get('50', 0):.3f} p99 {h['percentiles'].get('99', 0):.3f} "
                f"max {h['max']:.4f} | frac>0.95 {h['frac_gt']['0.95']:.4%} frac>0.99 {h['frac_gt']['0.99']:.4%}")
    flag_frac = max(v["per_row"]["frac_gt"]["0.95"] for k, v in overlap["hists"].items() if "131k_dec" in k)
    overlap["flag_rows_gt_0.95_max_frac"] = float(flag_frac); overlap["flag_threshold"] = overlap_flag_frac
    overlap["FLAG_train_dictionary_overlaps_eval_dictionary"] = bool(flag_frac > overlap_flag_frac)
    if overlap["FLAG_train_dictionary_overlaps_eval_dictionary"]:
        log(f"!!! FLAG: {flag_frac:.2%} of bank rows have a 2M direction within cos > 0.95 of a 131k eval-SAE decoder direction (threshold {overlap_flag_frac:.0%})")
    else:
        log(f"overlap OK: at most {flag_frac:.3%} of rows within cos > 0.95 of a 131k eval-SAE decoder direction")
    del W_enc131, W_dec131; torch.cuda.empty_cache()
    timings["overlap_s"] = time.time() - t0

    # ---------------------------------------------------------------- 8. assemble + publish (seeded shuffle; vecs streamed to the volume)
    t0 = time.time()
    row_fam, row_win, dropped = assemble_rows(Wn, win_feat_pos, {fam: leak_feat[fam] for fam in (FAM_ENC, FAM_DEC)}, (FAM_ENC, FAM_DEC), seed + 1)
    N_out = len(row_fam)
    counts = {FAM_ENC: int((row_fam == 0).sum()), FAM_DEC: int((row_fam == 1).sum())}
    log(f"assembling {N_out} rows {counts} (leak-dropped {dropped}); writing vecs.f32 ({N_out * D * 4 / 2**30:.1f} GB) to {out}")
    os.makedirs(out, exist_ok=True)
    win_pos_all = pos_of[cf]                                         # candidate -> U position (only valid for final windows)
    nrm_min, nrm_max = 1.0, 1.0
    with open(f"{out}/vecs.f32.tmp", "wb") as fout:
        CHW = 32768
        for c0 in range(0, N_out, CHW):
            fr = row_fam[c0:c0 + CHW]; wp = torch.from_numpy(win_pos_all[row_win[c0:c0 + CHW]]).to(dev)
            x = torch.empty(len(fr), D, dtype=torch.float32, device=dev)
            me = torch.from_numpy(fr == 0).to(dev)
            if me.any():
                x[me] = unit_rows(enc_tab, wp[me])
            if (~me).any():
                x[~me] = unit_rows(dec_tab, wp[~me])
            nn = x.norm(dim=-1)
            nrm_min = min(nrm_min, float(nn.min())); nrm_max = max(nrm_max, float(nn.max()))
            fout.write(x.cpu().numpy().tobytes())
    os.replace(f"{out}/vecs.f32.tmp", f"{out}/vecs.f32")
    assert os.path.getsize(f"{out}/vecs.f32") == N_out * D * 4
    log(f"vecs.f32 written ({time.time() - t0:.0f}s); unit norms [{nrm_min:.6f}, {nrm_max:.6f}]")
    t1 = time.time()
    # per-candidate anchor arrays (indexed by candidate index) for the records
    a_argpos = np.full(n_cand, -1, np.int64); a_last = np.full(n_cand, np.nan, np.float32); a_max = np.full(n_cand, np.nan, np.float32)
    a_argpos[S] = argpos; a_last[S] = act_last; a_max[S] = act_max
    with open(f"{out}/records.jsonl.tmp", "w") as fh:
        for i in range(N_out):
            j = int(row_win[i]); f = int(cf[j]); r = int(cr[j]); fam = FAM_ENC if row_fam[i] == 0 else FAM_DEC
            rec = make_record(i, fam, f, r, texts[j], len(ids_lists[j]), corpus_peak[f], max_acts[f, r], fire[f], doc_ids[f, r], positions[f, r],
                              {"argpos": a_argpos[j], "act_last": a_last[j] if np.isfinite(a_last[j]) else -1.0, "act_max": a_max[j] if np.isfinite(a_max[j]) else -1.0},
                              enc_dec_cos[pos_of[f]])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(f"{out}/records.jsonl.tmp", f"{out}/records.jsonl")
    log(f"records.jsonl written ({time.time() - t1:.0f}s)")
    timings["publish_s"] = time.time() - t0

    # ---------------------------------------------------------------- 9. stats + meta
    n_tok_final = n_tok_sel[keep_anchor]
    fam_recipe = {
        FAM_ENC: {"source": ae, "dir": "unit(encoder.weight[f]) (2M SAE encoder row, RAW layer-42 space; norm factor folded by merge)",
                  "target": f"stored top-{N_st} {L_win}-token max-activating window (Ultra-FineWeb, {ma_meta.get('tokens_seen')} tokens) decoded so the text ENDS at the peak token; "
                            f"kept iff decode->re-encode == ids, >= {min_tok} tokens, no pad/BOS id, text-distinct within the feature, and the feature PEAKS AT THE LAST TOKEN when the "
                            f"text is re-forwarded standalone ([BOS]+ids, layers 0..{LAYER}, rule {anchor_rule})",
                  "taken": counts[FAM_ENC], "features_taken": int(len(np.unique(cf[row_win[row_fam == 0]]))) if counts[FAM_ENC] else 0,
                  "windows_per_feature": K, "windows_per_feature_hist_live_features": live_cov_final, "min_tok": min_tok, "end_anchored": True,
                  "leak_dropped_rows": dropped[FAM_ENC]},
        FAM_DEC: {"source": ae, "dir": "unit(decoder.weight[:, f]) (2M SAE decoder column = the feature's write direction, RAW space)",
                  "target": "the SAME windows as the feature's sae2m rows", "taken": counts[FAM_DEC],
                  "features_taken": int(len(np.unique(cf[row_win[row_fam == 1]]))) if counts[FAM_DEC] else 0,
                  "windows_per_feature": K, "windows_per_feature_hist_live_features": live_cov_final, "min_tok": min_tok, "end_anchored": True,
                  "enc_dec_cos": {str(q): float(np.percentile(enc_dec_cos, q)) for q in (5, 25, 50, 75, 95)} | {"mean": float(enc_dec_cos.mean())} if nU else {},
                  "leak_dropped_rows": dropped[FAM_DEC]},
    }
    stats = {"kind": "sae2m midtrain bank: 2,097,152-feature layer-42 BatchTopK SAE (encoder-row + decoder-column directions) x end-anchored max-activating windows",
             "n_examples": N_out, "n_vecs": N_out, "families": counts, "seed": seed, "d_model": D, "model": MODEL, "layer": LAYER,
             "layout": "seeded shuffle of all rows (records.jsonl line i == vec_idx i); each window appears once per family with the same target_text",
             "sae": {"ae": ae, "F": int(Fd), "d": int(d), "k": k_sae, "threshold": thr, "bytes": os.path.getsize(ae)},
             "maxacts": {"path": maxacts, "N": int(N_st), "L": int(L_win), "pad_id": pad_id, "threshold": thr_ma, "meta": ma_meta,
                         "live_features": n_live, "dead_features": int(Fd - n_live), "candidate_windows": int(n_cand)},
             "selection": {"windows_per_feature": K, "min_tok": min_tok, "max_rows_per_family": int(max_rows_per_family), "max_features_debug": int(max_features), "feature_split": split_info,
                           "roundtrip": rt_stats, "breadth_first": sel_info, "selected_windows": int(n_sel),
                           "coverage_hist_selected_live_features": live_cov_sel, "coverage_hist_final_live_features": live_cov_final,
                           "features_in_bank": int(nU), "n_tok_hist_final": {str(int(a)): int(b) for a, b in zip(*np.unique(n_tok_final, return_counts=True))} if len(n_tok_final) else {}},
             "end_anchor_filter": {**anchor_summ, "families": {fam: {"n": int(n_sel), "n_keep": int(keep_anchor.sum()), "n_fail": int((~keep_anchor).sum()),
                                                                     "pass_last": anchor_summ["pass_last"], "pass_last2": anchor_summ["pass_last2"]} for fam in (FAM_ENC, FAM_DEC)},
                                   "protocol": f"each selected window re-tokenised standalone (ids verified by roundtrip), [BOS={bos}]+ids -> Qwen3.6-27B layers 0..{LAYER} (truncated, bf16, "
                                               f"sdpa) -> relu((x - b_dec) . W_enc[f] + b_enc[f]) fp32 per content token; keep iff argmax == last token ('last')",
                                   "workers": anchor_info, "anchor_batch": anchor_batch},
             "leak_check": {"threshold": LEAK_COS, "reference_sets": ref_names, "n_reference_dirs": int(ref_offs[-1]), "rows_dropped_by_family": dropped,
                            "features_flagged": {fam: int(len(leak_feat[fam])) for fam in (FAM_ENC, FAM_DEC)}, "examples": leak_examples,
                            "max_cos_table": leak_tbl, "eval_cache": eval_cache, "pool_heldout": pool_heldout},
             "eval_sae_overlap": {**overlap, "sae131k": sae131k, "note": "report only (no rows dropped): max cos of each 2M direction vs the 131k eval SAE's directions"},
             "unit_norm_range": [nrm_min, nrm_max], "timings_s": timings, "created": time.time(), "wall_s": time.time() - T0}
    meta = {**stats, "bank": out, "n_rows": N_out, "family_recipes": fam_recipe,
            "trainer_args": {"--data-dir": out, "--bank-file": "vecs.f32"},
            "mix_5m_group": {"name": "sae2m", "banks": [out], "dtype": "f32", "families": [FAM_ENC, FAM_DEC], "n": {FAM_ENC: counts[FAM_ENC], FAM_DEC: counts[FAM_DEC]}}}
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1, default=str)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1, default=str)
    vol.commit()
    # ---------------------------------------------------------------- 10. verification from the written files
    t0 = time.time()
    n_chk, fams_chk = check_bank_files(out)
    assert n_chk == N_out and fams_chk == counts
    vecs = np.memmap(f"{out}/vecs.f32", np.float32, "r", shape=(N_out, D))
    ref, ref_names2, ref_offs2, _, _ = _load_refs(eval_cache, pool_heldout, dev)
    worst = 0
    with torch.no_grad():
        for c0 in range(0, N_out, 16384):
            x = torch.from_numpy(np.ascontiguousarray(vecs[c0:c0 + 16384])).to(dev)
            nn = x.norm(dim=-1)
            assert float((nn - 1).abs().max()) < 1e-3, f"non-unit rows in chunk {c0}"
            worst += int(((x @ ref.T).max(1).values > LEAK_COS).sum())
    assert worst == 0, f"{worst} written rows within cos > {LEAK_COS} of an eval/held-out direction"
    del ref; torch.cuda.empty_cache()
    log(f"verification passed ({time.time() - t0:.0f}s): {N_out} rows, line i == vec_idx i, unit norms, 0 leak rows")
    shutil.rmtree(stage, ignore_errors=True)
    res = {"out": out, "n_examples": N_out, "families": counts, "features_in_bank": int(nU), "live_features": n_live, "selected_windows": int(n_sel),
           "coverage_hist_final_live_features": live_cov_final, "roundtrip": rt_stats, "end_anchor": {k: anchor_summ[k] for k in ("pass_last", "pass_last2", "fire_last_rate(>thr)", "n_keep", "n_fail")},
           "leak_dropped": dropped, "eval_sae_overlap_flag": overlap["FLAG_train_dictionary_overlaps_eval_dictionary"], "overlap_rows_gt_0.95_max_frac": float(flag_frac),
           "anchor_peak_mem_gb": [i.get("peak_mem_gb") for i in anchor_info.get("ranks", [])], "timings_s": timings, "minutes": (time.time() - T0) / 60}
    log(f"DONE {json.dumps(res, default=str)}")
    return res


# ----------------------------------------------------------------- inputs / peek / verify (cheap)
@app.function(image=image, cpu=4, memory=32768, volumes={"/data": vol}, timeout=1800)
def inputs_status(root: str = ROOT):
    """What the bank needs: ae.pt / maxacts files + sizes, maxacts summary, eval cache keys, pool_heldout families."""
    import json, glob
    import torch
    vol.reload()
    res = {"files": {}}
    for p in (f"{root}/trainer_0/ae.pt", f"{root}/trainer_0/config.json", f"{root}/maxacts_top5.pt", f"{root}/maxacts_top5.summary.json",
              f"{root}/maxacts_top5_smoke.pt", f"{root}/maxacts_top5_smoke.summary.json", f"{root}/shards/TRAIN_DONE", f"{root}/verify.json",
              EVAL_CACHE_V2, f"{POOL_HELDOUT}/vecs.f32", SAE131K):
        res["files"][p] = os.path.getsize(p) if os.path.exists(p) else None
    for p in glob.glob(f"{root}/maxacts_top5*.summary.json"):
        res[os.path.basename(p)] = json.load(open(p))
    if os.path.exists(f"{root}/trainer_0/config.json"):
        res["sae_config"] = json.load(open(f"{root}/trainer_0/config.json"))
    es = torch.load(EVAL_CACHE_V2, map_location="cpu", weights_only=False)
    res["eval_cache"] = {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in es.items()}
    res["banks"] = sorted(os.path.basename(p) for p in glob.glob("/data/banks/sae2m*"))
    return res


@app.function(image=image, cpu=4, memory=16384, volumes={"/data": vol}, timeout=1800)
def peek(out_name: str = "sae2m_bank", n: int = 3):
    import json
    import numpy as np
    vol.reload()
    out = f"/data/banks/{out_name}"
    st = json.load(open(f"{out}/build_stats.json"))
    N = st["n_examples"]
    vecs = np.memmap(f"{out}/vecs.f32", np.float32, "r", shape=(N, D))
    seen = {}
    with open(f"{out}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line); f = r["family"]
            if seen.get(f, 0) >= n:
                if len(seen) == len(st["families"]) and all(c >= n for c in seen.values()):
                    break
                continue
            seen[f] = seen.get(f, 0) + 1
            print(f"[{f}] row {i} |v|={np.linalg.norm(vecs[i]):.4f} feat={r['feature']} rank={r['window_rank']} n_tok={r['n_tok']} peak={r['window_peak']} "
                  f"anchor_last={r['anchor_act_last']} :: {r['target_text'][-120:]!r}", flush=True)
    return {k: st[k] for k in ("n_examples", "families", "end_anchor_filter", "leak_check") if k in st}


@app.function(image=image, gpu=SMALL_GPUS, cpu=8, memory=65536, volumes={"/data": vol}, timeout=4 * 3600)
def verify(out_name: str = "sae2m_bank"):
    """From the FINAL files: n_examples == lines == vecs rows, line i == vec_idx i, unit norms, max cos < 0.999 vs every eval/held-out dir."""
    import json, sys, time
    import numpy as np
    import torch
    sys.path.insert(0, "/pmx/sae2m_bank")
    from bank_lib import check_bank_files
    vol.reload()
    out = f"/data/banks/{out_name}"
    t0 = time.time()
    N, fams = check_bank_files(out)
    vecs = np.memmap(f"{out}/vecs.f32", np.float32, "r", shape=(N, D))
    dev = "cuda:0"
    ref, names, offs, _, _ = _load_refs(EVAL_CACHE_V2, POOL_HELDOUT, dev)
    worst = 0; nrm = [1.0, 1.0]; leak = np.full(len(names), -1.0)
    with torch.no_grad():
        for c0 in range(0, N, 16384):
            x = torch.from_numpy(np.ascontiguousarray(vecs[c0:c0 + 16384])).to(dev)
            nn = x.norm(dim=-1); nrm = [min(nrm[0], float(nn.min())), max(nrm[1], float(nn.max()))]
            cos = x @ ref.T
            worst += int((cos.max(1).values > LEAK_COS).sum())
            for ri in range(len(names)):
                leak[ri] = max(leak[ri], float(cos[:, offs[ri]:offs[ri + 1]].max()))
    res = {"out": out, "n": N, "families": fams, "unit_norm_range": nrm, "leak_rows": worst, "max_cos_by_ref": {n: float(leak[i]) for i, n in enumerate(names)},
           "seconds": time.time() - t0}
    print(json.dumps(res, indent=1), flush=True)
    assert worst == 0 and abs(nrm[0] - 1) < 1e-3 and abs(nrm[1] - 1) < 1e-3
    return res
