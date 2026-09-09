"""Modal app `maemm-bsf-verbatim`: the BSF-VERBATIM midtrain bank at /data/banks/<out_name> (default bsf_verbatim_1m) on `maemm-data`.

WHY: the bsf rows of /data/banks/everything_5m_fresh were mined from a 512-token store — the direction is the block projection of
the activation at position p computed with up to 511 tokens of context, but the SFT target is only the trailing 16-64-token window,
so the direction carries context the target lacks. Here every direction comes from a SHORT text forwarded ALONE:
    ids = [BOS] + tok(target_text, add_special_tokens=False)            (data/bsf_verbatim_worker.py, one process per GPU)
    x   = layer-42 residual at the LAST token (fp32)
    y   = normalize((x - mu_bsf) @ zca);  gn_g = ||(y @ E).view(G, b)[g]||;  block b = the top active block (rank <= --bsf-ranks)
          that still has capacity (<= cap_per_block rows per block, level-by-level fill == data/modal_bank_everything.py's bsf selection)
    dir = unit((Q[b]^T Q[b] ((x - mu_bsf) @ zca)) @ zca^-1)             (BSF = HF ceselder/qwen36-27b-bsf-l42-1b, cached /data/bsf27b_1b)
so the target text IS the literal verbatim context that produced the direction (ctx_len == n_tok), and the direction is derived from
the last token by construction. Texts = target windows of the UNUSED realact_short_20m parts (default p..x; fresh FineFineWeb windows,
already eval-hash-excluded, never trained on) — 8-32 source tokens each.

END-ANCHOR BY MEASUREMENT: for every candidate (text, block) the worker measures cos(h_t, dir) (RAW residual, == the RL reward's
"cosine" metric; also h_t @ dir == its default "proj" metric) over every content token t of the standalone forward and records
peak_from_end = n_tok - 1 - argmax_t. The bank reports the pass rates of the rule-free selection (dir_peak_last / dir_peak_last2 +
peak-offset histogram) and then, for rule != "none", DROPS rows whose peak is not at the last token ("last") / within the last 2
("last2") and tops the selection up with rule-passing (candidate, block) pairs so the bank still reaches ~n_rows.

Block EXCLUSIONS == everything builder: (a) the literal block ids of the pool_heldout bsf rows, (b) the top-1 block under THIS BSF
of every eval bsf direction (eval_sets_heldout_v2.pt). LEAK GUARD == everything builder: every row within cos > 0.999 of ANY v2
eval direction (all *_dirs families) or ANY pool_heldout row is dropped, then asserted. 10x-median norm filter on ||x - mu_acts27b||.

Output (EXACT bank format of the everything bank / the mix compositor data/modal_mix_5m_bank.py `_scan_records`):
    vecs.f32          float32 [N, 5120] UNIT rows
    records.jsonl     line i == vec_idx i, {"vec_idx", "family": "bsf", "target_text", "block", "rank", "gnorm", "cos_x", "whiten_frac",
                      "n_tok", "ctx_len" (== n_tok), "peak_from_end", "peak_from_end_proj", "peak_pos", "fire_from_end", "cos_last",
                      "cos_max", "act_norm", "src_part", "src_vec_idx", "src", "src_ctx_len", "src_W"}
    build_stats.json  "n_examples" == N, "families": {"bsf": N}, kind/dir/target/source/bsf/end_anchor/leak_check/...
    meta.json         build_stats + trainer args + family_recipes + exclusions_summary + leakage_check
    exclusions.json   the full block-id exclusion lists applied
Compositor group: {"name": "bsf_verbatim", "banks": ["/data/banks/bsf_verbatim_1m"], "dtype": "f32", "families": ["bsf"], "n": N}

Run (MODAL_PROFILE=safety-sahan, from the repo root):
    modal deploy data/modal_bsf_verbatim.py
    # smoke (1 GPU):
    python -c "import modal; print(modal.Function.from_name('maemm-bsf-verbatim','build_small').spawn(out_name='bsf_verbatim_smoke', n_candidates=20000, topup_rounds=0).object_id)"
    # full (8 GPUs):
    python -c "import modal; print(modal.Function.from_name('maemm-bsf-verbatim','build').spawn(out_name='bsf_verbatim_1m', n_candidates=3000000).object_id)"
    modal run data/modal_bsf_verbatim.py::run_verify --out-name bsf_verbatim_1m
    modal run data/modal_bsf_verbatim.py::run_peek --out-name bsf_verbatim_1m
Needs Modal secret `maemm-hf` (HF_TOKEN) for the one-time BSF download if /data/bsf27b_1b is missing.
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = os.environ.get("MAEMM_BSFV_APP", "maemm-bsf-verbatim")
app = modal.App(APP_NAME)

# image == data/modal_collect_bank.py (the realact_short collection app) + this app's worker
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
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
    .pip_install("flash-linear-attention==0.5.2")   # GDN forward via fla's Triton chunk kernel (forward is fine on Hopper)
    .add_local_file(REPO / "data" / "collect_acts27b_worker.py", "/pmx/collect_acts27b_worker.py")
    .add_local_file(REPO / "data" / "collect_bank_worker.py", "/pmx/collect_bank_worker.py")
    .add_local_file(REPO / "data" / "bsf_verbatim_worker.py", "/pmx/bsf_verbatim_worker.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

BUILD_GPU = os.environ.get("BSFV_GPU", "H200:8")
SMOKE_GPU = os.environ.get("BSFV_SMOKE_GPU", "H200:1")
SMALL_GPUS = ["H100", "A100-80GB", "L40S", "A100-40GB"]          # verify (leak check only)
D = 5120
BSF_HF = "ceselder/qwen36-27b-bsf-l42-1b"
BSF_DIR = "/data/bsf27b_1b"                                       # volume cache of the HF BSF files (== everything builder)
BSF_FILES = ("sasa.pt", "blocks_Q.pt", "whiten_mu.npy", "whiten_zca.npy", "meta.json")
MU_ACTS = "/data/acts27b/whiten_mu.npy"                           # the suite's realact centering (norm filter)
POOL_HELDOUT = "/data/pool_heldout"
EVAL_CACHE_V2 = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
PART_TMPL = "/data/banks/realact_short_20m_{}"
LEAK_COS = 0.999
NORM_FILTER_MULT = 10.0
WORK = "/root/work"                                               # local NVMe staging (ephemeral disk)


def _pread_full(fd, n, off):
    buf = bytearray(n); mv = memoryview(buf); got = 0
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise IOError(f"short read at {off + got}")
        got += k
    return buf


def _read_selected_lines(path, want):
    """Stream a records.jsonl and parse only the lines whose (0-based) index is in the SORTED int array `want`."""
    import json
    out = [None] * len(want); ptr = 0; nw = len(want)
    if nw == 0:
        return out
    nxt = int(want[0])
    with open(path, "rb") as f:
        for li, line in enumerate(f):
            if li == nxt:
                o = json.loads(line)
                out[ptr] = (o["target_text"], int(o["vec_idx"]), o.get("src"), o.get("ctx_len"), o.get("W"))
                ptr += 1
                if ptr == nw:
                    break
                nxt = int(want[ptr])
    assert ptr == nw, (path, ptr, nw)
    return out


def _select_rows(top_i, elig, doc_of, n_want, cap, doc_cap, G, n_doc, rng, state=None, log=None):
    """Level-by-level block-capacity selection == data/modal_bank_everything.py bsf selection (same logic, verbatim):
    each block gets <= 1 new row per level; within a level prefer lower rank; one RANDOM token per block per (level, rank) pass;
    per-document cap. top_i [N, R] block per rank; elig [N, R] eligibility per (candidate, rank); doc_of [N] document id.
    state = {used [N], block_cnt [G], per_doc [n_doc]} is continued across calls. -> (sel_tok, sel_rank (1-based), state)."""
    import numpy as np
    N, R = top_i.shape
    st = state if state is not None else {"used": np.zeros(N, bool), "block_cnt": np.zeros(G, np.int32), "per_doc": np.zeros(n_doc, np.int32)}
    used, block_cnt, per_doc = st["used"], st["block_cnt"], st["per_doc"]
    sel_tok, sel_rank, total = [], [], 0
    for level in range(cap):
        lvl0 = total
        for r in range(R):
            if total >= n_want:
                break
            bl = top_i[:, r].astype(np.int64)
            cand = elig[:, r] & ~used & (block_cnt[bl] == level) & (per_doc[doc_of] < doc_cap)
            idx = np.flatnonzero(cand)
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            _, first = np.unique(bl[idx], return_index=True)            # one random token per block
            picks = idx[np.sort(first)]
            order = rng.permutation(len(picks)); picks = picks[order]
            s_p = doc_of[picks]
            so = np.argsort(s_p, kind="stable"); s_sorted = s_p[so]
            starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
            grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(s_sorted)]))
            within = np.arange(len(s_sorted)) - starts[grp]
            keep = within < (doc_cap - per_doc[s_sorted])
            picks = picks[so[keep]]
            if total + len(picks) > n_want:
                picks = picks[: n_want - total]
            used[picks] = True
            np.add.at(block_cnt, bl[picks], 1)
            np.add.at(per_doc, doc_of[picks], 1)
            sel_tok.append(picks); sel_rank.append(np.full(len(picks), r + 1, np.int8))
            total += len(picks)
        if log and (level < 3 or level % 5 == 4 or total >= n_want):
            log(f"    level {level + 1}/{cap}: +{total - lvl0} -> {total}/{n_want} (blocks with rows {int((block_cnt > 0).sum())}, "
                f"full {int((block_cnt >= cap).sum())})")
        if total >= n_want:
            break
    sel_tok = np.concatenate(sel_tok) if sel_tok else np.zeros(0, np.int64)
    sel_rank = np.concatenate(sel_rank) if sel_rank else np.zeros(0, np.int8)
    return sel_tok, sel_rank, st


def _qs(a):
    import numpy as np
    a = np.asarray(a, np.float64)
    if a.size == 0:
        return {}
    return {str(q): float(np.percentile(a, q)) for q in (5, 10, 25, 50, 75, 90, 95)} | {"mean": float(a.mean())}


def _anchor_stats(pk):
    import numpy as np
    pk = np.asarray(pk, np.int64)
    if pk.size == 0:
        return {"n": 0}
    return {"n": int(pk.size), "pass_last": float((pk == 0).mean()), "pass_last2": float((pk <= 1).mean()),
            "peak_offset_from_end_hist": {str(i): int(c) for i, c in enumerate(np.bincount(np.clip(pk, 0, 64), minlength=1)) if c},
            "mean_offset": float(pk.mean())}


# ----------------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------------
def _build_impl(out_name, n_candidates, n_rows, cap_per_block, bsf_ranks, doc_cap, seed, rule, rule_metric, parts, min_tok, max_tok,
                max_batch, batch_tokens, eval_cache, overwrite_smoke, topup_rounds, topup_margin, shortfall_tol, gpu_label):
    import json, shutil, subprocess, sys, time
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, "/pmx"); sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL, READ_LAYER
    from bsf_verbatim_worker import BSF
    assert D_MODEL == D
    T0 = time.time()

    def log(msg):
        print(f"[bsfv +{time.time() - T0:6.0f}s] {msg}", flush=True)

    assert rule in ("none", "last", "last2") and rule_metric in ("cos", "proj"), (rule, rule_metric)
    assert n_rows > 0 and n_candidates > 0 and 1 <= bsf_ranks <= 64 and cap_per_block >= 1 and doc_cap >= 1
    vol.reload()
    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json"):
        if overwrite_smoke and "smoke" in out_name:
            shutil.rmtree(out)
        else:
            raise RuntimeError(f"{out}/build_stats.json exists — bank already built (new out_name?)")
    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import hf_hub_download, snapshot_download
    t0 = time.time()
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    log(f"base model in /data/hf_cache ({time.time() - t0:.0f}s)")
    if not all(os.path.exists(f"{BSF_DIR}/{f}") for f in BSF_FILES):
        os.makedirs(BSF_DIR, exist_ok=True)
        for f in BSF_FILES:
            if os.path.exists(f"{BSF_DIR}/{f}"):
                continue
            t1 = time.time()
            p = hf_hub_download(BSF_HF, f, local_dir="/root/bsf_dl", token=os.environ.get("HF_TOKEN"))
            shutil.copyfile(p, f"{BSF_DIR}/{f}.part"); os.replace(f"{BSF_DIR}/{f}.part", f"{BSF_DIR}/{f}")
            log(f"BSF {f}: downloaded + cached ({os.path.getsize(p) / 2**30:.2f} GB, {time.time() - t1:.0f}s)")
    for p in (MU_ACTS, eval_cache, f"{POOL_HELDOUT}/records.jsonl", f"{POOL_HELDOUT}/vecs.f32") + tuple(f"{BSF_DIR}/{f}" for f in BSF_FILES):
        assert os.path.exists(p), f"missing input {p}"
    part_dirs = [PART_TMPL.format(c) for c in parts]
    n_part = []
    for pd in part_dirs:
        st = json.load(open(f"{pd}/build_stats.json"))
        assert st["families"] == {"realact": st["n_examples"]}, (pd, st["families"])
        assert os.path.exists(f"{pd}/records.jsonl")
        n_part.append(int(st["n_examples"]))
    vol.commit()
    world = len([ln for ln in subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines() if ln.strip()])
    R, G_cap = bsf_ranks, cap_per_block
    log(f"targets: {n_rows} rows from <= {n_candidates} candidates/round (+{topup_rounds} top-up rounds) | cap/block {G_cap} ranks {R} "
        f"doc_cap {doc_cap} | rule {rule} ({rule_metric}) | parts {list(parts)} ({sum(n_part)} source rows) | world {world} ({gpu_label}) | "
        f"eval cache {eval_cache}")
    shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)

    # ---- workers (one per GPU; they hold the model across rounds) ----
    procs = []
    for r in range(world):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(r)
        env["PYTHONPATH"] = "/pmx:/pmx/helpers"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["HF_HOME"] = "/data/hf_cache"
        cmd = [sys.executable, "/pmx/bsf_verbatim_worker.py", "--rank", str(r), "--work", WORK, "--bsf-dir", BSF_DIR, "--mu-acts", MU_ACTS,
               "--ranks", str(R), "--max-batch", str(max_batch), "--batch-tokens", str(batch_tokens), "--min-tok", str(min_tok), "--max-tok", str(max_tok)]
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(2)
    stopped = {"v": False}

    def stop_workers():
        if stopped["v"]:
            return
        stopped["v"] = True
        open(f"{WORK}/stop", "w").close()
        for p in procs:
            try:
                p.wait(timeout=900)
            except Exception:  # noqa
                p.kill()

    try:
        return _build_body(out, out_name, n_candidates, n_rows, G_cap, R, doc_cap, seed, rule, rule_metric, parts, part_dirs, n_part, min_tok, max_tok,
                           eval_cache, topup_rounds, topup_margin, shortfall_tol, gpu_label, world, procs, stop_workers, log, T0, BSF, MODEL, READ_LAYER)
    finally:
        stop_workers()


def _build_body(out, out_name, n_candidates, n_rows, G_cap, R, doc_cap, seed, rule, rule_metric, parts, part_dirs, n_part, min_tok, max_tok,
                eval_cache, topup_rounds, topup_margin, shortfall_tol, gpu_label, world, procs, stop_workers, log, T0, BSF, MODEL, READ_LAYER):
    import json, shutil, time
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    import torch.nn.functional as F
    row_b = D * 4
    rng = np.random.default_rng(seed)
    used_lines = [np.zeros(n, bool) for n in n_part]
    cand = []                       # global candidate id -> (text, part_idx, src_vec_idx, src, src_ctx_len, src_W)
    doc_ids, doc_of_list = {}, []
    round_info = []

    # ---- candidate sampling: seeded uniform lines (without replacement) per part, shuffled, dealt round-robin to the workers ----
    def sample_round(k, n_cand):
        t0 = time.time()
        P = len(part_dirs)
        avail = [int((~u).sum()) for u in used_lines]
        n_cand = min(n_cand, sum(avail))
        take = [n_cand // P + (1 if i < n_cand % P else 0) for i in range(P)]
        take = [min(t, a) for t, a in zip(take, avail)]
        left = n_cand - sum(take); i = 0
        while left > 0 and i < 10 * P:
            add = min(avail[i % P] - take[i % P], left); take[i % P] += add; left -= add; i += 1
        jobs = []
        for pi, pd in enumerate(part_dirs):
            free = np.flatnonzero(~used_lines[pi])
            pick = np.sort(rng.choice(free, take[pi], replace=False)) if take[pi] else np.zeros(0, np.int64)
            used_lines[pi][pick] = True
            jobs.append((f"{pd}/records.jsonl", pick))
        with ThreadPoolExecutor(P) as ex:
            res = list(ex.map(lambda j: _read_selected_lines(*j), jobs))
        rows = [(rec, pi) for pi, recs in enumerate(res) for rec in recs]
        order = rng.permutation(len(rows))
        g0 = len(cand)
        files = [open(f"{WORK}/cands_r{r}_k{k}.jsonl", "w") for r in range(world)]
        for j, oi in enumerate(order.tolist()):
            (text, vidx, src, cl, W), pi = rows[oi]
            g = g0 + j
            cand.append((text, pi, vidx, src, cl, W))
            key = (pi, src if src is not None else f"v{vidx}")
            did = doc_ids.get(key)
            if did is None:
                did = len(doc_ids); doc_ids[key] = did
            doc_of_list.append(did)
            files[g % world].write(json.dumps({"i": g, "text": text}) + "\n")
        for r, f in enumerate(files):
            f.close()
            open(f"{WORK}/cands_r{r}_k{k}.jsonl.ready", "w").close()
        info = {"round": k, "n_candidates": len(rows), "per_part": dict(zip(parts, take)), "sample_s": time.time() - t0}
        round_info.append(info)
        log(f"round {k}: sampled {len(rows)} candidates ({dict(zip(parts, take))}) in {time.time() - t0:.0f}s -> {world} worker files")
        return len(rows)

    def wait_round(k):
        t0 = time.time(); last = t0
        while True:
            if all(os.path.exists(f"{WORK}/r{r}_k{k}.done.json") for r in range(world)):
                break
            for r, p in enumerate(procs):
                if p.poll() is not None and not os.path.exists(f"{WORK}/r{r}_k{k}.done.json"):
                    raise RuntimeError(f"worker rank {r} exited with code {p.returncode} during round {k} (see its [r{r}] log lines)")
            if time.time() - last > 300:
                log(f"round {k}: waiting for workers ({(time.time() - t0) / 60:.0f} min; done: "
                    f"{[r for r in range(world) if os.path.exists(f'{WORK}/r{r}_k{k}.done.json')]})")
                last = time.time()
            time.sleep(10)
        dones = [json.load(open(f"{WORK}/r{r}_k{k}.done.json")) for r in range(world)]
        tot_tok = sum(d["tokens"] for d in dones); el = time.time() - t0
        round_info[k].update({"workers": dones, "wall_s_incl_wait": el, "tokens": tot_tok, "n_forwarded": sum(d["n_forwarded"] for d in dones),
                              "worker_tok_per_s_sum": sum(d["tok_per_s"] for d in dones)})
        log(f"round {k}: workers done — {sum(d['n_forwarded'] for d in dones)} texts / {tot_tok / 1e6:.1f}M tokens; per-worker "
            f"{[round(d['seconds'] / 60, 1) for d in dones]} min; sum throughput {sum(d['tok_per_s'] for d in dones):.0f} tok/s")
        return dones

    keys_r = ("top_i", "top_v", "cos_x", "wfrac", "peak_cos", "peak_proj", "cos_last", "cos_max", "proj_last", "proj_max")
    keys_1 = ("norm_c", "norm_raw", "ntok", "ok")

    def load_outputs(n_rounds):
        N = len(cand)
        A = {"top_i": np.zeros((N, R), np.int32), "top_v": np.zeros((N, R), np.float32), "cos_x": np.zeros((N, R), np.float32),
             "wfrac": np.zeros((N, R), np.float32), "peak_cos": np.full((N, R), -1, np.int16), "peak_proj": np.full((N, R), -1, np.int16),
             "cos_last": np.zeros((N, R), np.float32), "cos_max": np.zeros((N, R), np.float32), "proj_last": np.zeros((N, R), np.float32),
             "proj_max": np.zeros((N, R), np.float32), "norm_c": np.zeros(N, np.float32), "norm_raw": np.zeros(N, np.float32),
             "ntok": np.zeros(N, np.int16), "ok": np.zeros(N, bool)}
        loc = np.full((N, 3), -1, np.int64)
        xmaps = {}
        for k in range(n_rounds):
            for r in range(world):
                m = np.load(f"{WORK}/r{r}_k{k}.meta.npz")
                g = m["gidx"]
                for key in keys_r + keys_1:
                    A[key][g] = m[key]
                loc[g, 0] = r; loc[g, 1] = k; loc[g, 2] = np.arange(len(g))
                xmaps[(r, k)] = np.memmap(f"{WORK}/r{r}_k{k}.xlast.f32", np.float32, "r", shape=(max(len(g), 1), D))
        assert (loc[:, 0] >= 0).all(), "some candidates have no worker output"
        return A, loc, xmaps

    # ---- rounds ----
    k = 0; n_next = n_candidates
    bsf = None; es = eval_dirs = recs_ho = ho_vecs = ho_rows = None
    excl_blk = excl_bsf_old = eval_bsf_new = eval_bsf_old = None
    dev = "cuda:0"
    while True:
        sample_round(k, n_next)
        wait_round(k)
        A, loc, xmaps = load_outputs(k + 1)
        N = len(cand)
        # text + dedupe filters (candidate level)
        ok = A["ok"].copy()
        n_len_drop = int((~ok).sum())
        seen = set(); n_dup = n_txt = 0
        for g in range(N):
            if not ok[g]:
                continue
            t = cand[g][0]
            if len(t.strip()) < 3:
                ok[g] = False; n_txt += 1; continue
            if t in seen:
                ok[g] = False; n_dup += 1; continue
            seen.add(t)
        del seen
        # norm filter (10x median ||x - mu_acts|| over the last-token activations of all ok candidates)
        med = float(np.median(A["norm_c"][ok])); thr = NORM_FILTER_MULT * med
        norm_bad = ok & ~((A["norm_c"] > 1e-6) & (A["norm_c"] <= thr))
        n_norm_drop = int(norm_bad.sum()); ok &= ~norm_bad
        log(f"round {k}: {N} candidates | ok {int(ok.sum())} (drops: n_tok outside [{min_tok},{max_tok}] {n_len_drop}, empty text {n_txt}, "
            f"duplicate text {n_dup}, norm {n_norm_drop}; norm median {med:.1f} thr {thr:.1f}) | n_tok mean {A['ntok'][ok].mean():.1f}")

        if bsf is None:
            # ---- BSF on the driver GPU + exclusions (once) ----
            t0 = time.time()
            bsf = BSF(BSF_DIR, dev, with_Q=True)
            assert bsf.G > int(A["top_i"].max())
            es = torch.load(eval_cache, map_location="cpu", weights_only=False)
            eval_dirs = {kk[:-5]: F.normalize(es[kk].float(), dim=-1) for kk in es if kk.endswith("_dirs")}
            recs_ho = [json.loads(l) for l in open(f"{POOL_HELDOUT}/records.jsonl")]
            ho_sz = os.path.getsize(f"{POOL_HELDOUT}/vecs.f32")
            ho_vecs = np.memmap(f"{POOL_HELDOUT}/vecs.f32", np.float32, "r", shape=(ho_sz // row_b, D))
            assert len(recs_ho) == ho_vecs.shape[0]
            ho_rows = {}
            for rr in recs_ho:
                ho_rows.setdefault(rr["family"], []).append(int(rr["vec_idx"]))
            excl_bsf_old = sorted({int(rr["block"]) for rr in recs_ho if rr["family"] == "bsf"})
            eval_bsf_old = sorted({int(recs_ho[i]["block"]) for i in es["meta"]["rows"]["bsf"]})
            assert set(eval_bsf_old) <= set(excl_bsf_old)
            with torch.no_grad():
                ev = eval_dirs["bsf"].to(dev)
                yv = F.normalize(ev @ bsf.zca, dim=-1)
                gnv = (yv @ bsf.E).view(-1, bsf.G, bsf.b).norm(dim=-1)
                eval_bsf_new = sorted(set(gnv.argmax(1).cpu().tolist()))
            excl_blk = np.zeros(bsf.G, bool)
            excl_blk[np.array(excl_bsf_old, np.int64)] = True
            excl_blk[np.array(eval_bsf_new, np.int64)] = True
            log(f"BSF on driver GPU (G={bsf.G} b={bsf.b} k={bsf.k}) + eval cache {sorted(eval_dirs)} + pool_heldout {ho_vecs.shape[0]} rows "
                f"({time.time() - t0:.0f}s) | bsf exclusions: {len(excl_bsf_old)} literal old ids + {len(eval_bsf_new)} eval-dir top blocks -> "
                f"{int(excl_blk.sum())} excluded, {bsf.G - int(excl_blk.sum())} usable")

        # ---- selection ----
        t0 = time.time()
        G = bsf.G
        top_i = A["top_i"]
        doc_of = np.asarray(doc_of_list, np.int64); n_doc = len(doc_ids)
        elig_base = ok[:, None] & ~excl_blk[top_i.astype(np.int64)]
        pk_m = A["peak_cos"] if rule_metric == "cos" else A["peak_proj"]
        pk_o = A["peak_proj"] if rule_metric == "cos" else A["peak_cos"]
        srng = np.random.default_rng(seed + 17 + k)
        log(f"selection run 1 (rule-free) over {int(elig_base.any(1).sum())} eligible candidates:")
        tok1, rank1, st = _select_rows(top_i, elig_base, doc_of, n_rows, G_cap, doc_cap, G, n_doc, srng, None, log)
        r1 = rank1.astype(np.int64) - 1
        pk1 = pk_m[tok1, r1]; pk1o = pk_o[tok1, r1]
        pre = {"n_selected": int(len(tok1)), "metric": rule_metric, **_anchor_stats(pk1), "other_metric": {"name": "proj" if rule_metric == "cos" else "cos", **_anchor_stats(pk1o)},
               "blocks_used": int((st["block_cnt"] > 0).sum()), "rank_hist": np.bincount(rank1, minlength=R + 1)[1:].tolist()}
        all_r1 = {"metric": rule_metric, **_anchor_stats(pk_m[ok, 0]), "other_metric": {"name": "proj" if rule_metric == "cos" else "cos", **_anchor_stats(pk_o[ok, 0])}}
        log(f"run 1: {len(tok1)} rows, blocks used {pre['blocks_used']}, rank hist {pre['rank_hist']} | END-ANCHOR ({rule_metric}) dir_peak_last "
            f"{pre['pass_last']:.4f} dir_peak_last2 {pre['pass_last2']:.4f} | other metric peak_last {pre['other_metric']['pass_last']:.4f} | "
            f"all ok candidates @rank1: peak_last {all_r1['pass_last']:.4f} ({time.time() - t0:.0f}s)")
        if rule == "none":
            tok_f, rank_f = tok1, rank1
            post = None
        else:
            lim = 0 if rule == "last" else 1
            fail = pk1 > lim
            bl_f = top_i[tok1[fail], r1[fail]].astype(np.int64)
            np.add.at(st["block_cnt"], bl_f, -1); np.add.at(st["per_doc"], doc_of[tok1[fail]], -1); st["used"][tok1[fail]] = False
            tok_keep, rank_keep = tok1[~fail], rank1[~fail]
            elig2 = elig_base & (pk_m <= lim)
            log(f"rule {rule}: dropped {int(fail.sum())} rows -> {len(tok_keep)} kept; top-up run 2 (rule as eligibility, {int(elig2.any(1).sum())} eligible):")
            tok2, rank2, st = _select_rows(top_i, elig2, doc_of, n_rows - len(tok_keep), G_cap, doc_cap, G, n_doc, srng, st, log)
            tok_f = np.concatenate([tok_keep, tok2]); rank_f = np.concatenate([rank_keep, rank2])
            post = {"rule": rule, "metric": rule_metric, "n_dropped": int(fail.sum()), "n_kept_from_run1": int(len(tok_keep)), "n_topup": int(len(tok2)),
                    "n_final": int(len(tok_f))}
            log(f"run 2: +{len(tok2)} -> {len(tok_f)} rows")
        rf = rank_f.astype(np.int64) - 1
        blk_f = top_i[tok_f, rf].astype(np.int64)
        assert len(np.unique(tok_f)) == len(tok_f), "a candidate selected twice"
        assert not excl_blk[blk_f].any() and st["block_cnt"].max() <= G_cap and ok[tok_f].all()
        if rule != "none":
            assert (pk_m[tok_f, rf] <= (0 if rule == "last" else 1)).all()
        n_f = len(tok_f)
        log(f"round {k} selection: {n_f}/{n_rows} rows from {N} candidates ({time.time() - t0:.0f}s)")
        if n_f >= n_rows * (1 - shortfall_tol) or k >= topup_rounds:
            break
        yld = n_f / max(N, 1); need = n_rows - n_f
        avail = int(sum((~u).sum() for u in used_lines))
        n_next = int(min(max(need / max(yld, 1e-6) * topup_margin, 50_000), n_candidates, avail))
        if n_next < 1000:
            log("no more source candidates — stopping the rounds"); break
        log(f"shortfall {need}: top-up round {k + 1} with {n_next} more candidates (yield so far {yld:.3f} rows/candidate, margin {topup_margin})")
        k += 1
    stop_workers()
    log(f"workers stopped; {k + 1} round(s), {len(cand)} candidates total")

    # ---- mint the final directions from the stored last-token activations (same math as the worker / everything builder) ----
    t0 = time.time()
    ord_ = np.lexsort((loc[tok_f, 2], loc[tok_f, 1], loc[tok_f, 0]))           # storage order -> sequential-ish reads
    tok_s, rank_s, blk_s = tok_f[ord_], rank_f[ord_], blk_f[ord_]
    rs = rank_s.astype(np.int64) - 1
    stage = np.memmap(f"{WORK}/stage.f32", np.float32, "w+", shape=(n_f, D))
    cos_x_re = np.empty(n_f, np.float32); wf_re = np.empty(n_f, np.float32); rank_ok = 0
    CH = 4096
    with torch.no_grad():
        for c0 in range(0, n_f, CH):
            c1 = min(c0 + CH, n_f); ts = tok_s[c0:c1]
            lr = loc[ts]
            X = np.empty((c1 - c0, D), np.float32)
            for (r, kk) in {(int(a), int(b)) for a, b in lr[:, :2]}:
                m = (lr[:, 0] == r) & (lr[:, 1] == kk)
                X[m] = xmaps[(r, kk)][lr[m, 2]]
            x = torch.from_numpy(X).to(dev)
            bl = torch.from_numpy(blk_s[c0:c1]).to(dev)
            dirs, cx, wf = bsf.mint(x, bl[:, None])
            stage[c0:c1] = dirs[:, 0].cpu().numpy(); cos_x_re[c0:c1] = cx[:, 0].cpu().numpy(); wf_re[c0:c1] = wf[:, 0].cpu().numpy()
            ti, _ = bsf.block_topk(x, R)
            rank_ok += int((ti == bl[:, None]).any(1).sum())
    stage.flush()
    cos_x_w = A["cos_x"][tok_s, rs]
    max_dcos = float(np.abs(cos_x_re - cos_x_w).max())
    log(f"minted {n_f} directions ({time.time() - t0:.0f}s): block re-scan consistency {rank_ok}/{n_f}, max |cos_x(driver) - cos_x(worker)| {max_dcos:.2e}, "
        f"cos_x median {np.median(cos_x_re):.3f}")
    assert rank_ok == n_f, "selected block not among the recomputed top blocks"
    assert max_dcos < 5e-3, f"driver/worker direction mismatch {max_dcos}"

    # ---- leak guard: cos > LEAK_COS vs ANY eval-cache direction or ANY pool_heldout row -> drop, then assert ----
    t0 = time.time()
    ref_names = sorted(eval_dirs) + [f"pool_heldout/{f}" for f in sorted(ho_rows)]
    ref = torch.cat([eval_dirs[f] for f in sorted(eval_dirs)] +
                    [F.normalize(torch.from_numpy(np.asarray(ho_vecs[np.array(ho_rows[f], np.int64)])).float(), dim=-1) for f in sorted(ho_rows)]).to(dev)
    ref_off = np.cumsum([0] + [len(eval_dirs[f]) for f in sorted(eval_dirs)] + [len(ho_rows[f]) for f in sorted(ho_rows)])
    maxcos = np.empty(n_f, np.float32); leak_tbl = {n: -1.0 for n in ref_names}
    nmin = nmax = 1.0
    with torch.no_grad():
        for c0 in range(0, n_f, 8192):
            x = torch.from_numpy(np.ascontiguousarray(stage[c0:c0 + 8192])).to(dev)
            nn = x.norm(dim=-1); nmin = min(nmin, float(nn.min())); nmax = max(nmax, float(nn.max()))
            cos = x @ ref.T
            for ri, n in enumerate(ref_names):
                leak_tbl[n] = max(leak_tbl[n], float(cos[:, ref_off[ri]:ref_off[ri + 1]].max()))
            maxcos[c0:c0 + 8192] = cos.max(1).values.cpu().numpy()
    assert abs(nmin - 1) < 1e-3 and abs(nmax - 1) < 1e-3, f"non-unit rows [{nmin}, {nmax}]"
    leak_drop = maxcos > LEAK_COS
    keep = np.flatnonzero(~leak_drop)
    n_out = int(len(keep))
    assert n_out > 0 and (maxcos[keep] <= LEAK_COS).all()
    log(f"leak guard ({time.time() - t0:.0f}s): {int(leak_drop.sum())} rows dropped (cos > {LEAK_COS} vs {ref.shape[0]} refs); max cos after "
        f"{float(maxcos[keep].max()):.4f} | table " + " ".join(f"{n.split('/')[-1][:12]}={v:.3f}" for n, v in leak_tbl.items()))
    del ref

    # ---- assemble: seeded shuffle -> vecs.f32 + records.jsonl (line i == vec_idx i) ----
    t0 = time.time()
    order = keep[np.random.default_rng(seed + 99).permutation(n_out)]                 # final row i = staged row order[i]
    os.makedirs(f"{WORK}/out", exist_ok=True)
    vecs = np.memmap(f"{WORK}/out/vecs.f32", np.float32, "w+", shape=(n_out, D))
    for c0 in range(0, n_out, 8192):
        vecs[c0:c0 + 8192] = stage[np.sort(order[c0:c0 + 8192])][np.argsort(np.argsort(order[c0:c0 + 8192]))]
    vecs.flush()
    # spot-check the permutation logic on a few rows
    for i in np.random.default_rng(1).integers(0, n_out, size=min(64, n_out)).tolist():
        assert np.array_equal(vecs[i], stage[order[i]])
    part_names = [os.path.basename(pd) for pd in part_dirs]
    ntok_a = A["ntok"].astype(np.int64)
    pk_cos_f = A["peak_cos"][tok_s, rs].astype(np.int64); pk_proj_f = A["peak_proj"][tok_s, rs].astype(np.int64)
    fam_counts = {"bsf": n_out}
    n_chars = np.zeros(n_out, np.int64)
    with open(f"{WORK}/out/records.jsonl", "w") as rf_:
        for i, j in enumerate(order.tolist()):
            g = int(tok_s[j]); text, pi, vidx, src, cl, W = cand[g]
            nt = int(ntok_a[g]); n_chars[i] = len(text)
            rec = {"vec_idx": i, "family": "bsf", "target_text": text, "block": int(blk_s[j]), "rank": int(rank_s[j]),
                   "gnorm": round(float(A["top_v"][g, rs[j]]), 4), "cos_x": round(float(cos_x_re[j]), 4), "whiten_frac": round(float(wf_re[j]), 4),
                   "n_tok": nt, "ctx_len": nt, "peak_from_end": int(pk_cos_f[j]), "peak_from_end_proj": int(pk_proj_f[j]),
                   "peak_pos": nt - 1 - int(pk_cos_f[j]), "fire_from_end": 0,
                   "cos_last": round(float(A["cos_last"][g, rs[j]]), 4), "cos_max": round(float(A["cos_max"][g, rs[j]]), 4),
                   "act_norm": round(float(A["norm_c"][g]), 2), "src_part": part_names[pi], "src_vec_idx": int(vidx), "src": src,
                   "src_ctx_len": cl, "src_W": W}
            rf_.write(json.dumps(rec) + "\n")
    log(f"assembled {n_out} rows ({time.time() - t0:.0f}s)")

    # ---- stats ----
    blk_out = blk_s[order]; rank_out = rank_s[order]; ntok_out = ntok_a[tok_s[order]]
    block_cnt_out = np.bincount(blk_out, minlength=bsf.G)
    rows_per_block_hist = np.bincount(block_cnt_out[block_cnt_out > 0], minlength=G_cap + 1).tolist()
    final_anchor = {"cos": _anchor_stats(pk_cos_f[order]), "proj": _anchor_stats(pk_proj_f[order])}
    wall = time.time() - T0
    bsf_stats = {"hf": BSF_HF, "dir": BSF_DIR, "G": bsf.G, "b": bsf.b, "k": bsf.k, "cap_per_block": G_cap, "bsf_ranks": R, "doc_cap": doc_cap,
                 "taken": n_out, "target": n_rows, "shortfall": n_rows - n_out, "blocks_used": int((block_cnt_out > 0).sum()),
                 "blocks_full": int((block_cnt_out >= G_cap).sum()), "rows_per_block_hist": rows_per_block_hist,
                 "rank_hist": np.bincount(rank_out, minlength=R + 1)[1:].tolist(),
                 "excluded_blocks": {"literal_old_ids": len(excl_bsf_old), "eval_dir_top1_under_this_bsf": len(eval_bsf_new), "total": int(excl_blk.sum())},
                 "block_consistency_rescan": rank_ok / max(n_f, 1), "cos_x": _qs(cos_x_re[order]), "whiten_frac": _qs(wf_re[order]),
                 "gnorm": _qs(A["top_v"][tok_s, rs][order]), "zca_asym_max": bsf.zca_asym, "zca_inv_err_max": bsf.zca_inv_err,
                 "docs_used": int(len(np.unique(doc_of[tok_s[order]]))), "dir_math": "unit((Q[b]^T Q[b] ((x - mu_bsf) @ zca)) @ zca^-1)"}
    stats = {"kind": "bsf_verbatim: BSF (SASA block-sparse featurizer) top-block subspace direction of the layer-42 residual at the LAST token of a "
                     "SHORT text forwarded ALONE ([BOS]+text) — the target text IS the literal verbatim context that produced the direction",
             "n_examples": n_out, "n_vecs": n_out, "families": fam_counts, "layout": "seeded shuffle (records.jsonl line i == vec_idx i)",
             "seed": seed, "d_model": D, "model": MODEL, "layer": READ_LAYER,
             "dir": "unit((Q[b]^T Q[b] ((x - mu_bsf) @ zca)) @ zca^-1); x = L42 residual at the last token of [BOS]+target_text forwarded alone; "
                    f"b = top active block (rank <= {R}) with capacity (<= {G_cap} rows/block, level-by-level fill), excluded blocks skipped",
             "target": "the verbatim text itself (a realact_short_20m target window, 8-32 source tokens): ctx_len == n_tok, nothing outside the "
                       "target was in the model's context; the direction fires at the last token by construction and by measurement (end_anchor)",
             "source": {"parts": part_names, "part_dirs": part_dirs, "source_rows_per_part": dict(zip(part_names, n_part)),
                        "n_candidates_sampled": len(cand), "rounds": round_info,
                        "sampling": "seeded uniform line sample without replacement per part (equal split), shuffled, dealt round-robin to the GPU workers",
                        "doc_key": "(src_part, src) = the source collect window id r{rank}_w{window} (the parts carry no document id)", "n_docs": n_doc,
                        "note": "the 5M mix compositor also draws 1M realact rows from these parts (different family/direction; overlap acceptable)"},
             "bsf": bsf_stats,
             "candidates": {"n": len(cand), "ok_after_filters": int(ok.sum()), "n_tok_range_kept": [min_tok, max_tok],
                            "dropped": {"n_tok_out_of_range": n_len_drop, "empty_text": n_txt, "duplicate_text": n_dup, "norm": n_norm_drop},
                            "eligible_any_rank": int(elig_base.any(1).sum())},
             "norm_filter": {"mult": NORM_FILTER_MULT, "median": med, "thr": thr, "centering": MU_ACTS, "over": "last-token activations of all ok candidates"},
             "end_anchor": {"rule": rule, "metric": rule_metric,
                            "definition": "peak_from_end = n_tok-1-argmax_t m(h_t, dir) over the content tokens of the standalone forward (BOS dropped); "
                                          "m = cos(h_t, dir) [cos; RL --reward-metric cosine] or h_t @ dir [proj; RL default]; 0 == the last token",
                            "pre_filter_selection": pre, "all_ok_candidates_rank1": all_r1, "post_filter": post, "final_bank": final_anchor},
             "leak_check": {"threshold": LEAK_COS, "refs": ref_names, "n_refs": int(ref_off[-1]), "n_dropped": int(leak_drop.sum()),
                            "max_cos_table_pre_drop": {n: round(v, 4) for n, v in leak_tbl.items()}, "max_cos_after": float(maxcos[keep].max())},
             "n_tok": {"mean": float(ntok_out.mean()), "median": float(np.median(ntok_out)), "min": int(ntok_out.min()), "max": int(ntok_out.max()),
                       "hist": {str(i): int(c) for i, c in enumerate(np.bincount(ntok_out)) if c}},
             "target_chars": {"mean": float(n_chars.mean()), "median": float(np.median(n_chars))},
             "gpu": gpu_label, "world": world, "wall_s": wall, "created": time.time(), "eval_cache": eval_cache, "acts_mu": MU_ACTS, "fresh_store": True}
    meta_out = {**stats, "bank": out, "n_rows": n_out, "trainer_args": {"--data-dir": out, "--bank-file": "vecs.f32"},
                "family_recipes": {"bsf": {**bsf_stats, "source": stats["source"], "dir": stats["dir"], "target": stats["target"]}},
                "exclusions_summary": {"bsf_blocks": {"literal_old_ids": len(excl_bsf_old), "eval_dir_top1_under_this_bsf": len(eval_bsf_new),
                                                      "total": int(excl_blk.sum()),
                                                      "sources": [f"{POOL_HELDOUT}/records.jsonl family=bsf 'block' (ids of the ORIGINAL Aug-21 BSF)",
                                                                  f"{eval_cache} bsf_dirs -> argmax block under {BSF_HF}"]},
                                       "direction_level": f"cos > {LEAK_COS} vs every {eval_cache} *_dirs family and every pool_heldout row -> dropped ({int(leak_drop.sum())})",
                                       "texts": "realact_short_20m parts are FineFineWeb windows already excluded against the eval span hashes; never trained on"},
                "leakage_check": stats["leak_check"], "unit_norm_range": [nmin, nmax], "eval_cache_cos_families": es["meta"].get("cos_families")}
    excl_out = {"bsf_blocks_literal_old_ids": excl_bsf_old, "bsf_blocks_eval_dir_top1_under_this_bsf": eval_bsf_new,
                "bsf_blocks_all": sorted(np.flatnonzero(excl_blk).tolist()), "eval_bsf_blocks_old_ids_used_by_cache": eval_bsf_old}

    # ---- publish ----
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    for fn in ("vecs.f32", "records.jsonl"):
        shutil.copyfile(f"{WORK}/out/{fn}", f"{out}/{fn}.tmp"); os.replace(f"{out}/{fn}.tmp", f"{out}/{fn}")
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
    json.dump(meta_out, open(f"{out}/meta.json", "w"), indent=1)
    json.dump(excl_out, open(f"{out}/exclusions.json", "w"))
    vol.commit()
    vsize = os.path.getsize(f"{out}/vecs.f32")
    assert vsize == n_out * row_b, f"vecs.f32 {vsize} B != {n_out} x {row_b}"
    n_lines = sum(1 for _ in open(f"{out}/records.jsonl"))
    assert n_lines == n_out, (n_lines, n_out)
    log(f"DONE -> {out}: {n_out} rows ({vsize / 2**30:.1f} GB) | published in {time.time() - t0:.0f}s | total {wall / 60:.1f} min")
    return {"out": out, "n": n_out, "families": fam_counts, "shortfall": n_rows - n_out, "blocks_used": bsf_stats["blocks_used"],
            "rows_per_block_hist": rows_per_block_hist, "rank_hist": bsf_stats["rank_hist"], "end_anchor_pre": pre, "end_anchor_post": post,
            "end_anchor_final": final_anchor, "leak_dropped": int(leak_drop.sum()), "n_tok": stats["n_tok"] | {"hist": None}, "candidates": stats["candidates"],
            "rounds": [{kk: v for kk, v in ri.items() if kk != "workers"} for ri in round_info], "minutes": wall / 60, "gpu": gpu_label}


_BUILD_DOC = """out_name -> /data/banks/<out_name>. n_candidates: texts sampled (and forwarded) per round; n_rows: target rows; cap_per_block /
bsf_ranks / doc_cap: the everything builder's bsf selection knobs; rule: none|last|last2 end-anchor filter on rule_metric cos|proj (peak of
cos(h_t, dir) resp. h_t @ dir over the standalone forward must be the last token / within the last 2); parts: realact_short_20m part letters;
topup_rounds: extra candidate rounds (workers keep the model loaded) if the selection falls short of n_rows*(1-shortfall_tol)."""


@app.function(image=image, gpu=BUILD_GPU, cpu=32, memory=192 * 1024, ephemeral_disk=1024 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=12 * 3600)
def build(out_name: str = "bsf_verbatim_1m", n_candidates: int = 3_000_000, n_rows: int = 1_000_000, cap_per_block: int = 34, bsf_ranks: int = 8,
          doc_cap: int = 32, seed: int = 5003, rule: str = "last", rule_metric: str = "cos", parts: str = "pqrstuvwx", min_tok: int = 4,
          max_tok: int = 96, max_batch: int = 512, batch_tokens: int = 32768, eval_cache: str = EVAL_CACHE_V2, overwrite_smoke: bool = False,
          topup_rounds: int = 1, topup_margin: float = 1.5, shortfall_tol: float = 0.02):
    """8-GPU build (see _BUILD_DOC for the parameters)."""
    return _build_impl(out_name, n_candidates, n_rows, cap_per_block, bsf_ranks, doc_cap, seed, rule, rule_metric, parts, min_tok, max_tok,
                       max_batch, batch_tokens, eval_cache, overwrite_smoke, topup_rounds, topup_margin, shortfall_tol, BUILD_GPU)


@app.function(image=image, gpu=SMOKE_GPU, cpu=16, memory=96 * 1024, ephemeral_disk=512 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=6 * 3600)
def build_small(out_name: str = "bsf_verbatim_smoke", n_candidates: int = 20_000, n_rows: int = 1_000_000, cap_per_block: int = 34, bsf_ranks: int = 8,
                doc_cap: int = 32, seed: int = 5003, rule: str = "last", rule_metric: str = "cos", parts: str = "pqrstuvwx", min_tok: int = 4,
                max_tok: int = 96, max_batch: int = 512, batch_tokens: int = 32768, eval_cache: str = EVAL_CACHE_V2, overwrite_smoke: bool = True,
                topup_rounds: int = 0, topup_margin: float = 1.5, shortfall_tol: float = 0.02):
    """Same build on ONE GPU (smoke tests)."""
    return _build_impl(out_name, n_candidates, n_rows, cap_per_block, bsf_ranks, doc_cap, seed, rule, rule_metric, parts, min_tok, max_tok,
                       max_batch, batch_tokens, eval_cache, overwrite_smoke, topup_rounds, topup_margin, shortfall_tol, SMOKE_GPU)


# ----------------------------------------------------------------------------------------------------------------
# verify / anchor re-check / peek
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=SMALL_GPUS, cpu=8, memory=49152, volumes={"/data": vol}, timeout=3600)
def verify(out_name: str = "bsf_verbatim_1m"):
    """From the FINAL artifacts: sizes, vec_idx alignment, families, block exclusions, unit norms, direction-level leak check."""
    import json, sys, time
    import numpy as np
    import torch
    import torch.nn.functional as F
    dev = "cuda:0"
    vol.reload()
    out = f"/data/banks/{out_name}"
    st = json.load(open(f"{out}/build_stats.json")); mt = json.load(open(f"{out}/meta.json")); ex = json.load(open(f"{out}/exclusions.json"))
    N = st["n_examples"]; row_b = D * 4
    assert os.path.getsize(f"{out}/vecs.f32") == N * row_b, "vecs.f32 size != n_examples rows"
    vecs = np.memmap(f"{out}/vecs.f32", np.float32, "r", shape=(N, D))
    recs = [json.loads(l) for l in open(f"{out}/records.jsonl")]
    assert len(recs) == N and all(r["vec_idx"] == i for i, r in enumerate(recs)), "records.jsonl misaligned"
    counts = {}
    excl = set(ex["bsf_blocks_all"])
    pk = np.array([r["peak_from_end"] for r in recs]); nt = np.array([r["n_tok"] for r in recs])
    for r in recs:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
        assert r["family"] == "bsf" and r["target_text"] and r["block"] not in excl and r["ctx_len"] == r["n_tok"]
    assert counts == st["families"] == {"bsf": N}, (counts, st["families"])
    rule = st["end_anchor"]["rule"]
    if rule != "none":
        assert (pk <= (0 if rule == "last" else 1)).all(), "end-anchor rule violated in the final bank"
    es = torch.load(mt["eval_cache"], map_location="cpu", weights_only=False)
    names = sorted(k[:-5] for k in es if k.endswith("_dirs"))
    ho_n = os.path.getsize(f"{POOL_HELDOUT}/vecs.f32") // row_b
    ho = np.memmap(f"{POOL_HELDOUT}/vecs.f32", np.float32, "r", shape=(ho_n, D))
    ref = F.normalize(torch.cat([es[f"{n}_dirs"].float() for n in names] + [torch.from_numpy(np.asarray(ho)).float()]), dim=-1).to(dev)
    names_all = names + ["pool_heldout(all)"]
    offs = np.cumsum([0] + [len(es[f"{n}_dirs"]) for n in names] + [ho.shape[0]])
    leak = np.full(len(names_all), -1.0); nmin = nmax = 1.0; n_bad = 0
    t0 = time.time()
    with torch.no_grad():
        for c0 in range(0, N, 8192):
            x = torch.from_numpy(np.ascontiguousarray(vecs[c0:c0 + 8192])).to(dev)
            nn = x.norm(dim=-1); nmin, nmax = min(nmin, float(nn.min())), max(nmax, float(nn.max()))
            cos = x @ ref.T
            n_bad += int((cos.max(1).values > LEAK_COS).sum())
            for ri in range(len(names_all)):
                leak[ri] = max(leak[ri], float(cos[:, offs[ri]:offs[ri + 1]].max()))
    assert abs(nmin - 1) < 1e-3 and abs(nmax - 1) < 1e-3, (nmin, nmax)
    assert n_bad == 0, f"{n_bad} rows within cos>{LEAK_COS} of an eval/held-out direction"
    print(f"[verify] {out}: {N} rows, families {counts}, unit norms [{nmin:.5f},{nmax:.5f}], no block-id leakage, no direction leakage (>{LEAK_COS}) "
          f"vs {len(names_all)} ref sets ({time.time() - t0:.0f}s)")
    print("[verify] max-cos per ref set: " + " ".join(f"{n[:11]}={leak[ri]:.3f}" for ri, n in enumerate(names_all)))
    print(f"[verify] end-anchor (rule {rule}): peak_last {float((pk == 0).mean()):.4f} peak_last2 {float((pk <= 1).mean()):.4f} | n_tok mean {nt.mean():.1f} "
          f"median {np.median(nt):.0f} | blocks used {st['bsf']['blocks_used']} | rank hist {st['bsf']['rank_hist']}")
    for r in recs[:3]:
        print(f"  sample |v|={np.linalg.norm(vecs[r['vec_idx']]):.4f} {json.dumps({k: (v[:90] if k == 'target_text' else v) for k, v in r.items()})}")
    return {"n": N, "families": counts, "leak": dict(zip(names_all, leak.tolist())), "peak_last": float((pk == 0).mean())}


@app.function(image=image, gpu=SMOKE_GPU, cpu=8, memory=65536, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=3 * 3600)
def anchor_check(out_name: str = "bsf_verbatim_1m", n: int = 2048, seed: int = 0, batch: int = 64):
    """INDEPENDENT end-anchor re-measurement: re-tokenize n random FINAL targets, forward each alone ([BOS]+text), read the L42 residual at
    every token and locate the peak of cos(h_t, dir) / h_t @ dir against the STORED unit direction. Also compares with the stored peak_from_end."""
    import json, sys, time
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import MODEL, READ_LAYER
    from mxf.inject import read_resid
    os.environ["HF_HOME"] = "/data/hf_cache"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    vol.reload()
    out = f"/data/banks/{out_name}"
    st = json.load(open(f"{out}/build_stats.json")); N = st["n_examples"]
    recs = [json.loads(l) for l in open(f"{out}/records.jsonl")]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N, min(n, N), replace=False))
    fd = os.open(f"{out}/vecs.f32", os.O_RDONLY)
    dirs = np.stack([np.frombuffer(_pread_full(fd, D * 4, int(i) * D * 4), np.float32) for i in idx]); os.close(fd)
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else 248044
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True, device_map={"": dev}).eval()
    print(f"[anchor] model up {time.time() - t0:.0f}s; {len(idx)} rows", flush=True)
    ids_all = [tok(recs[i]["target_text"], add_special_tokens=False)["input_ids"] for i in idx.tolist()]
    L_all = np.array([len(x) for x in ids_all])
    same_len = float(np.mean(L_all == np.array([recs[i]["n_tok"] for i in idx.tolist()])))
    pk_cos = np.full(len(idx), -1); pk_proj = np.full(len(idx), -1); cos_last = np.zeros(len(idx), np.float32)
    D_t = torch.from_numpy(dirs).to(dev)
    with torch.no_grad():
        for L in np.unique(L_all).tolist():
            rows = np.flatnonzero(L_all == L)
            for s in range(0, len(rows), batch):
                rr = rows[s:s + batch]
                ids = torch.tensor([[bos] + ids_all[j] for j in rr.tolist()], device=dev)
                h, _ = read_resid(model, READ_LAYER, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, pool="all")
                content = h[:, 1:, :]
                d = D_t[torch.from_numpy(rr).to(dev)]
                cos_t = torch.einsum("bld,bd->bl", F.normalize(content, dim=-1), d); proj_t = torch.einsum("bld,bd->bl", content, d)
                pk_cos[rr] = (L - 1 - cos_t.argmax(1)).cpu().numpy(); pk_proj[rr] = (L - 1 - proj_t.argmax(1)).cpu().numpy()
                cos_last[rr] = cos_t[:, -1].cpu().numpy()
    stored = np.array([recs[i]["peak_from_end"] for i in idx.tolist()])
    res = {"n": int(len(idx)), "retokenized_len_equals_n_tok_rate": same_len,
           "cos": _anchor_stats(pk_cos), "proj": _anchor_stats(pk_proj), "agree_with_stored_peak_from_end": float(np.mean(pk_cos == stored)),
           "cos_last_median": float(np.median(cos_last)), "stored_rule": st["end_anchor"]["rule"]}
    print("[anchor]", json.dumps(res), flush=True)
    return res


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=4, memory=16384)
def peek(out_name: str = "bsf_verbatim_1m", n: int = 5):
    import json
    import numpy as np
    vol.reload()
    out = f"/data/banks/{out_name}"
    st = json.load(open(f"{out}/build_stats.json"))
    print(json.dumps({k: st[k] for k in ("n_examples", "families", "n_tok", "target_chars", "candidates", "norm_filter", "leak_check", "wall_s", "gpu")}, indent=1))
    print("bsf:", json.dumps({k: v for k, v in st["bsf"].items() if k != "rows_per_block_hist"}, indent=1))
    print("rows_per_block_hist:", st["bsf"]["rows_per_block_hist"])
    print("end_anchor:", json.dumps(st["end_anchor"], indent=1))
    fd = os.open(f"{out}/vecs.f32", os.O_RDONLY)
    with open(f"{out}/records.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = json.loads(line)
            v = np.frombuffer(_pread_full(fd, D * 4, r["vec_idx"] * D * 4), np.float32)
            print(f"[{r['family']}] row {r['vec_idx']} |v|={np.linalg.norm(v):.4f} blk {r['block']} r{r['rank']} n_tok {r['n_tok']} peak_from_end {r['peak_from_end']} "
                  f"cos_x {r['cos_x']} :: {r['target_text'][:120]!r}")
    return st["n_examples"]


@app.local_entrypoint()
def run_smoke(out_name: str = "bsf_verbatim_smoke", n_candidates: int = 20_000):
    print(build_small.remote(out_name=out_name, n_candidates=n_candidates, overwrite_smoke=True, topup_rounds=0))
    print(verify.remote(out_name=out_name))


@app.local_entrypoint()
def run_verify(out_name: str = "bsf_verbatim_1m"):
    print(verify.remote(out_name=out_name))


@app.local_entrypoint()
def run_anchor_check(out_name: str = "bsf_verbatim_1m", n: int = 2048):
    print(anchor_check.remote(out_name=out_name, n=n))


@app.local_entrypoint()
def run_peek(out_name: str = "bsf_verbatim_1m"):
    peek.remote(out_name=out_name)
