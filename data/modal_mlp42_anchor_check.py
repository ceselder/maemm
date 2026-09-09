"""Modal app `maemm-mlp42-anchor-check`: STANDALONE end-anchor / fire check of the layer-42 MLP-neuron families (mlp, mlp_pair,
mlp_triple) of a MAEMM midtrain bank (default /data/banks/mlp42_5m_fresh, built by data/modal_mlp42_bank.py + data/mlp42_bank_worker.py).

User rule: every SFT target text must END exactly at the token where its direction fires, verified STANDALONE — the text re-tokenized
on its own (tok(text, add_special_tokens=False)), [BOS/sink 248044] prepended, no long context in front — because the RL reward
(rl/rl.py: F.normalize(h_L42) . dir, sink dropped, max over the LAST 5 kept tokens) and the evaluator score the model's own output that
way. The bank's windows were cut from 256-token scan contexts, so a firing token at the end of the window does not guarantee a peak
at the end of the standalone text. The SAE families got this check in data/modal_sae_maxacts_fresh.py (verify_end_anchor /
filter_end_anchor); this file is the same design for the MLP families.

Definitions (== data/mlp42_bank_worker.py / data/mlp42_neurons_worker.py, the code that built the bank):
  neuron value  a_i = act_fn(gate_proj(x))_i * up_proj(x)_i = input i of layers[42].mlp.down_proj, captured with a forward_pre_hook
                on down_proj (mlp42_neurons_worker.forward_capture — reused verbatim; it also returns the layer-42 residual output,
                the same tensor mxf.inject.read_resid returns).
  polarity      per-neuron sign of the extreme corpus value (records: "polarity"); a_s = polarity * a is the polarity-signed value.
  fire          a_s >= REL_THR * corpus_max, REL_THR = 0.10, corpus_max = max|a| over the 1.02M-token statistics pass
                (/data/mlp42/neuron_stats.npz "max_abs"; every record carries it as "corpus_max", cross-checked here).
  direction     mlp: polarity * unit(down_proj[:, i]); mlp_pair / mlp_triple: unit(sum_m a_m col_m) with the RAW activations at the joint
                firing token — the row of <bank>/vecs.f32 (unit norm), which is what the reward / evaluator dot with.
Per row (BOS position dropped, target tokens t = 0..L-1):
  neuron level  per member m: peak_pos_m = argmax_t a_s_m[t]; pass_last_m = (peak_pos_m == L-1); pass_last2_m = (peak_pos_m >= L-2);
                fire_last_m = a_s_m[L-1] >= REL_THR * corpus_max_m; a_last_over_window_peak_m = a_s_m[L-1] / act_m(record, 256-ctx scan).
                pairs / triples: "all" = every member satisfies the condition (also reported per member).
  direction     cos_t = <normalize(h_L42[t]), vecs[vec_idx]>; dir_peak_last = (argmax_t cos_t == L-1); dir_peak_last2 = (>= L-2).
                This is exactly the quantity the RL reward / evaluator maximise.
Filter rules (RULES): "fire_and_dirpeak_last" (default: EVERY member fires at the last token AND the direction peak is the last token),
  "fire_and_dirpeak_last2", "last" / "last2" (direction-only), "fire_last" (neuron-only).

    source ~/modal_venv/bin/activate; export MODAL_PROFILE=safety-sahan
    modal deploy data/modal_mlp42_anchor_check.py
    # 2048-row sample per family (~2 min on one H200); write=True stores the summary under build_stats.json / meta.json ["mlp_anchor_check"]
    python -c "import modal; print(modal.Function.from_name('maemm-mlp42-anchor-check','verify_mlp_anchor').spawn().object_id)"
    # FULL pass, sharded: 8 single-GPU shards over contiguous row ranges -> <bank>/mlp_anchor_shards/shard_XX_of_08.npz, then a CPU merge
    # -> <bank>/mlp_anchor_rows.json (compositor schema, fail_vec_idx UNIONED with <bank>/eval_cos_rows.json) + build_stats/meta ["mlp_anchor_filter"]
    python - <<'EOF'
    import modal
    sh = modal.Function.from_name('maemm-mlp42-anchor-check', 'filter_mlp_anchor_shard')
    ids = [sh.spawn(shard=s, n_shards=8).object_id for s in range(8)]
    [modal.FunctionCall.from_id(i).get() for i in ids]
    print(modal.Function.from_name('maemm-mlp42-anchor-check', 'filter_mlp_anchor_merge').spawn(rule='fire_and_dirpeak_last', n_shards=8, write=True).object_id)
    EOF
    # single-GPU full pass (same code path, ~80 min at 200 rows/s):
    python -c "import modal; print(modal.Function.from_name('maemm-mlp42-anchor-check','filter_mlp_anchor').spawn(rule='fire_and_dirpeak_last', write=True).object_id)"
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = os.environ.get("MAEMM_MLP42_ANCHOR_APP", "maemm-mlp42-anchor-check")
app = modal.App(APP_NAME)

# same pins as data/modal_mlp42_bank.py + flash-linear-attention from data/modal_sae_maxacts_fresh.py (one environment across the suite).
# Measured 2026-09-09 (2048 rows/family sample): fla made no difference on H200 (202 rows/s with or without); B200 was 2.6x SLOWER (78 rows/s).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "peft==0.20.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet", "scipy==1.17.1")
    .pip_install("flash-linear-attention==0.5.2")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "data" / "mlp42_neurons_worker.py", "/pmx/helpers/mlp42_neurons_worker.py")
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
GPUS = ["H200", "B200"]                       # forward-only; H200 preferred (B200 measured 2.6x slower for these short-sequence forwards)
BANK_DEFAULT = "/data/banks/mlp42_5m_fresh"
FAMS_DEFAULT = "mlp,mlp_pair,mlp_triple"
NEURON_STATS = "/data/mlp42/neuron_stats.npz"   # corpus max|a| + polarity the bank used (run_build hard-codes this path)
REL_THR = 0.10                                # fire := polarity * a >= REL_THR * corpus max|a|  (== mlp42_bank_worker.REL_THR)
MAXM = 3                                      # members per direction (mlp 1, mlp_pair 2, mlp_triple 3)
DIR_TXT = "argmax_t cos(normalize(h_L42[t]), vecs[vec_idx]) over the standalone re-tokenized target (BOS dropped)"
FIRE_TXT = f"EVERY member neuron fires at the last token: polarity * a[last] >= {REL_THR} * corpus_max"
RULE_TXT = {"last": f"direction peak ({DIR_TXT}) must be the LAST token",
            "last2": f"direction peak ({DIR_TXT}) must be within the LAST 2 tokens",
            "fire_last": FIRE_TXT,
            "fire_and_dirpeak_last": f"{FIRE_TXT} AND the direction peak ({DIR_TXT}) is the LAST token",
            "fire_and_dirpeak_last2": f"{FIRE_TXT} AND the direction peak ({DIR_TXT}) is within the LAST 2 tokens"}
RULES = {"last": lambda m: m["dp1"], "last2": lambda m: m["dp2"], "fire_last": lambda m: m["fire_all"],
         "fire_and_dirpeak_last": lambda m: m["fire_all"] & m["dp1"], "fire_and_dirpeak_last2": lambda m: m["fire_all"] & m["dp2"]}
RULE_LAST_PAIR = {"last": ("last", "last2"), "last2": ("last", "last2"), "fire_last": ("fire_last", "fire_last"),
                  "fire_and_dirpeak_last": ("fire_and_dirpeak_last", "fire_and_dirpeak_last2"),
                  "fire_and_dirpeak_last2": ("fire_and_dirpeak_last", "fire_and_dirpeak_last2")}   # (pass_last, pass_last2) columns per rule


def _env():
    import sys
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, "/pmx/helpers")


# ----------------------------------------------------------------------------------------------------------------
# shared code path (sample, single-GPU full pass and the sharded full pass all go through _load_rows -> _score -> _summarize)
# ----------------------------------------------------------------------------------------------------------------
def _load_rows(bank, fams, log):
    """records.jsonl (line i == vec_idx i) -> compact arrays over ALL rows of the requested families (in vec_idx order).
    Returns dict: vec_idx [n], fam [n] (str), text list[n], n_tok_rec [n], neurons [n,3] (-1 pad), pol [n,3], cmax [n,3], act_rec [n,3]
    (polarity-signed activation at the firing token in the bank's 256-token scan context; "act" for mlp, "acts" for pair/triple)."""
    import json
    import numpy as np
    want = set(fams)
    vi, fa, tx, nt, ne, po, cm, ac = [], [], [], [], [], [], [], []
    with open(f"{bank}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            if r["family"] not in want:
                continue
            assert int(r["vec_idx"]) == i, (i, r["vec_idx"])
            if r["family"] == "mlp":
                ns, ps, cs, as_ = [int(r["neuron"])], [float(r["polarity"])], [float(r["corpus_max"])], [float(r["act"])]
            else:
                ns, ps, cs, as_ = [int(x) for x in r["neurons"]], [float(x) for x in r["polarity"]], [float(x) for x in r["corpus_max"]], [float(x) for x in r["acts"]]
            k = len(ns)
            assert 1 <= k <= MAXM and len(ps) == k and len(cs) == k and len(as_) == k, r
            vi.append(i); fa.append(r["family"]); tx.append(r["target_text"]); nt.append(int(r.get("n_tok", -1)))
            ne.append(ns + [-1] * (MAXM - k)); po.append(ps + [0.0] * (MAXM - k)); cm.append(cs + [0.0] * (MAXM - k)); ac.append(as_ + [0.0] * (MAXM - k))
    rows = {"vec_idx": np.asarray(vi, np.int64), "fam": np.asarray(fa), "text": tx, "n_tok_rec": np.asarray(nt, np.int64),
            "neurons": np.asarray(ne, np.int64).reshape(-1, MAXM), "pol": np.asarray(po, np.float32).reshape(-1, MAXM),
            "cmax": np.asarray(cm, np.float32).reshape(-1, MAXM), "act_rec": np.asarray(ac, np.float32).reshape(-1, MAXM)}
    log(f"{bank}: loaded {len(vi)} rows of {fams}: " + " ".join(f"{f}={int((rows['fam'] == f).sum())}" for f in fams))
    return rows


def _subset(rows, idx):
    import numpy as np
    idx = np.asarray(idx, np.int64)
    return {k: (v[idx] if k != "text" else [v[j] for j in idx]) for k, v in rows.items()}


def _check_neuron_stats(rows, log):
    """Cross-check the records' corpus_max / polarity against /data/mlp42/neuron_stats.npz (the file the bank builder read)."""
    import numpy as np
    st = np.load(NEURON_STATS)
    max_abs = st["max_abs"].astype(np.float32); pol = st["polarity"].astype(np.float32)
    m = rows["neurons"] >= 0
    nid = rows["neurons"][m]
    cm_ok = np.isclose(rows["cmax"][m], max_abs[nid], rtol=1e-4, atol=1e-6)
    po_ok = rows["pol"][m] == pol[nid]
    res = {"neuron_stats": NEURON_STATS, "n_members_checked": int(m.sum()), "corpus_max_matches_rate": float(cm_ok.mean()),
           "polarity_matches_rate": float(po_ok.mean()), "rel_thr": REL_THR, "d_ff": int(len(max_abs))}
    log(f"neuron_stats cross-check: {res}")
    return res


class _Vecs:
    """Rows of <bank>/vecs.f32 (float32 [n_bank, d]) for the rows we score, fetched ONCE up-front (random per-batch reads from the volume
    memmap were the bottleneck of the first sample run: ~1.3 s per 96-row batch). A contiguous vec_idx range (a shard) is one sequential
    slice read; a scattered small set is one ascending gather; a scattered huge set reads the whole file into RAM."""

    def __init__(self, bank, n_bank, d, needed, log, ram_min=200_000):
        import time
        import numpy as np
        p = f"{bank}/vecs.f32"
        assert os.path.getsize(p) == n_bank * d * 4, (os.path.getsize(p), n_bank, d)
        u = np.unique(np.asarray(needed, np.int64))                       # ascending
        t0 = time.time()
        mm = np.memmap(p, np.float32, "r", shape=(n_bank, d))
        if len(u) == int(u[-1] - u[0] + 1):
            self.v = np.ascontiguousarray(mm[u[0]:u[-1] + 1], dtype=np.float32)
            how = f"contiguous slice [{u[0]}, {u[-1] + 1})"
        elif len(u) >= ram_min:
            self.v = np.fromfile(p, np.float32).reshape(n_bank, d); u = np.arange(n_bank)
            how = "whole file"
        else:
            self.v = np.ascontiguousarray(mm[u], dtype=np.float32)
            how = f"gather of {len(u)} scattered rows"
        del mm
        self.pos = np.full(n_bank, -1, np.int64); self.pos[u] = np.arange(len(u))
        log(f"vecs.f32 -> RAM: {how} ({self.v.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f}s)")

    def get(self, idx):
        import numpy as np
        j = self.pos[np.asarray(idx, np.int64)]
        assert (j >= 0).all()
        return self.v[j]


def _score(model, tok, bos, rows, vecs, batch, dev, log, log_every=50_000, n_total_for_eta=None):
    """Forward every target STANDALONE ([bos] + tok(text, add_special_tokens=False)), same-length rows batched together (no padding),
    capture layers[42].mlp.down_proj input (neuron values) + the layer-42 residual (mlp42_neurons_worker.forward_capture), drop BOS.
    Returns per-row arrays: L (re-tokenized length; 0 = unscored), n_peak/n_last/n_max [n,3] (polarity-signed member values: argmax
    position, value at the last token, max), d_peak/d_last/d_max [n] (cosine with the bank direction: argmax position, last, max),
    plus throughput numbers."""
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    import mlp42_neurons_worker as W
    from mxf.config import READ_LAYER
    from mxf.inject import get_layer
    n = len(rows["text"])
    t_tok = time.time()
    enc = tok(rows["text"], add_special_tokens=False)["input_ids"]
    lens = np.array([len(e) for e in enc], np.int64)
    log(f"re-tokenized {n} targets ({time.time() - t_tok:.0f}s): len min {lens.min()} median {int(np.median(lens))} max {lens.max()}; "
        f"len == record n_tok: {(lens == rows['n_tok_rec']).mean():.4f}; empty {(lens < 1).sum()}")
    order = np.argsort(lens, kind="stable")
    n_peak = np.full((n, MAXM), -1, np.int64); n_last = np.zeros((n, MAXM), np.float32); n_max = np.zeros((n, MAXM), np.float32)
    d_peak = np.full(n, -1, np.int64); d_last = np.zeros(n, np.float32); d_max = np.zeros(n, np.float32)
    layer = get_layer(model, READ_LAYER); mlp = layer.mlp
    d_ff = mlp.down_proj.weight.shape[1]
    assert int(rows["neurons"].max()) < d_ff
    nid_all = torch.from_numpy(np.where(rows["neurons"] >= 0, rows["neurons"], 0)).to(dev)          # [n,3] (pad -> 0, masked below)
    pol_all = torch.from_numpy(rows["pol"]).to(dev)                                                    # [n,3] (pad -> 0 -> a_s = 0)
    t0 = time.time(); done = 0; done_tok = 0; nb = 0; next_log = log_every
    t_fwd = t_vec = t_post = 0.0
    with torch.no_grad():
        i0 = 0
        while i0 < n:
            L = int(lens[order[i0]])
            i1 = i0
            while i1 < n and i1 - i0 < batch and int(lens[order[i1]]) == L:
                i1 += 1
            kb = order[i0:i1]; i0 = i1
            if L < 1:
                continue
            B = len(kb)
            ids = torch.tensor([[bos] + enc[k] for k in kb], device=dev, dtype=torch.long)             # [B, 1+L]
            tf = time.time()
            a_bf, h = W.forward_capture(model, ids, layer, mlp)                                        # [B,1+L,d_ff] bf16, [B,1+L,d]
            torch.cuda.synchronize(); t_fwd += time.time() - tf; tp = time.time()
            a = a_bf[:, 1:, :].float()                                                                 # BOS dropped
            h = h[:, 1:, :].float()
            kb_t = torch.from_numpy(kb).to(dev)
            nid = nid_all[kb_t]                                                                        # [B,3]
            a_s = a.gather(2, nid[:, None, :].expand(B, L, MAXM)) * pol_all[kb_t][:, None, :]         # [B,L,3] polarity-signed
            n_peak[kb] = a_s.argmax(1).cpu().numpy(); n_last[kb] = a_s[:, -1, :].cpu().numpy(); n_max[kb] = a_s.amax(1).cpu().numpy()
            tv = time.time()
            v = torch.from_numpy(vecs.get(rows["vec_idx"][kb])).to(dev)                               # [B,d] unit bank directions
            t_vec += time.time() - tv
            cos = torch.einsum("bld,bd->bl", F.normalize(h, dim=-1), v)                                # == rl.py reward per token
            d_peak[kb] = cos.argmax(1).cpu().numpy(); d_last[kb] = cos[:, -1].cpu().numpy(); d_max[kb] = cos.amax(1).cpu().numpy()
            torch.cuda.synchronize(); t_post += time.time() - tp                                    # everything after the forward (incl. vec gather)
            done += B; done_tok += B * L; nb += 1
            if done >= next_log:
                next_log += log_every
                el = time.time() - t0; rps = done / max(el, 1e-6)
                eta = (n - done) / rps / 60
                log(f"{done}/{n} rows ({rps:.0f} rows/s, {done_tok / max(el, 1e-6):.0f} tok/s, {nb} forwards, ETA {eta:.1f} min)")
            del a_bf, a, h, a_s, cos, v
    el = time.time() - t0
    thr = {"rows": int(done), "tokens": int(done_tok), "forwards": int(nb), "seconds": el, "rows_per_s": done / max(el, 1e-6),
           "tok_per_s": done_tok / max(el, 1e-6), "batch": batch, "gpu": torch.cuda.get_device_name(0),
           "seconds_forward": t_fwd, "seconds_post_forward": t_post, "seconds_vec_gather": t_vec, "seconds_per_forward": el / max(nb, 1)}
    if n_total_for_eta:
        thr["projected_minutes_for_rows"] = {str(n_total_for_eta): n_total_for_eta / max(thr["rows_per_s"], 1e-6) / 60}
    log(f"scored {done} rows / {done_tok} tokens in {el:.0f}s: {thr['rows_per_s']:.0f} rows/s, {thr['tok_per_s']:.0f} tok/s (batch {batch}, {nb} forwards, "
        f"{el / max(nb, 1) * 1e3:.0f} ms/forward: fwd {t_fwd:.0f}s, post {t_post:.0f}s incl. vec gather {t_vec:.0f}s) on {thr['gpu']}")
    return {"L": lens, "n_peak": n_peak, "n_last": n_last, "n_max": n_max, "d_peak": d_peak, "d_last": d_last, "d_max": d_max, "throughput": thr}


SC_KEYS = ("L", "n_peak", "n_last", "n_max", "d_peak", "d_last", "d_max")
ROW_KEYS = ("vec_idx", "fam", "n_tok_rec", "neurons", "pol", "cmax", "act_rec")          # everything _summarize needs (no text)


def _hist(off, cap=10):
    import numpy as np
    return {str(int(d)): int(c) for d, c in zip(*np.unique(off, return_counts=True)) if d <= cap}


def _summarize(rows, sc, fam, rel_thr=REL_THR):
    """Per-family summary of the neuron-level (per member + all members) and direction-level anchor statistics + the per-row masks the
    filter rules are built from (dp1/dp2 direction peak last / last-2, fire_all every member fires at the last token, p1_all/p2_all every
    member's value peaks at the last / last-2, ok = scored)."""
    import numpy as np
    sel = np.flatnonzero(rows["fam"] == fam)
    L = sc["L"][sel]; ok = L >= 1
    k = int((rows["neurons"][sel] >= 0).sum(1).max()) if len(sel) else 0
    mem = rows["neurons"][sel] >= 0                                       # [n,3] real members
    off_n = (L[:, None] - 1 - sc["n_peak"][sel])                          # neuron peak offset from the end, per member
    p1 = (off_n == 0) | ~mem; p2 = (off_n <= 1) | ~mem                    # padded members pass trivially (-> "all" over real members)
    fire = (sc["n_last"][sel] >= rel_thr * rows["cmax"][sel]) | ~mem
    ratio_wp = np.where(rows["act_rec"][sel] > 0, sc["n_last"][sel] / np.maximum(rows["act_rec"][sel], 1e-9), np.nan)
    ratio_mx = sc["n_last"][sel] / np.where(np.abs(sc["n_max"][sel]) > 1e-9, sc["n_max"][sel], np.nan)
    off_d = L - 1 - sc["d_peak"][sel]
    dp1 = off_d == 0; dp2 = off_d <= 1
    fire_all = fire.all(1); p1_all = p1.all(1); p2_all = p2.all(1)
    okf = lambda x: float(x[ok].mean()) if ok.any() else float("nan")
    per_member = []
    for m in range(k):
        mm = ok & mem[:, m]
        per_member.append({"pass_last": float(p1[mm, m].mean()), "pass_last2": float(p2[mm, m].mean()), "fire_last_rate": float(fire[mm, m].mean()),
                           "a_last_over_window_peak_median": float(np.nanmedian(ratio_wp[mm, m])), "a_last_over_a_max_median": float(np.nanmedian(ratio_mx[mm, m])),
                           "peak_offset_from_end_hist": _hist(off_n[mm, m])})
    worst_off = np.where(mem, off_n, 0).max(1)                            # the offset that decides the all-members pass
    res = {"n": int(len(sel)), "n_scored": int(ok.sum()), "n_members": k,
           "retokenized_len_equals_n_tok_rate": float((L == rows["n_tok_rec"][sel])[ok].mean()) if ok.any() else float("nan"),
           # neuron level, ALL members
           "pass_last": okf(p1_all), "pass_last2": okf(p2_all), "fire_last_rate": okf(fire_all),
           "fire_last_rate_any_member": okf((fire & mem).any(1)),
           "a_last_over_window_peak_median_minmember": float(np.nanmedian(np.where(mem, ratio_wp, np.inf).min(1)[ok])) if ok.any() else float("nan"),
           "neuron_peak_offset_from_end_hist": _hist(worst_off[ok]),
           "per_member": per_member,
           # direction level (what the RL reward / evaluator measure)
           "dir_peak_last": okf(dp1), "dir_peak_last2": okf(dp2),
           "cos_last_median": float(np.median(sc["d_last"][sel][ok])) if ok.any() else float("nan"),
           "cos_max_median": float(np.median(sc["d_max"][sel][ok])) if ok.any() else float("nan"),
           "cos_last_over_cos_max_median": float(np.median((sc["d_last"][sel] / np.maximum(sc["d_max"][sel], 1e-9))[ok])) if ok.any() else float("nan"),
           "peak_offset_from_end_hist": _hist(off_d[ok]),
           # joint rates
           "fire_and_dir_peak_last": okf(fire_all & dp1), "fire_and_dir_peak_last2": okf(fire_all & dp2),
           "neuron_pass_and_dir_peak_last": okf(p1_all & dp1), "neuron_pass_not_dir_peak_last": okf(p1_all & ~dp1),
           "dir_peak_last_not_neuron_pass": okf(~p1_all & dp1)}
    return res, {"dp1": dp1, "dp2": dp2, "fire_all": fire_all, "p1_all": p1_all, "p2_all": p2_all, "ok": ok, "sel": sel}


def _write_summary(bank, key, summ, log):
    """build_stats.json / meta.json [key] (atomic tmp + rename, like the SAE app)."""
    import json
    for fn in ("build_stats.json", "meta.json"):
        p = f"{bank}/{fn}"
        d = json.load(open(p))
        d[key] = summ
        json.dump(d, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
    vol.commit()
    log(f"written [{key}] into {bank}/build_stats.json + meta.json")


def _finalize(bank, fams, rule, rows, sc, eval_cos_json, write, log, extra):
    """Apply `rule` to every scored row, UNION each family's failures with the ids already excluded by <bank>/eval_cos_rows.json (the
    compositor takes ONE exclusion file per bank), write <bank>/mlp_anchor_rows.json (compositor schema: {rule, families: {fam: {n,
    pass_last, pass_last2, fire_last_rate, n_fail, n_keep, fail_vec_idx: [...]}}} + extras) and a list-free summary into build_stats.json
    / meta.json ["mlp_anchor_filter"]. pass_last / pass_last2 are the rule's own 'last' / 'last2' variants (RULE_LAST_PAIR)."""
    import json
    import time
    import numpy as np
    assert rule in RULES, rule
    T0 = time.time()
    ev = json.load(open(eval_cos_json)) if eval_cos_json and os.path.exists(eval_cos_json) else None
    log(f"eval-cos exclusions: {eval_cos_json} -> " + ("MISSING (no union)" if ev is None else
        " ".join(f"{f}={ev['families'][f]['n_fail']}" for f in ev["families"])))
    r_last, r_last2 = RULE_LAST_PAIR[rule]
    out = {"rule": rule, "rule_text": RULE_TXT[rule], "bank": bank, "n_rows": int(len(rows["vec_idx"])), "rel_thr": REL_THR,
           "pass_definition": f"families[fam].pass_last = rate of rule '{r_last}', pass_last2 = rate of rule '{r_last2}' ({RULE_TXT[r_last2]}); "
                              f"fire_last_rate = {FIRE_TXT}; dir_peak_last(2) = direction peak at the last (2) token(s); neuron_pass_last(2) = every member's "
                              f"polarity-signed value peaks at the last (2) token(s); fail_vec_idx = rows failing '{rule}' UNION the eval-cos exclusions",
           "eval_cos_json": eval_cos_json if ev is not None else None, "families": {}, **extra}
    for f in fams:
        res, m = _summarize(rows, sc, f)
        vi = rows["vec_idx"][m["sel"]]
        keep = RULES[rule](m) & m["ok"]
        fail_anchor = set(int(x) for x in vi[~keep])
        fail_ev = set(int(x) for x in ev["families"][f]["fail_vec_idx"]) if ev is not None and f in ev["families"] else set()
        not_ours = fail_ev - set(int(x) for x in vi)
        assert not not_ours, f"{f}: {len(not_ours)} eval-cos ids are not rows of this family (e.g. {sorted(not_ours)[:5]})"
        fail = fail_anchor | fail_ev
        out["families"][f] = {"n": int(len(vi)),
                              "pass_last": float((RULES[r_last](m) & m["ok"])[m["ok"]].mean()), "pass_last2": float((RULES[r_last2](m) & m["ok"])[m["ok"]].mean()),
                              "fire_last_rate": res["fire_last_rate"], "n_fail": int(len(fail)), "n_keep": int(len(vi) - len(fail)),
                              "fail_vec_idx": sorted(fail),
                              "n_fail_anchor_rule": int(len(fail_anchor)), "n_fail_eval_cos": int(len(fail_ev)),
                              "n_fail_eval_cos_only": int(len(fail_ev - fail_anchor)), "n_fail_both": int(len(fail_ev & fail_anchor)),
                              "n_keep_before_eval_cos_union": int(len(vi) - len(fail_anchor)),
                              "n_unscored_empty_retok": int((~m["ok"]).sum()),
                              "dir_peak_last": res["dir_peak_last"], "dir_peak_last2": res["dir_peak_last2"],
                              "neuron_pass_last": res["pass_last"], "neuron_pass_last2": res["pass_last2"],
                              "fire_last_rate_any_member": res["fire_last_rate_any_member"],
                              "fire_and_dir_peak_last": res["fire_and_dir_peak_last"], "fire_and_dir_peak_last2": res["fire_and_dir_peak_last2"],
                              "per_member": res["per_member"], "retokenized_len_equals_n_tok_rate": res["retokenized_len_equals_n_tok_rate"],
                              "a_last_over_window_peak_median_minmember": res["a_last_over_window_peak_median_minmember"],
                              "cos_last_median": res["cos_last_median"], "cos_max_median": res["cos_max_median"],
                              "peak_offset_from_end_hist": res["peak_offset_from_end_hist"], "neuron_peak_offset_from_end_hist": res["neuron_peak_offset_from_end_hist"],
                              "neuron_pass_and_dir_peak_last": res["neuron_pass_and_dir_peak_last"], "neuron_pass_not_dir_peak_last": res["neuron_pass_not_dir_peak_last"],
                              "dir_peak_last_not_neuron_pass": res["dir_peak_last_not_neuron_pass"]}
        d = out["families"][f]
        log(f"{f}: n {d['n']} | rule '{rule}' pass {d['pass_last']:.4f} (last2 variant {d['pass_last2']:.4f}) | fire_last(all) {d['fire_last_rate']:.4f} "
            f"dir_peak_last {d['dir_peak_last']:.4f} neuron_pass_last {d['neuron_pass_last']:.4f} | fail: anchor {d['n_fail_anchor_rule']} + eval-cos {d['n_fail_eval_cos']} "
            f"({d['n_fail_eval_cos_only']} new) = {d['n_fail']} -> keep {d['n_keep']} (before union {d['n_keep_before_eval_cos_union']})")
    out["n_keep_total"] = int(sum(d["n_keep"] for d in out["families"].values()))
    out["n_fail_total"] = int(sum(d["n_fail"] for d in out["families"].values()))
    out["finalize_seconds"] = time.time() - T0
    summ = {**out, "families": {f: {k: v for k, v in d.items() if k != "fail_vec_idx"} for f, d in out["families"].items()},
            "rows_file": f"{bank}/mlp_anchor_rows.json"}
    if write:
        p = f"{bank}/mlp_anchor_rows.json"
        if os.path.exists(p):
            log(f"NOTE: {p} exists — replacing it (atomic)")
        json.dump(out, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
        _write_summary(bank, "mlp_anchor_filter", summ, log)
        log(f"written {p} ({os.path.getsize(p) / 1e6:.1f} MB): keep {out['n_keep_total']} / drop {out['n_fail_total']} of {out['n_rows']}")
    else:
        log("write=False: nothing written")
    return summ


def _setup(dev, log):
    import time
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import mlp42_neurons_worker as W
    from mxf.config import MODEL
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    bos = W.BOS                                                           # 248044 == the bank scan's sink token
    assert tok.bos_token_id is None or tok.bos_token_id == bos, (tok.bos_token_id, bos)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": dev}).eval()
    log(f"model up ({time.time() - t0:.0f}s); bos={bos}; tok.bos_token_id={tok.bos_token_id}")
    return tok, bos, model


def _extra(sc_thr, stats_chk, batch):
    from mxf.config import MODEL, READ_LAYER
    import mlp42_neurons_worker as W
    return {"model": MODEL, "layer": READ_LAYER, "bos": W.BOS, "batch": batch, "neuron_stats_check": stats_chk, "throughput": sc_thr,
            "standalone": "tok(target_text, add_special_tokens=False), [248044] prepended, BOS position dropped, no other context"}


# ----------------------------------------------------------------------------------------------------------------
# 1. sample check
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPUS, cpu=8, memory=98304, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")],
              timeout=3 * 3600)
def verify_mlp_anchor(bank: str = BANK_DEFAULT, families: str = FAMS_DEFAULT, n_per_family: int = 2048, seed: int = 0, batch: int = 96,
                      write: bool = False):
    """Sample n_per_family rows per family, re-tokenize each target_text STANDALONE, forward [bos]+tokens, and report per family the
    neuron-level (per member / all members) and direction-level anchor statistics (see module docstring). write=True stores the summary
    under <bank>/build_stats.json + meta.json ["mlp_anchor_check"]. Same code path as the full passes."""
    _env()
    import json
    import time
    import numpy as np
    from mxf.config import D_MODEL, MODEL, READ_LAYER
    dev = "cuda:0"
    T0 = time.time()

    def log(m):
        print(f"[mlp-anchor +{time.time() - T0:6.0f}s] {m}", flush=True)

    vol.reload()
    fams = [f for f in families.split(",") if f]
    st = json.load(open(f"{bank}/build_stats.json"))
    n_bank = int(st["n_examples"])
    rows_all = _load_rows(bank, fams, log)
    rng = np.random.default_rng(seed)
    pick = []
    for f in fams:
        idx = np.flatnonzero(rows_all["fam"] == f)
        pick.append(idx[np.sort(rng.choice(len(idx), min(n_per_family, len(idx)), replace=False))])
    rows = _subset(rows_all, np.concatenate(pick))
    n_total = len(rows_all["text"]); del rows_all
    log(f"sample: {len(rows['text'])} rows (" + " ".join(f"{f}={int((rows['fam'] == f).sum())}" for f in fams) + f") of {n_total}; seed {seed}")
    stats_chk = _check_neuron_stats(rows, log)
    vecs = _Vecs(bank, n_bank, D_MODEL, rows["vec_idx"], log)
    tok, bos, model = _setup(dev, log)
    sc = _score(model, tok, bos, rows, vecs, batch, dev, log, log_every=2048, n_total_for_eta=n_total)
    out = {"bank": bank, "n_per_family": n_per_family, "seed": seed, "batch": batch, "model": MODEL, "layer": READ_LAYER, "bos": bos,
           "rel_thr": REL_THR, "neuron_stats_check": stats_chk, "throughput": sc["throughput"], "n_bank_rows_in_families": n_total,
           "projected_full_pass_minutes": n_total / max(sc["throughput"]["rows_per_s"], 1e-6) / 60,
           "definitions": {"neuron_value": "a_i = input i of layers[42].mlp.down_proj (act_fn(gate)*up), forward_pre_hook, bf16 -> fp32; a_s = polarity * a",
                           "fire": f"a_s[last] >= {REL_THR} * corpus_max (records' corpus_max == /data/mlp42/neuron_stats.npz max_abs)",
                           "pass_last": "argmax_t a_s[t] == last target token (pairs/triples: ALL members)", "pass_last2": "argmax within the last 2 (ALL members)",
                           "dir_peak_last": "argmax_t <normalize(h_L42[t]), vecs[vec_idx]> == last token (== RL reward / evaluator per-token cosine)",
                           "rules": RULE_TXT,
                           "standalone": "tok(target_text, add_special_tokens=False), [248044] prepended, BOS position dropped, no other context"},
           "families": {}}
    for f in fams:
        res, m = _summarize(rows, sc, f)
        res["rule_pass_rates"] = {r: float((fn(m) & m["ok"])[m["ok"]].mean()) for r, fn in RULES.items()}
        out["families"][f] = res
        log(f"{f}: n {res['n']} | NEURON all-members pass_last {res['pass_last']:.4f} pass_last2 {res['pass_last2']:.4f} fire_last {res['fire_last_rate']:.4f} "
            f"| DIRECTION dir_peak_last {res['dir_peak_last']:.4f} dir_peak_last2 {res['dir_peak_last2']:.4f} | fire&dirpeak_last {res['fire_and_dir_peak_last']:.4f} "
            f"| retok==n_tok {res['retokenized_len_equals_n_tok_rate']:.4f}")
        log(f"{f}: {json.dumps(res)}")
    out["seconds"] = time.time() - T0
    if write:
        _write_summary(bank, "mlp_anchor_check", out, log)
    log(f"DONE in {out['seconds'] / 60:.1f} min; projected full pass over {n_total} rows at batch {batch}: {out['projected_full_pass_minutes']:.1f} min")
    return out


# ----------------------------------------------------------------------------------------------------------------
# 2a. full pass, single GPU (same code path; ~80 min at 200 rows/s)
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPUS, cpu=8, memory=98304, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")],
              timeout=4 * 3600)
def filter_mlp_anchor(bank: str = BANK_DEFAULT, families: str = FAMS_DEFAULT, rule: str = "fire_and_dirpeak_last", batch: int = 96,
                      write: bool = True, eval_cos_json: str | None = None):
    """FULL pass over every row of the given families on ONE GPU, then _finalize (rule + eval-cos union + mlp_anchor_rows.json)."""
    _env()
    import json
    import time
    from mxf.config import D_MODEL
    assert rule in RULES, rule
    dev = "cuda:0"
    T0 = time.time()

    def log(m):
        print(f"[mlp-anchor-full +{time.time() - T0:6.0f}s] {m}", flush=True)

    vol.reload()
    fams = [f for f in families.split(",") if f]
    n_bank = int(json.load(open(f"{bank}/build_stats.json"))["n_examples"])
    rows = _load_rows(bank, fams, log)
    stats_chk = _check_neuron_stats(rows, log)
    vecs = _Vecs(bank, n_bank, D_MODEL, rows["vec_idx"], log)
    tok, bos, model = _setup(dev, log)
    sc = _score(model, tok, bos, rows, vecs, batch, dev, log, log_every=50_000)
    extra = _extra(sc["throughput"], stats_chk, batch)
    summ = _finalize(bank, fams, rule, rows, sc, eval_cos_json or f"{bank}/eval_cos_rows.json", write, log, extra)
    summ["seconds"] = time.time() - T0
    log(f"DONE in {summ['seconds'] / 60:.1f} min")
    return summ


# ----------------------------------------------------------------------------------------------------------------
# 2b. full pass, sharded: N single-GPU shards over contiguous row ranges -> npz, then a CPU merge
# ----------------------------------------------------------------------------------------------------------------
def _shard_path(bank, shard, n_shards):
    return f"{bank}/mlp_anchor_shards/shard_{shard:02d}_of_{n_shards:02d}.npz"


@app.function(image=image, gpu=GPUS, cpu=8, memory=65536, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")],
              timeout=2 * 3600)
def filter_mlp_anchor_shard(shard: int, n_shards: int = 8, bank: str = BANK_DEFAULT, families: str = FAMS_DEFAULT, batch: int = 96):
    """Score rows [shard*n/n_shards, (shard+1)*n/n_shards) of the family row list (vec_idx order, so a contiguous vecs.f32 slice) and save
    the per-row arrays to <bank>/mlp_anchor_shards/shard_XX_of_NN.npz (atomic). Nothing else is written."""
    _env()
    import json
    import time
    import numpy as np
    from mxf.config import D_MODEL
    dev = "cuda:0"
    T0 = time.time()

    def log(m):
        print(f"[mlp-anchor-shard {shard}/{n_shards} +{time.time() - T0:6.0f}s] {m}", flush=True)

    assert 0 <= shard < n_shards
    vol.reload()
    fams = [f for f in families.split(",") if f]
    n_bank = int(json.load(open(f"{bank}/build_stats.json"))["n_examples"])
    rows_all = _load_rows(bank, fams, log)
    n = len(rows_all["text"])
    lo, hi = shard * n // n_shards, (shard + 1) * n // n_shards
    rows = _subset(rows_all, np.arange(lo, hi)); del rows_all
    log(f"shard rows [{lo}, {hi}) = {hi - lo} rows (vec_idx {rows['vec_idx'][0]}..{rows['vec_idx'][-1]}); " + " ".join(f"{f}={int((rows['fam'] == f).sum())}" for f in fams))
    stats_chk = _check_neuron_stats(rows, log)
    vecs = _Vecs(bank, n_bank, D_MODEL, rows["vec_idx"], log)
    tok, bos, model = _setup(dev, log)
    sc = _score(model, tok, bos, rows, vecs, batch, dev, log, log_every=25_000)
    p = _shard_path(bank, shard, n_shards)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    meta = {"shard": shard, "n_shards": n_shards, "lo": lo, "hi": hi, "bank": bank, "families": fams, "batch": batch, "neuron_stats_check": stats_chk,
            "throughput": sc["throughput"], "seconds": time.time() - T0}
    np.savez(p + ".tmp.npz", **{k: rows[k] for k in ROW_KEYS}, **{k: sc[k] for k in SC_KEYS}, meta=json.dumps(meta))
    os.replace(p + ".tmp.npz", p)
    vol.commit()
    log(f"saved {p} ({os.path.getsize(p) / 1e6:.1f} MB); DONE in {(time.time() - T0) / 60:.1f} min")
    return meta


@app.function(image=image, cpu=8, memory=32768, volumes={"/data": vol}, timeout=3600)
def filter_mlp_anchor_merge(bank: str = BANK_DEFAULT, families: str = FAMS_DEFAULT, rule: str = "fire_and_dirpeak_last", n_shards: int = 8,
                            write: bool = True, eval_cos_json: str | None = None):
    """CPU merge of the N shard files -> _finalize (rule + eval-cos union + <bank>/mlp_anchor_rows.json + build_stats/meta summaries).
    Asserts the shards cover every row of the families exactly once. Shard files are left in place."""
    _env()
    import json
    import time
    import numpy as np
    assert rule in RULES, rule
    T0 = time.time()

    def log(m):
        print(f"[mlp-anchor-merge +{time.time() - T0:6.0f}s] {m}", flush=True)

    vol.reload()
    fams = [f for f in families.split(",") if f]
    parts, metas = [], []
    for s in range(n_shards):
        p = _shard_path(bank, s, n_shards)
        assert os.path.exists(p), f"missing shard file {p}"
        z = np.load(p, allow_pickle=False)
        parts.append({k: z[k] for k in ROW_KEYS + SC_KEYS}); metas.append(json.loads(str(z["meta"])))
        log(f"shard {s}: rows [{metas[-1]['lo']}, {metas[-1]['hi']}) {metas[-1]['throughput']['rows_per_s']:.0f} rows/s on {metas[-1]['throughput'].get('gpu')} "
            f"({metas[-1]['seconds'] / 60:.1f} min)")
    rows = {k: np.concatenate([q[k] for q in parts]) for k in ROW_KEYS}
    sc = {k: np.concatenate([q[k] for q in parts]) for k in SC_KEYS}
    order = np.argsort(rows["vec_idx"], kind="stable")
    rows = {k: v[order] for k, v in rows.items()}; sc = {k: v[order] for k, v in sc.items()}
    # coverage: every family row exactly once
    counts = json.load(open(f"{bank}/build_stats.json"))["families"]
    n_expect = sum(int(counts[f]) for f in fams)
    assert len(rows["vec_idx"]) == n_expect and len(np.unique(rows["vec_idx"])) == n_expect, (len(rows["vec_idx"]), n_expect)
    for f in fams:
        assert int((rows["fam"] == f).sum()) == int(counts[f]), (f, int((rows["fam"] == f).sum()), counts[f])
    log(f"merged {n_expect} rows from {n_shards} shards; coverage OK")
    thr = {"shards": [m["throughput"] for m in metas], "rows_per_s_sum": float(sum(m["throughput"]["rows_per_s"] for m in metas)),
           "max_shard_minutes": max(m["seconds"] for m in metas) / 60}
    extra = _extra(thr, metas[0]["neuron_stats_check"], metas[0]["batch"])
    extra["shards"] = [{k: m[k] for k in ("shard", "n_shards", "lo", "hi", "seconds")} for m in metas]
    summ = _finalize(bank, fams, rule, rows, sc, eval_cos_json or f"{bank}/eval_cos_rows.json", write, log, extra)
    summ["seconds"] = time.time() - T0
    log(f"DONE in {summ['seconds'] / 60:.1f} min")
    return summ
