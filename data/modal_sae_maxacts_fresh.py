"""Modal app `maemm-sae-maxacts-fresh`: per-feature max-activating 32-token windows of OUR layer-42 SAE (/data/sae/ae.pt,
F=131072 k=64) over the FRESH activation store's documents -> /data/sae/maxacts_fresh.pt (the original /data/sae/maxacts.pt is
never touched).

Same construction as scripts/sae27b_maxacts.py (the file that made /data/sae/maxacts.pt): every 32-token window is forwarded
STANDALONE ([BOS] + 32 tokens, BOS position dropped) to layer 42, encoded with the SAE's threshold gate
(relu((x - b_dec) @ W_enc + b_enc), zeroed below the learned threshold), and each feature keeps its top-N windows by peak
activation. The corpus is /data/acts27b_fresh/toks.i32 cut into 16 aligned 32-token windows per 512-token row (rows are
FineFineWeb documents disjoint from every existing store / training bank, see data/modal_acts27b_fresh.py), so every window is
fresh text. Output: {"max_tokens": [F, N, 32] int32, "max_acts": [F, N, 32] float32, "max_src": [F, N, 2] int32 (store row,
window slot), "meta": {...}}.

    MODAL_PROFILE=safety-sahan modal deploy data/modal_sae_maxacts_fresh.py
    python -c "import modal; print(modal.Function.from_name('maemm-sae-maxacts-fresh','maxacts').spawn().object_id)"
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = os.environ.get("MAEMM_SAE_MAXACTS_APP", "maemm-sae-maxacts-fresh")
app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet")
    .pip_install("flash-linear-attention==0.5.2")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
GPUS = ["B200", "H200"]
SAE_PT = "/data/sae/ae.pt"
SAE_HF_SIZE = 5369256453                 # identity check (== data/modal_bank_everything.py)
ACTS_FRESH = "/data/acts27b_fresh"
OUT_DEFAULT = "/data/sae/maxacts_fresh.pt"


@app.function(image=image, gpu=GPUS, cpu=8, memory=98304, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")],
              timeout=8 * 3600)
def maxacts(acts_dir: str = ACTS_FRESH, out: str = OUT_DEFAULT, n_rows: int | None = None, ctx_len: int = 32, topn: int = 16,
            batch_wins: int = 256, wait_s: int = 4 * 3600):
    import json
    import sys
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL, READ_LAYER
    from mxf.inject import read_resid
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda:0"
    T0 = time.time()

    def log(m):
        print(f"[maxacts +{time.time() - T0:6.0f}s] {m}", flush=True)

    assert not os.path.exists(out), f"{out} exists — refusing to overwrite a max-acts file"
    assert out != "/data/sae/maxacts.pt"
    # wait for the fresh store's tokens (finalize publishes toks/meta before acts)
    t_wait = time.time()
    while True:
        vol.reload()
        if os.path.exists(f"{acts_dir}/meta.json") and os.path.exists(f"{acts_dir}/toks.i32"):
            break
        assert time.time() - t_wait < wait_s, f"{acts_dir}/toks.i32 not published within {wait_s}s"
        log(f"waiting for {acts_dir}/toks.i32 ...")
        time.sleep(120)
    meta = json.load(open(f"{acts_dir}/meta.json"))
    NS, T = int(meta["n_seq"]), int(meta["seq_len"])
    assert T % ctx_len == 0
    n_rows = NS if n_rows is None else min(int(n_rows), NS)
    toks = np.fromfile(f"{acts_dir}/toks.i32", dtype=np.int32).reshape(NS, T)[:n_rows]
    n_slot = T // ctx_len
    n_win = n_rows * n_slot
    log(f"store {acts_dir}: {NS} rows x {T}; using {n_rows} rows -> {n_win} standalone {ctx_len}-token windows ({n_win * ctx_len:,} tokens)")

    assert os.path.getsize(SAE_PT) == SAE_HF_SIZE, "different SAE on the volume; refusing to guess"
    params = torch.load(SAE_PT, map_location="cpu", weights_only=False)
    W_enc = params["encoder.weight"].to(dev, torch.float32)              # nn.Linear [F, d]
    b_enc = params["encoder.bias"].to(dev, torch.float32)
    b_dec = (params.get("b_dec", params.get("bias"))).to(dev, torch.float32)
    thr = float(params["threshold"].item()) if "threshold" in params else 0.0
    Fd = W_enc.shape[0]
    assert W_enc.shape[1] == D_MODEL
    del params
    log(f"SAE: F={Fd} threshold={thr:.4f} (keys had threshold: {thr > 0})")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else int(meta.get("bos_id", 248044))
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
                                                 device_map={"": dev}).eval()
    log(f"model up; bos={bos}")

    L, N = ctx_len, topn
    topk_peak = torch.full((Fd, N), -1.0, device=dev)
    topk_tok = torch.zeros((Fd, N, L), dtype=torch.int32, device=dev)
    topk_act = torch.zeros((Fd, N, L), dtype=torch.float32, device=dev)
    topk_src = torch.full((Fd, N, 2), -1, dtype=torch.int32, device=dev)
    n_fire_tok = torch.zeros(Fd, dtype=torch.int64, device=dev)

    @torch.no_grad()
    def process(win, src):
        """win [W, L] int32 token ids (cuda), src [W, 2] int32 (row, slot)."""
        nonlocal topk_peak, topk_tok, topk_act, topk_src
        Wn = win.shape[0]
        inp = torch.cat([torch.full((Wn, 1), bos, device=dev, dtype=torch.long), win.long()], dim=1)
        h, _ = read_resid(model, READ_LAYER, {"input_ids": inp, "attention_mask": torch.ones_like(inp)}, pool="all")
        a = h[:, 1:, :].reshape(-1, D_MODEL)                             # fp32 [W*L, d], BOS dropped
        pre = torch.addmm(b_enc, a - b_dec, W_enc.T)                     # [W*L, F]
        feats = torch.relu(pre)
        if thr > 0:
            feats = feats * (feats > thr)
        n_fire_tok.add_((feats > 0).sum(0))
        feats = feats.view(Wn, L, Fd)
        peak = feats.amax(dim=1)                                         # [W, F]
        cand_peak = torch.cat([topk_peak, peak.T], dim=1)                # [F, N+W]
        vals, idx = cand_peak.topk(N, dim=1)
        gidx = idx.unsqueeze(-1).expand(Fd, N, L)
        cand_tok = torch.cat([topk_tok, win.unsqueeze(0).expand(Fd, Wn, L)], dim=1)
        topk_tok = torch.gather(cand_tok, 1, gidx).contiguous(); del cand_tok
        cand_act = torch.cat([topk_act, feats.permute(2, 0, 1)], dim=1)
        topk_act = torch.gather(cand_act, 1, gidx).contiguous(); del cand_act
        cand_src = torch.cat([topk_src, src.unsqueeze(0).expand(Fd, Wn, 2)], dim=1)
        topk_src = torch.gather(cand_src, 1, idx.unsqueeze(-1).expand(Fd, N, 2)).contiguous(); del cand_src
        topk_peak = vals

    t0 = time.time()
    done = 0
    for r0 in range(0, n_rows, max(1, batch_wins // n_slot)):
        r1 = min(r0 + max(1, batch_wins // n_slot), n_rows)
        rows = np.arange(r0, r1)
        win = toks[r0:r1].reshape(-1, L)                                  # [(r1-r0)*n_slot, L], aligned slots
        src = np.stack([np.repeat(rows, n_slot), np.tile(np.arange(n_slot), r1 - r0)], 1).astype(np.int32)
        process(torch.from_numpy(win).to(dev), torch.from_numpy(src).to(dev))
        done += win.shape[0]
        if (r0 // max(1, batch_wins // n_slot)) % 100 == 0:
            el = time.time() - t0
            log(f"{done}/{n_win} windows ({done * L / max(el, 1):.0f} tok/s, ETA {(n_win - done) * L / max(done * L / max(el, 1), 1) / 60:.0f} min)")
    nwin_f = (topk_peak > 0).sum(1)
    live = int((topk_peak.max(1).values > 0).sum())
    hist = torch.bincount(nwin_f, minlength=N + 1).tolist()
    meta_out = {"source": acts_dir, "rows_used": n_rows, "seq_len": T, "ctx_len": L, "topn": N, "n_windows": n_win, "n_tokens": n_win * L,
                "sae": SAE_PT, "threshold": thr, "encode": "relu((x - b_dec) @ W_enc + b_enc) * (. > threshold), windows forwarded standalone [BOS]+32",
                "live_features": live, "windows_per_feature_hist": hist, "n_fire_tokens_per_feature_sum": int(n_fire_tok.sum()),
                "model": MODEL, "layer": READ_LAYER, "seconds": time.time() - T0, "created": time.time()}
    torch.save({"max_tokens": topk_tok.cpu(), "max_acts": topk_act.cpu(), "max_src": topk_src.cpu(), "meta": meta_out}, out + ".tmp")
    os.replace(out + ".tmp", out)
    vol.commit()
    log(f"DONE -> {out}: live {live}/{Fd} features; windows/feature hist(0..{N}) {hist}; {(time.time() - T0) / 60:.1f} min")
    return meta_out


@app.function(image=image, gpu=GPUS, cpu=8, memory=65536, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")],
              timeout=2 * 3600)
def verify_end_anchor(bank: str = "/data/banks/everything_5m_fresh", families: str = "sae,sae_dec", n_per_family: int = 2048, seed: int = 0,
                      min_pass: float = 0.9, write: bool = True, batch: int = 64):
    """END-ANCHOR CHECK of a bank's SAE rows (user rule: the target must end AT the token where the feature fires). Samples n_per_family
    rows per family, re-tokenizes each target_text STANDALONE (the RL-reward / evaluator path: raw text -> [BOS]+tokens -> layer 42 ->
    SAE encoder relu((x - b_dec) @ W_enc[:, f] + b_enc[f])), and records where the feature's activation peaks: pass_last (peak == last
    token), pass_last2 (peak within the last 2 tokens), fire rate at the last token (> threshold), act_last / stored window_peak.
    Writes the per-family summary into <bank>/build_stats.json["end_anchor_check"] and meta.json family_recipes[fam]["end_anchor_check"]
    (write=True), then asserts pass_last2 >= min_pass for every family."""
    import json
    import sys
    import time
    import numpy as np
    import torch
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL, READ_LAYER
    from mxf.inject import read_resid
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dev = "cuda:0"
    T0 = time.time()
    vol.reload()
    fams = [f for f in families.split(",") if f]
    rows = {f: [] for f in fams}
    with open(f"{bank}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            if r["family"] in rows:
                assert int(r["vec_idx"]) == i
                rows[r["family"]].append(r)
    rng = np.random.default_rng(seed)
    sample = {f: [rows[f][j] for j in np.sort(rng.choice(len(rows[f]), min(n_per_family, len(rows[f])), replace=False))] for f in fams}
    print(f"[anchor] {bank}: " + " ".join(f"{f}={len(rows[f])} rows (sample {len(sample[f])})" for f in fams), flush=True)
    params = torch.load(SAE_PT, map_location="cpu", weights_only=False)
    W_enc = params["encoder.weight"].to(dev, torch.float32); b_enc = params["encoder.bias"].to(dev, torch.float32)
    b_dec = (params.get("b_dec", params.get("bias"))).to(dev, torch.float32)
    thr = float(params["threshold"].item()) if "threshold" in params else 0.0
    del params
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else 248044
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
                                                 device_map={"": dev}).eval()
    out = {}
    for f in fams:
        recs = sample[f]
        enc = [tok(r["target_text"], add_special_tokens=False)["input_ids"] for r in recs]
        keep = [k for k, e in enumerate(enc) if len(e) >= 2]
        by_len = {}
        for k in keep:
            by_len.setdefault(len(enc[k]), []).append(k)
        argpos = np.full(len(recs), -1); act_last = np.zeros(len(recs)); act_max = np.zeros(len(recs)); n_retok = np.zeros(len(recs), np.int64)
        with torch.no_grad():
            for L, ks in by_len.items():
                for c0 in range(0, len(ks), batch):
                    kb = ks[c0:c0 + batch]
                    ids = torch.tensor([[bos] + enc[k] for k in kb], device=dev)
                    h, _ = read_resid(model, READ_LAYER, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, pool="all")
                    a = h[:, 1:, :]                                                      # [B, L, d], BOS dropped
                    fidx = torch.tensor([int(recs[k]["feature"]) for k in kb], device=dev)
                    pre = torch.relu(((a - b_dec) * W_enc[fidx][:, None, :]).sum(-1) + b_enc[fidx][:, None])   # [B, L]
                    am = pre.argmax(1).cpu().numpy(); al = pre[:, -1].cpu().numpy(); mx = pre.max(1).values.cpu().numpy()
                    for j, k in enumerate(kb):
                        argpos[k] = am[j]; act_last[k] = al[j]; act_max[k] = mx[j]; n_retok[k] = L
        ok = argpos >= 0
        Ls = n_retok
        pass_last = (argpos == Ls - 1) & ok; pass_last2 = (argpos >= Ls - 2) & ok
        wp = np.array([float(r.get("window_peak", np.nan)) for r in recs])
        ratio = np.where(wp > 0, act_last / np.maximum(wp, 1e-9), np.nan)
        n_tok_rec = np.array([int(r["n_tok"]) for r in recs])
        res = {"n_sampled": int(len(recs)), "n_scored": int(ok.sum()), "pass_last": float(pass_last.sum() / max(ok.sum(), 1)),
               "pass_last2": float(pass_last2.sum() / max(ok.sum(), 1)), "fire_last_rate(>thr)": float(((act_last > thr) & ok).sum() / max(ok.sum(), 1)),
               "act_last_over_window_peak_median": float(np.nanmedian(ratio[ok])), "act_last_over_act_max_median": float(np.median((act_last / np.maximum(act_max, 1e-9))[ok])),
               "retokenized_len_equals_n_tok_rate": float((n_retok == n_tok_rec)[ok].mean()), "peak_offset_from_end_hist": {str(int(d)): int(c) for d, c in
               zip(*np.unique((Ls - 1 - argpos)[ok], return_counts=True)) if d <= 8}, "threshold": thr, "seed": seed,
               "rule": "peak of relu((x-b_dec)@W_enc[:,f]+b_enc[f]) over the standalone re-tokenized target must be the last token (pass_last) or within the last 2 (pass_last2)"}
        out[f] = res
        print(f"[anchor] {f}: {json.dumps(res)}", flush=True)
    if write:
        for fn in ("build_stats.json", "meta.json"):
            d = json.load(open(f"{bank}/{fn}"))
            d["end_anchor_check"] = out
            if fn == "meta.json":
                for f in fams:
                    d.setdefault("family_recipes", {}).setdefault(f, {})["end_anchor_check"] = out[f]
            json.dump(d, open(f"{bank}/{fn}.tmp", "w"), indent=1); os.replace(f"{bank}/{fn}.tmp", f"{bank}/{fn}")
        vol.commit()
        print(f"[anchor] written into {bank}/build_stats.json + meta.json ({time.time() - T0:.0f}s)", flush=True)
    bad = {f: r["pass_last2"] for f, r in out.items() if r["pass_last2"] < min_pass}
    assert not bad, f"end-anchor check below {min_pass}: {bad}"
    return out
