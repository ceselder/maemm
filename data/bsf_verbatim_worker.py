"""BSF-VERBATIM worker (one GPU per process) for data/modal_bsf_verbatim.py.

Each candidate is a SHORT text (a realact_short_20m target window). The worker forwards EVERY text ALONE:
    ids = [BOS] + tok(text, add_special_tokens=False)
reads the layer-42 residual at all positions (mxf.inject.read_resid, pool="all", BOS dropped), takes x = the residual at
the LAST token (fp32) and, with the block-sparse featurizer (SASA / BSF) files of --bsf-dir, computes EXACTLY as
data/modal_bank_everything.py does:
    y = normalize((x - mu_bsf) @ zca);  gn_g = ||(y @ E).view(G, b)[g]||;  top --ranks blocks (ids + gn)
    per candidate block b:  y_b = Q[b]^T Q[b] ((x - mu_bsf) @ zca);  x_b = y_b @ zca^-1;  dir_b = unit(x_b)
and the END-ANCHOR MEASUREMENT for every candidate block: cos(h_t, dir_b) and h_t @ dir_b (RAW residual, the RL reward's
"cosine" / "proj" metrics, sink dropped) over every content token t -> peak_from_end = (n_tok - 1) - argmax_t.

Batches are formed from texts of IDENTICAL token length (no padding, no attention-mask games), largest lengths first.

Protocol (shared local work dir, driver <-> worker, rounds k = 0, 1, ...):
    driver writes   {work}/cands_r{r}_k{k}.jsonl   lines {"i": global candidate id, "text": ...}   then touches .ready
    worker writes   {work}/r{r}_k{k}.xlast.f32     float32 [n, 5120] raw last-token residual, row j == line j
                    {work}/r{r}_k{k}.meta.npz      gidx, ntok, ok, top_i [n,R], top_v, cos_x, wfrac, peak_cos, peak_proj,
                                                   cos_last, cos_max, proj_last, proj_max, norm_c (||x - mu_acts||), norm_raw
                    {work}/r{r}_k{k}.done.json     counters (atomic: the driver waits for this file)
    driver touches  {work}/stop                    -> worker exits 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/pmx")
sys.path.insert(0, "/pmx/helpers")
from mxf.config import D_MODEL, MODEL, READ_LAYER  # noqa: E402
from mxf.inject import read_resid  # noqa: E402


def log(rank, *a):
    print(f"[r{rank}]", *a, flush=True)


class BSF:
    """The BSF files of `bsf_dir` on one device + the direction math of data/modal_bank_everything.py (verbatim)."""

    def __init__(self, bsf_dir, dev, with_Q=True):
        meta = json.load(open(f"{bsf_dir}/meta.json"))
        sasa = torch.load(f"{bsf_dir}/sasa.pt", map_location="cpu", weights_only=False)
        self.G, self.b = int(sasa["G"]), int(sasa["b"])
        self.k = int(sasa.get("k", meta.get("k", 32)))
        assert int(sasa["d"]) == D_MODEL and self.G * self.b == sasa["E"].shape[1]
        self.E = sasa["E"].float().to(dev)                                            # [d, G*b]
        del sasa
        self.Q = None
        if with_Q:
            self.Q = torch.load(f"{bsf_dir}/blocks_Q.pt", map_location="cpu", weights_only=False)["Q"].float().to(dev)   # [G, b, d]
            assert tuple(self.Q.shape) == (self.G, self.b, D_MODEL)
        self.mu = torch.from_numpy(np.load(f"{bsf_dir}/whiten_mu.npy").astype(np.float32)).to(dev)
        zca_np = np.load(f"{bsf_dir}/whiten_zca.npy").astype(np.float64)
        self.zca_asym = float(np.abs(zca_np - zca_np.T).max())
        zca_inv_np = np.linalg.inv(zca_np)
        self.zca_inv_err = float(np.abs(zca_np @ zca_inv_np - np.eye(D_MODEL)).max())
        self.zca = torch.from_numpy(zca_np.astype(np.float32)).to(dev)
        self.zca_inv = torch.from_numpy(zca_inv_np.astype(np.float32)).to(dev)
        self.dev = dev
        self.meta = meta

    @torch.no_grad()
    def block_topk(self, x, n_ranks):
        """x [n, d] RAW acts (fp32, on dev) -> (top block ids [n, n_ranks] int64, gn values [n, n_ranks])."""
        y = F.normalize((x - self.mu) @ self.zca, dim=-1)
        outs_i, outs_v = [], []
        for c0 in range(0, y.shape[0], 4096):
            z = (y[c0:c0 + 4096] @ self.E).view(-1, self.G, self.b)
            gn = z.norm(dim=-1)
            v, i = gn.topk(n_ranks, dim=-1)
            outs_i.append(i); outs_v.append(v)
        return torch.cat(outs_i), torch.cat(outs_v)

    @torch.no_grad()
    def mint(self, x, blocks):
        """x [n, d] RAW acts fp32; blocks [n, R] int64 -> (dirs [n, R, d] unit, cos_x [n, R] = cos(x_b, x - mu_bsf),
        whiten_frac [n, R] = ||y_b|| / ||y||)."""
        xc = x - self.mu                                       # centered raw act
        y = xc @ self.zca                                      # whitened (unnormalized)
        Qb = self.Q[blocks]                                    # [n, R, b, d] orthonormal rows (whitened space)
        coords = torch.einsum("nd,nrbd->nrb", y, Qb)
        yb = torch.einsum("nrb,nrbd->nrd", coords, Qb)         # component of y in the block subspace
        xb = yb @ self.zca_inv                                 # back to residual space (additive component of x - mu)
        dirs = F.normalize(xb, dim=-1)
        cos_x = F.cosine_similarity(xb, xc[:, None, :].expand_as(xb), dim=-1)
        wf = yb.norm(dim=-1) / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return dirs, cos_x, wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--work", required=True, help="shared local work dir (see module docstring)")
    ap.add_argument("--bsf-dir", required=True)
    ap.add_argument("--mu-acts", required=True, help="acts27b whiten_mu.npy (norm-filter centering, suite convention)")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--batch-tokens", type=int, default=32768)
    ap.add_argument("--min-tok", type=int, default=4)
    ap.add_argument("--max-tok", type=int, default=96)
    a = ap.parse_args()
    r, R = a.rank, a.ranks
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False           # exact fp32 BSF math (== the everything builder's defaults for mint)
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else 248044      # suite convention
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 local_files_only=True, device_map={"": dev}).eval()
    assert model.config.hidden_size == D_MODEL
    log(r, f"model up in {time.time() - t0:.0f}s (bos={bos})")
    t1 = time.time()
    bsf = BSF(a.bsf_dir, dev, with_Q=True)
    mu_acts = torch.from_numpy(np.load(a.mu_acts).astype(np.float32)).to(dev)
    assert mu_acts.shape == (D_MODEL,)
    log(r, f"BSF up in {time.time() - t1:.0f}s: G={bsf.G} b={bsf.b} k={bsf.k} | zca asym {bsf.zca_asym:.1e} inv err {bsf.zca_inv_err:.1e} | "
           f"ranks={R} max_batch={a.max_batch} batch_tokens={a.batch_tokens} | GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GB")

    @torch.no_grad()
    def forward(ids_list):
        """ids_list: list of equal-length token-id lists -> dict of per-row numpy outputs."""
        ids = torch.tensor([[bos] + w for w in ids_list], device=dev)
        h, _ = read_resid(model, READ_LAYER, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, pool="all")
        content = h[:, 1:, :]                                            # fp32 [B, L, d]; row t = content token t (BOS dropped)
        L = content.shape[1]
        x = content[:, -1, :].contiguous()                               # the LAST token's residual -> the direction
        ti, tv = bsf.block_topk(x, R)                                    # [B, R]
        dirs, cx, wf = bsf.mint(x, ti)                                   # [B, R, d]
        cn = F.normalize(content, dim=-1)
        cos_t = torch.einsum("bld,brd->blr", cn, dirs)                   # cosine(raw h_t, dir)   [B, L, R]
        proj_t = torch.einsum("bld,brd->blr", content, dirs)             # raw projection h_t @ dir
        pk_c = cos_t.argmax(1); pk_p = proj_t.argmax(1)                  # [B, R] (argmax over content tokens)
        return {"x": x.cpu().numpy(), "top_i": ti.int().cpu().numpy(), "top_v": tv.cpu().numpy(),
                "cos_x": cx.cpu().numpy(), "wfrac": wf.cpu().numpy(),
                "peak_cos": (L - 1 - pk_c).short().cpu().numpy(), "peak_proj": (L - 1 - pk_p).short().cpu().numpy(),
                "cos_last": cos_t[:, -1, :].cpu().numpy(), "cos_max": cos_t.max(1).values.cpu().numpy(),
                "proj_last": proj_t[:, -1, :].cpu().numpy(), "proj_max": proj_t.max(1).values.cpu().numpy(),
                "norm_c": (x - mu_acts).norm(dim=-1).cpu().numpy(), "norm_raw": x.norm(dim=-1).cpu().numpy()}

    def process_round(k):
        cpath = f"{a.work}/cands_r{r}_k{k}.jsonl"
        texts, gidx = [], []
        with open(cpath) as f:
            for line in f:
                o = json.loads(line)
                texts.append(o["text"]); gidx.append(int(o["i"]))
        n = len(texts)
        gidx = np.asarray(gidx, np.int64)
        tr = time.time()
        ntok = np.zeros(n, np.int32); ids_all = [None] * n
        for c0 in range(0, n, 8192):
            enc = tok(texts[c0:c0 + 8192], add_special_tokens=False)["input_ids"]
            for j, ids in enumerate(enc):
                ids_all[c0 + j] = ids; ntok[c0 + j] = len(ids)
        ok = (ntok >= a.min_tok) & (ntok <= a.max_tok)
        log(r, f"round {k}: {n} candidates tokenized in {time.time() - tr:.0f}s; n_tok in [{a.min_tok},{a.max_tok}]: {int(ok.sum())} "
               f"(min {ntok.min() if n else 0}, max {ntok.max() if n else 0}, mean {ntok.mean() if n else 0:.1f})")
        pre = f"{a.work}/r{r}_k{k}"
        xlast = np.memmap(pre + ".xlast.f32", np.float32, "w+", shape=(max(n, 1), D_MODEL))
        out = {"top_i": np.zeros((n, R), np.int32), "top_v": np.zeros((n, R), np.float32), "cos_x": np.zeros((n, R), np.float32),
               "wfrac": np.zeros((n, R), np.float32), "peak_cos": np.full((n, R), -1, np.int16), "peak_proj": np.full((n, R), -1, np.int16),
               "cos_last": np.zeros((n, R), np.float32), "cos_max": np.zeros((n, R), np.float32),
               "proj_last": np.zeros((n, R), np.float32), "proj_max": np.zeros((n, R), np.float32),
               "norm_c": np.zeros(n, np.float32), "norm_raw": np.zeros(n, np.float32)}
        idx_ok = np.flatnonzero(ok)
        lens = np.unique(ntok[idx_ok])[::-1]                              # largest first (biggest activations first -> early OOM if any)
        n_done = n_tok_done = 0; n_batches = 0
        tf = time.time(); last_log = tf
        for L in lens.tolist():
            sel = idx_ok[ntok[idx_ok] == L]
            B = max(1, min(a.max_batch, a.batch_tokens // (L + 1)))
            for s in range(0, len(sel), B):
                rows = sel[s:s + B]
                o = forward([ids_all[j] for j in rows.tolist()])
                xlast[rows] = o.pop("x")
                for key, arr in o.items():
                    out[key][rows] = arr
                n_done += len(rows); n_tok_done += len(rows) * (L + 1); n_batches += 1
                if time.time() - last_log > 60:
                    el = time.time() - tf
                    log(r, f"round {k}: {n_done}/{len(idx_ok)} texts, {n_tok_done / 1e6:.1f}M tokens, {n_tok_done / el:.0f} tok/s, "
                           f"{n_done / el:.0f} texts/s, L now {L}, B {B} ({el / 60:.1f} min)")
                    last_log = time.time()
        xlast.flush(); del xlast
        el = time.time() - tf
        np.savez(pre + ".meta.npz.tmp.npz", gidx=gidx, ntok=ntok.astype(np.int16), ok=ok, **out)
        os.replace(pre + ".meta.npz.tmp.npz", pre + ".meta.npz")
        done = {"rank": r, "round": k, "n": n, "n_ok": int(ok.sum()), "n_forwarded": n_done, "tokens": n_tok_done, "batches": n_batches,
                "seconds": el, "tok_per_s": n_tok_done / max(el, 1e-9), "ntok_hist": np.bincount(ntok, minlength=a.max_tok + 2).tolist(),
                "bos_id": int(bos), "ranks": R}
        with open(pre + ".done.json.tmp", "w") as f:
            json.dump(done, f)
        os.replace(pre + ".done.json.tmp", pre + ".done.json")
        log(r, f"round {k} DONE: {n_done} texts / {n_tok_done / 1e6:.1f}M tokens in {el / 60:.1f} min ({n_tok_done / max(el, 1e-9):.0f} tok/s)")

    k = 0
    while True:
        cpath = f"{a.work}/cands_r{r}_k{k}.jsonl"
        while not os.path.exists(cpath + ".ready"):
            if os.path.exists(f"{a.work}/stop"):
                log(r, f"stop after {k} rounds ({(time.time() - t0) / 60:.1f} min alive)")
                return
            time.sleep(2)
        process_round(k)
        k += 1


if __name__ == "__main__":
    main()
