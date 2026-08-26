"""Build the MIXED "universal" match-activation bank for the Qwen3.6-27B direction->text inverter.

FIVE direction families -> one balanced pool of (UNIT inject direction, target span) pairs, in the
EXACT on-disk format SL/pretrain.py consumes (build_stats.json carries n_examples so RL scripts
that read the bank also work unchanged):

  {out}/pool_train/{vecs.f32, records.jsonl, build_stats.json}     training pairs (SFT+RL)
  {out}/pool_heldout/{vecs.f32, records.jsonl, build_stats.json}   held-out pairs (EVAL ONLY)
  {out}/blocks.npz    bsf sidecar {block_ids [Gsel], Qraw [Gsel,d,b], mu [d]} (RL subspace reward)
  {out}/split.json    bsf block split ({"train":[...],"heldout":[...]}, template-compatible)
                      + per-family held-out splits under "families"

Each pool: vecs.f32 = float32 [Nex, d=5120] unit inject directions, row-major (row i = vec_idx i);
records.jsonl = one json/line {"vec_idx": int, "target_text": str, "family": str, ...family fields}.

TARGET SPANS: the acts dump has NO ctx.jsonl -- for a token at (seq s, pos p) the span is
tok.decode(toks[s, max(0, p-L+1) : p+1]) with L ~ Uniform[16, 64]; only positions p >= 16 are used.

FAMILIES (--n-per-family examples minted per family TOTAL; each family's own split then routes
~eval-frac of them to pool_heldout, the rest to pool_train):
  1. bsf      adaptation of SL/build_subspace_data.py to the 27B SASA formats: SASA-encode the
              corpus; each sampled anchor token picks a RANDOM block among its top-k ACTIVE blocks
              (deliberately not always top-1, for direction diversity); Qraw[g] = top-b PCA (CPU
              SVD) of the block's top-1-assigned members' raw-centered acts, computed ONCE per
              block; inject = unit(Qraw @ Qraw^T @ (act - mu)). BLOCK-level train/heldout split.
  2. realact  inject = unit(act - mu) at a sampled (s, p) -- simplest on-manifold family; raw-norm
              filtered at 10x a presampled median (mirrors eval_universal's realact hygiene).
              SEQUENCE-level train/heldout split.
  3. sae      inject = unit SAE encoder column (sae.enc_dirs); target = the scanned (s, p) with
              the max activation of that feature. Peak search runs over the SAME subsampled scan
              as bsf member mining (--scan-tokens caps the cost; peaks are argmax over that
              subsample, not the full dump -- documented tradeoff). FEATURE-level split.
  4. jlens    inject = unit(W_U[t] @ J_READ_LAYER) for vocab tokens t; target = a random corpus
              occurrence of t at p >= 16 (tokens absent from toks are skipped). Needs a
              precomputed --wu-path ([vocab, d] tensor, e.g. torch.save(base.lm_head.weight
              .detach().cpu(), "wu.pt")); WITHOUT it the family is SKIPPED with a warning.
              TOKEN-level split.
  5. cluster  NOT re-mined: --cluster-pool (default /root/pmx/data/pool_sft_1M, the existing
              cluster dir->span bank from SL/build_data.py) is subsampled and re-emitted with
              family="cluster" (directions re-unit-normalized, source fields preserved).
              CLUSTER-id split (falls back to record-level if no "cluster" field).

HELD-OUT HYGIENE: eval_universal.py reserves the LAST 5% of acts-dump SEQUENCES for its realact
eval family, so NO pair here ever draws from a sequence with index >= ceil(0.95 * n_seq).

SASA formats (bsf/train_sasa.py -- DIFFERENT from the 8B bsf.pt template): sasa.pt =
{"E":[d, G*b], "D":[G*b, d], "bias":[d] (DECODER bias, unused here), "G","b","d"}; whitening is
SEPARATE files whiten_mu.npy [d] + whiten_zca.npy [d,d] (ZCA). Encode a raw act x:
y = normalize((x - mu) @ zca, dim=-1); z = (y @ E).view(-1, G, b); gnorm_g = ||z[:, g, :]||;
active blocks = top-k by gnorm (k from {sasa}/meta.json, default 32, --k overrides).
Acts dump (bsf/collect_acts.py): meta.json {"n_seq","seq_len","d_model"};
acts.f16 memmap [n_seq, seq_len, d]; toks.i32 memmap [n_seq, seq_len].

    PYTHONPATH=/root/pmx/helpers python SL/build_universal_bank.py \
        --sae-path /root/pmx/sae27b/ae.pt --wu-path /root/pmx/lenses/qwen3.6-27b/wu.pt \
        --out /root/pmx/data/pool_universal --n-per-family 200000
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as tF
from transformers import AutoTokenizer

from mxf.config import D_MODEL, MODEL, READ_LAYER
from mxf.sae import load_sae

HELDOUT_FRAC = 0.05        # eval_universal.py: last 5% of sequences are eval-only -- never touched
NORM_FILTER_MULT = 10.0    # realact raw-norm filter (same 10x-median rule as the eval protocol)
SPAN_MIN, SPAN_MAX = 16, 64
MIN_SPAN_CHARS = 3         # skip effectively-empty decoded spans (build_sae_data hygiene)
MAX_TARGET_CHARS = 1200    # target_text char cap (matches build_subspace_data / build_data)
SAE_SUB = 4096             # rows per SAE encode sub-chunk ([SAE_SUB, d_sae] fp32 transient)


class PoolWriter:
    """Streams one pretrain pool: vecs.f32 (float32 unit rows appended row-major, so row i ==
    vec_idx i) + records.jsonl. EVERY direction is (re-)unit-normalized on write; non-finite or
    ~zero-norm directions are refused (add() returns False)."""

    def __init__(self, pdir):
        os.makedirs(pdir, exist_ok=True)
        self.pdir, self.n, self.fam = pdir, 0, {}
        self.vf = open(f"{pdir}/vecs.f32", "wb")
        self.rf = open(f"{pdir}/records.jsonl", "w")

    def add(self, vec, record):
        v = np.asarray(vec, np.float32).reshape(-1)
        assert v.shape[0] == D_MODEL, f"direction dim {v.shape[0]} != {D_MODEL}"
        nrm = float(np.linalg.norm(v))
        if not np.isfinite(nrm) or nrm < 1e-6:
            return False
        self.vf.write((v / nrm).tobytes())
        self.rf.write(json.dumps({"vec_idx": self.n, **record}) + "\n")
        self.fam[record["family"]] = self.fam.get(record["family"], 0) + 1
        self.n += 1
        return True

    def close(self, params):
        self.vf.close(); self.rf.close()
        json.dump({"n_examples": self.n, "families": self.fam, "kind": "universal-mixed",
                   "pool": os.path.basename(self.pdir), **params},
                  open(f"{self.pdir}/build_stats.json", "w"), indent=1)


def decode_span(tokz, toks_ram, s, p, rng):
    """Target span for token (s, p): decode toks[s, max(0, p-L+1) : p+1], L ~ U[16, 64]."""
    L = int(rng.integers(SPAN_MIN, SPAN_MAX + 1))
    return tokz.decode(toks_ram[s, max(0, p - L + 1): p + 1].tolist())[:MAX_TARGET_CHARS]


@torch.no_grad()
def corpus_scan(acts, scan_ids, T, d, mu, zca, E, G, b, k, sae, dev, chunk):
    """ONE pass over the scan subsample computing BOTH per-token SASA top-k (bsf members +
    anchors) and per-SAE-feature running max/argmax (sae targets; positions < SPAN_MIN masked so
    a context span always exists). Flat row r <-> (scan_ids[r // T], r % T).
    Returns (topk_idx [n,k] i32, topk_val [n,k] f32, feat_max [F] f32, feat_arg [F] i64) -- the
    topk arrays cost ~n*k*8 bytes of RAM (~2 GB at 8M tokens, k=32)."""
    n_scan = len(scan_ids)
    topk_idx = np.empty((n_scan * T, k), np.int32)
    topk_val = np.empty((n_scan * T, k), np.float32)
    feat_max = torch.full((sae.d_sae,), -1.0, device=dev)
    feat_arg = torch.zeros(sae.d_sae, dtype=torch.long, device=dev)
    seq_per = max(1, chunk // T)
    for it, i0 in enumerate(range(0, n_scan, seq_per)):
        sl = scan_ids[i0: i0 + seq_per]
        x = torch.from_numpy(np.asarray(acts[sl]).astype(np.float32)).to(dev).view(-1, d)
        bad = ~torch.isfinite(x).all(dim=-1)      # corrupt bf16 rows: nan_to_num maps inf to a
        x = torch.nan_to_num(x)                   # huge FINITE value, so mask them out explicitly
        # SASA encode (matches train_sasa): whiten -> unit-norm -> E -> per-block gnorm -> top-k
        y = tF.normalize((x - mu) @ zca, dim=-1)
        gn = (y @ E).view(-1, G, b).norm(dim=-1)                               # [c*T, G]
        v, ix = gn.topk(k, dim=-1)                                             # desc-sorted
        v[bad], ix[bad] = -1.0, -1                # never a member (block -1) nor an anchor (val<=0)
        row0 = i0 * T
        topk_idx[row0: row0 + len(x)] = ix.cpu().numpy().astype(np.int32)
        topk_val[row0: row0 + len(x)] = v.cpu().numpy()
        # SAE peak tracking (pre-topk post-ReLU, sae.encode_features convention), p<SPAN_MIN masked
        good = (torch.arange(T, device=dev).repeat(len(sl)) >= SPAN_MIN) & ~bad
        for s0 in range(0, len(x), SAE_SUB):
            xa = x[s0: s0 + SAE_SUB]
            a_ = torch.relu((xa - sae.b_dec) @ sae.W_enc + sae.b_enc)          # [c, F]
            a_[~good[s0: s0 + len(xa)]] = -1.0
            m, amax = a_.max(0)
            upd = m > feat_max
            feat_max[upd] = m[upd]
            feat_arg[upd] = row0 + s0 + amax[upd]
        if it % 20 == 0:
            print(f"[scan] {min(i0 + seq_per, n_scan) * T:,}/{n_scan * T:,} tokens", flush=True)
    return topk_idx, topk_val, feat_max.cpu().numpy(), feat_arg.cpu().numpy()


# -----------------------------------------------------------------------------------------------
# family 1: bsf (subspace) -- adapted from SL/build_subspace_data.py
# -----------------------------------------------------------------------------------------------

def build_bsf(a, acts, toks_ram, scan_ids, T, topk_idx, topk_val, G, b, mu, dev, writers, rng,
              tokz, out):
    ntok = topk_idx.shape[0]
    pcol = (np.arange(ntok, dtype=np.int64) % T)

    # member mining: per-token TOP-1 block (col 0 of the desc-sorted top-k); position-0 rows
    # (attention sink) excluded. Vectorized equivalent of the template's capped-fill loop:
    # members of g = its top-1-assigned scanned tokens, desc by gnorm, capped at --members.
    top1, top1g = topk_idx[:, 0].astype(np.int64), topk_val[:, 0]
    vrows = np.flatnonzero(pcol >= 1)
    sv = vrows[np.lexsort((-top1g[vrows], top1[vrows]))]      # by block, then desc gnorm
    bs = top1[sv]
    starts = np.searchsorted(bs, np.arange(G), "left")
    counts = np.searchsorted(bs, np.arange(G), "right") - starts
    usable = np.flatnonzero(counts >= a.min_members)
    print(f"[bsf] usable blocks (>= {a.min_members} top-1 members in scan): {len(usable)}/{G}",
          flush=True)
    if not len(usable):
        print("[bsf] WARN no usable blocks -- skipping family", flush=True)
        return {"train": [], "heldout": []}, {"usable_blocks": 0}
    usable_mask = np.zeros(G, bool)
    usable_mask[usable] = True
    heldout_blocks = set(int(g) for g in rng.choice(
        usable, max(1, int(len(usable) * a.eval_frac)), replace=False))
    print(f"[bsf] held-out blocks: {len(heldout_blocks)}/{len(usable)} (eval-frac {a.eval_frac})",
          flush=True)

    # anchor selection: sampled tokens (p >= SPAN_MIN so a span exists); each anchor picks a
    # RANDOM usable block among its top-k ACTIVE blocks (deliberate, for direction diversity).
    perm = rng.permutation(np.flatnonzero(pcol >= SPAN_MIN))
    rows_l, g_l, gv_l, got, ptr, no_usable = [], [], [], 0, 0, 0
    while got < a.n_per_family and ptr < len(perm):
        take = perm[ptr: ptr + max(65536, 2 * (a.n_per_family - got))]
        ptr += len(take)
        cnd, val = topk_idx[take], topk_val[take]
        okm = usable_mask[cnd] & (val > 0)
        score = rng.random(okm.shape)
        score[~okm] = -1.0
        col = score.argmax(1)
        ar = np.arange(len(take))
        keep = okm[ar, col]
        no_usable += int((~keep).sum())
        cut = a.n_per_family - got
        rows_l.append(take[keep][:cut])
        g_l.append(cnd[ar, col][keep][:cut].astype(np.int64))
        gv_l.append(val[ar, col][keep][:cut])
        got += len(rows_l[-1])
    rows, gsel, gval = np.concatenate(rows_l), np.concatenate(g_l), np.concatenate(gv_l)
    print(f"[bsf] {len(rows)} anchors ({no_usable} candidates had no usable active block)",
          flush=True)

    # group anchors by chosen block -> each block's Qraw (raw member-PCA) is computed exactly ONCE
    ordg = np.argsort(gsel, kind="stable")
    rows, gsel, gval = rows[ordg], gsel[ordg], gval[ordg]
    groups = np.split(np.arange(len(rows)), np.flatnonzero(np.diff(gsel)) + 1)
    blk_ids, Qraws, n_emit, n_svd, n_span = [], [], 0, 0, 0
    for gi, grp in enumerate(groups):
        g = int(gsel[grp[0]])
        mem = sv[starts[g]: starts[g] + min(a.members, int(counts[g]))]
        ms, mp = scan_ids[mem // T], mem % T
        o = np.argsort(ms * T + mp)                          # sorted gather: memmap locality
        raw = torch.nan_to_num(torch.from_numpy(
            np.asarray(acts[ms[o], mp[o]]).astype(np.float32)).to(dev)) - mu   # [m,d] raw-centered
        rc = (raw - raw.mean(0)).cpu()   # CPU SVD: cuSOLVER errors (err89) on ill-conditioned blocks
        try:
            Vt = torch.linalg.svd(rc, full_matrices=False).Vh
        except Exception as e:
            n_svd += len(grp)
            print(f"[bsf]  skip block {g}: svd failed ({e})", flush=True)
            continue
        Qraw = Vt[:b].t().contiguous().to(dev)                                 # [d, b]
        blk_ids.append(g)
        Qraws.append(Qraw.cpu().numpy().astype(np.float32))
        arows = rows[grp]
        s_arr, p_arr = scan_ids[arows // T], arows % T
        oa = np.argsort(s_arr * T + p_arr)
        s_arr, p_arr, agv = s_arr[oa], p_arr[oa], gval[grp][oa]
        X = torch.nan_to_num(torch.from_numpy(
            np.asarray(acts[s_arr, p_arr]).astype(np.float32)).to(dev)) - mu   # [n_g, d]
        coords = X @ Qraw                                                      # [n_g, b]
        proj = coords @ Qraw.t()                                               # projection into S
        inj = (proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)).cpu().numpy()
        co = coords.cpu().numpy()
        dst = "heldout" if g in heldout_blocks else "train"
        for j in range(len(s_arr)):
            text = decode_span(tokz, toks_ram, int(s_arr[j]), int(p_arr[j]), rng)
            if len(text.strip()) < MIN_SPAN_CHARS:
                n_span += 1
                continue
            n_emit += writers[dst].add(inj[j], {
                "target_text": text, "family": "bsf", "block": g,
                "coords": [float(x) for x in co[j]], "gnorm": float(agv[j]),
                "seq": int(s_arr[j]), "pos": int(p_arr[j])})
        if (gi + 1) % 500 == 0:
            print(f"[bsf]  {gi + 1}/{len(groups)} blocks, {n_emit} examples", flush=True)

    # sidecar: every block that minted examples (RL subspace reward reads Qraw by block id)
    if not blk_ids:
        print("[bsf] WARN every selected block failed SVD -- no blocks.npz written", flush=True)
        return {"train": [], "heldout": []}, {"usable_blocks": int(len(usable)),
                                              "blocks_built": 0, "dropped_svd": n_svd}
    np.savez(f"{out}/blocks.npz", block_ids=np.array(blk_ids, np.int32),
             Qraw=np.stack(Qraws).astype(np.float32), mu=mu.cpu().numpy().astype(np.float32))
    split = {"train": [g for g in blk_ids if g not in heldout_blocks],
             "heldout": sorted(g for g in blk_ids if g in heldout_blocks)}
    print(f"[bsf] done: {n_emit} examples over {len(blk_ids)} blocks "
          f"({len(split['train'])} train / {len(split['heldout'])} heldout); "
          f"dropped {n_svd} anchors (svd) + {n_span} (empty span)", flush=True)
    return split, {"usable_blocks": int(len(usable)), "blocks_built": len(blk_ids),
                   "anchors_no_usable_block": no_usable, "dropped_svd": n_svd,
                   "dropped_span": n_span}


# -----------------------------------------------------------------------------------------------
# family 2: realact
# -----------------------------------------------------------------------------------------------

def build_realact(a, acts, toks_ram, lo, T, mu_np, writers, rng, tokz):
    heldout_seqs = set(int(s) for s in rng.choice(
        lo, max(1, int(lo * a.eval_frac)), replace=False))
    # raw-norm filter threshold from a seeded presample (eval_universal's 10x-median rule)
    ss, pp = rng.integers(0, lo, 16384), rng.integers(1, T, 16384)
    o = np.argsort(ss * T + pp)
    pre = np.nan_to_num(np.asarray(acts[ss[o], pp[o]]).astype(np.float32)).astype(np.float64)
    med = float(np.median(np.linalg.norm(pre, axis=1)))     # f64: fp32 squaring overflows on the
                                                            # huge nan_to_num'd corrupt-row values
    print(f"[realact] presample median raw norm {med:.1f} (keep <= {NORM_FILTER_MULT}x)", flush=True)
    got, n_drop = 0, 0
    for _ in range(64):                                       # safety-capped top-up loop
        if got >= a.n_per_family:
            break
        m = min(65536, max(8192, int((a.n_per_family - got) * 1.3)))
        s_arr, p_arr = rng.integers(0, lo, m), rng.integers(SPAN_MIN, T, m)
        o = np.argsort(s_arr * T + p_arr)
        s_arr, p_arr = s_arr[o], p_arr[o]
        X = np.nan_to_num(np.asarray(acts[s_arr, p_arr]).astype(np.float32))
        nrm = np.linalg.norm(X.astype(np.float64), axis=1)
        ok = (nrm > 1e-3) & (nrm <= NORM_FILTER_MULT * med)
        n_drop += int((~ok).sum())
        # emission order is PERMUTED: the gather is (s,p)-sorted for memmap locality, so stopping
        # at the budget mid-chunk would otherwise bias the family toward low sequence ids
        for j in rng.permutation(np.flatnonzero(ok)):
            if got >= a.n_per_family:
                break
            s, p = int(s_arr[j]), int(p_arr[j])
            text = decode_span(tokz, toks_ram, s, p, rng)
            if len(text.strip()) < MIN_SPAN_CHARS:
                n_drop += 1
                continue
            got += writers["heldout" if s in heldout_seqs else "train"].add(
                X[j] - mu_np, {"target_text": text, "family": "realact", "seq": s, "pos": p,
                               "act_norm": round(float(nrm[j]), 2)})
        print(f"[realact] {got}/{a.n_per_family}", flush=True)
    if got < a.n_per_family:
        print(f"[realact] WARN only minted {got} (norm filter too tight?)", flush=True)
    return heldout_seqs, {"minted": got, "dropped": n_drop, "median_norm": med}


# -----------------------------------------------------------------------------------------------
# family 3: sae
# -----------------------------------------------------------------------------------------------

def build_sae_family(a, sae, feat_max, feat_arg, scan_ids, T, toks_ram, writers, rng, tokz):
    alive = np.flatnonzero(feat_max > 0)                      # fired on the scan at some p>=16
    n_take = min(a.n_per_family, len(alive))
    if n_take < a.n_per_family:
        print(f"[sae] WARN only {len(alive)} scan-alive features < n-per-family "
              f"{a.n_per_family} -- family capped at {n_take}", flush=True)
    feats = rng.choice(alive, n_take, replace=False)
    heldout_feats = set(int(f) for f in rng.choice(
        feats, max(1, int(n_take * a.eval_frac)), replace=False))
    got, n_drop = 0, 0
    for c0 in range(0, n_take, 32768):
        fs = [int(f) for f in feats[c0: c0 + 32768]]
        dirs = sae.enc_dirs(fs).float().cpu().numpy()         # unit encoder columns [c, d]
        for j, f in enumerate(fs):
            row = int(feat_arg[f])
            s, p = int(scan_ids[row // T]), int(row % T)
            text = decode_span(tokz, toks_ram, s, p, rng)
            if len(text.strip()) < MIN_SPAN_CHARS:
                n_drop += 1
                continue
            got += writers["heldout" if f in heldout_feats else "train"].add(
                dirs[j], {"target_text": text, "family": "sae", "feature": f,
                          "act": round(float(feat_max[f]), 3), "seq": s, "pos": p})
        print(f"[sae] {min(c0 + 32768, n_take)}/{n_take} features, {got} examples", flush=True)
    return heldout_feats, {"alive_on_scan": int(len(alive)), "minted": got, "dropped_span": n_drop}


# -----------------------------------------------------------------------------------------------
# family 4: jlens (optional -- needs --wu-path)
# -----------------------------------------------------------------------------------------------

def load_wu(path):
    """[vocab, d] unembedding: torch-saved tensor (or a dict holding one; .npy also accepted).
    Produce it on the GPU box with: torch.save(base.lm_head.weight.detach().cpu(), "wu.pt")."""
    if path.endswith(".npy"):
        wu = torch.from_numpy(np.load(path))
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        wu = obj if torch.is_tensor(obj) else next(v for v in obj.values() if torch.is_tensor(v))
    wu = wu.float()
    assert wu.ndim == 2 and wu.shape[1] == D_MODEL, f"wu shape {tuple(wu.shape)} != [vocab, {D_MODEL}]"
    return wu


def build_jlens(a, toks_ram, T, writers, rng, tokz, dev):
    if not a.wu_path:
        print("[jlens] WARN --wu-path not given -> SKIPPING the jlens family "
              "(supply W_U by dumping base.lm_head.weight to enable it)", flush=True)
        return None, {"skipped": True}
    wu = load_wu(a.wu_path)
    J = torch.load(a.lens_path, map_location="cpu", weights_only=False)["J"][READ_LAYER]
    J = J.float().to(dev)
    assert J.shape == (D_MODEL, D_MODEL), f"J{READ_LAYER} shape {tuple(J.shape)}"
    vocab = wu.shape[0]
    print(f"[jlens] W_U {tuple(wu.shape)}  J{READ_LAYER} {tuple(J.shape)}", flush=True)

    # occurrence index over positions >= SPAN_MIN (flat f <-> (f // Tin, f % Tin + SPAN_MIN))
    Tin = T - SPAN_MIN
    flat_tok = toks_ram[:, SPAN_MIN:].reshape(-1)
    order = np.argsort(flat_tok, kind="stable")
    sorted_tok = flat_tok[order]
    present = np.unique(sorted_tok)
    present = present[(present >= 0) & (present < vocab)]
    perm = rng.permutation(present)
    heldout_toks = set(int(t) for t in rng.choice(
        present, max(1, int(len(present) * a.eval_frac)), replace=False))
    tpt = max(1, math.ceil(a.n_per_family / len(present)))    # spans per token to reach the budget
    print(f"[jlens] {len(present)} vocab tokens present at p>={SPAN_MIN}; targets/token {tpt}",
          flush=True)

    got, n_drop = 0, 0
    for c0 in range(0, len(perm), 8192):
        if got >= a.n_per_family:
            break
        ts = perm[c0: c0 + 8192].astype(np.int64)
        dirs = tF.normalize(wu[torch.from_numpy(ts)].to(dev) @ J, dim=-1).cpu().numpy()
        for j, t in enumerate(ts):
            if got >= a.n_per_family:
                break
            t = int(t)
            l = np.searchsorted(sorted_tok, t, "left")
            r = np.searchsorted(sorted_tok, t, "right")
            occ = order[l:r]
            for f in rng.choice(occ, min(tpt, len(occ), a.n_per_family - got), replace=False):
                s, p = int(f // Tin), int(f % Tin) + SPAN_MIN
                text = decode_span(tokz, toks_ram, s, p, rng)
                if len(text.strip()) < MIN_SPAN_CHARS:
                    n_drop += 1
                    continue
                got += writers["heldout" if t in heldout_toks else "train"].add(
                    dirs[j], {"target_text": text, "family": "jlens", "token_id": t,
                              "token_str": tokz.decode([t]), "seq": s, "pos": p})
        print(f"[jlens] {got}/{a.n_per_family} ({min(c0 + 8192, len(perm))}/{len(perm)} tokens)",
              flush=True)
    if got < a.n_per_family:
        print(f"[jlens] WARN only minted {got} (corpus occurrences exhausted)", flush=True)
    return heldout_toks, {"present_tokens": int(len(present)), "targets_per_token": tpt,
                          "minted": got, "dropped_span": n_drop}


# -----------------------------------------------------------------------------------------------
# family 5: cluster (re-emit the existing bank -- NOT re-mined)
# -----------------------------------------------------------------------------------------------

def build_cluster(a, writers, rng):
    """pool_sft_1M is the existing cluster (dir->span) bank (SL/build_data.py, target-mode
    cluster). Its probe directions are already raw-space unit rows; we subsample, tag
    family="cluster", keep the source fields, and re-unit-normalize on write."""
    rp, vp = f"{a.cluster_pool}/records.jsonl", f"{a.cluster_pool}/vecs.f32"
    if not (os.path.exists(rp) and os.path.exists(vp)):
        raise FileNotFoundError(f"--cluster-pool {a.cluster_pool} missing records.jsonl/vecs.f32 "
                                "(expected the existing cluster dir->span bank, e.g. pool_sft_1M)")
    lines = open(rp).read().splitlines()
    n_vec = os.path.getsize(vp) // (4 * D_MODEL)
    mm = np.memmap(vp, dtype=np.float32, mode="r", shape=(n_vec, D_MODEL))
    n_take = min(a.n_per_family, len(lines))
    recs = [json.loads(lines[i]) for i in np.sort(rng.choice(len(lines), n_take, replace=False))]
    cids = sorted({int(r["cluster"]) for r in recs if "cluster" in r})
    if cids:
        mode = "cluster-id"
        held = set(int(c) for c in rng.choice(
            np.array(cids), max(1, int(len(cids) * a.eval_frac)), replace=False))
    else:                                                     # no cluster field -> record-level
        mode = "record-row"
        held = set(int(i) for i in rng.choice(
            n_take, max(1, int(n_take * a.eval_frac)), replace=False))
    print(f"[cluster] {n_take}/{len(lines)} records from {a.cluster_pool} "
          f"({len(cids)} clusters, split mode {mode})", flush=True)
    got, n_bad = 0, 0
    for i, r in enumerate(recs):
        vi = int(r["vec_idx"])
        if not (0 <= vi < n_vec):
            n_bad += 1
            continue
        ho = (int(r.get("cluster", -1)) in held) if mode == "cluster-id" else (i in held)
        got += writers["heldout" if ho else "train"].add(
            np.asarray(mm[vi], np.float32),
            {**{k: v for k, v in r.items() if k != "vec_idx"}, "family": "cluster"})
        if (i + 1) % 50000 == 0:
            print(f"[cluster] {i + 1}/{n_take}", flush=True)
    print(f"[cluster] done: {got} examples ({n_bad} bad vec_idx skipped)", flush=True)
    return {"mode": mode, "heldout": sorted(held)}, {"source_records": len(lines), "minted": got,
                                                     "source_clusters": len(cids)}


# -----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sasa", default="/root/pmx/bsf27b/sasa",
                    help="dir with sasa.pt + whiten_mu.npy + whiten_zca.npy (+ meta.json for k)")
    ap.add_argument("--acts-dir", default="/root/pmx/bsf27b/acts",
                    help="acts dump dir: meta.json + acts.f16 [n_seq,seq_len,d] + toks.i32")
    ap.add_argument("--sae-path", required=True, help="ae.pt for the 27B READ_LAYER SAE")
    ap.add_argument("--lens-path", default="/root/pmx/lenses/qwen3.6-27b/j-lens/lens.pt",
                    help="lens.pt; jlens uses [\"J\"][READ_LAYER]")
    ap.add_argument("--wu-path", default=None,
                    help="saved [vocab,d] unembedding tensor; OMIT to skip the jlens family")
    ap.add_argument("--cluster-pool", default="/root/pmx/data/pool_sft_1M",
                    help="existing cluster (dir->span) bank to re-emit as family=cluster")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-family", type=int, default=200000,
                    help="examples minted per family TOTAL; each family's split routes "
                         "~eval-frac of them to pool_heldout")
    ap.add_argument("--k", type=int, default=0,
                    help="SASA top-k active blocks; 0 = read from {sasa}/meta.json (32 if absent)")
    ap.add_argument("--members", type=int, default=256, help="top members per block for the PCA")
    ap.add_argument("--min-members", type=int, default=48,
                    help="blocks with fewer top-1 scan members are unusable")
    ap.add_argument("--eval-frac", type=float, default=0.1,
                    help="per-family held-out fraction (bsf: blocks, realact: sequences, "
                         "sae: features, jlens: tokens, cluster: cluster ids)")
    ap.add_argument("--scan-tokens", type=int, default=8_000_000,
                    help="corpus tokens in the shared SASA/SAE scan (whole random sequences, "
                         "rounded up; caps bsf member mining AND the sae peak search)")
    ap.add_argument("--chunk", type=int, default=16384,
                    help="token rows per GPU encode chunk (16384 with G=32768,b=8 peaks ~25 GB "
                         "of transient GPU memory -- lower it on smaller cards)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    dev = a.device if torch.cuda.is_available() else "cpu"
    fams = ["scan", "bsf", "realact", "sae", "jlens", "cluster"]
    rngs = {n: np.random.default_rng([a.seed, i]) for i, n in enumerate(fams)}

    # ---- substrate: acts dump (first 95% of sequences ONLY -- eval owns the last 5%) ----
    am = json.load(open(f"{a.acts_dir}/meta.json"))
    n_seq, T, d = am["n_seq"], am["seq_len"], am["d_model"]
    assert d == D_MODEL, f"acts d_model {d} != D_MODEL {D_MODEL}"
    acts = np.memmap(f"{a.acts_dir}/acts.f16", dtype=np.float16, mode="r", shape=(n_seq, T, d))
    toks_mm = np.memmap(f"{a.acts_dir}/toks.i32", dtype=np.int32, mode="r", shape=(n_seq, T))
    lo = int(np.ceil((1.0 - HELDOUT_FRAC) * n_seq))           # usable sequences: [0, lo)
    toks_ram = np.asarray(toks_mm[:lo])                       # ~2KB/seq -- spans decoded from RAM
    tokz = AutoTokenizer.from_pretrained(MODEL)
    print(f"[universal] acts {n_seq}x{T}x{d}; training draws from seqs [0,{lo}) "
          f"(last {n_seq - lo} reserved for the realact EVAL family)", flush=True)

    # ---- SASA featurizer (27B format: E + separate ZCA whitening; bias is the DECODER bias) ----
    ck = torch.load(f"{a.sasa}/sasa.pt", map_location="cpu")
    G, b = ck["G"], ck["b"]
    assert ck["d"] == d and ck["E"].shape == (d, G * b), "sasa.pt shape mismatch"
    E = ck["E"].float().to(dev)
    mu = torch.from_numpy(np.load(f"{a.sasa}/whiten_mu.npy")).float().to(dev)
    zca = torch.from_numpy(np.load(f"{a.sasa}/whiten_zca.npy")).float().to(dev)
    smeta_p = f"{a.sasa}/meta.json"
    k = a.k or (json.load(open(smeta_p)).get("k", 32) if os.path.exists(smeta_p) else 32)
    assert 0 < k <= G
    sae = load_sae(a.sae_path, device=dev, dtype=torch.float32)
    print(f"[universal] SASA G={G} b={b} k={k} | SAE d_sae={sae.d_sae} | device {dev}", flush=True)

    # ---- shared scan (bsf top-k + members AND sae feature peaks in one pass over the acts) ----
    n_scan = min(lo, math.ceil(a.scan_tokens / T))
    scan_ids = np.sort(rngs["scan"].choice(lo, n_scan, replace=False))
    print(f"[universal] scanning {n_scan} seqs = {n_scan * T:,} tokens (--scan-tokens "
          f"{a.scan_tokens:,})", flush=True)
    topk_idx, topk_val, feat_max, feat_arg = corpus_scan(
        acts, scan_ids, T, d, mu, zca, E, G, b, k, sae, dev, a.chunk)
    del E
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- mint all five families into the two pools ----
    writers = {"train": PoolWriter(f"{a.out}/pool_train"),
               "heldout": PoolWriter(f"{a.out}/pool_heldout")}
    fam_stats = {}
    bsf_split, fam_stats["bsf"] = build_bsf(
        a, acts, toks_ram, scan_ids, T, topk_idx, topk_val, G, b, mu, dev, writers,
        rngs["bsf"], tokz, a.out)
    heldout_seqs, fam_stats["realact"] = build_realact(
        a, acts, toks_ram, lo, T, mu.cpu().numpy(), writers, rngs["realact"], tokz)
    heldout_feats, fam_stats["sae"] = build_sae_family(
        a, sae, feat_max, feat_arg, scan_ids, T, toks_ram, writers, rngs["sae"], tokz)
    heldout_toks, fam_stats["jlens"] = build_jlens(
        a, toks_ram, T, writers, rngs["jlens"], tokz, dev)
    cluster_split, fam_stats["cluster"] = build_cluster(a, writers, rngs["cluster"])

    # ---- splits + stats ----
    json.dump({"train": bsf_split["train"], "heldout": bsf_split["heldout"],   # bsf blocks
               "families": {"realact_heldout_seqs": sorted(heldout_seqs),
                            "sae_heldout_features": sorted(heldout_feats),
                            "jlens_heldout_tokens": (sorted(heldout_toks)
                                                     if heldout_toks is not None else None),
                            "cluster_heldout": cluster_split}},
              open(f"{a.out}/split.json", "w"))
    params = {"args": vars(a), "k": k, "G": G, "b": b, "d": d, "n_scan_seq": int(n_scan),
              "usable_seqs": int(lo), "heldout_frac_seqs": HELDOUT_FRAC,
              "family_stats": fam_stats}
    for pw in writers.values():
        pw.close(params)
    counts = " ".join(f"{f}={writers['train'].fam.get(f, 0)}/{writers['heldout'].fam.get(f, 0)}"
                      for f in ["bsf", "realact", "sae", "jlens", "cluster"])
    print(f"BUILD_UNIVERSAL_DONE {counts} (train/heldout) | total train={writers['train'].n} "
          f"heldout={writers['heldout'].n} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
