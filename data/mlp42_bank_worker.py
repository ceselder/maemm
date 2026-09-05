"""Layer-42 MLP neurons of Qwen/Qwen3.6-27B as a SIXTH inverter direction family (container side; launched by
data/modal_mlp42_bank.py). Builds, from the /data/mlp42 dump of data/modal_mlp42_neurons.py:

  1. scan   (GPU, forward-only) -> /data/mlp42/bank_scan.npz
     Re-runs layer 42 on the first `n_windows` (default 1600 x 256 = 409,600 content tokens) of the SAME seeded
     acts27b TRAIN windows the neuron statistics used ([BOS 248044] + 256 tokens, BOS position dropped). For every
     neuron: polarity-signed value a_s = polarity * a, fire := a_s >= REL_THR * corpus max|a| (the 1.02M-token max of
     neuron_stats.npz), fire count, running top-32 (a_s, global token) contexts. For the SPARSE set (< 0.1% of tokens
     & non-trivial write, 2,820 neurons): every firing event (token, neuron, a_s) and the sparse x sparse co-firing
     count matrix.
  2. build  (small GPU: leak check) -> /data/banks/mlp42 + /data/eval_universal_ho/eval_sets_heldout_v2.pt
     Neuron universe = neurons firing on < 1% of tokens (frequency from the stats histogram, as in the report's
     bands) with >= 1 usable firing context in the scan. 10% of them (seeded, stratified by frequency band) are
     HELD OUT together with every co-firing pair touching a held-out neuron.
       family "mlp"       direction = polarity * unit(down_proj[:, i]); target = W~U[16,32]-token window ENDING at a
                          firing token (firing token last); the neuron's top-8 firing tokens -> up to 8 rows/neuron.
       family "mlp_pair"  strong both-sparse co-firing pairs (>= 10 joint firings, lift >= 10, Poisson p < 1e-10 on
                          the scan tokens); bank direction = unit(a_i * col_i + a_j * col_j) with the two raw
                          activations AT the joint-firing token, target = window ending at that token (top-4 joint
                          tokens per pair by min normalized activation). Eval direction of a pair = the same
                          composite with the MEAN joint-firing activations (one fixed direction per pair).
     Eval cache v2 = the old cache (every old family byte-identical, meta["cos_families"] unchanged) + 'mlp_dirs'
     [512, d] + 'mlp_pair_dirs' [<=256, d] + per-family neuron/polarity/corpus_max tensors ([n, k]) for the
     fire-back metric + meta["extra_families"] = ["mlp", "mlp_pair"] + meta["mlp42"] (definitions, held-out ids).
     Bank rows are TRAIN neurons / both-train pairs only; every staged row within cos > LEAK_COS of ANY v2 eval
     direction or pool_heldout row is dropped, then the leak assertion is re-checked.
  3. merge  (CPU) -> /data/banks/mix_1m_mlp = seeded shuffle of /data/banks/mix_1m_v2 + /data/banks/mlp42.
"""
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from mxf.config import D_MODEL, MODEL, READ_LAYER
from mxf.inject import get_layer
import mlp42_neurons_worker as W

OUT = W.OUT                                   # /data/mlp42
BANK_OUT = "/data/banks/mlp42"
MIX_SRC = "/data/banks/mix_1m_v2"
MIX_OUT = "/data/banks/mix_1m_mlp"
EVAL_CACHE_V1 = "/data/eval_universal_ho/eval_sets_heldout.pt"
EVAL_CACHE_V2 = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
POOL_HELDOUT = "/data/pool_heldout"
REL_THR = 0.10                                # fire := polarity * a >= REL_THR * corpus max|a|
CAND_MAX = 1e-2                               # neuron universe: fires on < 1% of tokens
SPARSE_MAX = 1e-3                             # pairs: both members fire on < 0.1% of tokens (+ write floor), == report
LEAK_COS = 0.999
BAND_EDGES = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.01]
BAND_NAMES = ["<0.01%", "0.01-0.1%", "0.1-1%", "1-10%", ">=10%"]
EXTRA_FAMILIES = ["mlp", "mlp_pair"]


def log(msg):
    W.log(msg)


# ----------------------------------------------------------------------------------------------------------------
# neuron sets from the 1.02M-token statistics (== scripts/mlp42_neurons_analyze.py)
# ----------------------------------------------------------------------------------------------------------------
def freq_above(hist, edges, lo, bpd, thr):
    """hist [NB, N] counts of |a| per log bin (bin 0 underflow, bin NB-1 overflow); thr [N] or scalar. Per-neuron count
    of tokens with |a| >= thr (log-uniform interpolation inside the straddling bin)."""
    NB, N = hist.shape
    thr = np.broadcast_to(np.asarray(thr, np.float64), (N,))
    pos = (np.log10(np.maximum(thr, 1e-30)) - lo) * bpd
    b = np.floor(pos).astype(np.int64) + 1
    frac_above = 1.0 - (pos - np.floor(pos))
    b = np.clip(b, 0, NB - 1)
    csum = np.cumsum(hist[::-1], axis=0)[::-1]
    above_full = np.where(b + 1 < NB, csum[np.minimum(b + 1, NB - 1), np.arange(N)], 0)
    straddle = hist[b, np.arange(N)] * frac_above
    out = above_full + straddle
    out[thr <= 0] = hist.sum(0)[thr <= 0]
    return out


def neuron_sets(st, rel_thr=REL_THR, cand_max=CAND_MAX, sparse_max=SPARSE_MAX):
    """Returns freq_rel [N] (fraction of the 1.02M tokens with polarity*a >= rel_thr*max), band [N] (index into
    BAND_NAMES), cand_ids (< cand_max, max|a| > 0), sparse_ids (< sparse_max & write_max >= its 25th percentile)."""
    hist = (st["hist_pos"] + st["hist_neg"]).astype(np.float64)
    edges, lo, bpd = st["hist_edges"], float(st["hist_lo"]), int(st["hist_bpd"])
    max_abs = st["max_abs"].astype(np.float64); wnorm = st["wnorm"].astype(np.float64)
    freq_rel = freq_above(hist, edges, lo, bpd, rel_thr * max_abs) / float(st["n_tok"])
    band = np.digitize(freq_rel, BAND_EDGES[1:-1])
    write_max = max_abs * wnorm
    floor = float(np.quantile(write_max, 0.25))
    cand = (freq_rel < cand_max) & (max_abs > 0)
    sparse = (freq_rel < sparse_max) & (write_max >= floor)
    return freq_rel, band, np.flatnonzero(cand), np.flatnonzero(sparse), floor


# ----------------------------------------------------------------------------------------------------------------
# stage: scan
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def run_scan(model, n_windows=1600, win_len=256, batch=16, topk=32, dev="cuda:0"):
    st = np.load(f"{OUT}/neuron_stats.npz"); sw = np.load(f"{OUT}/sel_windows.npz")
    N = int(st["N"])
    assert int(sw["ids"].shape[1]) == win_len and n_windows <= sw["ids"].shape[0], (sw["ids"].shape, n_windows)
    ids_np = sw["ids"][:n_windows]; sel = sw["sel_windows"][:n_windows]
    max_abs = torch.from_numpy(st["max_abs"].astype(np.float32)).to(dev)
    pol = torch.from_numpy(st["polarity"].astype(np.float32)).to(dev)
    freq_rel, band, cand_ids, sparse_ids, floor = neuron_sets(st)
    log(f"scan: {n_windows} windows x {win_len} = {n_windows * win_len} tokens | universe (<{CAND_MAX:.0%}) {len(cand_ids)} "
        f"| sparse (<{SPARSE_MAX:.1%} & write>={floor:.3g}) {len(sparse_ids)} | bands {np.bincount(band, minlength=5).tolist()}")
    layer = get_layer(model, READ_LAYER); mlp = layer.mlp
    assert mlp.down_proj.weight.shape[1] == N
    thr = REL_THR * max_abs
    S = torch.as_tensor(sparse_ids, device=dev, dtype=torch.long)
    C_ss = torch.zeros(len(S), len(S), device=dev)
    n_fire = torch.zeros(N, dtype=torch.int64, device=dev)
    sum_fire = torch.zeros(N, dtype=torch.float64, device=dev)
    topv = torch.full((topk, N), -float("inf"), device=dev); topi = torch.full((topk, N), -1, dtype=torch.int64, device=dev)
    ev_tok, ev_nid, ev_val = [], [], []
    bos_col = np.full((batch, 1), W.BOS, np.int32)
    t0 = time.time()
    for b0 in range(0, n_windows, batch):
        b1 = min(b0 + batch, n_windows); B = b1 - b0
        ids = torch.from_numpy(np.concatenate([bos_col[:B], ids_np[b0:b1]], 1).astype(np.int64)).to(dev)
        a_bf, _ = W.forward_capture(model, ids, layer, mlp)
        a = a_bf[:, 1:].reshape(-1, N).float() * pol[None, :]           # polarity-signed neuron values, BOS dropped
        g0 = b0 * win_len
        fire = a >= thr[None, :]
        n_fire += fire.sum(0); sum_fire += (a * fire).sum(0, dtype=torch.float64)
        v, i = a.topk(topk, dim=0)
        cv = torch.cat([topv, v]); ci = torch.cat([topi, i + g0])
        keep = cv.topk(topk, dim=0).indices
        topv = cv.gather(0, keep); topi = ci.gather(0, keep)
        aS = a[:, S]; fS = fire[:, S]
        fSf = fS.float()
        C_ss.addmm_(fSf.T, fSf)
        nz = fS.nonzero()
        ev_tok.append((nz[:, 0] + g0).to(torch.int32).cpu()); ev_nid.append(nz[:, 1].to(torch.int16).cpu())
        ev_val.append(aS[nz[:, 0], nz[:, 1]].cpu())
        del a, fire, aS, fS, fSf, a_bf
        if (b0 // batch) % 20 == 0:
            log(f"scan {b1}/{n_windows} windows ({b1 * win_len / max(time.time() - t0, 1e-6):.0f} tok/s)")
    T = n_windows * win_len
    ev_tok = torch.cat(ev_tok).numpy(); ev_nid = torch.cat(ev_nid).numpy(); ev_val = torch.cat(ev_val).numpy()
    nf = n_fire.cpu().numpy()
    log(f"scan done in {time.time() - t0:.0f}s: sparse events {len(ev_tok)} | universe neurons with >=1 firing "
        f"{int((nf[cand_ids] > 0).sum())}/{len(cand_ids)} | sparse pairs C>=10: {int((torch.triu(C_ss, 1) >= 10).sum())}")
    np.savez(f"{OUT}/bank_scan.npz", n_windows=n_windows, win_len=win_len, T=T, sel=sel, rel_thr=REL_THR,
             n_fire=nf, sum_fire=sum_fire.cpu().numpy(), topv=topv.cpu().numpy(), topi=topi.cpu().numpy(),
             ev_tok=ev_tok, ev_nid=ev_nid, ev_val=ev_val, C_ss=C_ss.to(torch.int32).cpu().numpy(),
             sparse_ids=sparse_ids, cand_ids=cand_ids, freq_rel=freq_rel, band=band, write_floor=floor)
    log(f"saved {OUT}/bank_scan.npz")
    return {"T": T, "n_events": int(len(ev_tok)), "n_universe_firing": int((nf[cand_ids] > 0).sum())}


# ----------------------------------------------------------------------------------------------------------------
# stage: build (selection + bank + eval cache v2 + leak check)
# ----------------------------------------------------------------------------------------------------------------
def _poisson_sf(k, mu):
    from scipy.stats import poisson
    return poisson.sf(np.asarray(k, np.float64) - 1.0, np.asarray(mu, np.float64))


def _alloc(counts, total, floor=0):
    """Proportional allocation of `total` over strata with `counts` available (largest remainder), capped by
    availability, at least `floor` per stratum where available."""
    counts = np.asarray(counts, np.int64)
    if counts.sum() <= total:
        return counts.copy()
    raw = total * counts / counts.sum()
    take = np.floor(raw).astype(np.int64)
    lo = np.minimum(counts, floor)
    take = np.minimum(np.maximum(take, lo), counts)
    rem = total - take.sum()
    for q in np.argsort(-take):                       # floors pushed us over the target: trim the largest strata
        if rem >= 0:
            break
        cut = int(min(-rem, take[q] - lo[q]))
        take[q] -= cut; rem += cut
    for q in np.argsort(-(raw - np.floor(raw))):      # largest remainder fills what is left
        if rem <= 0:
            break
        if take[q] < counts[q]:
            take[q] += 1; rem -= 1
    assert take.sum() == total and (take <= counts).all(), (take, counts, total)
    return take


@torch.no_grad()
def run_build(tok, dev="cuda:0", seed=2026, heldout_frac=0.10, n_eval_single=512, n_eval_pair=256, k_single=8, k_pair=4,
              w_lo=16, w_hi=32, min_tok=8, min_c=10, min_lift=10.0, max_p=1e-10, check_mix=True):
    T0 = time.time()
    for p in (f"{OUT}/bank_scan.npz", f"{OUT}/neuron_stats.npz", f"{OUT}/sel_windows.npz", f"{OUT}/down_proj_cols.f16",
              EVAL_CACHE_V1, f"{POOL_HELDOUT}/vecs.f32"):
        assert os.path.exists(p), f"missing {p}"
    assert not os.path.exists(f"{BANK_OUT}/build_stats.json"), f"{BANK_OUT} already built"
    assert not os.path.exists(EVAL_CACHE_V2), f"{EVAL_CACHE_V2} exists — refusing to overwrite an eval cache"
    sc = np.load(f"{OUT}/bank_scan.npz"); st = np.load(f"{OUT}/neuron_stats.npz"); sw = np.load(f"{OUT}/sel_windows.npz")
    N = int(st["N"]); T = int(sc["T"]); win_len = int(sc["win_len"]); n_windows = int(sc["n_windows"])
    ids_np = sw["ids"][:n_windows]; sel = sc["sel"]
    max_abs = st["max_abs"].astype(np.float64); pol = st["polarity"].astype(np.float32)
    freq_rel, band = sc["freq_rel"], sc["band"]
    cand_ids, sparse_ids = sc["cand_ids"], sc["sparse_ids"]
    n_fire = sc["n_fire"]
    cols = np.fromfile(f"{OUT}/down_proj_cols.f16", np.float16).reshape(N, D_MODEL).astype(np.float32)
    wnorm = np.linalg.norm(cols, axis=1)
    signed = cols / wnorm[:, None] * pol[:, None]                       # direction written when the neuron fires
    rng = np.random.default_rng(seed)
    wrng = np.random.default_rng(seed + 1)

    # ---- usable firing contexts per neuron (fire condition + enough tokens before the firing token) ----
    topv, topi = sc["topv"], sc["topi"]                                 # [K, N], sorted descending
    def contexts(i):
        v, g = topv[:, i], topi[:, i]
        ok = (g >= 0) & (v >= REL_THR * max_abs[i]) & ((g % win_len) >= min_tok - 1)
        return g[ok], v[ok]
    usable = np.array([i for i in cand_ids if len(contexts(i)[0]) > 0], np.int64)
    log(f"build: universe {len(cand_ids)} neurons (<{CAND_MAX:.0%}), {len(usable)} with >=1 usable firing context in the "
        f"{T}-token scan | bands(usable) {np.bincount(band[usable], minlength=5).tolist()}")

    # ---- hold-out split: heldout_frac per band (seeded) ----
    heldout, train = [], []
    for b in range(5):
        ids_b = usable[band[usable] == b]
        if len(ids_b) == 0:
            continue
        ids_b = rng.permutation(ids_b)
        n_ho = int(round(heldout_frac * len(ids_b)))
        heldout += ids_b[:n_ho].tolist(); train += ids_b[n_ho:].tolist()
    heldout = np.array(sorted(heldout), np.int64); train = np.array(sorted(train), np.int64)
    ho_set = set(heldout.tolist())
    # eval singles: proportional across bands (largest remainder), floor 8 where available
    ho_bands = [heldout[band[heldout] == b] for b in range(5)]
    take = _alloc([len(x) for x in ho_bands], n_eval_single, floor=8)
    eval_single = np.concatenate([np.sort(rng.choice(x, int(k), replace=False)) if k > 0 else np.zeros(0, np.int64)
                                  for x, k in zip(ho_bands, take)]).astype(np.int64)
    log(f"hold-out: {len(heldout)} neurons ({heldout_frac:.0%}, per band {[len(x) for x in ho_bands]}) | train {len(train)} | "
        f"eval singles {len(eval_single)} per band {take.tolist()}")

    # ---- co-firing pairs on the scan tokens (sparse x sparse) ----
    S = sparse_ids; C = sc["C_ss"].astype(np.int64)
    iu, ju = np.triu_indices(len(S), 1)
    c = C[iu, ju]
    nfS = n_fire[S].astype(np.float64)
    E = nfS[iu] * nfS[ju] / T
    lift = np.where(E > 0, c / np.maximum(E, 1e-12), 0.0)
    m = (c >= min_c) & (lift >= min_lift)
    pv = np.ones(len(c)); pv[m] = _poisson_sf(c[m], E[m])
    strong = m & (pv < max_p)
    pi, pj = S[iu[strong]], S[ju[strong]]
    pc, pl, pp = c[strong], lift[strong], pv[strong]
    log(f"pairs: sparse set {len(S)} | strong (C>={min_c}, lift>={min_lift}, p<{max_p}) {len(pi)} of {int(m.sum())} passing C/lift")
    # joint-firing tokens per strong pair from the sparse event list
    ev_tok, ev_nid, ev_val = sc["ev_tok"].astype(np.int64), sc["ev_nid"].astype(np.int64), sc["ev_val"].astype(np.float64)
    order = np.lexsort((ev_tok, ev_nid))
    ev_tok, ev_nid, ev_val = ev_tok[order], ev_nid[order], ev_val[order]
    bounds = np.searchsorted(ev_nid, np.arange(len(S) + 1))
    def events(sidx):
        lo, hi = bounds[sidx], bounds[sidx + 1]
        return ev_tok[lo:hi], ev_val[lo:hi]
    sidx_of = {int(n): k for k, n in enumerate(S.tolist())}
    usable_set = set(usable.tolist())
    pairs = []
    for q in range(len(pi)):
        i, j = int(pi[q]), int(pj[q])
        if i not in usable_set or j not in usable_set:
            continue
        ti, vi = events(sidx_of[i]); tj, vj = events(sidx_of[j])
        joint, ii, jj = np.intersect1d(ti, tj, assume_unique=True, return_indices=True)
        assert len(joint) == pc[q], (len(joint), pc[q])
        a_i, a_j = vi[ii], vj[jj]                                        # polarity-signed activations at the joint tokens
        pairs.append({"i": i, "j": j, "C": int(pc[q]), "lift": float(pl[q]), "p": float(pp[q]), "joint": joint, "a_i": a_i, "a_j": a_j,
                      "a_i_co": float(a_i.mean()), "a_j_co": float(a_j.mean()), "usable_pos": (joint % win_len) >= min_tok - 1,
                      "n_heldout": int(i in ho_set) + int(j in ho_set), "band_min": int(min(band[i], band[j]))})
    ho_pairs = [p for p in pairs if p["n_heldout"] > 0]
    tr_pairs = [p for p in pairs if p["n_heldout"] == 0 and p["usable_pos"].any()]
    # eval pairs: proportional across the min-member band
    hb = [[p for p in ho_pairs if p["band_min"] == b] for b in range(5)]
    take_p = _alloc([len(x) for x in hb], n_eval_pair, floor=8)
    eval_pairs = []
    for x, k in zip(hb, take_p):
        if k > 0:
            eval_pairs += [x[q] for q in np.sort(rng.choice(len(x), int(k), replace=False))]
    log(f"pairs: usable {len(pairs)} | touching a held-out neuron {len(ho_pairs)} (per band {[len(x) for x in hb]}) | "
        f"both-train with a usable joint token {len(tr_pairs)} | eval pairs {len(eval_pairs)} per band {take_p.tolist()} "
        f"| eval pairs with 2 held-out members {sum(p['n_heldout'] == 2 for p in eval_pairs)}")

    # ---- eval directions ----
    def composite(i, j, a_i, a_j):
        d = a_i * wnorm[i] * signed[i] + a_j * wnorm[j] * signed[j]      # == a_raw_i * col_i + a_raw_j * col_j
        return d / np.linalg.norm(d)
    mlp_dirs = torch.from_numpy(signed[eval_single]).float()
    mlp_pair_dirs = torch.from_numpy(np.stack([composite(p["i"], p["j"], p["a_i_co"], p["a_j_co"]) for p in eval_pairs])).float()
    es = torch.load(EVAL_CACHE_V1, map_location="cpu", weights_only=False)
    for k in list(es):
        assert not k.startswith("mlp"), f"v1 cache already has {k}"
    v2 = dict(es)
    v2["mlp_dirs"] = F.normalize(mlp_dirs, dim=-1)
    v2["mlp_neuron"] = torch.as_tensor(eval_single, dtype=torch.long)[:, None]
    v2["mlp_polarity"] = torch.as_tensor(pol[eval_single], dtype=torch.float32)[:, None]
    v2["mlp_corpus_max"] = torch.as_tensor(max_abs[eval_single], dtype=torch.float32)[:, None]
    v2["mlp_pair_dirs"] = F.normalize(mlp_pair_dirs, dim=-1)
    v2["mlp_pair_neuron"] = torch.as_tensor([[p["i"], p["j"]] for p in eval_pairs], dtype=torch.long)
    v2["mlp_pair_polarity"] = torch.as_tensor([[pol[p["i"]], pol[p["j"]]] for p in eval_pairs], dtype=torch.float32)
    v2["mlp_pair_corpus_max"] = torch.as_tensor([[max_abs[p["i"]], max_abs[p["j"]]] for p in eval_pairs], dtype=torch.float32)
    meta2 = dict(es["meta"]); meta2["extra_families"] = list(EXTRA_FAMILIES)
    meta2["mlp42"] = {
        "source": OUT, "model": MODEL, "layer": READ_LAYER, "d_ff": N,
        "fire": f"polarity * a >= {REL_THR} * corpus max|a| (max over the 1.02M-token statistics pass)",
        "universe": f"neurons firing on < {CAND_MAX:.0%} of tokens (stats histogram) with >= 1 usable firing context in the scan",
        "scan": {"n_windows": n_windows, "win_len": win_len, "T": T, "windows": "first n_windows of /data/mlp42/sel_windows.npz (acts27b TRAIN rows, seed 0)"},
        "heldout_frac": heldout_frac, "seed": seed, "band_edges": BAND_EDGES, "band_names": BAND_NAMES,
        "n_universe": int(len(usable)), "n_train_neurons": int(len(train)), "n_heldout_neurons": int(len(heldout)),
        "heldout_neurons": heldout.tolist(), "train_neurons": train.tolist(),
        "mlp": {"n": int(len(eval_single)), "dir": "polarity * unit(down_proj[:, i])", "band": band[eval_single].tolist(),
                "freq_rel": freq_rel[eval_single].tolist(), "n_fire_scan": n_fire[eval_single].tolist(),
                "stratification": "proportional to held-out band sizes (floor 8)", "per_band": take.tolist()},
        "mlp_pair": {"n": int(len(eval_pairs)), "dir": "unit(a_i_co * col_i + a_j_co * col_j), a_*_co = mean raw activation over the pair's joint-firing tokens",
                     "pair_rule": f"both in the sparse set (<{SPARSE_MAX:.1%} & write >= p25), C >= {min_c}, lift >= {min_lift}, Poisson p < {max_p}, on the scan tokens",
                     "heldout_rule": "pair touches >= 1 held-out neuron", "n_heldout_pairs_available": int(len(ho_pairs)),
                     "n_heldout_members": [p["n_heldout"] for p in eval_pairs], "C": [p["C"] for p in eval_pairs],
                     "lift": [p["lift"] for p in eval_pairs], "acts_co": [[p["a_i_co"], p["a_j_co"]] for p in eval_pairs],
                     "band_min": [p["band_min"] for p in eval_pairs], "per_band": take_p.tolist()},
        "fireback_metric": "max over the last 5 kept tokens of polarity * a_i / corpus_max_i (pairs: min over the two members), best of bo",
        "bank": BANK_OUT, "created": time.time()}
    v2["meta"] = meta2
    # every OLD key must be identical
    for k in es:
        if k == "meta":
            continue
        if torch.is_tensor(es[k]):
            assert torch.equal(es[k], v2[k])
        else:
            assert es[k] == v2[k]
    for k in es["meta"]:
        assert es["meta"][k] == meta2[k], k
    tmp = EVAL_CACHE_V2 + ".tmp"
    torch.save(v2, tmp); os.replace(tmp, EVAL_CACHE_V2)
    log(f"eval cache v2 written -> {EVAL_CACHE_V2}: mlp {len(eval_single)} dirs, mlp_pair {len(eval_pairs)} dirs; cos_families unchanged "
        f"{meta2['cos_families']}; extra_families {meta2['extra_families']}")

    # ---- bank rows (train neurons / both-train pairs) ----
    def window(g):
        w, p = int(g // win_len), int(g % win_len)
        Wn = int(wrng.integers(w_lo, w_hi + 1)); start = max(0, p - Wn + 1)
        ids = ids_np[w, start:p + 1].tolist()
        return {"seq": int(sel[w]), "window": w, "pos": p, "start": start, "n_tok": len(ids), "fire_from_end": 0,
                "target_text": tok.decode(ids)}
    rows_dir, recs = [], []
    n_short_txt = 0
    for i in train.tolist():
        g, v = contexts(i)
        for r, (gg, vv) in enumerate(zip(g[:k_single].tolist(), v[:k_single].tolist())):
            rec = window(gg)
            if len(rec["target_text"].strip()) < 3:
                n_short_txt += 1; continue
            rec.update({"family": "mlp", "neuron": i, "polarity": float(pol[i]), "corpus_max": float(max_abs[i]), "act": float(vv),
                        "norm_act": float(vv / max_abs[i]), "window_rank": r, "freq_rel": float(freq_rel[i]), "band": BAND_NAMES[band[i]],
                        "n_fire_scan": int(n_fire[i])})
            recs.append(rec); rows_dir.append(signed[i])
    n_single_rows = len(recs)
    for p in tr_pairs:
        i, j = p["i"], p["j"]
        strength = np.minimum(p["a_i"] / max_abs[i], p["a_j"] / max_abs[j])
        strength = np.where(p["usable_pos"], strength, -np.inf)
        top = np.argsort(-strength)[:k_pair]
        for r, q in enumerate(top.tolist()):
            if not np.isfinite(strength[q]):
                break
            rec = window(int(p["joint"][q]))
            if len(rec["target_text"].strip()) < 3:
                n_short_txt += 1; continue
            a_i, a_j = float(p["a_i"][q]), float(p["a_j"][q])
            rec.update({"family": "mlp_pair", "neurons": [i, j], "polarity": [float(pol[i]), float(pol[j])],
                        "corpus_max": [float(max_abs[i]), float(max_abs[j])], "acts": [a_i, a_j],
                        "norm_acts": [a_i / float(max_abs[i]), a_j / float(max_abs[j])], "acts_co": [p["a_i_co"], p["a_j_co"]],
                        "C": p["C"], "lift": p["lift"], "window_rank": r, "band_min": BAND_NAMES[p["band_min"]]})
            recs.append(rec); rows_dir.append(composite(i, j, a_i, a_j))
    n_pair_rows = len(recs) - n_single_rows
    X = torch.from_numpy(np.stack(rows_dir).astype(np.float32))
    X = F.normalize(X, dim=-1)
    log(f"staged {len(recs)} rows: mlp {n_single_rows} (from {len(train)} neurons), mlp_pair {n_pair_rows} (from {len(tr_pairs)} pairs); "
        f"dropped short texts {n_short_txt} ({time.time() - T0:.0f}s)")

    # ---- leak check vs EVERY v2 eval direction family + pool_heldout rows; drop leakers ----
    ref_names = sorted(k[:-5] for k in v2 if k.endswith("_dirs"))
    refs = [F.normalize(v2[f"{n}_dirs"].float(), dim=-1) for n in ref_names]
    ho_n = os.path.getsize(f"{POOL_HELDOUT}/vecs.f32") // (4 * D_MODEL)
    ho = np.memmap(f"{POOL_HELDOUT}/vecs.f32", np.float32, "r", shape=(ho_n, D_MODEL))
    refs.append(F.normalize(torch.from_numpy(np.asarray(ho)).float(), dim=-1)); ref_names.append("pool_heldout(all)")
    ref = torch.cat(refs).to(dev)
    offs = np.cumsum([0] + [len(r) for r in refs])
    fam_arr = np.array([0 if r["family"] == "mlp" else 1 for r in recs])
    leak = np.full((2, len(ref_names)), -1.0)
    maxcos = np.zeros(len(recs), np.float32); argref = np.zeros(len(recs), np.int64)
    for c0 in range(0, len(recs), 8192):
        x = X[c0:c0 + 8192].to(dev)
        cos = x @ ref.T
        mx, am = cos.max(1)
        maxcos[c0:c0 + 8192] = mx.cpu().numpy(); argref[c0:c0 + 8192] = am.cpu().numpy()
        for ri in range(len(ref_names)):
            cm = cos[:, offs[ri]:offs[ri + 1]].max(1).values.cpu().numpy()
            for fi in range(2):
                mm = fam_arr[c0:c0 + 8192] == fi
                if mm.any():
                    leak[fi, ri] = max(leak[fi, ri], float(cm[mm].max()))
    bad = maxcos > LEAK_COS
    leak_examples = []
    for q in np.flatnonzero(bad)[:20].tolist():
        ri = int(np.searchsorted(offs, argref[q], side="right") - 1)
        leak_examples.append({"row": q, "family": recs[q]["family"], "ref": ref_names[ri], "cos": float(maxcos[q]),
                              "neuron": recs[q].get("neuron", recs[q].get("neurons"))})
    log(f"leak check: {int(bad.sum())} staged rows within cos>{LEAK_COS} of an eval/held-out direction -> dropped; examples {leak_examples[:5]}")
    for fi, f in enumerate(("mlp", "mlp_pair")):
        log(f"  max-cos {f:9s} " + " ".join(f"{n[:12]}={leak[fi, ri]:.3f}" for ri, n in enumerate(ref_names)))
    keep = ~bad
    recs = [r for r, k in zip(recs, keep) if k]; X = X[torch.from_numpy(keep)]
    worst = max(float((X[c0:c0 + 8192].to(dev) @ ref.T).max()) for c0 in range(0, len(recs), 8192))
    assert worst <= LEAK_COS, f"leak re-check failed: max cos {worst}"
    counts = {"mlp": sum(r["family"] == "mlp" for r in recs), "mlp_pair": sum(r["family"] == "mlp_pair" for r in recs)}

    # ---- optional: how close are the NEW eval dirs to any row of the existing training mix? (reported, not asserted) ----
    mix_max = None
    if check_mix and os.path.exists(f"{MIX_SRC}/vecs.f32"):
        t1 = time.time()
        NA = json.load(open(f"{MIX_SRC}/build_stats.json"))["n_examples"]
        A = np.memmap(f"{MIX_SRC}/vecs.f32", np.float32, "r", shape=(NA, D_MODEL))
        new_ref = torch.cat([v2["mlp_dirs"], v2["mlp_pair_dirs"]]).to(dev)
        best = torch.full((new_ref.shape[0],), -1.0, device=dev)
        for c0 in range(0, NA, 16384):
            x = torch.from_numpy(np.ascontiguousarray(A[c0:c0 + 16384])).to(dev)
            best = torch.maximum(best, (new_ref @ x.T).max(1).values)
        best = best.cpu().numpy()
        mix_max = {"mlp": {"max": float(best[:len(eval_single)].max()), "median": float(np.median(best[:len(eval_single)])),
                           "n_gt_0.9": int((best[:len(eval_single)] > 0.9).sum()), "n_gt_0.999": int((best[:len(eval_single)] > LEAK_COS).sum())},
                   "mlp_pair": {"max": float(best[len(eval_single):].max()), "median": float(np.median(best[len(eval_single):])),
                                "n_gt_0.9": int((best[len(eval_single):] > 0.9).sum()), "n_gt_0.999": int((best[len(eval_single):] > LEAK_COS).sum())}}
        log(f"new eval dirs vs {MIX_SRC} rows ({NA}): {json.dumps(mix_max)} ({time.time() - t1:.0f}s)")

    # ---- assemble (seeded shuffle; records.jsonl line i == vec_idx i) ----
    Nb = len(recs)
    perm = rng.permutation(Nb)
    os.makedirs(BANK_OUT, exist_ok=True)
    vecs = np.memmap(f"{BANK_OUT}/vecs.f32.tmp", np.float32, "w+", shape=(Nb, D_MODEL))
    Xn = X.numpy()
    for c0 in range(0, Nb, 8192):
        vecs[c0:c0 + 8192] = Xn[perm[c0:c0 + 8192]]
    vecs.flush(); del vecs
    os.replace(f"{BANK_OUT}/vecs.f32.tmp", f"{BANK_OUT}/vecs.f32")
    with open(f"{BANK_OUT}/records.jsonl.tmp", "w") as fh:
        for i, g in enumerate(perm.tolist()):
            fh.write(json.dumps({"vec_idx": i, **recs[g]}, ensure_ascii=False) + "\n")
    os.replace(f"{BANK_OUT}/records.jsonl.tmp", f"{BANK_OUT}/records.jsonl")
    assert os.path.getsize(f"{BANK_OUT}/vecs.f32") == Nb * D_MODEL * 4
    stats = {"kind": "layer-42 MLP neuron directions: mlp = polarity*unit(down_proj col) x top-8 firing windows (TRAIN neurons only); "
                     "mlp_pair = unit(a_i col_i + a_j col_j) at joint-firing tokens of strong both-sparse co-firing pairs (both-train pairs)",
             "n_examples": Nb, "n_vecs": Nb, "families": counts, "layout": "seeded shuffle of all rows (records.jsonl line i == vec_idx i)",
             "seed": seed, "d_model": D_MODEL, "model": MODEL, "layer": READ_LAYER, "source": OUT, "eval_cache_v2": EVAL_CACHE_V2,
             "created": time.time()}
    meta_out = {"bank": BANK_OUT, "n_rows": Nb, "families": counts, "seed": seed,
                "neurons": {"universe": int(len(usable)), "train": int(len(train)), "heldout": int(len(heldout)), "heldout_frac": heldout_frac,
                            "per_band_usable": np.bincount(band[usable], minlength=5).tolist(), "per_band_heldout": [len(x) for x in ho_bands],
                            "band_names": BAND_NAMES, "rows_per_neuron_max": k_single, "min_tok": min_tok, "window": [w_lo, w_hi]},
                "pairs": {"strong_total": int(len(pi)), "usable": int(len(pairs)), "heldout": int(len(ho_pairs)), "train_with_usable_token": int(len(tr_pairs)),
                          "rows_per_pair_max": k_pair, "rule": meta2["mlp42"]["mlp_pair"]["pair_rule"]},
                "eval": {"mlp": int(len(eval_single)), "mlp_pair": int(len(eval_pairs))},
                "leak_check": {"threshold": LEAK_COS, "dropped_rows": int(bad.sum()), "examples": leak_examples,
                               "max_cos_table": {f: {n: round(float(leak[fi, ri]), 4) for ri, n in enumerate(ref_names)} for fi, f in enumerate(("mlp", "mlp_pair"))}},
                "new_eval_dirs_vs_mix_1m_v2": mix_max, "dropped_short_text": n_short_txt, "scan": meta2["mlp42"]["scan"], "created": time.time()}
    json.dump(stats, open(f"{BANK_OUT}/build_stats.json", "w"), indent=1)
    json.dump(meta_out, open(f"{BANK_OUT}/meta.json", "w"), indent=1)
    json.dump({"train_neurons": train.tolist(), "heldout_neurons": heldout.tolist(), "eval_single": eval_single.tolist(),
               "eval_pairs": [[p["i"], p["j"]] for p in eval_pairs], "train_pairs": [[p["i"], p["j"]] for p in tr_pairs],
               "heldout_pairs": [[p["i"], p["j"]] for p in ho_pairs]}, open(f"{OUT}/bank_selection.json", "w"))
    log(f"DONE -> {BANK_OUT}: {Nb} rows {counts} | eval v2 {EVAL_CACHE_V2} | {(time.time() - T0) / 60:.1f} min")
    return {"bank": BANK_OUT, "n": Nb, "families": counts, "neurons": meta_out["neurons"], "pairs": meta_out["pairs"],
            "eval": meta_out["eval"], "leak_dropped": int(bad.sum()), "mix_check": mix_max}


# ----------------------------------------------------------------------------------------------------------------
# stage: merge (mix_1m_v2 + mlp42 -> mix_1m_mlp)
# ----------------------------------------------------------------------------------------------------------------
def run_merge(src_a=MIX_SRC, src_b=BANK_OUT, out=MIX_OUT, seed=17, stage_dir="/root/merge"):
    t0 = time.time()
    assert not os.path.exists(f"{out}/build_stats.json"), f"{out} already built"
    sa = json.load(open(f"{src_a}/build_stats.json")); sb = json.load(open(f"{src_b}/build_stats.json"))
    NA, NB = int(sa["n_examples"]), int(sb["n_examples"]); N = NA + NB
    assert os.path.getsize(f"{src_a}/vecs.f32") == NA * D_MODEL * 4 and os.path.getsize(f"{src_b}/vecs.f32") == NB * D_MODEL * 4
    log(f"merge: {src_a} ({NA}) + {src_b} ({NB}) -> {out} ({N})")
    recs = []
    for name, src, n in ((os.path.basename(src_a), src_a, NA), (os.path.basename(src_b), src_b, NB)):
        with open(f"{src}/records.jsonl") as fh:
            for i, line in enumerate(fh):
                r = json.loads(line)
                assert int(r["vec_idx"]) == i
                r["src_bank"] = name; r["src_vec_idx"] = i
                recs.append(r)
        assert len(recs) == (NA if name == os.path.basename(src_a) else N)
    log(f"records loaded ({time.time() - t0:.0f}s)")
    A = np.fromfile(f"{src_a}/vecs.f32", np.float32).reshape(NA, D_MODEL)
    B = np.fromfile(f"{src_b}/vecs.f32", np.float32).reshape(NB, D_MODEL)
    log(f"vectors loaded ({time.time() - t0:.0f}s)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)                      # output row i <- staged row order[i]
    os.makedirs(stage_dir, exist_ok=True)
    vecs = np.memmap(f"{stage_dir}/vecs.f32", np.float32, "w+", shape=(N, D_MODEL))
    for c0 in range(0, N, 16384):
        src = order[c0:c0 + 16384]
        chunk = np.empty((len(src), D_MODEL), np.float32)
        ma = src < NA
        chunk[ma] = A[src[ma]]; chunk[~ma] = B[src[~ma] - NA]
        vecs[c0:c0 + len(src)] = chunk
    vecs.flush(); del vecs, A, B
    fams = {}
    with open(f"{stage_dir}/records.jsonl", "w") as fh:
        for i, g in enumerate(order.tolist()):
            r = dict(recs[g]); r["vec_idx"] = i
            fams[r["family"]] = fams.get(r["family"], 0) + 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"staged {N} rows ({time.time() - t0:.0f}s): {fams}")
    os.makedirs(out, exist_ok=True)
    for fn in ("vecs.f32", "records.jsonl"):
        shutil.copyfile(f"{stage_dir}/{fn}", f"{out}/{fn}.tmp"); os.replace(f"{out}/{fn}.tmp", f"{out}/{fn}")
    assert os.path.getsize(f"{out}/vecs.f32") == N * D_MODEL * 4
    stats = {"kind": f"merge of {os.path.basename(src_a)} (realact/realact_long/sae/bsf/cluster) + {os.path.basename(src_b)} (mlp/mlp_pair): "
                     "all-families training mix + layer-42 MLP neuron directions",
             "n_examples": N, "n_vecs": N, "families": fams, "layout": "seeded shuffle of all rows (records.jsonl line i == vec_idx i; "
             "src_bank/src_vec_idx point back to the part)", "parts": {os.path.basename(src_a): {"path": src_a, "n_examples": NA, "families": sa["families"]},
                                                                      os.path.basename(src_b): {"path": src_b, "n_examples": NB, "families": sb["families"]}},
             "seed": seed, "d_model": D_MODEL, "model": MODEL, "created": time.time()}
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
    # spot-check: 64 random output rows must equal their source rows
    vecs = np.memmap(f"{out}/vecs.f32", np.float32, "r", shape=(N, D_MODEL))
    Am = np.memmap(f"{src_a}/vecs.f32", np.float32, "r", shape=(NA, D_MODEL)); Bm = np.memmap(f"{src_b}/vecs.f32", np.float32, "r", shape=(NB, D_MODEL))
    with open(f"{out}/records.jsonl") as fh:
        lines = [next(fh) for _ in range(2000)]
    for i in rng.choice(2000, 64, replace=False).tolist():
        r = json.loads(lines[i]); assert r["vec_idx"] == i
        src = Am[r["src_vec_idx"]] if r["src_bank"] == os.path.basename(src_a) else Bm[r["src_vec_idx"]]
        assert np.array_equal(np.asarray(vecs[i]), np.asarray(src)), i
    shutil.rmtree(stage_dir, ignore_errors=True)
    log(f"DONE -> {out}: {N} rows {fams} ({(time.time() - t0) / 60:.1f} min)")
    return {"out": out, "n_examples": N, "families": fams}
