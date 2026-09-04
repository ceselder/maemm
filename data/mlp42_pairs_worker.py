"""Sparse COMBINATIONS of layer-42 MLP neurons: co-firing pairs / triples, their composite write directions, SAE matching
and inverter verbalization (container side; launched by modal_mlp42_neurons.py::pairs).

Co-firing: on ~200k tokens (the first n_windows of the SAME seeded acts27b train windows as the single-neuron stage),
neuron n "fires" at a token when polarity_n * a_n >= rel_thr * max|a_n| (max from the 1.02M-token stage). For every pair
(i, j): C_ij = #tokens where both fire, E_ij = n_i n_j / T (independence), lift = C / E, Poisson tail p-value, and the
mean polarity-signed activation of each member on the co-firing tokens (S_ij / C_ij). STRONG pair = C >= 10 & lift >= 10
& p < 1e-10.

Composite write of a combination with activations a_k: unit( sum_k a_k * polarity_k * down_proj[:, k] ), variants
  "co"   a_k = mean activation on the co-firing tokens (co-firing pairs only)
  "typ"  a_k = the neuron's mean activation over its own firing tokens (defined for ANY neuron -> random pairs, triples)
  "unit" a_k = 1 / ||down_proj[:, k]||  (plain sum of unit directions)
Controls: random pairs (sparse x sparse and all x all), "token top-2" (the two most active neurons, by activation
relative to own max, at randomly chosen tokens, composite with their actual activations), and 3-hot analogues.

Outputs (/data/mlp42): pairs_cofire.npz (per-neuron fire stats, strong-pair list, sparse x sparse C/E block, token top-3
records), pairs_sae.npz (nearest-SAE cosines per set x variant + singles reference), verbalize_pairs.jsonl/.json.
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from mxf.config import READ_LAYER
from mxf.inject import get_layer
import mlp42_neurons_worker as W

OUT = W.OUT


def log(msg):
    W.log(msg)


def _poisson_sf(k, mu):
    """P(X >= k) for Poisson(mu), elementwise (scipy)."""
    from scipy.stats import poisson
    return poisson.sf(np.asarray(k, np.float64) - 1.0, np.asarray(mu, np.float64))


@torch.no_grad()
def cofire_pass(model, ids_np, max_abs, pol, rel_thr, batch, win_len, dev, n_tok_probe=40, seed=0):
    """Returns dict with C [N,N] (co-fire counts, fp32 on dev), S [N,N] (sum of polarity-signed activation of ROW neuron
    on tokens where both fire), n_fire [N], sum_a_fire [N], T, and token-top-3 probe records."""
    layer = get_layer(model, READ_LAYER); mlp = layer.mlp
    N = max_abs.shape[0]
    thr = (rel_thr * max_abs).to(dev)
    C = torch.zeros(N, N, device=dev); S = torch.zeros(N, N, device=dev)
    n_fire = torch.zeros(N, dtype=torch.float64, device=dev); sum_a = torch.zeros(N, dtype=torch.float64, device=dev)
    rng = np.random.default_rng(seed)
    probe_ids, probe_a, probe_tok = [], [], []
    n_windows = ids_np.shape[0]
    bos_col = np.full((batch, 1), W.BOS, np.int32)
    t0 = time.time()
    for b0 in range(0, n_windows, batch):
        b1 = min(b0 + batch, n_windows); B = b1 - b0
        ids = torch.from_numpy(np.concatenate([bos_col[:B], ids_np[b0:b1]], 1).astype(np.int64)).to(dev)
        a_bf, _ = W.forward_capture(model, ids, layer, mlp)
        a = a_bf[:, 1:].reshape(-1, N).float() * pol[None, :]                 # polarity-signed neuron values
        Fm = (a >= thr[None, :]).float()                                     # [tokens, N] firing indicator
        C.addmm_(Fm.T, Fm)
        S.addmm_((a * Fm).T, Fm)
        n_fire += Fm.sum(0, dtype=torch.float64); sum_a += (a * Fm).sum(0, dtype=torch.float64)
        # token probe: top-3 neurons by activation relative to own max at random tokens
        rows = rng.choice(a.shape[0], n_tok_probe, replace=False)
        rel = a[rows] / max_abs.to(dev)[None, :]
        v, i = rel.topk(3, dim=1)
        probe_ids.append(i.cpu().numpy()); probe_a.append(a[torch.as_tensor(rows, device=dev)[:, None], i].cpu().numpy())
        probe_tok.append(b0 * win_len + rows)
        del a, Fm, a_bf
        if (b0 // batch) % 10 == 0:
            log(f"cofire {b1}/{n_windows} windows ({b1 * win_len / max(time.time() - t0, 1e-6):.0f} tok/s)")
    T = n_windows * win_len
    return {"C": C, "S": S, "n_fire": n_fire, "sum_a": sum_a, "T": T,
            "probe_ids": np.concatenate(probe_ids), "probe_a": np.concatenate(probe_a), "probe_tok": np.concatenate(probe_tok)}


def select_strong(C, n_fire, T, min_c=10, min_lift=10.0, max_p=1e-10, cap=300_000):
    """Upper-triangular strong co-firing pairs. Returns (i, j, C, E, lift, p) numpy arrays."""
    N = C.shape[0]
    E = torch.outer(n_fire.float(), n_fire.float()) / T
    lift = C / E.clamp_min(1e-9)
    upper = torch.ones(N, N, dtype=torch.bool, device=C.device).triu(1)
    cand = upper & (C >= min_c) & (lift >= min_lift)
    idx = cand.nonzero()
    if idx.shape[0] > cap:
        top = lift[idx[:, 0], idx[:, 1]].topk(cap).indices
        idx = idx[top]
    i, j = idx[:, 0], idx[:, 1]
    c = C[i, j].cpu().numpy(); e = E[i, j].cpu().numpy(); lf = lift[i, j].cpu().numpy()
    p = _poisson_sf(c, e)
    keep = p < max_p
    del E, lift, upper, cand
    return i.cpu().numpy()[keep], j.cpu().numpy()[keep], c[keep], e[keep], lf[keep], p[keep]


@torch.no_grad()
def nn_sae(dirs, Wd, chunk=1024):
    """dirs [n, d] unit -> (max |cos| [n], argmax id [n], signed cos [n]) vs unit decoder rows Wd [F, d]."""
    out_v, out_i, out_s = [], [], []
    for c0 in range(0, dirs.shape[0], chunk):
        c = dirs[c0:c0 + chunk] @ Wd.T
        v, i = c.abs().max(1)
        out_v.append(v); out_i.append(i); out_s.append(c.gather(1, i[:, None])[:, 0])
    return torch.cat(out_v), torch.cat(out_i), torch.cat(out_s)


def composite(signed_unit, wnorm, members, acts, variant):
    """members [n,k] long, acts [n,k] (>0) -> unit composite dirs [n,d]. variant: 'co'/'typ' use acts * ||w||, 'unit' uses 1."""
    m = torch.as_tensor(members, dtype=torch.long, device=signed_unit.device)
    if variant == "unit":
        wts = torch.ones(m.shape, device=signed_unit.device)
    else:
        wts = torch.as_tensor(acts, dtype=torch.float32, device=signed_unit.device).clamp_min(1e-3) * wnorm[m]
    d = (signed_unit[m] * wts[:, :, None]).sum(1)
    return F.normalize(d, dim=1)


@torch.no_grad()
def run_pairs(model, tok, sae, sparse_ids, n_windows=800, win_len=256, batch=16, rel_thr=0.10, seed=0, dev="cuda:0",
              n_set=4096, n_ctrl=2048, n_tri=512):
    st = np.load(f"{OUT}/neuron_stats.npz"); sm = np.load(f"{OUT}/sae_match.npz"); sw = np.load(f"{OUT}/sel_windows.npz")
    N = int(st["N"])
    max_abs = torch.from_numpy(st["max_abs"].astype(np.float32)).to(dev)
    pol = torch.from_numpy(st["polarity"].astype(np.float32)).to(dev)
    single_nn = torch.from_numpy(sm["nn_dec_abs"].astype(np.float32)).to(dev); single_id = torch.from_numpy(sm["nn_dec_id"].astype(np.int64)).to(dev)
    ids_np = sw["ids"][:n_windows]
    assert ids_np.shape == (n_windows, win_len), ids_np.shape
    sparse = np.zeros(N, bool); sparse[np.asarray(sparse_ids, np.int64)] = True
    layer = get_layer(model, READ_LAYER); mlp = layer.mlp
    Wm = mlp.down_proj.weight
    wcols = Wm.T.float().contiguous(); wnorm = wcols.norm(dim=1); signed = wcols / wnorm[:, None] * pol[:, None]
    rng = np.random.default_rng(seed)
    log(f"pairs: {n_windows} windows x {win_len} = {n_windows * win_len} tokens, rel_thr {rel_thr}, sparse set {sparse.sum()}")

    # ---------------------------------------------------------------- co-firing pass
    R = cofire_pass(model, ids_np, max_abs, pol, rel_thr, batch, win_len, dev, seed=seed)
    C, S, n_fire, T = R["C"], R["S"], R["n_fire"], R["T"]
    f = (n_fire / T).float()
    a_typ = (R["sum_a"] / n_fire.clamp_min(1)).float()                       # mean activation on own firing tokens
    i, j, c, e, lf, p = select_strong(C, n_fire, T)
    both_sp = sparse[i] & sparse[j]; any_sp = sparse[i] | sparse[j]
    a_i_co = (S[torch.as_tensor(i, device=dev), torch.as_tensor(j, device=dev)] / torch.as_tensor(c, device=dev).clamp_min(1)).cpu().numpy()
    a_j_co = (S[torch.as_tensor(j, device=dev), torch.as_tensor(i, device=dev)] / torch.as_tensor(c, device=dev).clamp_min(1)).cpu().numpy()
    nf = n_fire.cpu().numpy()
    coact = c / np.minimum(nf[i], nf[j]).clip(min=1)                          # P(other fires | rarer one fires)
    # moderate pairs count (C>=10 & lift>=3) and the sparse x sparse block for local analysis
    E_full = torch.outer(n_fire.float(), n_fire.float()) / T
    upper = torch.ones(N, N, dtype=torch.bool, device=dev).triu(1)
    n_mod = int(((C >= 10) & (C / E_full.clamp_min(1e-9) >= 3) & upper).sum())
    n_c10 = int(((C >= 10) & upper).sum())
    sp_idx = torch.as_tensor(np.flatnonzero(sparse), device=dev)
    C_ss = C[sp_idx][:, sp_idx].cpu().numpy().astype(np.int32); E_ss = E_full[sp_idx][:, sp_idx].cpu().numpy().astype(np.float32)
    del E_full, upper
    log(f"strong pairs (C>=10, lift>=10, p<1e-10): {len(i)} | both-sparse {int(both_sp.sum())} | any-sparse {int(any_sp.sum())} | "
        f"C>=10 pairs {n_c10} | moderate (lift>=3) {n_mod} | fire-freq median {np.median(f.cpu().numpy()):.4g} | "
        f"neurons that never fire in this subset {int((nf == 0).sum())}")
    if len(i):
        log(f"strong pairs: lift median {np.median(lf):.1f} p90 {np.quantile(lf, .9):.1f} | coact median {np.median(coact):.3f} | C median {np.median(c):.0f}")
    np.savez(f"{OUT}/pairs_cofire.npz", i=i, j=j, C=c, E=e, lift=lf, p=p, coact=coact, a_i_co=a_i_co, a_j_co=a_j_co,
             both_sparse=both_sp, any_sparse=any_sp, n_fire=nf, freq=f.cpu().numpy(), a_typ=a_typ.cpu().numpy(), T=T,
             rel_thr=rel_thr, n_windows=n_windows, sparse_ids=np.flatnonzero(sparse), C_ss=C_ss, E_ss=E_ss,
             n_pairs_c10=n_c10, n_pairs_moderate=n_mod,
             probe_ids=R["probe_ids"], probe_a=R["probe_a"], probe_tok=R["probe_tok"])

    # ---------------------------------------------------------------- triangles among strong pairs (3-hot)
    adj = {}
    for a_, b_ in zip(i.tolist(), j.tolist()):
        adj.setdefault(a_, set()).add(b_); adj.setdefault(b_, set()).add(a_)
    tri = set()
    order = rng.permutation(len(i))
    for q in order:
        a_, b_ = int(i[q]), int(j[q])
        common = adj[a_] & adj[b_]
        for k_ in sorted(common)[:4]:
            tri.add(tuple(sorted((a_, b_, k_))))
        if len(tri) >= n_tri * 3:
            break
    tri = np.array(sorted(tri), np.int64).reshape(-1, 3)
    if len(tri) > n_tri:
        tri = tri[rng.choice(len(tri), n_tri, replace=False)]
    tri_sp = sparse[tri].all(1) if len(tri) else np.zeros(0, bool)
    log(f"3-hot: {len(tri)} co-firing triangles sampled ({int(tri_sp.sum())} all-sparse)")

    # ---------------------------------------------------------------- direction sets
    a_typ_np = a_typ.cpu().numpy()
    sets = {}
    if len(i):
        sel = rng.choice(len(i), min(n_set, len(i)), replace=False)
        sets["cofire_all"] = {"members": np.stack([i[sel], j[sel]], 1), "acts_co": np.stack([a_i_co[sel], a_j_co[sel]], 1), "meta": {"lift": lf[sel], "C": c[sel], "coact": coact[sel], "both_sparse": both_sp[sel]}}
        ss = np.flatnonzero(both_sp)
        if len(ss):
            sel2 = ss if len(ss) <= n_set else rng.choice(ss, n_set, replace=False)
            sets["cofire_sparse"] = {"members": np.stack([i[sel2], j[sel2]], 1), "acts_co": np.stack([a_i_co[sel2], a_j_co[sel2]], 1), "meta": {"lift": lf[sel2], "C": c[sel2], "coact": coact[sel2]}}
    sp_list = np.flatnonzero(sparse)
    rp = np.stack([rng.choice(sp_list, n_ctrl), rng.choice(sp_list, n_ctrl)], 1); rp = rp[rp[:, 0] != rp[:, 1]]
    sets["random_sparse"] = {"members": rp}
    ra = np.stack([rng.integers(0, N, n_ctrl), rng.integers(0, N, n_ctrl)], 1); ra = ra[ra[:, 0] != ra[:, 1]]
    sets["random_all"] = {"members": ra}
    pi, pa = R["probe_ids"], R["probe_a"]
    ok2 = pa[:, 1] > 0
    sets["token_top2"] = {"members": pi[ok2][:, :2], "acts_co": pa[ok2][:, :2]}
    if len(tri):
        sets["cofire_tri"] = {"members": tri}
    rt = np.stack([rng.integers(0, N, n_tri) for _ in range(3)], 1); rt = rt[(rt[:, 0] != rt[:, 1]) & (rt[:, 1] != rt[:, 2]) & (rt[:, 0] != rt[:, 2])]
    sets["random_tri"] = {"members": rt}
    ok3 = pa[:, 2] > 0
    sets["token_top3"] = {"members": pi[ok3][:, :3], "acts_co": pa[ok3][:, :3]}

    Wd = F.normalize(sae.W_dec, dim=1)
    out = {}
    for name, Sd in sets.items():
        mem = Sd["members"]
        k = mem.shape[1]
        variants = {"typ": a_typ_np[mem], "unit": np.ones(mem.shape, np.float32)}
        if "acts_co" in Sd:
            variants["co"] = Sd["acts_co"]
        for var, acts in variants.items():
            d = composite(signed, wnorm, mem, acts, var)
            v, fid, sc = nn_sae(d, Wd)
            out[f"{name}__{var}__nn_abs"] = v.cpu().numpy(); out[f"{name}__{var}__nn_id"] = fid.cpu().numpy(); out[f"{name}__{var}__nn_signed"] = sc.cpu().numpy()
        mt = torch.as_tensor(mem, device=dev)
        out[f"{name}__members"] = mem
        out[f"{name}__single_nn_abs"] = single_nn[mt].cpu().numpy()                      # [n,k]
        out[f"{name}__single_nn_id"] = single_id[mt].cpu().numpy()
        for key, val in Sd.get("meta", {}).items():
            out[f"{name}__meta_{key}"] = val
        ref = "co" if "acts_co" in Sd else "typ"
        pair_nn = out[f"{name}__{ref}__nn_abs"]; best_single = out[f"{name}__single_nn_abs"].max(1)
        same = (out[f"{name}__{ref}__nn_id"][:, None] == out[f"{name}__single_nn_id"]).any(1)
        log(f"SAE NN [{name}, k={k}, n={len(mem)}, {ref}]: composite mean {pair_nn.mean():.3f} median {np.median(pair_nn):.3f} | "
            f"best single mean {best_single.mean():.3f} median {np.median(best_single):.3f} | composite > best single {np.mean(pair_nn > best_single):.2%} | "
            f"composite NN feature == a member's NN feature {same.mean():.2%}")
    np.savez(f"{OUT}/pairs_sae.npz", **out)
    log("pairs_cofire.npz + pairs_sae.npz saved")
    return {"i": i, "j": j, "c": c, "lift": lf, "coact": coact, "a_i_co": a_i_co, "a_j_co": a_j_co, "both_sparse": both_sp, "any_sparse": any_sp,
            "tri": tri, "a_typ": a_typ_np, "signed": signed, "wnorm": wnorm, "max_abs": st["max_abs"], "polarity": st["polarity"], "sparse": sparse}


def build_verbalize_sets(Rp, n_pairs=128, n_random=64, n_single_pairs=32, n_tri=32, seed=0):
    """Direction sets for run_verbalize: co-firing pairs (composite with co-firing activations), random sparse pairs
    (typical activations), the members of the first n_single_pairs co-firing pairs as singles, co-firing triples."""
    rng = np.random.default_rng(seed)
    signed, wnorm, max_abs, pol, sparse = Rp["signed"], Rp["wnorm"], Rp["max_abs"], Rp["polarity"], Rp["sparse"]
    i, j = Rp["i"], Rp["j"]
    # prefer both-sparse strong pairs, then any-sparse, then any (report the mix)
    pools = [np.flatnonzero(Rp["both_sparse"]), np.flatnonzero(Rp["any_sparse"] & ~Rp["both_sparse"]), np.flatnonzero(~Rp["any_sparse"])]
    chosen = []
    for pool in pools:
        need = n_pairs - len(chosen)
        if need <= 0:
            break
        take = pool if len(pool) <= need else rng.choice(pool, need, replace=False)
        chosen.extend(take.tolist())
    chosen = np.array(chosen, np.int64)
    # rank chosen by lift so the strongest come first (singles sample = members of the first n_single_pairs)
    chosen = chosen[np.argsort(-Rp["lift"][chosen])]
    mem = np.stack([i[chosen], j[chosen]], 1); acts = np.stack([Rp["a_i_co"][chosen], Rp["a_j_co"][chosen]], 1)
    sets = []
    d = composite(signed, wnorm, mem, acts, "co").cpu()
    sets.append({"name": "pair_cofire", "ids": [int(q) for q in chosen], "dirs": d, "neuron": mem.tolist(),
                 "polarity": [[float(pol[a]), float(pol[b])] for a, b in mem], "ref_max": [[float(max_abs[a]), float(max_abs[b])] for a, b in mem],
                 "extra": {"lift": Rp["lift"][chosen].tolist(), "C": Rp["c"][chosen].tolist(), "coact": Rp["coact"][chosen].tolist(),
                           "both_sparse": Rp["both_sparse"][chosen].tolist(), "acts_co": acts.tolist()}})
    sp_list = np.flatnonzero(sparse)
    rp = np.stack([rng.choice(sp_list, n_random * 2), rng.choice(sp_list, n_random * 2)], 1); rp = rp[rp[:, 0] != rp[:, 1]][:n_random]
    d = composite(signed, wnorm, rp, Rp["a_typ"][rp], "typ").cpu()
    sets.append({"name": "pair_random", "ids": list(range(len(rp))), "dirs": d, "neuron": rp.tolist(),
                 "polarity": [[float(pol[a]), float(pol[b])] for a, b in rp], "ref_max": [[float(max_abs[a]), float(max_abs[b])] for a, b in rp]})
    singles = sorted(set(mem[:n_single_pairs].ravel().tolist()))
    sets.append({"name": "single_member", "ids": singles, "dirs": signed[torch.as_tensor(singles, device=signed.device)].cpu(), "neuron": singles,
                 "polarity": [float(pol[a]) for a in singles], "ref_max": [float(max_abs[a]) for a in singles]})
    tri = Rp["tri"]
    if len(tri):
        tsel = tri if len(tri) <= n_tri else tri[rng.choice(len(tri), n_tri, replace=False)]
        d = composite(signed, wnorm, tsel, Rp["a_typ"][tsel], "typ").cpu()
        sets.append({"name": "tri_cofire", "ids": list(range(len(tsel))), "dirs": d, "neuron": tsel.tolist(),
                     "polarity": [[float(pol[a]) for a in t] for t in tsel], "ref_max": [[float(max_abs[a]) for a in t] for t in tsel]})
    return sets


def save_verbalize_sets_meta(sets, path):
    meta = {}
    for S in sets:
        meta[S["name"]] = {"ids": [int(x) for x in S["ids"]], "neuron": S["neuron"], "extra": S.get("extra", {})}
    json.dump(meta, open(path, "w"))
