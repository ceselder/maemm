#!/usr/bin/env python3
"""Sparse COMBINATIONS of layer-42 MLP neurons (co-firing pairs / triples): figures + numbers for the report.

Reads /data/mlp42/{pairs_cofire.npz, pairs_sae.npz, verbalize_pairs.jsonl, verbalize_pairs_sets.json} (downloaded next to the
single-neuron dump) plus neuron_stats.npz / sel_windows.npz for contexts, and writes to the report folder:
    fig7_cofiring.{png,pdf}  fig8_pair_sae.{png,pdf}  fig9_pair_verbalization.{png,pdf}
    data/pairs_cofire.json  data/pairs_sae.json  data/pairs_verbalize.json  data/examples_pairs.json

Usage: python scripts/mlp42_pairs_analyze.py --raw <raw>/mlp42 --report <report> [--sae-examples <raw>/data/examples.parquet]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlp42_neurons_analyze import (C_ALL, C_DENSE, C_GRID, C_INK, C_MUTED, C_RANDOM, C_SPARSE, decode_ctx, ecdf,  # noqa: E402
                                   headline, qcurve, save, style, summ)

C_PAIR, C_TRI = "#e87ba4", "#008300"        # next categorical slots (magenta, green)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--sae-examples", default=None)
    ap.add_argument("--tag", default="pairs")
    a = ap.parse_args()
    raw, report = Path(a.raw), Path(a.report)
    pc = np.load(raw / "pairs_cofire.npz"); ps = np.load(raw / "pairs_sae.npz")
    st = np.load(raw / "neuron_stats.npz"); sw = np.load(raw / "sel_windows.npz"); sm = np.load(raw / "sae_match.npz")
    sets_meta = json.load(open(raw / f"verbalize_{a.tag}_sets.json"))
    N = int(st["N"]); T = int(pc["T"])
    i, j, c, lift, p, coact = pc["i"], pc["j"], pc["C"], pc["lift"], pc["p"], pc["coact"]
    both_sp, any_sp = pc["both_sparse"], pc["any_sparse"]
    sparse_ids = pc["sparse_ids"]; sparse = np.zeros(N, bool); sparse[sparse_ids] = True
    n_fire, freq = pc["n_fire"], pc["freq"]
    C_ss, E_ss = pc["C_ss"].astype(np.float64), pc["E_ss"].astype(np.float64)
    ns = len(sparse_ids)
    iu = np.triu_indices(ns, 1)
    css, ess = C_ss[iu], E_ss[iu]
    lift_ss = css / np.maximum(ess, 1e-9)
    n_sparse_pairs = len(css)
    # degree: strong partners per neuron
    deg = np.bincount(np.concatenate([i, j]), minlength=N)
    cof = {"description": f"co-firing on {T:,} tokens; fire = polarity*a >= 10% of own corpus max; strong pair = C>=10 & lift>=10 & Poisson p<1e-10",
           "T": T, "n_windows": int(pc["n_windows"]), "n_neurons": N, "n_sparse": int(ns),
           "n_strong": int(len(i)), "n_strong_both_sparse": int(both_sp.sum()), "n_strong_any_sparse": int(any_sp.sum()),
           "n_strong_no_sparse": int((~any_sp).sum()), "n_pairs_c_ge10": int(pc["n_pairs_c10"]), "n_pairs_moderate_lift_ge3": int(pc["n_pairs_moderate"]),
           "n_all_pairs": int(N * (N - 1) // 2), "n_sparse_pairs": int(n_sparse_pairs),
           "sparse_pairs_with_C_ge1": int((css >= 1).sum()), "sparse_pairs_with_C_ge10": int((css >= 10).sum()),
           "sparse_pairs_C_ge10_and_lift_ge10": int(((css >= 10) & (lift_ss >= 10)).sum()),
           "sparse_pairs_expected_C_median": float(np.median(ess)), "sparse_pairs_expected_C_p90": float(np.quantile(ess, .9)),
           "fire_freq_subset_median": float(np.median(freq)), "neurons_never_firing_in_subset": int((n_fire == 0).sum()),
           "sparse_neurons_never_firing_in_subset": int((n_fire[sparse_ids] == 0).sum()),
           "strong": {"lift": summ(lift), "C": summ(c), "coact": summ(coact), "neg_log10_p": summ(-np.log10(np.maximum(p, 1e-300))),
                      "both_sparse": {"lift": summ(lift[both_sp]), "C": summ(c[both_sp]), "coact": summ(coact[both_sp])},
                      "not_sparse": {"lift": summ(lift[~any_sp]), "C": summ(c[~any_sp]), "coact": summ(coact[~any_sp])}},
           "degree": {"sparse_neurons_with_ge1_strong_partner": int((deg[sparse_ids] > 0).sum()), "sparse_degree": summ(deg[sparse_ids]),
                      "all_neurons_with_ge1_strong_partner": int((deg > 0).sum()), "all_degree": summ(deg),
                      "sparse_degree_hist": np.bincount(np.minimum(deg[sparse_ids], 20), minlength=21).tolist(),
                      "all_degree_hist": np.bincount(np.minimum(deg, 20), minlength=21).tolist()},
           "lift_hist_sparse_pairs_C_ge1": {"log10_edges": np.linspace(-1, 4, 51).tolist(),
                                            "counts": np.histogram(np.log10(np.maximum(lift_ss[css >= 1], 1e-1)), bins=np.linspace(-1, 4, 51))[0].tolist()},
           "lift_hist_strong": {"log10_edges": np.linspace(0.9, 4.5, 37).tolist(), "counts": np.histogram(np.log10(lift), bins=np.linspace(0.9, 4.5, 37))[0].tolist()},
           "coact_hist_strong": {"edges": np.linspace(0, 1, 21).tolist(), "counts_both_sparse": np.histogram(coact[both_sp], bins=np.linspace(0, 1, 21))[0].tolist(),
                                 "counts_other": np.histogram(coact[~both_sp], bins=np.linspace(0, 1, 21))[0].tolist()}}
    json.dump(cof, open(report / "data" / "pairs_cofire.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in cof.items() if not isinstance(v, dict)}, indent=0))
    print("strong lift", cof["strong"]["lift"]["median"], "coact", cof["strong"]["coact"]["median"], "degree sparse", cof["degree"]["sparse_degree"]["median"])

    # ---- fig7: co-firing statistics
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    hb = np.linspace(-1, 4, 51)
    ax.hist(np.log10(np.maximum(lift_ss[css >= 1], 1e-1)), bins=hb, color=C_SPARSE, alpha=0.8, lw=0, label=f"sparse×sparse pairs with ≥1 co-firing (n={int((css >= 1).sum()):,})")
    ax.hist(np.log10(np.maximum(lift_ss[css >= 10], 1e-1)), bins=hb, color=C_INK, alpha=0.6, lw=0, label=f"… with ≥10 co-firings (n={int((css >= 10).sum()):,})")
    ax.axvline(1, color=C_MUTED, lw=1); ax.text(1.05, 3, "lift = 10", fontsize=8, color=C_MUTED)
    ax.set_yscale("log"); ax.set_xlabel("log10 lift = observed / expected co-firings", fontsize=9); ax.set_ylabel("pairs", fontsize=9)
    ax.set_title(f"Lift among the {n_sparse_pairs:,} sparse×sparse pairs", fontsize=9.5); ax.legend(frameon=False, fontsize=7.5, loc="lower left"); style(ax)
    ax = axs[1]
    dh = np.arange(0, 21)
    ax.bar(dh - 0.2, np.bincount(np.minimum(deg[sparse_ids], 20), minlength=21) / ns, width=0.4, color=C_SPARSE, lw=0, label=f"sparse neurons (n={ns:,})")
    ax.bar(dh + 0.2, np.bincount(np.minimum(deg, 20), minlength=21) / N, width=0.4, color=C_ALL, lw=0, label=f"all neurons (n={N:,})")
    ax.set_xlabel("number of strong co-firing partners (20 = ≥20)", fontsize=9); ax.set_ylabel("fraction of neurons", fontsize=9); ax.set_yscale("log")
    ax.set_title(f"{cof['degree']['sparse_neurons_with_ge1_strong_partner']:,} of {ns:,} sparse neurons have ≥1 strong partner", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8); style(ax)
    ax = axs[2]
    eb = np.linspace(0, 1, 21)
    ax.hist(coact[~both_sp], bins=eb, color=C_ALL, alpha=0.6, lw=0, density=True, label=f"strong pairs, not both sparse (n={int((~both_sp).sum()):,})")
    ax.hist(coact[both_sp], bins=eb, color=C_SPARSE, alpha=0.7, lw=0, density=True, label=f"strong pairs, both sparse (n={int(both_sp.sum()):,})")
    ax.set_xlabel("co-activation strength = P(other fires | rarer member fires)", fontsize=9); ax.set_ylabel("density", fontsize=9)
    ax.set_title(f"Typical strength: median {np.median(coact[both_sp]) if both_sp.any() else float('nan'):.2f} (both sparse), {np.median(coact):.2f} (all strong)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8); style(ax)
    headline(fig, f"Co-firing is rare but real: {len(i):,} strongly co-firing neuron pairs (lift ≥ 10, ≥10 joint firings, p < 1e-10) on {T / 1e3:.0f}k tokens, "
                  f"{int(both_sp.sum()):,} of them within the sparse set",
             "fire = polarity·a ≥ 10% of the neuron's own corpus max; expected co-firings = n_i n_j / T (independence); sparse set = the 2,820 rare neurons of the single-neuron analysis")
    fig.tight_layout(); save(fig, report, "fig7_cofiring")

    # ---- SAE matching of composite directions
    order = [("cofire_sparse", "co"), ("cofire_all", "co"), ("random_sparse", "typ"), ("random_all", "typ"), ("token_top2", "co"),
             ("cofire_tri", "typ"), ("random_tri", "typ"), ("token_top3", "co")]
    labels = {"cofire_sparse": "co-firing pairs, both sparse", "cofire_all": "co-firing pairs, all", "random_sparse": "random pairs, sparse×sparse",
              "random_all": "random pairs, all", "token_top2": "top-2 neurons at random tokens", "cofire_tri": "co-firing triples",
              "random_tri": "random triples", "token_top3": "top-3 neurons at random tokens"}
    sae = {"description": "nearest SAE decoder |cos| of composite directions unit(sum_k a_k polarity_k down_proj[:,k]); variants co (co-firing-token mean "
                          "activations), typ (member's own typical firing activation), unit (unit-direction sum); 'best single' = max over members of the "
                          "member's own nearest-SAE |cos|", "sets": {}}
    for name, ref in order:
        if f"{name}__members" not in ps:
            continue
        mem = ps[f"{name}__members"]; sn = ps[f"{name}__single_nn_abs"]; sid = ps[f"{name}__single_nn_id"]
        d = {"n": int(len(mem)), "k": int(mem.shape[1]), "ref_variant": ref, "best_single": summ(sn.max(1)), "mean_single": summ(sn.mean(1)),
             "worst_single": summ(sn.min(1)), "q_best_single": qcurve(sn.max(1))}
        for var in ("co", "typ", "unit"):
            key = f"{name}__{var}__nn_abs"
            if key in ps:
                v = ps[key]; fid = ps[f"{name}__{var}__nn_id"]
                d[var] = {"nn_abs": summ(v), "frac_gt_0.5": float((v > .5).mean()), "frac_gt_0.3": float((v > .3).mean()),
                          "frac_composite_gt_best_single": float((v > sn.max(1)).mean()), "frac_composite_gt_best_single_by_0.05": float((v > sn.max(1) + 0.05).mean()),
                          "frac_nn_feature_is_a_members_nn": float((fid[:, None] == sid).any(1).mean()), "q_nn_abs": qcurve(v),
                          "delta_vs_best_single": summ(v - sn.max(1))}
        for mk in ("lift", "C", "coact", "both_sparse"):
            if f"{name}__meta_{mk}" in ps:
                d[f"meta_{mk}"] = summ(ps[f"{name}__meta_{mk}"].astype(np.float64))
        sae["sets"][name] = d
        print(name, "n", d["n"], "composite", ref, round(d[ref]["nn_abs"]["mean"], 3), round(d[ref]["nn_abs"]["median"], 3), "| best single",
              round(d["best_single"]["mean"], 3), round(d["best_single"]["median"], 3), "| unit", round(d["unit"]["nn_abs"]["mean"], 3), round(d["unit"]["nn_abs"]["median"], 3),
              "| frac comp>best", round(d[ref]["frac_composite_gt_best_single"], 3), "| same feat", round(d[ref]["frac_nn_feature_is_a_members_nn"], 3))
    # singles reference from the single-neuron stage
    nn1 = sm["nn_dec_abs"].astype(np.float64)
    sae["singles_reference"] = {"sparse": summ(nn1[sparse]), "all": summ(nn1), "random_unit_vectors": summ(sm["rand_dec_abs"].astype(np.float64))}
    json.dump(sae, open(report / "data" / "pairs_sae.json", "w"), indent=1)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axs[0]
    rows = [("single sparse neurons", nn1[sparse], C_SPARSE)]
    for name, ref in order:
        if name in sae["sets"]:
            rows.append((labels[name] + f" [{ref}]", ps[f"{name}__{ref}__nn_abs"].astype(np.float64),
                         C_PAIR if name.startswith("cofire") and "tri" not in name else C_TRI if "tri" in name else C_RANDOM if "random" in name else C_DENSE))
    rows.append(("random unit vectors", sm["rand_dec_abs"].astype(np.float64), C_GRID))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, col) in zip(y, rows):
        ax.barh(yy, v.mean(), color=col, height=0.62, lw=0)
        ax.plot([np.median(v)], [yy], marker="|", ms=14, color=C_INK, mew=1.8)
        ax.text(v.mean() + 0.005, yy, f"mean {v.mean():.3f} · median {np.median(v):.3f} · n={len(v):,}", va="center", fontsize=7.5, color=C_INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.set_xlim(0, 0.62)
    ax.set_xlabel("nearest SAE decoder |cos| (bar = mean, tick = median)", fontsize=9); ax.set_title("Composite directions vs singles vs controls", fontsize=9.5); style(ax)
    ax = axs[1]
    series = [(nn1[sparse], C_SPARSE, "single sparse neurons")]
    if "cofire_sparse" in sae["sets"]:
        series.append((ps["cofire_sparse__co__nn_abs"], C_PAIR, "co-firing sparse pairs (co-firing acts)"))
        series.append((ps["cofire_sparse__unit__nn_abs"], C_DENSE, "co-firing sparse pairs (unit sum)"))
    series += [(ps["random_sparse__typ__nn_abs"], C_RANDOM, "random sparse pairs"), (ps["token_top2__co__nn_abs"], C_ALL, "top-2 neurons at random tokens")]
    if "cofire_tri" in sae["sets"]:
        series.append((ps["cofire_tri__typ__nn_abs"], C_TRI, "co-firing triples"))
    ecdf(ax, series, "nearest SAE decoder |cos|", "ECDFs")
    ax = axs[2]
    key = "cofire_sparse" if "cofire_sparse" in sae["sets"] else "cofire_all"
    if key in sae["sets"]:
        bs = ps[f"{key}__single_nn_abs"].max(1); pv = ps[f"{key}__co__nn_abs"]
        ax.scatter(bs, pv, s=8, color=C_PAIR, alpha=0.6, lw=0, label=labels[key])
        rs = ps["random_sparse__single_nn_abs"].max(1); rv = ps["random_sparse__typ__nn_abs"]
        ax.scatter(rs, rv, s=6, color=C_RANDOM, alpha=0.5, lw=0, label="random sparse pairs")
        ax.plot([0, 0.95], [0, 0.95], color=C_MUTED, lw=0.8)
        ax.set_xlabel("best member's own nearest-SAE |cos|", fontsize=9); ax.set_ylabel("composite pair direction's nearest-SAE |cos|", fontsize=9)
        ax.set_title(f"Pair beats its best member in {100 * np.mean(pv > bs):.0f}% of co-firing pairs (random pairs: {100 * np.mean(rv > rs):.0f}%)", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, markerscale=2); style(ax)
    d0 = sae["sets"].get("cofire_sparse", sae["sets"].get("cofire_all"))
    headline(fig, f"Summed writes of co-firing pairs are NOT closer to SAE features than their best single member "
                  f"(mean {d0['co']['nn_abs']['mean']:.3f} vs {d0['best_single']['mean']:.3f}; median {d0['co']['nn_abs']['median']:.3f} vs {d0['best_single']['median']:.3f}) — "
                  f"and only slightly closer than random sparse pairs ({sae['sets']['random_sparse']['typ']['nn_abs']['mean']:.3f} / {sae['sets']['random_sparse']['typ']['nn_abs']['median']:.3f})" if d0 else "composite directions",
             "composite = unit(Σ a_k·polarity_k·down_proj[:,k]) with a_k = mean activation on the co-firing tokens (or the member's typical firing activation for controls); SAE = 131k-feature layer-42 dictionary")
    fig.tight_layout(); save(fig, report, "fig8_pair_sae")

    # ---- verbalization of pairs
    vpath = raw / f"verbalize_{a.tag}.jsonl"
    verb = None
    if os.path.exists(vpath):
        recs = [json.loads(l) for l in open(vpath)]
        by = {}
        for r in recs:
            by.setdefault((r["set"], r["row"]), []).append(r)
        verb = {"source": str(vpath), "sets": {}}
        for sname in ("pair_cofire", "pair_random", "single_member", "tri_cofire"):
            groups = [v for (s, _), v in by.items() if s == sname]
            if not groups:
                continue
            def bo(key):
                return np.array([max(x.get(key, np.nan) for x in g) for g in groups], np.float64)
            k = len(groups[0][0].get("members", [0]))
            d = {"n_dirs": len(groups), "k": k, "cos_last5_bo": summ(bo("cos_last5")), "cos_all_bo": summ(bo("cos_all")), "cosc_all_bo": summ(bo("cosc_all")),
                 "cos_all_mean_single": float(np.mean([x["cos_all"] for g in groups for x in g])), "q_cos_all_bo": qcurve(bo("cos_all"))}
            na = bo("norm_act")          # weakest member (all members fire)
            d.update({"all_norm_act_bo": summ(na), "all_fired10_bo": float(np.mean(na > .1)), "all_fired25_bo": float(np.mean(na > .25)),
                      "all_fired50_bo": float(np.mean(na > .5)), "all_beat_corpus_bo": float(np.mean(na > 1.0))})
            if k > 1:
                nx = bo("norm_act_max")
                d.update({"any_norm_act_bo": summ(nx), "any_fired10_bo": float(np.mean(nx > .1)), "any_fired25_bo": float(np.mean(nx > .25)), "any_fired50_bo": float(np.mean(nx > .5))})
                # per-member fire-back (best of 4 per member)
                per = np.array([[max(x["norm_act_members"][m] for x in g) for m in range(k)] for g in groups])
                d["per_member_fired10_bo"] = float((per > .1).mean()); d["per_member_fired25_bo"] = float((per > .25).mean()); d["per_member_fired50_bo"] = float((per > .5).mean())
            verb["sets"][sname] = d
            print(sname, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in d.items() if not isinstance(vv, (dict, list))},
                  {kk: round(vv["mean"], 3) for kk, vv in d.items() if isinstance(vv, dict) and "mean" in vv})
        # chance for pairs: both members cross 10% in a random 48-token span (independent tokens, from subset frequencies)
        L = 48.0
        for sname, memlist in (("pair_cofire", sets_meta["pair_cofire"]["neuron"]), ("pair_random", sets_meta["pair_random"]["neuron"])):
            if sname in verb["sets"]:
                fr = freq[np.array(memlist)]                                   # [n,2] fire freq (10% of max) in the subset
                p1 = 1 - (1 - fr) ** L                                          # each member alone
                both = (p1[:, 0] * p1[:, 1]); both_bo = 1 - (1 - both) ** 4
                verb["sets"][sname]["chance_all_fired10_best_of_4_independent"] = float(both_bo.mean())
                verb["sets"][sname]["chance_any_fired10_best_of_4_independent"] = float((1 - (1 - (1 - (1 - p1[:, 0]) * (1 - p1[:, 1]))) ** 4).mean())
        verb["sets"]["pair_cofire"]["members_both_sparse_frac"] = float(np.mean(sets_meta["pair_cofire"]["extra"]["both_sparse"]))
        verb["sets"]["pair_cofire"]["lift"] = summ(np.array(sets_meta["pair_cofire"]["extra"]["lift"]))
        verb["sets"]["pair_cofire"]["coact"] = summ(np.array(sets_meta["pair_cofire"]["extra"]["coact"]))
        json.dump(verb, open(report / "data" / "pairs_verbalize.json", "w"), indent=1)

        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
        names = [s for s in ("pair_cofire", "pair_random", "single_member", "tri_cofire") if s in verb["sets"]]
        vl = {"pair_cofire": "co-firing\npairs", "pair_random": "random\npairs", "single_member": "single\nmembers", "tri_cofire": "co-firing\ntriples"}
        vc = {"pair_cofire": C_PAIR, "pair_random": C_RANDOM, "single_member": C_SPARSE, "tri_cofire": C_TRI}
        ax = axs[0]
        vals = [verb["sets"][s]["cos_all_bo"]["mean"] for s in names]
        ax.bar(range(len(names)), vals, color=[vc[s] for s in names], width=0.6, lw=0)
        for q_, v in enumerate(vals):
            ax.text(q_, v + 0.002, f"{v:.3f}", ha="center", fontsize=8, color=C_INK)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([vl[s] for s in names], fontsize=8); ax.set_ylim(0, max(vals) * 1.25)
        ax.set_ylabel("best-of-4 max cos(composite dir, clean L42 act) over all tokens", fontsize=8.5); ax.set_title("Cosine reproduced by the inverter", fontsize=9.5); style(ax)
        ax = axs[1]
        keys = [("all_fired10_bo", "all members\n≥10%"), ("all_fired25_bo", "all\n≥25%"), ("all_fired50_bo", "all\n≥50%"), ("any_fired10_bo", "any member\n≥10%"), ("any_fired50_bo", "any\n≥50%")]
        w = 0.8 / len(names)
        for jn, s in enumerate(names):
            vals = [verb["sets"][s].get(k_, verb["sets"][s].get(k_.replace("any_", "all_"), np.nan)) for k_, _ in keys]
            ax.bar(np.arange(len(keys)) + (jn - (len(names) - 1) / 2) * w, vals, width=w - 0.03, color=vc[s], lw=0, label=vl[s].replace("\n", " "))
            for q_, v in enumerate(vals):
                if v == v:
                    ax.text(q_ + (jn - (len(names) - 1) / 2) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, color=C_INK)
        if "chance_all_fired10_best_of_4_independent" in verb["sets"]["pair_cofire"]:
            ax.scatter([0 + (0 - (len(names) - 1) / 2) * w], [verb["sets"]["pair_cofire"]["chance_all_fired10_best_of_4_independent"]], marker="_", s=200, color=C_INK, zorder=5, label="chance (independent members, random span)")
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([k for _, k in keys], fontsize=8); ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of directions (best of 4 generations)", fontsize=8.5); ax.set_title("Fire-back of the member neurons on the generated text", fontsize=9.5)
        ax.legend(frameon=False, fontsize=7); style(ax)
        ax = axs[2]
        gp = [v for (s, _), v in by.items() if s == "pair_cofire"]
        m1 = np.array([max(x["norm_act_members"][0] for x in g) for g in gp]); m2 = np.array([max(x["norm_act_members"][1] for x in g) for g in gp])
        ax.scatter(np.clip(m1, 5e-3, None), np.clip(m2, 5e-3, None), s=10, color=C_PAIR, alpha=0.7, lw=0)
        ax.axhline(0.1, color=C_MUTED, lw=0.8); ax.axvline(0.1, color=C_MUTED, lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("member 1 fire-back (value / corpus max, best over its 4 samples; clipped at 0.005)", fontsize=8.5); ax.set_ylabel("member 2 fire-back (best over its 4 samples)", fontsize=8.5)
        ax.set_title(f"Per pair (each member's own best sample): both ≥10% in {100 * np.mean((m1 > .1) & (m2 > .1)):.0f}%, one in {100 * np.mean((m1 > .1) ^ (m2 > .1)):.0f}%, neither in {100 * np.mean((m1 <= .1) & (m2 <= .1)):.0f}%", fontsize=9)
        style(ax)
        vp, vr_, vsg = verb["sets"]["pair_cofire"], verb["sets"].get("pair_random", {}), verb["sets"].get("single_member", {})
        headline(fig, f"The inverter fires BOTH neurons of a co-firing pair {100 * vp['all_fired10_bo']:.0f}% of the time (≥10% of each corpus max; random pairs "
                      f"{100 * vr_.get('all_fired10_bo', float('nan')):.0f}%, chance {100 * vp.get('chance_all_fired10_best_of_4_independent', float('nan')):.0f}%); "
                      f"single members alone: {100 * vsg.get('all_fired10_bo', float('nan')):.0f}%",
                 f"RL-A adapter, composite direction injected norm-matched at layer 1, T=1, 16–48 tokens, 4 samples/direction; {len(gp)} co-firing pairs "
                 f"({100 * verb['sets']['pair_cofire']['members_both_sparse_frac']:.0f}% both-sparse), {vr_.get('n_dirs', 0)} random sparse pairs, {vsg.get('n_dirs', 0)} member singles")
        fig.tight_layout(); save(fig, report, "fig9_pair_verbalization")

    # ---- examples: 8 co-firing pairs (top-4 by both-fire, 2 median, 2 bottom) with member contexts + generations
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    ids_all = sw["ids"]; win_len = ids_all.shape[1]; topv, topi = st["topv"], st["topi"]; pol = st["polarity"]
    sae_ex = None
    if a.sae_examples and os.path.exists(a.sae_examples):
        import pandas as pd
        df = pd.read_parquet(a.sae_examples); sae_ex = df[df["rank"] == 0].set_index("feature")["window"].to_dict()

    def ctx(n, k=2):
        out = []
        for q_ in range(k):
            g = int(topi[q_, n]); wnd, pos = g // win_len, g % win_len
            out.append({"value": float(pol[n] * topv[q_, n]), "text": decode_ctx(tok, ids_all[wnd], pos)})
        return out

    examples = []
    if verb:
        gp = {row: v for (s, row), v in by.items() if s == "pair_cofire"}
        rows_sorted = sorted(gp, key=lambda r: -min(max(x["norm_act_members"][m] for x in gp[r]) for m in range(2)))
        pick = rows_sorted[:4] + rows_sorted[len(rows_sorted) // 2 - 1: len(rows_sorted) // 2 + 1] + rows_sorted[-2:]
        kinds = ["top-4 by both-fire"] * 4 + ["median"] * 2 + ["bottom-2"] * 2
        ex_meta = sets_meta["pair_cofire"]
        for r, kind in zip(pick, kinds):
            mem = ex_meta["neuron"][r]; ex = ex_meta["extra"]
            gens = sorted(gp[r], key=lambda x: -x["norm_act"])
            # nearest SAE feature of the composite (from pairs_sae if this pair is in cofire_all/cofire_sparse — else recompute is not possible locally; use members')
            examples.append({"pick": kind, "members": mem, "lift": ex["lift"][r], "C": ex["C"][r], "coact": ex["coact"][r], "both_sparse": ex["both_sparse"][r],
                             "acts_co": ex["acts_co"][r],
                             "member_info": [{"neuron": int(n), "freq_subset": float(freq[n]), "max_abs": float(st["max_abs"][n]), "sparse": bool(sparse[n]),
                                              "nn_sae": {"feature": int(sm["nn_dec_id"][n]), "cos": float(sm["nn_dec_cos"][n]), "example": sae_ex.get(int(sm["nn_dec_id"][n])) if sae_ex else None},
                                              "contexts": ctx(int(n))} for n in mem],
                             "generations": [{"text": g["text"], "cos_all": g["cos_all"], "norm_act_members": g["norm_act_members"]} for g in gens[:3]]})
    json.dump({"examples": examples}, open(report / "data" / "examples_pairs.json", "w"), indent=1, ensure_ascii=False)
    print("done")


if __name__ == "__main__":
    main()
