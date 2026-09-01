"""Compare snippet-locality between TWO adapters' ON-POLICY rollout builds (the
locality_<name>.json files written by `snippet_locality.py build --adapter ...`), sharing one
testbed (same held-out features, same real max-act examples, same gen seed).

Per arm (firing rollouts only, peak act > fire): pooled + per-feature means of the locality
metrics, firing fraction, and the paired-vs-real diff. Across arms: PAIRED per-feature diff
(arm A - arm B) over features with >=1 firing rollout in BOTH arms — the apples-to-apples
"did the last-5 reward fix the smearing" test (headline metric: win5_share).

    python MAEMMBench/analysis/compare_locality_arms.py \
        --arm last5_step75=locality_last5_step75.json \
        --arm v2_step225=locality_v2_step225.json \
        --out snippet_locality_last5_vs_baseline.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from snippet_locality import METRICS, MORE_LOCAL_IS  # noqa: E402


def _mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return None, None
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)


def arm_stats(L):
    """Aggregate one locality.json: rollout ('maemm') + real arms, firing texts only."""
    pooled = {arm: {m: [] for m in METRICS} for arm in ("maemm", "real")}
    peaks = {arm: [] for arm in ("maemm", "real")}
    fired_n = {arm: [0, 0] for arm in ("maemm", "real")}
    ntok = {arm: [] for arm in ("maemm", "real")}
    per_feat = {}
    for r in L["features"]:
        fmeans = {}
        for arm in ("maemm", "real"):
            rows = [t for t in r["texts"]
                    if (t["kind"] == "real") == (arm == "real")]
            fired = [t for t in rows if t["fired"]]
            fired_n[arm][0] += len(fired)
            fired_n[arm][1] += len(rows)
            ntok[arm] += [t["n_tokens"] for t in rows]
            peaks[arm] += [t["peak"] for t in fired]
            for m in METRICS:
                pooled[arm][m] += [t["metrics"][m] for t in fired]
            fmeans[arm] = ({m: float(np.mean([t["metrics"][m] for t in fired]))
                            for m in METRICS} | {"n_fired": len(fired)}) if fired else None
        per_feat[r["feature"]] = fmeans
    agg = {"fired_frac": {arm: fired_n[arm][0] / fired_n[arm][1] for arm in fired_n},
           "n_texts": {arm: fired_n[arm][1] for arm in fired_n},
           "mean_n_tokens": {arm: float(np.mean(ntok[arm])) for arm in ntok},
           "mean_peak_act_fired": {arm: float(np.mean(peaks[arm])) if peaks[arm] else None
                                   for arm in peaks},
           "pooled": {arm: {m: dict(zip(("mean", "sem"), _mean_sem(pooled[arm][m])),
                                    median=(float(np.median(pooled[arm][m]))
                                            if pooled[arm][m] else None),
                                    n=len(pooled[arm][m]))
                            for m in METRICS} for arm in pooled}}
    return agg, per_feat


def paired(per_feat_a, per_feat_b, key_a="maemm", key_b="maemm"):
    """Per-feature paired diff (a - b) over features present+firing in both. Returns per metric:
    diff mean/sem/ci95 + each side's mean over the common features."""
    feats = [f for f in per_feat_a
             if f in per_feat_b and per_feat_a[f][key_a] and per_feat_b[f][key_b]]
    out = {"n_features": len(feats), "features": feats, "metrics": {}}
    for m in METRICS:
        va = np.array([per_feat_a[f][key_a][m] for f in feats])
        vb = np.array([per_feat_b[f][key_b][m] for f in feats])
        mu, sem = _mean_sem(va - vb)
        out["metrics"][m] = {"diff_mean": mu, "diff_sem": sem,
                             "ci95": [mu - 1.96 * sem, mu + 1.96 * sem],
                             "a_mean": float(va.mean()), "b_mean": float(vb.mean()),
                             "more_local_is": MORE_LOCAL_IS[m]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True,
                    help="name=locality.json (give twice; first = the candidate, second = the "
                         "baseline for the headline diff)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arms = [a.split("=", 1) for a in args.arm]
    assert len(arms) == 2, "exactly two --arm specs"
    (name_a, path_a), (name_b, path_b) = arms
    La, Lb = json.load(open(path_a)), json.load(open(path_b))

    agg_a, pf_a = arm_stats(La)
    agg_b, pf_b = arm_stats(Lb)
    # the real arm is the same texts under the same clean-base read in both builds — verify
    ra, rb = agg_a["pooled"]["real"], agg_b["pooled"]["real"]
    for m in METRICS:
        assert abs(ra[m]["mean"] - rb[m]["mean"]) < 1e-9, f"real arms differ on {m}!"

    pw = paired(pf_a, pf_b)                                     # candidate - baseline
    vs_real = {name_a: paired(pf_a, pf_a, "maemm", "real"),
               name_b: paired(pf_b, pf_b, "maemm", "real")}

    hw = pw["metrics"]["win5_share"]
    out = {"config": {"arms": {name_a: {"locality": os.path.abspath(path_a),
                                        **{k: La["config"].get(k) for k in
                                           ("adapter", "rollout_source", "gen_seed")}},
                               name_b: {"locality": os.path.abspath(path_b),
                                        **{k: Lb["config"].get(k) for k in
                                           ("adapter", "rollout_source", "gen_seed")}}},
                      "testbed": La["config"]["testbed"], "fire": La["config"]["fire"],
                      "n_features": La["config"]["n_features"],
                      "headline_order": f"{name_a} - {name_b}"},
           "headline": {"metric": "win5_share",
                        name_a: hw["a_mean"], name_b: hw["b_mean"],
                        "real": ra["win5_share"]["mean"],
                        "paired_diff": {k: hw[k] for k in ("diff_mean", "diff_sem", "ci95")},
                        "n_paired_features": pw["n_features"]},
           "aggregate": {name_a: agg_a, name_b: agg_b},
           "paired_arms": pw, "paired_vs_real": vs_real,
           "per_feature": [{"feature": f,
                            name_a: pf_a[f]["maemm"],
                            name_b: pf_b.get(f, {}).get("maemm"),
                            "real": pf_a[f]["real"]} for f in pf_a]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"=== SNIPPET LOCALITY: {name_a} vs {name_b} (firing rollouts; paired over "
          f"{pw['n_features']} features) ===")
    print(f"  fired frac:  {name_a} {agg_a['fired_frac']['maemm']:.3f}  "
          f"{name_b} {agg_b['fired_frac']['maemm']:.3f}  real {agg_a['fired_frac']['real']:.3f}")
    print(f"  mean tokens: {name_a} {agg_a['mean_n_tokens']['maemm']:.1f}  "
          f"{name_b} {agg_b['mean_n_tokens']['maemm']:.1f}  "
          f"real {agg_a['mean_n_tokens']['real']:.1f}")
    print(f"  peak act:    {name_a} {agg_a['mean_peak_act_fired']['maemm']:.2f}  "
          f"{name_b} {agg_b['mean_peak_act_fired']['maemm']:.2f}  "
          f"real {agg_a['mean_peak_act_fired']['real']:.2f}")
    for m in METRICS:
        d = pw["metrics"][m]
        arrow = "MORE local" if d["diff_mean"] * MORE_LOCAL_IS[m] > 0 else "LESS local"
        print(f"  {m:>12}  {name_a} {d['a_mean']:.4f}  {name_b} {d['b_mean']:.4f}  "
              f"real {ra[m]['mean']:.4f}  paired diff {d['diff_mean']:+.4f} "
              f"±{d['diff_sem']:.4f} -> {name_a} {arrow}")
    print(f"COMPARE_DONE {args.out}")


if __name__ == "__main__":
    main()
