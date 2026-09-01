"""Merge two autointerp-detection score results (same features/positives/negatives, on-policy
rollouts from two different adapters) into one adapter-vs-adapter comparison JSON.

The two testbeds MUST come from `autointerp_detection.py build` at the same seed (see
modal_autointerp_detection.py::compare) so the 64 held-out features + test spans are identical
and the comparison is paired per feature. The `examples` arm (real max-act descriptions) is
judged in BOTH batches on byte-identical prompts, so its A-vs-B gap is a judge-noise floor.

Usage:
  python autointerp_compare.py \
      --a-name last5_step75 --a-results out/results_last5_step75.json \
      --a-testbed out/testbed_last5_step75.json \
      --b-name v2_step225 --b-results out/results_v2_step225.json \
      --b-testbed out/testbed_v2_step225.json \
      --out comparison.json
"""
import argparse
import json

import numpy as np


def _firing(tb):
    """Rollout firing stats from the build stage's clean-base self-acts diagnostic."""
    fire = tb["config"]["fire"]
    g = [r["rollout_self_acts"]["greedy"] for r in tb["features"]]
    t = [x for r in tb["features"] for x in r["rollout_self_acts"]["temp"]]
    per_feat = [float(np.mean([x > fire for x in r["rollout_self_acts"]["temp"]]))
                for r in tb["features"]]
    return {"fire_threshold": fire,
            "greedy_fired_frac": float(np.mean([x > fire for x in g])),
            "temp_fired_frac": float(np.mean([x > fire for x in t])),
            "per_feature_temp_fired_frac_mean": float(np.mean(per_feat)),
            "frac_features_majority_firing": float(np.mean([p >= 0.5 for p in per_feat]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for side in "ab":
        ap.add_argument(f"--{side}-name", required=True)
        ap.add_argument(f"--{side}-results", required=True)
        ap.add_argument(f"--{side}-testbed", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    R = {s: json.load(open(getattr(a, f"{s}_results"))) for s in "ab"}
    T = {s: json.load(open(getattr(a, f"{s}_testbed"))) for s in "ab"}
    names = {s: getattr(a, f"{s}_name") for s in "ab"}

    feats = [r["feature"] for r in R["a"]["per_feature"]]
    assert feats == [r["feature"] for r in R["b"]["per_feature"]], "feature sets differ!"
    for s in "ab":   # positives/negatives must be shared verbatim for a paired comparison
        assert [f["feature"] for f in T[s]["features"]] == feats
    for ra, rb in zip(T["a"]["features"], T["b"]["features"]):
        assert [p["text"] for p in ra["positives"]] == [p["text"] for p in rb["positives"]]
        assert [n["text"] for n in ra["negatives"]] == [n["text"] for n in rb["negatives"]]

    n_list = R["a"]["state"]["n_list"]
    assert n_list == R["b"]["state"]["n_list"]
    comp = {"names": names,
            "adapters": {s: T[s]["config"]["adapter"] for s in "ab"},
            "batch_ids": {s: R[s]["state"]["batch_id"] for s in "ab"},
            "judge_model": R["a"]["state"]["judge_model"],
            "n_features": len(feats), "n_list": n_list,
            "n_pos": T["a"]["config"]["n_pos"], "n_neg": T["a"]["config"]["n_neg"],
            "firing": {names[s]: _firing(T[s]) for s in "ab"},
            "auc": {}, "paired": {}, "examples_arm_noise_floor": {}, "per_feature": []}

    for N in n_list:
        vk = f"maemm_N{N}"
        va = np.array([r["auc"][vk] for r in R["a"]["per_feature"]])
        vb = np.array([r["auc"][vk] for r in R["b"]["per_feature"]])
        d = va - vb
        comp["auc"][f"N{N}"] = {
            names["a"]: {"mean": float(va.mean()),
                         "sem": float(va.std(ddof=1) / np.sqrt(len(va)))},
            names["b"]: {"mean": float(vb.mean()),
                         "sem": float(vb.std(ddof=1) / np.sqrt(len(vb)))}}
        comp["paired"][f"N{N}"] = {"delta_mean": float(d.mean()),
                                   "delta_sem": float(d.std(ddof=1) / np.sqrt(len(d))),
                                   "n": int(len(d)),
                                   "frac_a_ge_b": float((d >= 0).mean())}
        # examples arm: byte-identical prompts in both batches -> pure judge-resample noise
        ea = np.array([r["auc"][f"examples_N{N}"] for r in R["a"]["per_feature"]])
        eb = np.array([r["auc"][f"examples_N{N}"] for r in R["b"]["per_feature"]])
        de = ea - eb
        comp["examples_arm_noise_floor"][f"N{N}"] = {
            "mean_a": float(ea.mean()), "mean_b": float(eb.mean()),
            "delta_mean": float(de.mean()),
            "delta_sem": float(de.std(ddof=1) / np.sqrt(len(de)))}

    for i, f in enumerate(feats):
        row = {"feature": f}
        for N in n_list:
            row[f"auc_maemm_N{N}"] = {names[s]: R[s]["per_feature"][i]["auc"][f"maemm_N{N}"]
                                      for s in "ab"}
        for s in "ab":
            r = T[s]["features"][i]
            row[f"temp_fired_frac_{names[s]}"] = float(
                np.mean([x > T[s]["config"]["fire"] for x in r["rollout_self_acts"]["temp"]]))
        comp["per_feature"].append(row)

    json.dump(comp, open(a.out, "w"), indent=1)
    print(f"=== {names['a']} vs {names['b']} (maemm-rollout detection AUC) ===")
    for N in n_list:
        c, p = comp["auc"][f"N{N}"], comp["paired"][f"N{N}"]
        nf = comp["examples_arm_noise_floor"][f"N{N}"]
        print(f"  N={N}: {names['a']} {c[names['a']]['mean']:.4f} | "
              f"{names['b']} {c[names['b']]['mean']:.4f} | "
              f"paired d {p['delta_mean']:+.4f} +/-{p['delta_sem']:.4f} | "
              f"examples-arm noise {nf['delta_mean']:+.4f}")
    for s in "ab":
        fi = comp["firing"][names[s]]
        print(f"  firing {names[s]}: temp {fi['temp_fired_frac']:.3f} "
              f"greedy {fi['greedy_fired_frac']:.3f}")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
