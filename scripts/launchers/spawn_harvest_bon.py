"""Launch the best-of-N harvest (app maemm-harvest-bon-s2m, data/modal_harvest_bon.py). Records call ids in
~/shared/overnight/simple2m/ids.json["harvest"][<out_name>].

    cd ~/maemm-pub-simple2m && MODAL_PROFILE=safety-sahan modal deploy data/modal_harvest_bon.py
    python scripts/launchers/spawn_harvest_bon.py --probe                 # 256 targets x N=256, 5 sampler configs (1 GPU each)
    python scripts/launchers/spawn_harvest_bon.py --probe --report        # best-of-n curves once the shards are done
    python scripts/launchers/spawn_harvest_bon.py --full --n-samples 32 --n-shards 8 --configs s250_t1.0,s250_t1.3   # the real harvest

Sampler config name = s<step>_t<temperature>[_w<window>] over the main simple2m arm's checkpoints (CKPTS below)."""
import argparse
import json
import os
import sys
import time

import modal

os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
APP = os.environ.get("HARVEST_APP", "maemm-harvest-bon-s2m")
IDS = os.path.expanduser("~/shared/overnight/simple2m/ids.json")
CKPTS = {"s25": "/data/ckpts_rl_simple2m_8x2048_anywin/step_25", "s50": "/data/ckpts_rl_simple2m_8x2048_anywin/step_50",
         "s100": "/data/ckpts_rl_simple2m_8x2048_anywin/step_100", "s150": "/data/ckpts_rl_simple2m_8x2048_anywin/step_150",
         "s200": "/data/ckpts_rl_simple2m_8x2048_anywin/step_200", "s250": "/data/ckpts_rl_simple2m_8x2048_anywin/step_250",
         "s300": "/data/ckpts_rl_simple2m_8x2048_anywin/final", "sft": "/data/sft_mix/simple2m_sft/final",
         "v16k": "/data/ckpts_rl_simple2m_16x1024_anywin/step_100",
         "l16s100": "/data/ckpts_rl_simple2m_8x2048_last16_lr1e-6/step_100", "l16s150": "/data/ckpts_rl_simple2m_8x2048_last16_lr1e-6/step_150",
         "l16s200": "/data/ckpts_rl_simple2m_8x2048_last16_lr1e-6/step_200", "l16s250": "/data/ckpts_rl_simple2m_8x2048_last16_lr1e-6/step_250",
         "l16s300": "/data/ckpts_rl_simple2m_8x2048_last16_lr1e-6/final"}
SFT_MIX, RL_POOL = "/data/banks/mix_simple2m_sft", "/data/banks/mix_simple2m_rl"
PROBE_SPEC = [{"bank": SFT_MIX, "families": {"realact": 64}}, {"bank": RL_POOL, "families": {"realact_ctx64_2048": 64, "sae2m": 64, "sae2m_dec": 64}}]
FULL_SPEC = [{"bank": SFT_MIX, "families": {"realact": 400000, "sae2m": 100000, "sae2m_dec": 100000}},
             {"bank": RL_POOL, "families": {"realact_ctx64_2048": 300000, "sae2m": 50000, "sae2m_dec": 50000}}]
PROBE_CONFIGS = ["s250_t1.0", "s250_t1.3", "s100_t1.0", "s50_t1.0", "l16s150_t1.0_w16", "l16s150_t1.3_w16"]   # every rollout is ALSO scored under the other window


def parse_cfg(name):
    parts = name.split("_")
    ck = CKPTS[parts[0]]
    temp = float(parts[1][1:])
    win = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("w") else 0
    return ck, temp, win


def load_ids():
    return json.load(open(IDS)) if os.path.exists(IDS) else {}


def save_ids(d):
    json.dump(d, open(IDS, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--full", action="store_true")
    ap.add_argument("--out-name", default=""); ap.add_argument("--configs", default=""); ap.add_argument("--n-samples", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=0); ap.add_argument("--n-shards", type=int, default=0); ap.add_argument("--seed", type=int, default=2050)
    ap.add_argument("--report", action="store_true", help="probe: run probe_report over the finished shards and print it")
    ap.add_argument("--finalize", default="", help="bank_out name: compose the distilled bank from --configs")
    ap.add_argument("--k-keep", type=int, default=1); ap.add_argument("--require-beat-orig", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="worker --limit (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--quant", default="", help="sampler quantization (fp8) passed to the worker"); ap.add_argument("--queue-mult", type=int, default=1)
    ap.add_argument("--also-window", type=int, default=-1, help="second scoring window (-1 off; the probe used 16)")
    ap.add_argument("--select", choices=["raw", "centered"], default="raw", help="selection cosine convention (both are stored)")
    ap.add_argument("--spec", default="", help="JSON plan spec override (list of {bank, families:{fam:n}})")
    a = ap.parse_args()
    assert a.probe != a.full or a.report or a.finalize, "pick --probe or --full"
    out_name = a.out_name or ("harvest_probe_v1" if a.probe else "harvest_full_v1")
    configs = a.configs.split(",") if a.configs else (PROBE_CONFIGS if a.probe else ["s250_t1.0"])
    n_samples = a.n_samples or (256 if a.probe else 32)
    top_k = a.top_k or (8 if a.probe else 4)
    n_shards = a.n_shards or (1 if a.probe else 8)
    spec = json.loads(a.spec) if a.spec else (PROBE_SPEC if a.probe else FULL_SPEC)
    ids = load_ids(); H = ids.setdefault("harvest", {}); rec = H.setdefault(out_name, {"out_name": out_name, "app": APP})

    if a.report:
        rep = modal.Function.from_name(APP, "probe_report").remote(out_name, json.dumps(configs))
        p = os.path.expanduser(f"~/shared/overnight/simple2m/{out_name}_probe_report.json"); json.dump(rep, open(p, "w"), indent=1)
        print(f"report -> {p}")
        for tag, e in rep["per_tag"].items():
            for fam, f in e["families"].items():
                cur = f.get("best_of_n", {})
                print(f"{tag:>16} {fam:<20} n={f['n']:>4} orig {f['orig_cos']:.3f} | mean {f['cos_mean']:.3f} std_within {f['cos_std_within']:.3f} | "
                      + " ".join(f"bo{n} {v:.3f}" for n, v in cur.items()) + f" | beat-orig {f['beat_orig_frac']:.2f} len {f['len_mean']:.0f}")
                for k in [k for k in f if k.startswith("best_of_n_w")]:
                    w = k[len("best_of_n_"):]
                    print(f"{'':>16} {'  (window ' + w + ')':<20} {'':>6} orig {f.get('orig_cos_' + w, float('nan')):.3f} | mean {f.get('cos_mean_' + w, float('nan')):.3f} {'':>16} | "
                          + " ".join(f"bo{n} {v:.3f}" for n, v in f[k].items()))
        for fam, f in rep["pooled"].items():
            print(f"    pooled {fam:<20} best-over-tags {f['best_over_tags']:.3f} orig {f['orig_cos']:.3f} beat {f['beat_orig_frac']:.2f} winners {f['best_tag_hist']}")
        return
    if a.finalize:
        bs = modal.Function.from_name(APP, "finalize").remote(out_name, json.dumps(configs), a.finalize, k_keep=a.k_keep, select="reward",
                                                              require_beat_orig=a.require_beat_orig, overwrite=False)
        rec.setdefault("banks", {})[a.finalize] = {"configs": configs, "k_keep": a.k_keep, "require_beat_orig": a.require_beat_orig,
                                                    "n_examples": bs["n_examples"], "families": bs["families"], "stats": bs["stats"], "out": f"/data/banks/{a.finalize}",
                                                    "created": bs["created"]}
        save_ids(ids); print(json.dumps(rec["banks"][a.finalize], indent=1)); return

    # ---- plan (blocking; cheap for the probe, ~10-20 min for the full harvest) ----
    if "plan" not in rec:
        t0 = time.time()
        pl = modal.Function.from_name(APP, "plan").remote(out_name, json.dumps(spec), seed=a.seed, n_shards=n_shards)
        rec["plan"] = {k: pl[k] for k in ("n_targets", "n_shards", "families", "seed", "created", "wall_s")}; rec["spec"] = spec; save_ids(ids)
        print(f"plan: {pl['n_targets']} targets in {pl['n_shards']} shards {pl['families']} ({time.time() - t0:.0f}s)")
    n_shards = rec["plan"]["n_shards"]
    # ---- shards ----
    fn = modal.Function.from_name(APP, "harvest_shard")
    calls = rec.setdefault("calls", {})
    for cfg in configs:
        ck, temp, win = parse_cfg(cfg)
        for sh in range(n_shards):
            key = f"{cfg}/{sh:02d}"
            if key in calls and not calls[key].get("failed"):
                continue
            fc = fn.spawn(out_name, sh, cfg, ck, n_samples=n_samples, temperature=temp, top_k=top_k, max_new_tokens=a.max_new_tokens,
                          reward_window_last=win, probe=a.probe, seed=a.seed + sh, limit=a.limit, also_window=(16 if a.probe and a.also_window < 0 else a.also_window),
                          queue_mult=a.queue_mult, quant=a.quant, select=a.select)
            calls[key] = {"call": fc.object_id, "ckpt": ck, "temperature": temp, "window_last": win, "n_samples": n_samples, "top_k": top_k,
                          "probe": a.probe, "quant": a.quant, "queue_mult": a.queue_mult, "select": a.select, "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            print(f"spawned {key}: {fc.object_id}")
    rec.update({"n_samples": n_samples, "top_k": top_k, "configs": sorted(set(rec.get("configs", []) + configs))}); save_ids(ids)
    print(json.dumps({k: v for k, v in rec.items() if k != "calls"}, indent=1))


if __name__ == "__main__":
    main()
