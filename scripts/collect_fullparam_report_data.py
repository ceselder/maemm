"""Collect the numbers behind ~/shared/reports/maemm-rl-fullparam/ (rl_disagg --full-param runs vs the LoRA ablation arms).

  wandb (celestedeschamphelaere-personal/maxact-fast): per-step reward/entropy/|dlogp|/gnorm/timings/memory of the full-param
      runs (rl_fullparam_smoke6, rl_fullparam_val50_lr*) and the LoRA arms (rl_abl_initboth_8x512_lr7e-6, rl_abl_initnewfft_8x512_lr7e-6)
  Modal volume maemm-data /disagg_runs/<ts>/: trainer_steps.json, fullparam_summary.json, fullparam_sync_r*.json, trainer_mb.json,
      verify_r*.json of each run (modal_rl_disagg._collect copies the work dir's json artefacts there)
  -> data/steps_fullparam.json, data/steps_lora.json, data/publish.json, data/memory.json (+ raw copies under data/raw/)

    source ~/modal_venv/bin/activate; export MODAL_PROFILE=safety-sahan
    python3 scripts/collect_fullparam_report_data.py --val50 rl_fullparam_val50_lr1e-6 --disagg-run 20260908_xxxxxx
"""
import argparse
import json
import os
import subprocess
import sys

OUT = os.path.expanduser("~/shared/reports/maemm-rl-fullparam/data")
KEYS = {"reward": "reward/mean", "reward_max": "reward/max", "entropy": "policy/entropy", "dlogp": "policy/sampler_abs_dlogp",
        "gnorm": "grad_norm", "step_s": "time/step_s", "update_s": "time/update_s", "score_s": "time/score_s", "wait_s": "time/wait_rollouts_s",
        "t_pub": "time/publish_s", "fwd_bwd_s": "time/fwd_bwd_s", "peak_gb": "mem/hf_peak_gb", "peak_gb_max_rank": "mem/hf_peak_gb_max_rank",
        "len": "rollout/len_mean", "gen_s": "rollout/gen_s", "lr": "lr", "lag": "policy/offpolicy_lag_steps", "ratio": "ratio/mean",
        "is_trunc_frac": "scalerl/is_trunc_frac", "kl": "policy/kl_to_init", "mb": "micro_batch",
        "pub_wait": "publish/t_wait_ready", "pub_transfer": "publish/t_transfer", "pub_gbps": "publish/gbps", "pub_gb": "publish/gb",
        "pub_total": "publish/total_s", "pub_hnorm_s": "publish/hnorm_s", "queue": "rollout/queue_depth"}


def wandb_rows(name):
    import wandb
    api = wandb.Api()
    runs = list(api.runs("celestedeschamphelaere-personal/maxact-fast", filters={"display_name": name}))
    if not runs:
        print(f"[collect] no wandb run named {name}", file=sys.stderr)
        return None, []
    r = sorted(runs, key=lambda x: x.created_at)[-1]
    rows = []
    for rec in r.scan_history(page_size=2000):      # every logged row (history(keys=...) drops rows lacking ANY key)
        if rec.get("_step") is None or rec.get("reward/mean") is None:
            continue
        row = {"step": int(rec["_step"])}
        for k, wk in KEYS.items():
            v = rec.get(wk)
            if v is not None and v == v:
                row[k] = float(v)
        rows.append(row)
    rows.sort(key=lambda x: x["step"])
    cfg = {k: r.config.get(k) for k in ("lr", "n_rollout", "n_trainer", "micro_batch_used", "policy_base_resolved", "init_adapter", "full_param",
                                        "publish_mode", "groups_per_step", "group_size", "max_new_tokens", "total_steps", "save_steps", "recipe")}
    return {"id": r.id, "name": r.name, "state": r.state, "created_at": str(r.created_at), "config": cfg, "url": r.url}, rows


def volume_get(remote, local):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    p = subprocess.run(["modal", "volume", "get", "maemm-data", remote, local, "--force"], capture_output=True, text=True)
    return p.returncode == 0 and os.path.exists(local)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default="rl_fullparam_smoke6")
    ap.add_argument("--val50", default="")
    ap.add_argument("--lora", nargs="*", default=["initboth:rl_abl_initboth_8x512_lr7e-6", "initnewfft:rl_abl_initnewfft_8x512_lr7e-6"])
    ap.add_argument("--disagg-run", nargs="*", default=[], help="<tag>:<ts> dirs under /data/disagg_runs to fetch (tag = smoke | val50)")
    a = ap.parse_args()
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    steps, meta = {}, {}
    for tag, name in (("smoke", a.smoke), ("val50", a.val50)):
        if name:
            m, rows = wandb_rows(name)
            if m:
                steps[tag], meta[tag] = rows, m
                steps[f"{tag}_lr"] = m["config"].get("lr")
    lora = {}
    for spec in a.lora:
        tag, name = spec.split(":", 1)
        m, rows = wandb_rows(name)
        if m:
            lora[tag], meta[f"lora_{tag}"] = rows, m
    for spec in a.disagg_run:
        tag, ts = spec.split(":", 1)
        for fn in ("trainer_steps.json", "fullparam_summary.json", "trainer_mb.json", "fullparam_sync_r0.json", "fullparam_sync_r1.json",
                   "fullparam_sync_r2.json", "verify_r0.json"):
            ok = volume_get(f"/disagg_runs/{ts}/{fn}", f"{OUT}/raw/{tag}_{fn}")
            print(f"[collect] {tag} {fn}: {'ok' if ok else 'missing'}")
    json.dump(steps, open(f"{OUT}/steps_fullparam.json", "w"), indent=1)
    json.dump(lora, open(f"{OUT}/steps_lora.json", "w"), indent=1)
    json.dump(meta, open(f"{OUT}/wandb_runs.json", "w"), indent=1)
    print(f"[collect] wrote steps for {sorted(k for k in steps if not k.endswith('_lr'))} + lora {sorted(lora)}")


if __name__ == "__main__":
    main()
