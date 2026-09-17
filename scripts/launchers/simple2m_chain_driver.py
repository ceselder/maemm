"""SIMPLE2M chain driver (paper-clean recipe, 2026-09-17): NO pretrain/midtrain distinction.

  SFT  = ONE full-FT run from the BASE model on an exact 50/50 mix: N short-context Ultra-FineWeb activations + N 2M-SAE rows
         (N/2 encoder-column + N/2 decoder-row directions, max-act windows of the SFT feature split), N = min(4M, SAE rows available)
  RL   = full-parameter CISPO/GRPO (8 x 2048, lr 1e-6, 300 steps, reward = max cosine over the WHOLE rollout span) on an exact
         1/3 short-ctx : 1/3 long-ctx : 1/3 SAE (RL feature split) pool, every row from documents / features disjoint from SFT and eval
  eval = cache v3 (v2 families unchanged for mean_all + held-out 2M-SAE enc/dec slice families), evaluator maemm-eval-ckpt-s2m

Inputs (spawned by hand, ids in ~/shared/overnight/simple2m/ids.json): bank_sae2m_sft, bank_sae2m_rl (maemm-sae2m-bank-s2m),
coll_sft_4m, coll_rl_500k, coll_eval_100k (maemm-collect-bank-s2m), store_rl_long (maemm-acts-ufw-s2m). This driver waits for them and
runs: store finalize -> long bank | compose SFT mix -> SFT + evaluator | compose RL pool | (SFT final) -> RL + evaluator -> done.
Idempotent: every stage is recorded in ids.json; rerun = resume. Discord on launches / completion / failure.

    cd /tmp && nohup python3 ~/maemm-pub-simple2m/scripts/launchers/simple2m_chain_driver.py >> ~/shared/overnight/simple2m/driver.out 2>&1 &
"""
import json
import math
import os
import subprocess
import sys
import time

import modal

IDS = os.path.expanduser("~/shared/overnight/simple2m/ids.json")
LOG = os.path.expanduser("~/shared/overnight/simple2m/driver.log")
V3 = "/data/eval_universal_ho/eval_sets_heldout_v3.pt"
APPS = {"bank": "maemm-sae2m-bank-s2m", "collect": "maemm-collect-bank-s2m", "store": "maemm-acts-ufw-s2m", "long": "maemm-ufw-long-bank-s2m",
        "compose": "maemm-mix-5m-bank-s2m", "sft": "maemm-sft-fullft-s2m", "eval": "maemm-eval-ckpt-s2m", "rl": "maemm-rl-disagg-fullparam-s2m"}
SFT_RUN, SFT_MIX = "simple2m_sft", "mix_simple2m_sft"
RL_RUN, RL_POOL = "rl_simple2m_8x2048_anywin", "mix_simple2m_rl"
EFF_BATCH = 4096                                   # micro 32 x grad-accum 16 x 8 GPUs
SFT_EXTRA = ("--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 16 --pad-multiple 1 --prefix-share-step "
             "--fsdp-prefetch 2")                  # NO --init-adapter: the policy starts from the base model
RL_RECIPE = ("--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 "
             "--max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 2048 "
             "--group-size 8 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 0 "
             "--prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --transcript-every 5 "
             "--lr 1e-6 --save-steps 25,50,100,150,200,250,300 --eval-cache " + V3)   # inline eval on cache v3 too
RL_STEPS = 300


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(m):
    line = f"[{now()}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def discord(m):
    subprocess.run(["notify-discord", f"simple2m: {m}"])


def load():
    return json.load(open(IDS))


def save(d):
    json.dump(d, open(IDS, "w"), indent=1)


def wait_call(fc_id, what, poll=120):
    fc = modal.FunctionCall.from_id(fc_id)
    t0 = time.time()
    while True:
        try:
            r = fc.get(timeout=0)
            log(f"{what} DONE after {(time.time() - t0) / 60:.0f} min: {str(r)[:400]}")
            return r
        except TimeoutError:
            time.sleep(poll)
        except Exception as e:  # noqa
            log(f"{what} FAILED: {type(e).__name__}: {str(e)[:600]}")
            discord(f"{what} FAILED ({type(e).__name__}) — chain paused; see driver.log")
            raise SystemExit(1)


def vol_get_json(path):
    out = f"/tmp/simple2m_{path.strip('/').replace('/', '_')}"
    subprocess.run(["modal", "volume", "get", "--force", "maemm-data", path, out], check=True, capture_output=True)
    return json.load(open(out))


def vol_has(path):
    r = subprocess.run(["modal", "volume", "ls", "maemm-data", os.path.dirname(path)], capture_output=True, text=True)
    return os.path.basename(path) in r.stdout


def stage(d, key):
    return key in d and d[key].get("done")


# ---------------------------------------------------------------------------------------------------------------------
d = load()
log("driver start; stages done: " + ", ".join(k for k in d if isinstance(d[k], dict) and d[k].get("done")))

# ---- A. long-context store -> finalize -> long bank ------------------------------------------------------------------
if not stage(d, "long_bank"):
    wait_call(d["store_rl_long"]["call"], "store acts_ufw_rl (collect)")
    if "store_finalize" not in d:
        c = modal.Function.from_name(APPS["store"], "finalize").spawn(out_name="acts_ufw_rl", keep_shards=False)
        d["store_finalize"] = {"call": c.object_id, "spawned": now()}; save(d); log(f"store finalize spawned {c.object_id}")
    wait_call(d["store_finalize"]["call"], "store acts_ufw_rl (finalize)")
    d["store_finalize"]["done"] = True; save(d)
    if "long_bank" not in d:
        c = modal.Function.from_name(APPS["long"], "build").spawn(acts_dir="/data/acts_ufw_rl", out_name="ufw_long_rl_500k", n_rows=500_000,
                                                                   p_lo=256, p_hi=511, w_lo=16, w_hi=64, max_per_doc=8, seed=11)
        d["long_bank"] = {"call": c.object_id, "out": "/data/banks/ufw_long_rl_500k", "spawned": now()}; save(d); log(f"long bank spawned {c.object_id}")
    r = wait_call(d["long_bank"]["call"], "long bank ufw_long_rl_500k")
    d["long_bank"].update({"done": True, "result": r}); save(d)

# ---- B. SFT mix (exact 50/50) -> SFT + evaluator ----------------------------------------------------------------------
if not stage(d, "compose_sft"):
    wait_call(d["bank_sae2m_sft"]["call"], "SAE bank sae2m_sft")
    wait_call(d["coll_sft_4m"]["call"], "collection ufw_short_sft_4m")
    st = vol_get_json("/banks/sae2m_sft/build_stats.json"); fam = st["families"]
    n_enc, n_dec = int(fam["sae2m"]), int(fam["sae2m_dec"])
    acts_avail = int(vol_get_json("/banks/ufw_short_sft_4m/build_stats.json")["n_examples"])
    n_sae = min(n_enc + n_dec, acts_avail) // 2 * 2                    # even, so enc == dec
    n_acts = n_sae                                                       # EXACT 50/50: activations == SAE rows
    spec = [{"name": "acts_ufw_sft", "banks": ["/data/banks/ufw_short_sft_4m"], "dtype": "f16", "families": ["realact"], "n": n_acts,
             "all_one_family": "realact", "fresh": "Ultra-FineWeb docs [3.0M, ...) ordered stream; short-ctx pairs p in [8,256], W in [8,32]"},
            {"name": "sae2m_sft", "banks": ["/data/banks/sae2m_sft"], "dtype": "f32", "families": ["sae2m", "sae2m_dec"],
             "n": {"sae2m": n_sae // 2, "sae2m_dec": n_sae // 2},
             "fresh": "2M-SAE SFT feature split (1,847,152 features; eval 100k + RL 150k held out); enc + dec directions x end-anchored max-act windows"}]
    if "compose_sft" not in d:
        c = modal.Function.from_name(APPS["compose"], "build").spawn(out_name=SFT_MIX, spec_json=json.dumps(spec), seed=2040, threads=48,
                                                                      dense_frac=0.0, eval_cache=V3)
        d["compose_sft"] = {"call": c.object_id, "out": f"/data/banks/{SFT_MIX}", "spec": spec, "n_acts": n_acts, "n_sae": n_sae,
                            "sae_bank_rows": {"sae2m": n_enc, "sae2m_dec": n_dec}, "acts_avail": acts_avail, "spawned": now()}; save(d)
        log(f"compose SFT spawned {c.object_id}: acts {n_acts} + sae {n_sae} (bank had enc {n_enc} dec {n_dec}; acts avail {acts_avail})")
    wait_call(d["compose_sft"]["call"], f"compose {SFT_MIX}")
    st = vol_get_json(f"/banks/{SFT_MIX}/build_stats.json")
    d["compose_sft"].update({"done": True, "n_examples": st["n_examples"], "families": st["families"]}); save(d)
    discord(f"SFT mix composed: {st['n_examples']} rows {st['families']} -> launching full-FT SFT from base")

if not stage(d, "sft"):
    n_rows = d["compose_sft"]["n_examples"]; steps = math.ceil(n_rows / EFF_BATCH)
    if "sft" not in d:
        t = modal.Function.from_name(APPS["sft"], "train").spawn(run_name=SFT_RUN, data_dir=f"/data/banks/{SFT_MIX}", n_ckpts=8, epochs=1,
                                                                  batch_size=32, lr=1e-5, max_seq=192, backend="nccl", extra_args=SFT_EXTRA)
        e = modal.Function.from_name(APPS["eval"], "fullmodel_daemon").spawn(ckpt_dir=f"/data/sft_mix/{SFT_RUN}", tag=f"sft_{SFT_RUN}",
                                                                              wandb_name=f"{SFT_RUN}_eval", final_step=steps,
                                                                              extra_args=f"--eval-cache {V3} --no-extra-evals")
        d["sft"] = {"train": t.object_id, "eval": e.object_id, "run": SFT_RUN, "save": f"/data/sft_mix/{SFT_RUN}", "data": f"/data/banks/{SFT_MIX}",
                    "eff_batch": EFF_BATCH, "micro_batch": 32, "grad_accum": 16, "lr": 1e-5, "max_seq": 192, "steps_expected": steps, "init": "base model",
                    "extra": SFT_EXTRA, "app": APPS["sft"], "eval_app": APPS["eval"], "eval_cache": V3, "spawned": now()}; save(d)
        log(f"SFT spawned: train {t.object_id} eval {e.object_id} ({steps} steps expected)")
        discord(f"SFT launched from BASE on {SFT_MIX} ({n_rows} rows 50/50, {steps} steps ≈ {steps * 12 / 3600:.1f} h at 12 s/step); RL follows")
    wait_call(d["sft"]["train"], "SFT train")
    ok = False
    for _ in range(40):
        if vol_has(f"/sft_mix/{SFT_RUN}/final/SAVE_DONE"):
            ok = True; break
        time.sleep(60)
    if not ok:
        discord("SFT final has no SAVE_DONE — RL NOT launched"); log("SFT final missing"); raise SystemExit(1)
    d["sft"]["done"] = True; save(d)

# ---- C. RL pool (exact thirds) -------------------------------------------------------------------------------------------
if not stage(d, "compose_rl"):
    wait_call(d["bank_sae2m_rl"]["call"], "SAE bank sae2m_rl")
    wait_call(d["coll_rl_500k"]["call"], "collection ufw_short_rl_500k")
    fam = vol_get_json("/banks/sae2m_rl/build_stats.json")["families"]
    n_sae = int(fam["sae2m"]) + int(fam["sae2m_dec"])
    n_short = int(vol_get_json("/banks/ufw_short_rl_500k/build_stats.json")["n_examples"])
    n_long = int(vol_get_json("/banks/ufw_long_rl_500k/build_stats.json")["n_examples"])
    n_each = min(n_sae, n_short, n_long)                                 # EXACT thirds
    spec = [{"name": "acts_short_rl", "banks": ["/data/banks/ufw_short_rl_500k"], "dtype": "f16", "families": ["realact"], "n": n_each,
             "all_one_family": "realact", "fresh": "Ultra-FineWeb docs [6.0M, ...) ordered stream; short-ctx pairs"},
            {"name": "acts_long_rl", "banks": ["/data/banks/ufw_long_rl_500k"], "dtype": "f32", "families": ["realact_long"], "n": {"realact_long": n_each},
             "fresh": "Ultra-FineWeb docs [7.0M, ...) 512-token store; deep positions p in [256,511], W in [16,64]"},
            {"name": "sae2m_rl", "banks": ["/data/banks/sae2m_rl"], "dtype": "f32", "families": ["sae2m", "sae2m_dec"],
             "n": {"sae2m": n_each // 2, "sae2m_dec": n_each - n_each // 2} if n_each < n_sae else None,
             "fresh": "2M-SAE RL feature split (150,000 features; disjoint from SFT and eval); enc + dec"}]
    if "compose_rl" not in d:
        c = modal.Function.from_name(APPS["compose"], "build").spawn(out_name=RL_POOL, spec_json=json.dumps(spec), seed=2041, threads=48,
                                                                      dense_frac=0.0, eval_cache=V3)
        d["compose_rl"] = {"call": c.object_id, "out": f"/data/banks/{RL_POOL}", "spec": spec, "n_each": n_each,
                           "avail": {"sae": n_sae, "short": n_short, "long": n_long}, "spawned": now()}; save(d)
        log(f"compose RL spawned {c.object_id}: {n_each} x 3 (avail sae {n_sae} short {n_short} long {n_long})")
    wait_call(d["compose_rl"]["call"], f"compose {RL_POOL}")
    st = vol_get_json(f"/banks/{RL_POOL}/build_stats.json")
    d["compose_rl"].update({"done": True, "n_examples": st["n_examples"], "families": st["families"]}); save(d)
    log(f"RL pool: {st['n_examples']} rows {st['families']}")

# ---- D. full-parameter RL (all-token reward) + evaluator ---------------------------------------------------------------
if not stage(d, "rl"):
    save_dir = f"/data/ckpts_{RL_RUN}"
    if "rl" not in d:
        extra = f"{RL_RECIPE} --run-name {RL_RUN} --save-dir {save_dir}"
        t = modal.Function.from_name(APPS["rl"], "train").spawn(n_rollout=2, n_trainer=6, total_steps=RL_STEPS, extra_args=extra,
                                                                 pool_dir=f"/data/banks/{RL_POOL}", policy_base=f"/data/sft_mix/{SFT_RUN}/final", full_param=True)
        e = modal.Function.from_name(APPS["eval"], "fullmodel_daemon").spawn(ckpt_dir=save_dir, tag=RL_RUN, wandb_name=f"{RL_RUN}_eval", final_step=RL_STEPS,
                                                                              extra_args=f"--eval-cache {V3} --no-extra-evals")
        d["rl"] = {"train": t.object_id, "eval": e.object_id, "run": RL_RUN, "save": save_dir, "policy_base": f"/data/sft_mix/{SFT_RUN}/final",
                   "pool": f"/data/banks/{RL_POOL}", "lr": "1e-6", "group_size": 8, "groups_per_step": 2048, "steps": RL_STEPS, "split": "2+6",
                   "reward": "max cosine over the WHOLE rollout span (--reward-window-last 0)", "extra": extra, "app": APPS["rl"], "eval_app": APPS["eval"],
                   "eval_cache": V3, "spawned": now()}; save(d)
        log(f"RL spawned: train {t.object_id} eval {e.object_id}")
        discord(f"SFT DONE -> full-param RL launched: {RL_RUN} (lr 1e-6, 2+6, 8x2048, whole-span reward, 300 steps) on {RL_POOL}; train {t.object_id}")
    wait_call(d["rl"]["train"], "RL train")
    d["rl"]["done"] = True; save(d)
    discord(f"RL DONE: {RL_RUN} 300 steps; evaluator {d['rl']['eval']} scores the last checkpoints (cache v3: mean_all + held-out 2M enc/dec)")
log("chain complete")
