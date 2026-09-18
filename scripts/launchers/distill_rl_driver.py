"""Distillation chain, stage 3: wait for the distilled-bank SFT (ids.json["distill"]["sft_raw"]) to finish, then launch full-parameter RL
with the ORIGINAL (raw-cosine) reward from its final checkpoint — the simple2m RL recipe verbatim (8x2048, lr 1e-6, whole-span reward,
300 steps, pool mix_simple2m_rl) on the Sep-17 raw-reward deployments — plus the raw evaluator daemon. Records ids.json["distill"]["rl"].

    cd ~/maemm-pub-simple2m && nohup python3 scripts/launchers/distill_rl_driver.py >> ~/shared/overnight/simple2m/distill_rl_driver.log 2>&1 &
"""
import json
import os
import subprocess
import time

import modal

os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
IDS = os.path.expanduser("~/shared/overnight/simple2m/ids.json")
V3 = "/data/eval_universal_ho/eval_sets_heldout_v3.pt"
APPS = {"rl": "maemm-rl-disagg-fullparam-s2m", "eval": "maemm-eval-ckpt-s2m"}          # Sep-17 deployments = raw-cosine reward / scorer
RL_RUN, RL_POOL, RL_STEPS = "rl_distill_fresh2m_bo16_8x2048_anywin", "mix_simple2m_rl", 300
RL_RECIPE = ("--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 "
             "--max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 2048 "
             "--group-size 8 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 0 "
             "--prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --transcript-every 5 "
             "--lr 1e-6 --save-steps 25,50,100,150,200,250,300 --eval-cache " + V3)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(m):
    print(f"[{now()}] {m}", flush=True)


def load():
    return json.load(open(IDS))


def save(d):
    json.dump(d, open(IDS, "w"), indent=1)


def vol_has(path):
    r = subprocess.run(["modal", "volume", "ls", "maemm-data", path], capture_output=True, text=True, env={**os.environ, "MODAL_PROFILE": "safety-sahan"})
    return r.returncode == 0 and bool(r.stdout.strip())


def discord(msg):
    try:
        subprocess.run(["notify-discord", msg], check=False, timeout=30)
    except Exception:  # noqa
        pass


d = load(); S = d["distill"]["sft_raw"]
log(f"driver start; waiting for SFT {S['run']} (train {S['train']})")
fc = modal.FunctionCall.from_id(S["train"])
while True:
    try:
        fc.get(timeout=0); break
    except TimeoutError:
        time.sleep(120)
    except Exception as e:
        log(f"SFT train call FAILED: {type(e).__name__}: {str(e)[:300]}")
        if vol_has(f"/sft_mix/{S['run']}/final/SAVE_DONE"):
            log("final/SAVE_DONE exists anyway — continuing"); break
        discord(f"distill SFT {S['run']} FAILED ({type(e).__name__}); RL not launched"); raise SystemExit(1)
ok = False
for _ in range(40):
    if vol_has(f"/sft_mix/{S['run']}/final/SAVE_DONE"):
        ok = True; break
    time.sleep(60)
if not ok:
    discord("distill SFT final has no SAVE_DONE — RL NOT launched"); log("SFT final missing"); raise SystemExit(1)
d = load(); d["distill"]["sft_raw"]["done"] = True; d["distill"]["sft_raw"]["finished"] = now(); save(d)
log("SFT done")
if "rl" in d["distill"]:
    log(f"RL already recorded: {d['distill']['rl']['train']}"); raise SystemExit(0)
save_dir = f"/data/ckpts_{RL_RUN}"
policy_base = d["distill"].get("rl_policy_base") or f"/data/sft_mix/{S['run']}/final"
extra = f"{RL_RECIPE} --run-name {RL_RUN} --save-dir {save_dir}"
t = modal.Function.from_name(APPS["rl"], "train").spawn(n_rollout=2, n_trainer=6, total_steps=RL_STEPS, extra_args=extra,
                                                         pool_dir=f"/data/banks/{RL_POOL}", policy_base=policy_base, full_param=True)
e = modal.Function.from_name(APPS["eval"], "fullmodel_daemon").spawn(ckpt_dir=save_dir, tag=RL_RUN, wandb_name=f"{RL_RUN}_eval", final_step=RL_STEPS,
                                                                      extra_args=f"--eval-cache {V3} --no-extra-evals")
d = load()
d["distill"]["rl"] = {"train": t.object_id, "eval": e.object_id, "run": RL_RUN, "save": save_dir, "policy_base": policy_base, "pool": f"/data/banks/{RL_POOL}",
                      "lr": "1e-6", "group_size": 8, "groups_per_step": 2048, "steps": RL_STEPS, "split": "2+6",
                      "reward": "ORIGINAL raw cosine, max over the whole rollout span (--reward-window-last 0), length penalty .00025/tok beyond 8",
                      "extra": extra, "app": APPS["rl"], "eval_app": APPS["eval"], "eval_cache": V3, "spawned": now()}
save(d)
log(f"RL spawned: train {t.object_id} eval {e.object_id}")
discord(f"distill chain: SFT on the best-of-16 harvest DONE -> RL launched from {policy_base}: {RL_RUN} (raw reward, lr 1e-6, 2+6, 8x2048, whole-span, 300 steps); train {t.object_id}")
