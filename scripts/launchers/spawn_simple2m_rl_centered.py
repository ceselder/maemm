"""The simple2m main RL arm AGAIN, identical in every flag, from the SAME SFT final on the SAME RL pool, with the ONE change of the
2026-09-18 scorer fix: the reward is cos(unit(h - mu), d) instead of cos(unit(h), d) (see eval_universal.center_mu). Apps deployed from
this worktree after the fix: maemm-rl-disagg-fullparam-s2m-c (trainer, B200:8, transformers fork pin) and maemm-eval-ckpt-s2m-c
(evaluator, centered). Records ids.json["rl_centered"].

    source ~/modal_venv/bin/activate && MODAL_PROFILE=safety-sahan python3 scripts/launchers/spawn_simple2m_rl_centered.py
"""
import json
import os
import time

import modal

os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
APP, EVAL_APP = "maemm-rl-disagg-fullparam-s2m-c", "maemm-eval-ckpt-s2m-c"
V3 = "/data/eval_universal_ho/eval_sets_heldout_v3.pt"
RUN = "rl_simple2m_8x2048_anywin_centered"
SAVE = f"/data/ckpts_{RUN}"
POOL = "/data/banks/mix_simple2m_rl"
POLICY = "/data/sft_mix/simple2m_sft/final"
IDS = os.path.expanduser("~/shared/overnight/simple2m/ids.json")
# == simple2m_chain_driver.RL_RECIPE, verbatim (2026-09-17 main arm) -- nothing re-tuned, so any difference is the reward fix
RL_RECIPE = ("--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 "
             "--max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 2048 "
             "--group-size 8 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 0 "
             "--prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --transcript-every 5 "
             "--lr 1e-6 --save-steps 25,50,100,150,200,250,300 --eval-cache " + V3)
STEPS = 300


def main():
    d = json.load(open(IDS))
    assert "rl_centered" not in d, f"already launched: {d['rl_centered']}"
    extra = f"{RL_RECIPE} --run-name {RUN} --save-dir {SAVE}"
    t = modal.Function.from_name(APP, "train").spawn(n_rollout=2, n_trainer=6, total_steps=STEPS, extra_args=extra, pool_dir=POOL, policy_base=POLICY, full_param=True)
    e = modal.Function.from_name(EVAL_APP, "fullmodel_daemon").spawn(ckpt_dir=SAVE, tag=RUN, wandb_name=f"{RUN}_eval", final_step=STEPS,
                                                                     extra_args=f"--eval-cache {V3} --no-extra-evals")
    d["rl_centered"] = {"train": t.object_id, "eval": e.object_id, "run": RUN, "save": SAVE, "policy_base": POLICY, "pool": POOL, "steps": STEPS, "split": "2+6",
                        "lr": "1e-6", "group_size": 8, "groups_per_step": 2048, "reward": "CENTERED max cosine over the whole span: cos(unit(h - mu), d) (scorer fix f9b5095)",
                        "differs_from_main_arm": "reward/eval scorer centering only; every flag identical; same SFT final; same pool",
                        "extra": extra, "app": APP, "eval_app": EVAL_APP, "eval_cache": V3, "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(d, open(IDS, "w"), indent=1)
    print(json.dumps({k: d["rl_centered"][k] for k in ("train", "eval", "run", "save")}))


if __name__ == "__main__":
    main()
