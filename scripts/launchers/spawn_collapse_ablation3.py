"""Collapse ablation round 3: variance-floor filter. Resume RL-D from step_250 (last healthy ckpt) for 60 steps, identical to rounds 1/2
(4xB200 = 1 rollout + 3 trainer, 16x256, lr 1e-5, --max-lag 8, bank mix_1m_v2), with --zero-var-eps raised so groups whose reward std is
below the threshold leave the effective batch (loss weight 0, out of the denominators). Default eps 1e-6 never fires (continuous cosine reward)."""
import json, modal, os, sys
INIT = "/data/ckpts_rl_D_mix1m_lr1e-5/step_250"; REF = "/data/sft_mix/mix1m_from_realact23m/final"
BASE = ("--recipe scalerl --loss cispo --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 8 --fp32-head --autocast-bf16 --length-control penalty "
        "--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 256 --group-size 16 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
        f"--init-adapter {INIT} --ref-adapter {REF} --step-offset 251 --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --save-steps 280 --transcript-every 5 ")
ARMS = {
    "zv01": "--lr 1e-5 --cispo-eps-max 5 --zero-var-eps 0.01",   # drop groups with within-group reward std <= 0.01 (mean group std ~0.028 at step 250)
    "zv02": "--lr 1e-5 --cispo-eps-max 5 --zero-var-eps 0.02",   # ... <= 0.02
}
only = sys.argv[1:] or list(ARMS)
train = modal.Function.from_name("maemm-rl-disagg-x4", "train"); ev = modal.Function.from_name("maemm-eval-ckpt", "daemon")
P = "/home/celeste/shared/overnight/collapse_ablation3_ids.json"
ids = json.load(open(P)) if os.path.exists(P) else {}
for arm in only:
    name, save = f"rl_ablate3_{arm}", f"/data/ckpts_ablate3_{arm}"
    extra = BASE + ARMS[arm] + f" --run-name {name} --save-dir {save}"
    t = train.spawn(n_rollout=1, n_trainer=3, total_steps=311, extra_args=extra, pool_dir="/data/banks/mix_1m_v2")
    e = ev.spawn(ckpt_dir=save, tag=name, rl_run_id="", wandb_name=f"{name}_eval", final_step=310)
    ids[arm] = {"train": t.object_id, "eval": e.object_id, "save": save, "extra": ARMS[arm]}
    print(f"{arm:6s} train {t.object_id} eval {e.object_id}")
json.dump(ids, open(P, "w"), indent=1)
