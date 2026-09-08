"""Collapse ablation: resume RL-D from step_250 (last healthy ckpt) for 60 steps under one change each. 4xB200 (1 rollout + 3 trainer), same 16x256 batch."""
import json, modal, sys
INIT = "/data/ckpts_rl_D_mix1m_lr1e-5/step_250"; REF = "/data/sft_mix/mix1m_from_realact23m/final"
BASE = ("--recipe scalerl --loss cispo --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 8 --fp32-head --autocast-bf16 --length-control penalty "
        "--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 256 --group-size 16 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
        f"--init-adapter {INIT} --ref-adapter {REF} --step-offset 251 --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --save-steps 280 --transcript-every 5 ")
ARMS = {
    "control":  "--lr 1e-5 --cispo-eps-max 5",
    "lr3e-6":   "--lr 3e-6 --cispo-eps-max 5",
    "adam95":   "--lr 1e-5 --cispo-eps-max 5 --adam-betas 0.9 0.95 --adam-eps 1e-15",
    "iscap2":   "--lr 1e-5 --cispo-eps-max 2",
    "lrdecay":  "--lr 1e-5 --cispo-eps-max 5 --lr-decay linear --lr-decay-total-steps 400",
    "nolenpen": "--lr 1e-5 --cispo-eps-max 5 --no-len-penalty",
    "gclip03":  "--lr 1e-5 --cispo-eps-max 5 --max-grad-norm 0.3",
}
only = sys.argv[1:] or list(ARMS)
train = modal.Function.from_name("maemm-rl-disagg-x4", "train"); ev = modal.Function.from_name("maemm-eval-ckpt", "daemon")
ids = json.load(open("/home/celeste/shared/overnight/collapse_ablation_ids.json")) if __import__("os").path.exists("/home/celeste/shared/overnight/collapse_ablation_ids.json") else {}
for arm in only:
    name, save = f"rl_ablate_{arm}", f"/data/ckpts_ablate_{arm}"
    extra = BASE + ARMS[arm] + f" --run-name {name} --save-dir {save}"
    t = train.spawn(n_rollout=1, n_trainer=3, total_steps=311, extra_args=extra, pool_dir="/data/banks/mix_1m_v2")
    e = ev.spawn(ckpt_dir=save, tag=name, rl_run_id="", wandb_name=f"{name}_eval", final_step=310)
    ids[arm] = {"train": t.object_id, "eval": e.object_id, "save": save, "extra": ARMS[arm]}
    print(f"{arm:9s} train {t.object_id} eval {e.object_id}")
json.dump(ids, open("/home/celeste/shared/overnight/collapse_ablation_ids.json", "w"), indent=1)
