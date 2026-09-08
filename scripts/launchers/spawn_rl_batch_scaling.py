"""RL-F (16x512) and RL-G (16x1024): identical to RL-E (init mix1m_from_realact23m/final, bank mix_1m_mlp, constant lr 1e-5, ScaleRL/CISPO) except the
batch (directions per step) and the GPU split (3 rollout + 5 trainer on 8xB200). Constant lr on purpose: tests whether a bigger batch alone survives past
step ~270 (RL-E collapsed at 261); best checkpoints are picked by held-out eval regardless."""
import json, modal, sys
saves = "25,40,63,100,160,200,250,300,400"
def extra(groups, name, save):
    return (f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 8 --fp32-head --autocast-bf16 --length-control penalty "
            f"--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step {groups} --group-size 16 --lr 1e-5 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
            f"--init-adapter /data/sft_mix/mix1m_from_realact23m/final --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 --save-every 0 --save-steps {saves} --transcript-every 5 "
            f"--run-name {name} --save-dir {save}")
RUNS = {"F_16x512": (512, "rl_F_mixmlp_16x512", "/data/ckpts_rl_F_16x512"), "G_16x1024": (1024, "rl_G_mixmlp_16x1024", "/data/ckpts_rl_G_16x1024")}
only = sys.argv[1:] or list(RUNS)
train = modal.Function.from_name("maemm-rl-disagg", "train"); ev = modal.Function.from_name("maemm-eval-ckpt-mlp", "daemon")
ids = {}
for k in only:
    g, name, save = RUNS[k]
    t = train.spawn(n_rollout=3, n_trainer=5, total_steps=400, extra_args=extra(g, name, save), pool_dir="/data/banks/mix_1m_mlp")
    e = ev.spawn(ckpt_dir=save, tag=name, rl_run_id="", wandb_name=f"{name}_eval", final_step=400, extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt")
    ids[k] = {"train": t.object_id, "eval": e.object_id, "run": name, "save": save, "groups": g}; print(k, "train", t.object_id, "eval", e.object_id)
json.dump(ids, open("/home/celeste/shared/overnight/batch_scaling_ids.json", "w"), indent=1)
