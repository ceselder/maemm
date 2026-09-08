"""SFT (pretrain objective) lr sweep at a fixed 2M-example budget: same 23M realact bank (seeded random 2M subset), same
trainer/config as the 23M run except lr. One 8xB200 train() spawn + one 1xB200 evaluator (v2 cache, mlp families) per lr."""
import json, sys, time
import modal
LRS = [3e-4, 1e-3, 3e-5, 1e-4, 3e-3]
N = 2_000_000
extra = "--prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 20 --max-examples %d" % N
train = modal.Function.from_name("maemm-sft-8xb200", "train")
daemon = modal.Function.from_name("maemm-eval-ckpt-mlp", "daemon")
out = {}
for lr in LRS:
    name = f"lrsweep2m_lr{lr:g}"
    c = train.spawn(run_name=name, data_dir="/data/banks/realact_short_20m_all", n_ckpts=4, epochs=1,
                    batch_size=64, lr=lr, max_seq=160, backend="nccl", extra_args=extra)
    e = daemon.spawn(ckpt_dir=f"/data/sft_mix/{name}", tag=f"sft_{name}", rl_run_id="", wandb_name=f"{name}_eval",
                     final_step=3907, extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt")
    out[name] = {"lr": lr, "train_call": c.object_id, "eval_call": e.object_id, "spawned": time.time()}
    print(name, c.object_id, e.object_id, flush=True)
json.dump(out, open("/home/celeste/shared/overnight/lrsweep_ids.json", "w"), indent=1)
