"""Big-batch SFT on the scaled pretrain data (user: "SFT run with the scaled PT data on a much higher batch size"): the full 23M realact bank,
effective batch 4096 (mb64 x 8 GPUs x grad-accum 8; 8x the 512 used by every SFT so far), lr 3e-4 (sqrt-scaled from 1e-4 @512; the sweep's top
stable lr), otherwise the 23M run's recipe (prefix-cache trainer, OneCycle, max_seq 160). 5616 optimizer steps, ~12.5 h on 8xB200."""
import json, modal
name = "realact23m_bs4096_lr3e-4"
extra = "--prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 10 --grad-accum 8"
t = modal.Function.from_name("maemm-sft-8xb200", "train").spawn(run_name=name, data_dir="/data/banks/realact_short_20m_all", n_ckpts=10, epochs=1,
                                                                 batch_size=64, lr=3e-4, max_seq=160, backend="nccl", extra_args=extra)
e = modal.Function.from_name("maemm-eval-ckpt-mlp", "daemon").spawn(ckpt_dir=f"/data/sft_mix/{name}", tag=f"sft_{name}", rl_run_id="", wandb_name=f"{name}_eval",
                                                                    final_step=5616, extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt")
json.dump({name: {"train": t.object_id, "eval": e.object_id, "eff_batch": 4096, "lr": 3e-4, "steps": 5616, "data": "/data/banks/realact_short_20m_all"}},
          open("/home/celeste/shared/overnight/sft_bs4096_ids.json", "w"), indent=1)
print("SFT train", t.object_id, "eval", e.object_id)
