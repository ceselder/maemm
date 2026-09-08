"""SFT on the full 23M real-activation bank at EFFECTIVE BATCH 16,384 (8 GPUs x 64/GPU x grad-accum 32; the baseline realact20m_prefix_lr1e-4 used 512 = 8 x 64).
23M / 16384 = 1,404 optimizer steps (baseline 44,922). lr 5e-4 = sqrt(32) x the baseline's 1e-4 (the eff-512 lr sweep was flat .364-.369 from 3e-5 to 5e-4 and diverged at 1e-3).
Same everything else: LoRA r64/a16 rsLoRA, prefix-cache, max_seq 160, OneCycle linear anneal, 14 ckpts + example-count saves. Evaluator = same app/cache as the baseline's eval run."""
import json, modal
run = "realact23m_b16k_lr1e-4"
extra = ("--prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 5 --grad-accum 32 "
         "--save-examples 250000,500000,1000000,2000000,5000000,10000000")
t = modal.Function.from_name("maemm-sft-8xb200", "train").spawn(run_name=run, data_dir="/data/banks/realact_short_20m_all", n_ckpts=14, epochs=1,
                                                                  batch_size=64, lr=1e-4, max_seq=160, backend="nccl", extra_args=extra)
e = modal.Function.from_name("maemm-eval-ckpt", "daemon").spawn(ckpt_dir=f"/data/sft_mix/{run}", tag=f"sft_{run}", rl_run_id="", wandb_name=f"{run}_eval", final_step=1404)
json.dump({run: {"train": t.object_id, "eval": e.object_id, "save": f"/data/sft_mix/{run}", "eff_batch": 16384, "lr": 1e-4, "steps_total_expected": 1404,
                 "baseline": {"run": "realact20m_prefix_lr1e-4", "eff_batch": 512, "lr": 1e-4, "train_wandb": "da7cxuz3", "eval_wandb_name": "realact20m_prefix_lr1e-4_eval"}}},
          open("/home/celeste/shared/overnight/sft_b16k_ids.json", "w"), indent=1)
print("SFT train", t.object_id, "eval", e.object_id)
