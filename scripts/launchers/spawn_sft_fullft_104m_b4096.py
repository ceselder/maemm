"""FULL fine-tune of Qwen3.6-27B on the 104M real-activation corpus (7 part banks) at EFFECTIVE BATCH 4,096 (batch 64/GPU x grad-accum 8 x 8 GPUs), lr 1e-5
(the Sep-5 FFT sweep optimum at eff 512; user: 'fft at 4096 batch at 1e-5'). FSDP2 fp32 masters, prefix cache, --pad-multiple 1, loss logged every step.
104M / 4096 = 25,390 optimizer steps (~10.5 s each -> ~74 h over 24 h Modal legs; the supervisor resumes from full checkpoints)."""
import json, modal
run = "realact104m_fullft_b4096_lr1e-5"
data = "/data/banks/realact_short_50m_all,/data/banks/realact_short_20m_h,/data/banks/realact_short_20m_i,/data/banks/realact_short_20m_j,/data/banks/realact_short_20m_k,/data/banks/realact_short_20m_l,/data/banks/realact_short_20m_m"
extra = "--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 8 --pad-multiple 1 --save-examples 2000000,5000000,10000000,23000000,50000000"
t = modal.Function.from_name("maemm-sft-fullft", "train").spawn(run_name=run, data_dir=data, n_ckpts=14, epochs=1, batch_size=64, lr=1e-5, max_seq=160, backend="nccl", extra_args=extra)
e = modal.Function.from_name("maemm-eval-ckpt-fullft", "fullmodel_daemon").spawn(ckpt_dir=f"/data/sft_mix/{run}", tag=f"sft_{run}", wandb_name=f"{run}_eval", final_step=104_000_000 // 4096 + 1)
p = "/home/celeste/shared/overnight/sft_fullft_100m_ids.json"; d = json.load(open(p))
d[run] = {"train": t.object_id, "eval": e.object_id, "save": f"/data/sft_mix/{run}", "data": data, "eff_batch": 4096, "micro_batch": 64, "grad_accum": 8, "lr": 1e-5, "steps_expected": 104_000_000 // 4096, "app": "maemm-sft-fullft"}
json.dump(d, open(p, "w"), indent=1); print(f"FFT b4096 spawned: train {t.object_id} eval {e.object_id}")
