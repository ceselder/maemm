"""FULL fine-tune (FSDP2, fp32 masters, bf16 compute, prefix cache) of Qwen3.6-27B on the 104M real-activation bank at EFFECTIVE BATCH 16,384.
Usage: python3 /tmp/spawn_sft_fullft_100m.py <micro_batch_per_gpu> [grad_ckpt=0]   (grad_accum = 16384 / (8 * micro_batch))
lr 5e-5 (sqrt-scaled from the Sep-5 FFT sweep optimum 1e-5 @ eff 512; FFT lr ~ LoRA lr / 10). Logs every step. Evaluator = full-model daemon."""
import json, sys, modal
mb = int(sys.argv[1]) if len(sys.argv) > 1 else 64
gck = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ga = 16384 // (8 * mb); assert ga * 8 * mb == 16384, "micro batch must divide 2048"
run = "realact104m_fullft_b16k_lr5e-5"; data = "/data/banks/realact_short_50m_all,/data/banks/realact_short_20m_h,/data/banks/realact_short_20m_i,/data/banks/realact_short_20m_j,/data/banks/realact_short_20m_k,/data/banks/realact_short_20m_l,/data/banks/realact_short_20m_m"   # 7 part banks read directly (multi-bank loader 97b2a49) = 104M examples
extra = (f"--full-ft --prefix-cache --grad-ckpt {gck} --autocast-bf16 --log-steps 1 --grad-accum {ga} --pad-multiple 1 "
         f"--save-examples 2000000,5000000,10000000,23000000,50000000")
t = modal.Function.from_name("maemm-sft-fullft", "train").spawn(run_name=run, data_dir=data, n_ckpts=14, epochs=1, batch_size=mb, lr=5e-5, max_seq=160, backend="nccl", extra_args=extra)
e = modal.Function.from_name("maemm-eval-ckpt-fullft", "fullmodel_daemon").spawn(ckpt_dir=f"/data/sft_mix/{run}", tag=f"sft_{run}", wandb_name=f"{run}_eval", final_step=104_000_000 // 16384 + 1)
json.dump({run: {"train": t.object_id, "eval": e.object_id, "save": f"/data/sft_mix/{run}", "data": data, "eff_batch": 16384, "micro_batch": mb, "grad_accum": ga, "grad_ckpt": gck, "lr": 5e-5,
                 "steps_expected": 104_000_000 // 16384, "app": "maemm-sft-fullft"}}, open("/home/celeste/shared/overnight/sft_fullft_100m_ids.json", "w"), indent=1)
print(f"FFT spawned: train {t.object_id} eval {e.object_id} (mb {mb} x ga {ga} x 8 = 16384, grad-ckpt {gck})")
