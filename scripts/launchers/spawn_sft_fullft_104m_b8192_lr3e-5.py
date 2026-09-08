"""Comparison arm to realact104m_fullft_b4096_lr1e-5: SAME 104M corpus, full fine-tune, but EFFECTIVE BATCH 8,192 (64/GPU x ga 16 x 8) at lr 3e-5
(user 2026-09-08: 'I wonder if I put learning rate too low... try 3e-5 at 8k batch ... on a different node and see comparison').
Fast flags from the start (--prefix-share-step --fsdp-prefetch 2). 104M / 8192 = 12,695 steps. Same --save-examples points -> matched-examples evals."""
import json, modal
run = "realact104m_fullft_b8192_lr3e-5"
data = "/data/banks/realact_short_50m_all,/data/banks/realact_short_20m_h,/data/banks/realact_short_20m_i,/data/banks/realact_short_20m_j,/data/banks/realact_short_20m_k,/data/banks/realact_short_20m_l,/data/banks/realact_short_20m_m"
extra = "--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 16 --pad-multiple 1 --save-examples 2000000,5000000,10000000,23000000,50000000 --prefix-share-step --fsdp-prefetch 2"
t = modal.Function.from_name("maemm-sft-fullft", "train").spawn(run_name=run, data_dir=data, n_ckpts=14, epochs=1, batch_size=64, lr=3e-5, max_seq=160, backend="nccl", extra_args=extra)
e = modal.Function.from_name("maemm-eval-ckpt-fullft", "fullmodel_daemon").spawn(ckpt_dir=f"/data/sft_mix/{run}", tag=f"sft_{run}", wandb_name=f"{run}_eval", final_step=104_000_000 // 8192 + 1)
p = "/home/celeste/shared/overnight/sft_fullft_100m_ids.json"; d = json.load(open(p))
d[run] = {"train": t.object_id, "eval": e.object_id, "save": f"/data/sft_mix/{run}", "data": data, "eff_batch": 8192, "micro_batch": 64, "grad_accum": 16, "lr": 3e-5, "steps_expected": 104_000_000 // 8192, "app": "maemm-sft-fullft", "purpose": "lr/batch comparison arm vs b4096_lr1e-5 (matched --save-examples)"}
json.dump(d, open(p, "w"), indent=1); print(f"FFT b8192 lr3e-5 spawned: train {t.object_id} eval {e.object_id}")
