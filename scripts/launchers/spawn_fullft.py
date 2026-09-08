"""Full-fine-tuning arms. Usage: python spawn_fullft.py train <micro_batch> <grad_accum>   |   python spawn_fullft.py eval"""
import json, os, subprocess, sys, time
import modal
LRS = [1e-5, 3e-5]
P = "/home/celeste/shared/overnight/fullft_ids.json"
out = json.load(open(P)) if os.path.exists(P) else {}
mode = sys.argv[1]
if mode == "train":
    MB, GA = int(sys.argv[2]), int(sys.argv[3]); assert MB * GA * 8 == 512
    extra = f"--full-ft --prefix-cache --grad-ckpt 0 --log-steps 20 --max-examples 2000000 --grad-accum {GA}"
    train = modal.Function.from_name("maemm-sft-fullft", "train")
    open("/tmp/resume_paused", "w").write("full-FT run: the maemm-sft-8xb200 supervisor must NOT resume it (LoRA image has no --full-ft); resume manually via maemm-sft-fullft train(resume_from=...)\n")
    for lr in LRS:
        name = f"fullft2m_lr{lr:g}"
        subprocess.run(["modal", "volume", "put", "maemm-data", "/tmp/resume_paused", f"/sft_mix/{name}/resume_paused", "--force"], check=True, capture_output=True)
        c = train.spawn(run_name=name, data_dir="/data/banks/realact_short_20m_all", n_ckpts=4, epochs=1, batch_size=MB, lr=lr, max_seq=160,
                        backend="nccl", extra_args=extra)
        out.setdefault(name, {}).update({"lr": lr, "micro_batch": MB, "grad_accum": GA, "train_call": c.object_id, "train_spawned": time.time()})
        print(name, "train", c.object_id, flush=True)
else:
    evald = modal.Function.from_name("maemm-eval-ckpt-fullft", "fullmodel_daemon")
    for lr in LRS:
        name = f"fullft2m_lr{lr:g}"
        e = evald.spawn(ckpt_dir=f"/data/sft_mix/{name}", tag=f"sft_{name}", wandb_name=f"{name}_eval", final_step=3907,
                        extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt --no-extra-evals")
        out.setdefault(name, {}).update({"eval_call": e.object_id, "eval_spawned": time.time()})
        print(name, "eval", e.object_id, flush=True)
json.dump(out, open(P, "w"), indent=1)
