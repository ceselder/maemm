"""Switch the running 104M full fine-tune (realact104m_fullft_b4096_lr1e-5) onto the MFU-improved trainer at a checkpoint boundary.
Usage: python3 /tmp/resume_fft_b4096_fast.py <ckpt_step>     (the step_<N> dir must carry SAVE_DONE)
Steps: (1) per-run resume_paused marker (supervisor hands-off), (2) cancel the current train call + stop its container, (3) patch
run_meta.json extra_args on the volume so FUTURE supervisor resumes keep the fast flags, (4) spawn train(resume_from=step_N, skip_steps=N+1,
wandb_id=<same run>, extra_args=FAST), (5) update the ids file. AFTER the new leg's heartbeat is live: `modal volume rm maemm-data sft_mix/<run>/resume_paused`.
Fast flags (agent fullft-mfu, merged 4e5f1d0; 58 -> ~90 ex/s/rank): --prefix-share-step (exact) --fsdp-prefetch 2 (exact) --fsdp-keep-unsharded 40
(comm-only, exact; +16 GB) --optim adamw-mbf16 (bf16 first moment, fp32 second; optimizer numerics change — state is reset at every leg anyway).
NOT adding --length-bucket (changes the --skip-steps replay order mid-run)."""
import json, subprocess, sys, time
import modal
step = int(sys.argv[1]); run = "realact104m_fullft_b4096_lr1e-5"; P = "/home/celeste/shared/overnight/sft_fullft_100m_ids.json"
FAST = "--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 8 --pad-multiple 1 --save-examples 2000000,5000000,10000000,23000000,50000000 --prefix-share-step --fsdp-prefetch 2"   # EXACT set only: keep-unsharded 40 + mbf16 hit 162 GB and thrashed (8.5 s/step, erratic) on the real run
def vls(path): return subprocess.run(["modal", "volume", "ls", "maemm-data", path], capture_output=True, text=True).stdout
assert "SAVE_DONE" in vls(f"sft_mix/{run}/step_{step}"), f"step_{step} has no SAVE_DONE yet"
d = json.load(open(P)); cur = d[run]["train"]
open("/tmp/rp_fft", "w").write(f"paused {time.strftime('%FT%TZ', time.gmtime())}: manual resume onto the fast trainer from step_{step}\n")
subprocess.run(["modal", "volume", "put", "maemm-data", "/tmp/rp_fft", f"sft_mix/{run}/resume_paused", "--force"], check=True, capture_output=True)
try: modal.FunctionCall.from_id(cur).cancel(); print("cancelled current leg", cur)
except Exception as e: print("cancel:", e)
out = subprocess.run(["modal", "container", "list"], capture_output=True, text=True).stdout
for line in out.splitlines():
    if "ap-qBXyaRykZdwuOuYYCjmOgW" in line:
        ct = line.split()[1] if line.split()[0] == "│" else line.split()[0]
        logs = subprocess.run(["modal", "container", "logs", ct], capture_output=True, text=True).stdout
        if "ep0 step" in logs or "staging" in logs:
            subprocess.run(["modal", "container", "stop", ct, "--yes"], capture_output=True); print("stopped container", ct)
subprocess.run(["modal", "volume", "get", "maemm-data", f"sft_mix/{run}/run_meta.json", "/tmp/rm_fft.json", "--force"], check=True, capture_output=True)
meta = json.load(open("/tmp/rm_fft.json")); meta["extra_args_before_fast_switch"] = meta["extra_args"]; meta["extra_args"] = FAST
json.dump(meta, open("/tmp/rm_fft.json", "w"), indent=2)
subprocess.run(["modal", "volume", "put", "maemm-data", "/tmp/rm_fft.json", f"sft_mix/{run}/run_meta.json", "--force"], check=True, capture_output=True); print("run_meta.extra_args patched (supervisor resumes will use the fast flags)")
t = modal.Function.from_name("maemm-sft-fullft", "train").spawn(run_name=run, data_dir=d[run]["data"], n_ckpts=14, epochs=1, batch_size=64, lr=1e-5, max_seq=160, backend="nccl",
                                                                  extra_args=FAST, resume_from=f"/data/sft_mix/{run}/step_{step}", skip_steps=step + 1, wandb_id=meta["wandb_id"])
d[run].setdefault("legs", []).append({"from_step": step, "call": t.object_id, "flags": "fast-exact (share+prefetch)"}); d[run]["train"] = t.object_id; d[run]["extra_args"] = FAST
json.dump(d, open(P, "w"), indent=1); print(f"FAST leg spawned {t.object_id} from step_{step} (skip {step + 1}, wandb {meta['wandb_id']}). Remove sft_mix/{run}/resume_paused once its heartbeat is live.")
