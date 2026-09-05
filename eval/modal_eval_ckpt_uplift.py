"""Modal app `maemm-eval-ckpt-uplift`: the checkpoint-eval daemon (eval/eval_ckpt_daemon.py) for the cross-uplift matrix.

Same function as eval/modal_eval_ckpt.py (deploy + spawn, one B200 per daemon, image + PYTHONPATH identical) with two
launcher fixes learned on 2026-09-05 while sharing the `maemm-eval-ckpt-mlp` deployment:
  1. `vol.reload()` BEFORE the daemon subprocess starts. Modal reuses warm containers whose /data mount predates files
     written by other containers; a `--once` daemon then sees no checkpoint and exits "nothing pending".
  2. The daemon subprocess runs in its own process group and is KILLED when the function ends or is cancelled. Without
     this a cancelled daemon keeps its ~150 GB of GPU memory in the warm container and the next daemon placed there dies
     with CUDA OOM at model load (seen: "Process 1 has 151.93 GiB memory in use").
The evaluator code itself is mounted from EVAL_REPO (default: this repo). For the uplift matrix it is the mlp42-bank
worktree (v2 eval cache: the 11 cos families + sae + the extra mlp / mlp_pair families; mean_all unchanged):

    EVAL_REPO=/home/celeste/maemm-pub/.claude/worktrees/mlp42-bank MODAL_PROFILE=safety-sahan modal deploy eval/modal_eval_ckpt_uplift.py
    python -c "import modal; modal.Function.from_name('maemm-eval-ckpt-uplift', 'daemon').spawn(ckpt_dir=..., tag=..., final_step=...,
        once=True, only_step=0, extra_args='--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt --no-extra-evals')"
"""
import os
from pathlib import Path

import modal

REPO = Path(os.environ.get("EVAL_REPO", Path(__file__).resolve().parent.parent))
app = modal.App(os.environ.get("EVAL_APP", "maemm-eval-ckpt-uplift"))
GPU = os.environ.get("EVAL_GPU", "B200:1")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
    .pip_install(
        "transformers==5.15.0",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "wandb==0.28.2",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    .pip_install("flash-linear-attention==0.5.2")
    .pip_install("anthropic")
    .add_local_file(REPO / "rl" / "rl.py", "/pmx/RL/rl_hf.py")
    .add_local_file(REPO / "rl" / "rl_disagg.py", "/pmx/RL/rl_disagg.py")
    .add_local_file(REPO / "rl" / "fast_lens_ext.py", "/pmx/helpers/fast_lens_ext.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "eval" / "inline_extra_evals.py", "/pmx/RL/inline_extra_evals.py")
    .add_local_file(REPO / "eval" / "eval_universal.py", "/pmx/eval/eval_universal.py")
    .add_local_file(REPO / "eval" / "snippet_locality.py", "/pmx/eval/snippet_locality.py")
    .add_local_file(REPO / "eval" / "autointerp_detection.py", "/pmx/eval/autointerp_detection.py")
    .add_local_file(REPO / "eval" / "eval_ckpt_daemon.py", "/pmx/eval/eval_ckpt_daemon.py")
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=False)


@app.function(image=image, gpu=GPU, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb"),
                       modal.Secret.from_name("maemm-anthropic"), modal.Secret.from_name("maemm-openrouter")],
              timeout=24 * 3600)
def daemon(ckpt_dir: str, tag: str, rl_run_id: str = "", poll_s: int = 120, once: bool = False, only_step: int = -1,
           final_step: int = 1000, vllm_gpu_mem: float = 0.5, wandb_name: str = "", extra_args: str = ""):
    import signal
    import subprocess
    import threading
    vol.reload()   # FIX 1: a reused warm container must see checkpoints committed after it was created
    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers:/pmx/eval:/pmx/RL"
    env["HF_HOME"] = "/data/hf_cache"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs("/tmp/wandb", exist_ok=True)
    cmd = ["python", "/pmx/eval/eval_ckpt_daemon.py", "--ckpt-dir", ckpt_dir, "--tag", tag, "--rl-run-id", rl_run_id,
           "--poll-s", str(poll_s), "--final-step", str(final_step), "--vllm-gpu-mem", str(vllm_gpu_mem)]
    if wandb_name:
        cmd += ["--wandb-name", wandb_name]
    if once:
        cmd.append("--once")
    if only_step >= 0:
        cmd += ["--only-step", str(only_step)]
    if extra_args:
        cmd += extra_args.split()
    print("[modal] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)   # FIX 2: own process group -> killable on cancel
    # The daemon polls the volume from a SUBPROCESS, where `modal.Volume.from_name(...).reload()` is not the mounted
    # handle (it silently no-ops) -> refresh the mount from THIS process while the daemon is idle (a reload during its
    # model/asset load or a checkpoint eval invalidates open file handles).
    _stop = threading.Event()
    _state = {"loaded": False, "busy": False}

    def _reload_loop():
        while not _stop.wait(30):
            if not _state["loaded"] or _state["busy"]:
                continue
            try:
                vol.reload()
            except Exception as e:  # noqa
                print(f"[modal] vol.reload failed: {e}", flush=True)
    threading.Thread(target=_reload_loop, daemon=True).start()
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            if "previously evaled:" in line:
                _state["loaded"] = True
            if "evaluating LATEST step" in line or "-> evaluating" in line or "] evaluating step" in line:
                _state["busy"] = True
            if "evaled in" in line or "adapter load failed" in line or "nothing pending" in line:
                _state["busy"] = False
        rc = p.wait()
    finally:
        _stop.set()
        if p.poll() is None:   # cancelled (or errored) while the daemon runs: do not leave a GPU-holding zombie behind
            print("[modal] terminating eval daemon process group (cancel/error)", flush=True)
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=20)
            except Exception:  # noqa
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:  # noqa
                    pass
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"eval daemon exited rc={rc}")


@app.local_entrypoint()
def main(ckpt_dir: str, tag: str, rl_run_id: str = "", once: bool = False, only_step: int = -1, final_step: int = 1000,
         extra_args: str = ""):
    daemon.remote(ckpt_dir=ckpt_dir, tag=tag, rl_run_id=rl_run_id, once=once, only_step=only_step, final_step=final_step,
                  extra_args=extra_args)
