"""Modal app `maemm-eval-ckpt`: ONE-GPU checkpoint eval daemon for the RL runs (eval/eval_ckpt_daemon.py).

Evaluates every <ckpt_dir>/step_* (+ final) the trainer saves -- rl.py's full inline_eval protocol
(512/family, Bo4, T=1, 16-64 tokens, SAE norm_act + rank) and the inline_extra_evals suite (locality, autointerp
AUC, WildChat AUC, adversarial holds; Sonnet 5 judge via Anthropic native with OpenRouter fallback) -- on one
B200 (or H200) hosting the HF base for scoring and a vLLM engine (fast steering hook + CUDA graphs) for generation.
Same image as modal_rl_disagg.py. Separate wandb run `eval_ckpt_<tag>` (x-axis ckpt_step).

Deploy + spawn (profile safety-sahan) -- a deployed function survives the local client:
    MODAL_PROFILE=safety-sahan EVAL_GPU=B200:1 modal deploy modal_eval_ckpt.py
    MODAL_PROFILE=safety-sahan python -c "import modal; modal.Function.from_name('maemm-eval-ckpt', 'daemon').spawn(
        ckpt_dir='/data/ckpts_last5_disagg_2x6', tag='last5_disagg_2x6', rl_run_id='<wandb id>',
        wandb_name='rl_everything_8x256_disagg_entropy2.0_last5win_eval')"
ONE GPU container at a time: the daemon holds its GPU while polling, so launch it only once the first checkpoint exists.
One-off (the step_90 protocol check):
    ... .spawn(ckpt_dir='/data/ckpts_last5_v15_g8', tag='last5_v15_g8', once=True, only_step=90)
Set EVAL_GPU (default B200:1; e.g. H200:1) at deploy time.
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent   # repo root (this launcher lives one level down)
app = modal.App("maemm-eval-ckpt")
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
    .add_local_file(REPO / "rl" / "rl_disagg.py", "/pmx/RL/rl_disagg.py")               # _build_engine (fast hook + graphs)
    .add_local_file(REPO / "rl" / "fast_lens_ext.py", "/pmx/helpers/fast_lens_ext.py")
    .add_local_file(REPO / "eval" / "inline_extra_evals.py", "/pmx/RL/inline_extra_evals.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
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
    import subprocess
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
    p = subprocess.Popen(cmd, cwd="/pmx", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # The daemon polls the volume from a SUBPROCESS, where `modal.Volume.from_name(...).reload()` is not the mounted
    # handle (it silently no-ops) -> it would never see checkpoints committed after it started. Refresh the mount from
    # THIS process instead — but only while the daemon is idle: a reload during its model/asset load or a checkpoint
    # eval invalidates open file handles (that is how the first attempt died with LocalEntryNotFoundError).
    import threading
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
    for line in p.stdout:
        print(line, end="", flush=True)
        if "previously evaled:" in line:
            _state["loaded"] = True
        if "evaluating LATEST step" in line or "-> evaluating" in line or "] evaluating step" in line:
            _state["busy"] = True
        if "evaled in" in line or "adapter load failed" in line or "nothing pending" in line:
            _state["busy"] = False
    rc = p.wait()
    _stop.set()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"eval daemon exited rc={rc}")


@app.local_entrypoint()
def main(ckpt_dir: str = "/data/ckpts_last5_v15_g8", tag: str = "last5_v15_g8", rl_run_id: str = "", once: bool = False,
         only_step: int = -1, final_step: int = 1000, extra_args: str = ""):
    daemon.remote(ckpt_dir=ckpt_dir, tag=tag, rl_run_id=rl_run_id, once=once, only_step=only_step, final_step=final_step,
                  extra_args=extra_args)
