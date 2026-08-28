"""Modal app: Dr.GRPO RL (train/rl.py) on a single 8xB200 container.

The trainer lives at train/rl.py in this repo; it is mounted into the container at
/pmx/RL/rl_hf.py (its original path on the training box, which the commands below expect),
with mxf/ importable via PYTHONPATH=/pmx/helpers. Port of the B300-box run
`big_rl_longhz_dp4_lp025`. Data lives on the `maemm-data` Volume (uploaded via
`modal volume put`):
    /data/pool_rl_mix        direction bank (vecs.f32 750k x 5120 f32 + records.jsonl + build_stats.json)
    /data/sft_init           init LoRA adapter (also the frozen KL reference)
    /data/sae/{ae.pt,maxacts.pt}, /data/pool_heldout, /data/eval_universal_ho   (future evals)
    /data/hf_cache           HF_HOME (Qwen/Qwen3.6-27B downloads once, persists)
    /data/ckpts              output checkpoints (step_25, step_50, ... final)

Needs two Modal secrets in your workspace: `maemm-hf` (HF_TOKEN) and `maemm-wandb`
(WANDB_API_KEY). Launch from the repo root:
    modal run --detach modal_rl.py
Options:
    --backend gloo           if NCCL hangs at the first collective (box-specific bug; try nccl first)
    --total-steps N          override step count (default 400)
"""

from pathlib import Path

import modal

REPO = Path(__file__).parent

app = modal.App("maemm-rl-8xb200")

# torch 2.10.0+cu128 == the box venv; cu128 wheels carry sm_100 (B200) kernels.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
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
    .add_local_file(REPO / "train" / "rl.py", "/pmx/RL/rl_hf.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

TRAIN_ARGS = [
    "--bank-file", "vecs.f32",
    # Start RL FRESH from the SFT adapter (= start policy AND KL ref). Warm-starting from the
    # deep-RL dp4/step_100 collapsed even at LP 1.0 (fresh optimizer + weak KL anchor + policy
    # already near the reward-hack cliff). SFT-init is the proven-stable pattern (uni_rl: 270
    # steps no collapse; original longhz: 100 steps stable).
    "--init-adapter", "/data/sft_init",
    "--lr", "1e-5",
    "--reward-metric", "cosine",
    "--reward-scale", "1000",
    "--min-new-tokens", "16",
    "--max-new-tokens", "96",
    "--len-penalty-start", "16",
    # LP 1.0/tok is the PROVEN-STABLE value (dp4: 117 steps, gate 0.92-0.97, clip <1.3%).
    # 0.25/tok was the root cause of every box collapse — do NOT lower it again.
    "--len-penalty-per-tok", "1.0",
    # --div-coef is a launch parameter (see train()/main); div2000 is safe at LP 1.0 (dp4-proven).
    "--kl-coef", "0.03",
    "--groups-per-step", "32",   # 32 % 8 == 0 -> 4 whole groups per rank
    "--group-size", "16",
    "--rollout-chunk", "64",
    # micro-batch 4 (box used 8): update() peaked OOM on 178GB B200s at gen len ~42. Pure grad-
    # accumulation slicing — global-token-normalized loss makes gradients identical to mb=8.
    "--micro-batch", "4",
    "--score-batch", "24",
    "--save-every", "25",
    "--eval-every", "0",
    "--sae-eval-every", "0",
    "--save-dir", "/data/ckpts",
    "--run-name", "maemm-modal-8xb200",
]


@app.function(
    image=image,
    gpu="B200:8",
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=86400,
)
def train(backend: str = "gloo", total_steps: int = 400, div_coef: float = 0.0):
    # backend MUST be gloo: rl_hf.py's _ddp_sync_grads/all_gather run on CPU tensors by design
    # ("gloo: CPU tensor, no NCCL anywhere") — under NCCL they raise
    # "RuntimeError: No backend type associated with device type cpu" at the first collective.
    import os
    import shutil
    import subprocess
    import threading
    import time

    # ---- collapse-fix guard: the mounted trainer MUST carry the gate-masked diversity bonus
    # (_gmask). Refuse to train on unpatched code — the un-masked div bonus caused the box
    # collapse (reward 390->53, gate 90%->8%). ----
    with open("/pmx/RL/rl_hf.py") as f:
        _src_lines = f.read().splitlines()
    _hits = [f"{i + 1}: {l.strip()}" for i, l in enumerate(_src_lines) if "_gmask" in l]
    assert len(_hits) >= 2, f"collapse fix (_gmask) MISSING from mounted rl_hf.py — got {_hits}"
    print("[modal] collapse-fix check OK:\n  " + "\n  ".join(_hits), flush=True)

    os.environ["HF_HOME"] = "/data/hf_cache"

    # single-flight base-model download into the persistent volume (avoids 8 ranks racing)
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download("Qwen/Qwen3.6-27B")
    vol.commit()
    print(f"[modal] base model in cache ({time.time() - t0:.0f}s)", flush=True)

    # stage the direction bank onto container-local disk: memmap over the volume FUSE mount is
    # the one thing we don't trust, and per-step random row reads are faster off local NVMe.
    t0 = time.time()
    local_pool = "/root/pool_rl_mix"
    if not os.path.exists(local_pool):
        shutil.copytree("/data/pool_rl_mix", local_pool)
    print(f"[modal] pool staged to {local_pool} ({time.time() - t0:.0f}s)", flush=True)

    # periodic volume commit so checkpoints land even if the container dies mid-run
    def _committer():
        while True:
            time.sleep(300)
            try:
                vol.commit()
            except Exception as e:
                print(f"[modal] vol.commit failed: {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers"
    env["DDP_BACKEND"] = backend
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"          # /pmx is a read-only mount; wandb writes to cwd otherwise
    # ranks must load PURELY from the validated cache: 8 concurrent hub re-resolutions returned
    # spurious "does not appear to have a file named model-0000X-of-00015.safetensors" even though
    # the hub file exists and the cached snapshot (same sha) is complete. snapshot_download above
    # (driver, online) already guarantees completeness.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # B200 = 178GB (box B300 = 288GB): variable-length RL batches fragment the caching allocator —
    # rank 0 OOM'd at step 3 with 159GB allocated failing a 24MB alloc. expandable_segments is the
    # canonical fix (recommended by the OOM message itself).
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs("/tmp/wandb", exist_ok=True)

    cmd = [
        "torchrun", "--nproc_per_node=8", "--master_port=29531", "RL/rl_hf.py",
        "--data-dir", local_pool,
        "--total-steps", str(total_steps),
        "--div-coef", str(div_coef),
    ] + TRAIN_ARGS
    print("[modal] launching:", " ".join(cmd), f"(DDP_BACKEND={backend})", flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"torchrun exited rc={rc} (backend={backend})")
    print("[modal] training complete", flush=True)


@app.function(
    image=image,
    gpu="B200:1",
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=7200,
)
def smoke():
    """1xB200 pipeline validation: pre-warms /data/hf_cache (so the 8x run skips the 55GB
    download) and runs 2 tiny world=1 steps (no wandb, ckpts to /tmp)."""
    import os
    import shutil
    import subprocess
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download("Qwen/Qwen3.6-27B")
    vol.commit()
    print(f"[modal-smoke] base model cached ({time.time() - t0:.0f}s)", flush=True)

    local_pool = "/root/pool_rl_mix"
    if not os.path.exists(local_pool):
        shutil.copytree("/data/pool_rl_mix", local_pool)
    print("[modal-smoke] pool staged", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"
    os.makedirs("/tmp/wandb", exist_ok=True)
    cmd = [
        "python", "RL/rl_hf.py",
        "--data-dir", local_pool,
        "--bank-file", "vecs.f32",
        "--init-adapter", "/data/warmstart",
        "--lr", "1e-5", "--reward-metric", "cosine", "--reward-scale", "1000",
        "--min-new-tokens", "16", "--max-new-tokens", "96",
        "--len-penalty-start", "16", "--len-penalty-per-tok", "0.25",
        "--div-coef", "0", "--kl-coef", "0.03",
        "--groups-per-step", "2", "--group-size", "4",
        "--rollout-chunk", "8", "--micro-batch", "4", "--score-batch", "8",
        "--total-steps", "2", "--save-every", "0",
        "--eval-every", "0", "--sae-eval-every", "0",
        "--save-dir", "/tmp/smoke_ckpt", "--no-wandb",
    ]
    print("[modal-smoke] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"smoke exited rc={rc}")
    print("[modal-smoke] OK", flush=True)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=7200,
    cpu=8,
)
def prewarm():
    """CPU-only: download the base model into the persistent HF cache on the volume."""
    import os
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download("Qwen/Qwen3.6-27B")
    vol.commit()
    print(f"[modal-prewarm] base model cached ({time.time() - t0:.0f}s)", flush=True)


@app.local_entrypoint()
def main(backend: str = "gloo", total_steps: int = 400, div_coef: float = 0.0):
    train.remote(backend=backend, total_steps=total_steps, div_coef=div_coef)


@app.local_entrypoint()
def run_prewarm():
    prewarm.remote()


@app.local_entrypoint()
def run_smoke():
    smoke.remote()
