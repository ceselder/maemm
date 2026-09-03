"""Modal app: universal-inverter SFT (train/pretrain.py) on 8xB200, one container per datamix.

The trainer lives at train/pretrain.py in this repo; it is mounted into the container at
/pmx/SL/pretrain.py (its original path on the training box), with mxf/ importable via
PYTHONPATH=/pmx/helpers. Same image/volume as modal_rl.py (no vllm — pretrain is pure HF).
Data lives on the `maemm-data` Volume:
    /data/<bank>            SFT bank: records.jsonl ({"vec_idx", "target_text"} per line)
                            + vecs.f32 or vecs.f16 (N x 5120 f32/f16 memmap, N >= max(vec_idx)+1)
    /data/hf_cache          HF_HOME (Qwen/Qwen3.6-27B downloads once, persists)
    /data/sft_mix/<run>/    output: run_meta.json + heartbeat + step_* ckpts + final

Every run is an independent spawn — launch MANY datamixes in parallel:
    modal deploy modal_sft.py                     # registers train + the auto-resume supervisor
    modal run modal_sft.py::launch --run-name mix-a --data-dir /data/banks/mix_a
    modal run modal_sft.py::launch --run-name mix-b --data-dir /data/banks/mix_b
Each run saves --n-ckpts (default 14) evenly-spaced intermediate ckpts + final under
/data/sft_mix/<run-name>/ (the per-SFT-step curves; n-ckpts=0 = only `final` = the bug that
lost them last time — the harness refuses it). The supervisor respawns any run whose
heartbeat went stale before `final` exists (24h Modal cap / crash), resuming from the latest
step_N via --init-adapter + --skip-steps (batch order is deterministic) + --wandb-id.
Pause auto-resume: `touch /data/sft_mix/resume_paused` (global) or `<run>/resume_paused`.

Needs Modal secrets `maemm-hf` (HF_TOKEN) and `maemm-wandb` (WANDB_API_KEY).
"""

from pathlib import Path

import os

import modal

REPO = Path(__file__).resolve().parent.parent   # repo root (this launcher lives one level down)

APP_NAME = "maemm-sft-8xb200"
app = modal.App(APP_NAME)

# torch 2.10.0+cu128 == the box venv; cu128 wheels carry sm_100 (B200) kernels. Identical pins to
# modal_rl.py so both trainers see one environment; pretrain needs no vllm (pure HF fwd/bwd).
# --prefix-cache (sft/prefix_cache.py) needs the transformers fork with autograd-safe linear-attention cache writes:
#   SFT_TRANSFORMERS="transformers @ git+https://github.com/ceselder/transformers@e52940e567ab9a991a1c971c1094e340233baff3"
# Default stays the stock pin so existing deployments rebuild nothing.
_SFT_TRANSFORMERS = os.environ.get("SFT_TRANSFORMERS", "transformers==5.15.0")
image = modal.Image.debian_slim(python_version="3.11")
if "git+" in _SFT_TRANSFORMERS:
    image = image.apt_install("git")
image = (
    image
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        _SFT_TRANSFORMERS,
        "peft==0.20.0",
        "accelerate==1.14.0",
        "wandb==0.28.2",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    # fla: transformers' Qwen3.5/3.6 GatedDeltaNet uses flash-linear-attention's Triton chunk kernels when importable,
    # else a pure-torch fallback (the last5_rp/big_rp SFT runs used the fallback: 10.6% MFU). Same pin as the RL image.
    .pip_install("flash-linear-attention==0.5.2")
    # torchao: float8 training linears for --fp8-base (sft/fp8.py). 0.16.0 is the release built against torch 2.10.0
    # (pytorch/ao#2919 compat table; 0.17+ target 2.11). Pure-python path (casts + torch._scaled_mm), no torchao kernels.
    .pip_install("torchao==0.16.0")
)
# Hopper (H100/H200) opt-in at deploy/run time: SFT_TRITON=3.7.1. fla 0.5.2 refuses its gated chunk_bwd_dqkwg on Triton
# 3.4-3.7.0 on Hopper (fla #640). No vLLM in this image, so the global Triton can simply be upgraded.
_SFT_TRITON = os.environ.get("SFT_TRITON", "")
if _SFT_TRITON:
    image = image.pip_install(f"triton=={_SFT_TRITON}")
image = (
    image
    .add_local_file(REPO / "sft" / "pretrain.py", "/pmx/SL/pretrain.py")
    .add_local_file(REPO / "sft" / "prefix_cache.py", "/pmx/SL/prefix_cache.py")   # --prefix-cache sibling import
    .add_local_file(REPO / "sft" / "fp8.py", "/pmx/SL/fp8.py")            # --fp8-base (imported by pretrain.py)
    .add_local_file(REPO / "sft" / "fp8_eval.py", "/pmx/SL/fp8_eval.py")  # fp8 speed/fidelity harness (fp8_eval fn)
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

SFT_ROOT = "/data/sft_mix"
STALE_HEARTBEAT_S = 30 * 60   # live legs touch+commit the heartbeat every <=5 min; 30 min = dead
SPAWN_COOLDOWN_S = 3600       # never double-spawn while a fresh leg pulls image/model (~15-30 min)


VEC_BANK_FILES = (("vecs.f32", 4), ("vecs.f16", 2))   # (filename, bytes per element); same as pretrain.py


def _vec_bank_file(data_dir: str):
    """(filename, itemsize) of the vector bank in data_dir -- vecs.f32 preferred, else vecs.f16."""
    import os

    for fname, itemsize in VEC_BANK_FILES:
        if os.path.exists(f"{data_dir}/{fname}"):
            return fname, itemsize
    raise AssertionError(f"bank incomplete: neither vecs.f32 nor vecs.f16 in {data_dir}")


def _preflight(run_name: str, data_dir: str, n_ckpts: int, resume_from: str):
    """Shared guardrails + bank staging for train/smoke. Returns (save_dir, local_bank, d_model)."""
    import glob
    import json
    import os
    import shutil
    import sys
    import time

    # ---- ckpt-cadence guard: n_ckpts=0 means pretrain.py saves ONLY `final` — that is the exact
    # bug that lost the per-SFT-step curves. Refuse it for real runs. ----
    assert n_ckpts > 0, "n_ckpts must be > 0 (0 = only `final`, no per-SFT-step curve — the old bug)"
    assert run_name and "/" not in run_name, f"bad run_name {run_name!r} (path component, no slashes)"

    # ---- resume-support guard: the mounted trainer MUST carry the crash-resume flags the
    # supervisor relies on (--skip-steps fast-forward, --wandb-id, --n-ckpts). Refuse stale code. ----
    with open("/pmx/SL/pretrain.py") as f:
        _src = f.read()
    for _flag in ("--n-ckpts", "--skip-steps", "--wandb-id"):
        assert _flag in _src, f"mounted pretrain.py is stale: {_flag} missing (resume/ckpt support)"
    print("[modal] mounted-trainer check OK (--n-ckpts/--skip-steps/--wandb-id present)", flush=True)

    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import D_MODEL, MODEL  # single source of truth (5120, Qwen/Qwen3.6-27B)

    save_dir = f"{SFT_ROOT}/{run_name}"
    if os.path.exists(f"{save_dir}/final"):
        raise RuntimeError(f"{save_dir}/final exists — run complete; pick a new --run-name")
    prior = sorted(glob.glob(f"{save_dir}/step_*"))
    if prior and not resume_from:
        raise RuntimeError(f"{save_dir} already has {len(prior)} ckpts — a fresh leg would clobber "
                           "them; pick a new --run-name (the supervisor passes resume_from)")

    # single-flight base-model download into the persistent volume (avoids 8 ranks racing)
    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL)
    vol.commit()
    print(f"[modal] base model in cache ({time.time() - t0:.0f}s)", flush=True)

    # ---- bank schema guard, then stage onto container-local NVMe: memmap over the volume FUSE
    # mount is the one thing we don't trust, and per-batch random row reads are faster locally. ----
    assert os.path.isdir(data_dir), f"bank not found on volume: {data_dir}"
    assert os.path.exists(f"{data_dir}/records.jsonl"), f"bank incomplete: {data_dir}/records.jsonl missing"
    vec_file, itemsize = _vec_bank_file(data_dir)   # vecs.f32 or vecs.f16 (pretrain.py reads either)
    rec0 = json.loads(open(f"{data_dir}/records.jsonl").readline())
    assert "vec_idx" in rec0 and "target_text" in rec0, f"records.jsonl schema: got {sorted(rec0)}"
    vsize = os.path.getsize(f"{data_dir}/{vec_file}")
    assert vsize % (D_MODEL * itemsize) == 0, \
        f"{vec_file} = {vsize} B, not a multiple of one {D_MODEL}-x-{itemsize}B row"
    print(f"[modal] bank OK: {vsize // (D_MODEL * itemsize)} vecs x {D_MODEL} ({vec_file}) + records.jsonl", flush=True)

    t0 = time.time()
    local_bank = f"/root/bank_{os.path.basename(data_dir.rstrip('/'))}"
    if not os.path.exists(local_bank):
        shutil.copytree(data_dir, local_bank)
    print(f"[modal] bank staged to {local_bank} ({time.time() - t0:.0f}s)", flush=True)
    return save_dir, local_bank, D_MODEL


def _train_env(backend: str):
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers"
    # pretrain.py is STANDARD cuda-tensor DDP (unlike rl.py's by-design CPU collectives that force
    # gloo) — nccl is the correct default here; gloo is the training-box fallback ("NCCL deadlocks
    # on this box" was a box bug, not a trainer requirement).
    env["DDP_BACKEND"] = backend
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"          # /pmx is a read-only mount; wandb writes to cwd otherwise
    # ranks must load PURELY from the validated cache: 8 concurrent hub re-resolutions returned
    # spurious missing-shard errors on the RL app even though the cached snapshot was complete.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # variable-length padded batches fragment the caching allocator on 178GB B200s (RL app OOM'd on
    # a 24MB alloc with 159GB allocated); expandable_segments is the canonical fix.
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs("/tmp/wandb", exist_ok=True)
    return env


def _stream(cmd, env, tag):
    import subprocess

    print(f"[{tag}] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


@app.function(
    image=image,
    gpu=os.environ.get("SFT_GPU", "B200:8"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=86400,
    ephemeral_disk=600 * 1024,   # _preflight stages the bank locally: a 20M-example vecs.f16 is ~205 GB
    memory=256 * 1024,
)
def train(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
          batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
          backend: str = "nccl", extra_args: str = "",
          resume_from: str = "", skip_steps: int = 0, wandb_id: str = ""):
    # One SFT run (one datamix). run_name/data_dir/n_ckpts parameterize independent parallel
    # spawns. batch_size/lr/max_seq: 0 = TrainConfig defaults (64 / 3e-5 / 192). extra_args:
    # whitespace-split, appended last (argparse last-wins), e.g. "--compile --log-steps 50".
    # resume_from/skip_steps/wandb_id: supervisor crash-resume — --init-adapter <ckpt> +
    # --skip-steps fast-forward through the deterministic batch order + same wandb run.
    import glob
    import json
    import os
    import secrets as pysecrets
    import string
    import threading
    import time

    if not data_dir.startswith("/"):
        data_dir = f"/data/{data_dir}"
    save_dir, local_bank, _ = _preflight(run_name, data_dir, n_ckpts, resume_from)
    os.makedirs(save_dir, exist_ok=True)

    # ---- run_meta.json: everything the supervisor needs to respawn this run faithfully. Written
    # once on the first leg (wandb id minted here so every leg logs to ONE wandb run). ----
    meta_path = f"{save_dir}/run_meta.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        wandb_id = wandb_id or meta.get("wandb_id", "")
    else:
        wandb_id = wandb_id or "".join(
            pysecrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        meta = {"run_name": run_name, "data_dir": data_dir, "n_ckpts": n_ckpts, "epochs": epochs,
                "batch_size": batch_size, "lr": lr, "max_seq": max_seq, "backend": backend,
                "extra_args": extra_args, "wandb_id": wandb_id, "created": time.time(), "legs": []}
    meta["legs"].append({"ts": time.time(), "resume_from": resume_from, "skip_steps": skip_steps})
    json.dump(meta, open(meta_path, "w"), indent=2)

    # ---- heartbeat + committer: touch <run>/heartbeat every 60s (supervisor aliveness signal,
    # started BEFORE the slow model download so fresh legs read as alive immediately) and
    # vol.commit right after every new ckpt lands (else every 300s) so saves are durable even if
    # the container dies mid-run. ----
    def _committer():
        last_commit, last_ckpts = 0.0, -1
        while True:
            try:
                with open(f"{save_dir}/heartbeat", "w") as f:
                    f.write(str(time.time()))
                n = len(glob.glob(f"{save_dir}/step_*")) + len(glob.glob(f"{save_dir}/final"))
                if n != last_ckpts or time.time() - last_commit > 300:
                    vol.commit()
                    last_commit, last_ckpts = time.time(), n
            except Exception as e:
                print(f"[modal] heartbeat/commit failed: {e}", flush=True)
            time.sleep(60)

    threading.Thread(target=_committer, daemon=True).start()
    vol.commit()  # publish run_meta + heartbeat now: the supervisor must see this leg exists

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=8", "SL/pretrain.py",
        "--data-dir", local_bank,
        "--save-dir", save_dir,
        "--run-name", run_name,
        "--n-ckpts", str(n_ckpts),
        "--epochs", str(epochs),
        "--wandb-id", wandb_id,
    ]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if lr:
        cmd += ["--lr", str(lr)]
    if max_seq:
        cmd += ["--max-seq", str(max_seq)]
    if resume_from:
        cmd += ["--init-adapter", resume_from, "--skip-steps", str(skip_steps)]
    if extra_args:
        cmd += extra_args.split()

    rc = _stream(cmd, _train_env(backend), "modal")
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"torchrun exited rc={rc} (backend={backend}) — supervisor will resume")
    assert os.path.exists(f"{save_dir}/final"), "rc=0 but no final ckpt — trainer exited early?"
    print(f"[modal] SFT run {run_name} COMPLETE -> {save_dir}/final", flush=True)


@app.function(
    image=image,
    gpu=os.environ.get("SFT_SMOKE_GPU", "B200:1"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=7200,
)
def smoke(data_dir: str, n_records: int = 256, batch_size: int = 8, extra_args: str = ""):
    """1xB200 pipeline validation: pre-warms /data/hf_cache, carves a tiny bank (first n_records
    of records.jsonl + the full vecs.f32 / vecs.f16) and runs world=1 SFT over it (no wandb, ckpts
    to /tmp, including the --n-ckpts intermediate-save path). extra_args reach pretrain.py verbatim,
    e.g. "--compile --grad-ckpt 0 --autocast-bf16 --head-on-labels --parity-check --log-steps 10"."""
    import os
    import shutil

    if not data_dir.startswith("/"):
        data_dir = f"/data/{data_dir}"
    _, local_bank, _ = _preflight("smoke", data_dir, n_ckpts=2, resume_from="")
    vec_file, _ = _vec_bank_file(local_bank)

    tiny = "/root/bank_smoke"
    if not os.path.exists(tiny):
        os.makedirs(tiny)
        with open(f"{local_bank}/records.jsonl") as fin, open(f"{tiny}/records.jsonl", "w") as fout:
            for i, line in enumerate(fin):
                if i >= n_records:
                    break
                fout.write(line)
        shutil.copy(f"{local_bank}/{vec_file}", f"{tiny}/{vec_file}")
    print(f"[modal-smoke] tiny bank: first {n_records} records", flush=True)

    cmd = [
        "python", "SL/pretrain.py",
        "--data-dir", tiny,
        "--save-dir", "/tmp/smoke_ckpt",
        "--run-name", "smoke",
        "--n-ckpts", "2",
        "--epochs", "1",
        "--batch-size", str(batch_size),
        "--no-wandb",
    ]
    if extra_args:
        cmd += extra_args.split()   # e.g. "--compile --grad-ckpt 0 --autocast-bf16 --log-steps 10" (speed test)
    rc = _stream(cmd, _train_env("nccl"), "modal-smoke")
    if rc != 0:
        raise RuntimeError(f"smoke exited rc={rc}")
    saved = sorted(os.listdir("/tmp/smoke_ckpt"))
    assert "final" in saved and any(s.startswith("step_") for s in saved), f"ckpt cadence broken: {saved}"
    print(f"[modal-smoke] OK — saved {saved}", flush=True)


@app.function(
    image=image,
    gpu=os.environ.get("SFT_SMOKE_GPU", "B200:1"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=3 * 3600,
)
def fp8_eval(data_dir: str, n_records: int = 6400, extra_args: str = ""):
    """1 GPU: sft/fp8_eval.py -- bf16 vs --fp8-base speed (meter TFLOP/s, examples/s, peak mem), same-weights logit
    KL / top-1, and N-step loss curves from one LoRA init on one batch order. Tiny bank = first n_records (the harness
    needs train-steps*batch + kl batches). extra_args are appended (e.g. "--train-steps 300 --bench-batch-sizes 16,32").
    Result JSON is printed and copied to /data/sft_mix/_fp8_eval/<utc>.json."""
    import os
    import shutil
    import time

    if not data_dir.startswith("/"):
        data_dir = f"/data/{data_dir}"
    _, local_bank, _ = _preflight("smoke", data_dir, n_ckpts=2, resume_from="")
    tiny = "/root/bank_fp8eval"
    if not os.path.exists(tiny):
        os.makedirs(tiny)
        with open(f"{local_bank}/records.jsonl") as fin, open(f"{tiny}/records.jsonl", "w") as fout:
            for i, line in enumerate(fin):
                if i >= n_records:
                    break
                fout.write(line)
        shutil.copy(f"{local_bank}/vecs.f32", f"{tiny}/vecs.f32")
    print(f"[modal-fp8-eval] tiny bank: first {n_records} records", flush=True)
    out = "/tmp/fp8_eval.json"
    cmd = ["python", "SL/fp8_eval.py", "--data-dir", tiny, "--out", out]
    if extra_args:
        cmd += extra_args.split()
    rc = _stream(cmd, _train_env("nccl"), "modal-fp8-eval")
    if os.path.exists(out):
        dst = f"{SFT_ROOT}/_fp8_eval/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(out, dst)
        vol.commit()
        print(f"[modal-fp8-eval] result JSON -> {dst}", flush=True)
        print("FP8_EVAL_JSON_BEGIN"); print(open(out).read()); print("FP8_EVAL_JSON_END", flush=True)
    if rc != 0:
        raise RuntimeError(f"fp8_eval exited rc={rc}")


@app.function(image=image, timeout=600, cpu=2)
def env_check():
    """CPU-only image check: pins + the torchao float8 imports pretrain.py --fp8-base relies on."""
    import importlib
    import sys

    import torch
    print(f"python {sys.version.split()[0]} torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
    for pkg in ("torchao", "transformers", "peft", "accelerate", "fla", "triton"):
        try:
            print(f"{pkg} {getattr(importlib.import_module(pkg), '__version__', '?')}")
        except Exception as e:  # noqa
            print(f"{pkg} IMPORT FAILED: {e!r}")
    from torchao.float8 import Float8LinearConfig
    from torchao.float8.float8_linear import Float8Linear
    from torchao.float8.float8_linear_utils import swap_linear_layers
    sys.path.insert(0, "/pmx/SL")
    import fp8
    print(f"torchao float8 API OK: {Float8Linear.__name__}, {swap_linear_layers.__name__}")
    print(f"fp8.py OK: recipe default={fp8.DEFAULT_RECIPE} MIN_DIM={fp8.MIN_DIM} "
          f"config={Float8LinearConfig.from_recipe_name(fp8.DEFAULT_RECIPE)}")
    print("ENV_CHECK_OK", flush=True)


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
    import sys
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    sys.path.insert(0, "/pmx/helpers")
    from mxf.config import MODEL
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL)
    vol.commit()
    print(f"[modal-prewarm] base model cached ({time.time() - t0:.0f}s)", flush=True)


# ---- auto-resume supervisor: Modal caps functions at 24h; big banks can outrun that, and crashed
# legs must not silently stall a datamix sweep. Every 20 min: for each /data/sft_mix/<run>/ with a
# run_meta.json and no `final`, if the heartbeat is stale (live legs touch+commit it every <=5 min)
# and no spawn is cooling down, respawn `train` in RESUME mode: --init-adapter <latest step_N> +
# --skip-steps N+1 (step_N is saved after batch N of the deterministic order) + the same wandb id.
# No step ckpts yet -> clean fresh respawn. Unlike the RL supervisor this is fully generic across
# N parallel runs: per-run heartbeat/state files, nothing hardcoded. ----
@app.function(schedule=modal.Period(minutes=20), volumes={"/data": vol}, timeout=600)
def supervisor():
    import glob
    import json
    import os
    import time

    vol.reload()
    if os.path.exists(f"{SFT_ROOT}/resume_paused"):
        print(f"[supervisor] auto-resume PAUSED ({SFT_ROOT}/resume_paused present) — no spawns", flush=True)
        return
    for meta_path in sorted(glob.glob(f"{SFT_ROOT}/*/run_meta.json")):
        run_dir = os.path.dirname(meta_path)
        run = os.path.basename(run_dir)
        if os.path.exists(f"{run_dir}/final"):
            print(f"[supervisor] {run}: COMPLETE", flush=True)
            continue
        if os.path.exists(f"{run_dir}/resume_paused"):
            print(f"[supervisor] {run}: paused ({run}/resume_paused present)", flush=True)
            continue
        hb = f"{run_dir}/heartbeat"
        age = time.time() - (os.path.getmtime(hb) if os.path.exists(hb) else os.path.getmtime(meta_path))
        if age < STALE_HEARTBEAT_S:
            print(f"[supervisor] {run}: alive (heartbeat {age / 60:.0f} min old)", flush=True)
            continue
        st_path = f"{run_dir}/resume_state.json"
        st = json.load(open(st_path)) if os.path.exists(st_path) else {}
        if time.time() - st.get("last_spawn_ts", 0) < SPAWN_COOLDOWN_S:
            print(f"[supervisor] {run}: stale but in post-spawn cooldown ({st.get('call')})", flush=True)
            continue
        meta = json.load(open(meta_path))
        steps = sorted(int(p.rsplit("_", 1)[-1]) for p in glob.glob(f"{run_dir}/step_*")
                       if p.rsplit("_", 1)[-1].isdigit())
        resume_from = f"{run_dir}/step_{steps[-1]}" if steps else ""
        skip = steps[-1] + 1 if steps else 0  # step_N lands after batch N -> skip batches 0..N
        print(f"[supervisor] {run}: DEAD (heartbeat {age / 60:.0f} min old) — respawning "
              f"{'from ' + resume_from + f' skip={skip}' if steps else 'FRESH (no ckpts yet)'}", flush=True)
        call = modal.Function.from_name(APP_NAME, "train").spawn(
            run_name=meta["run_name"], data_dir=meta["data_dir"], n_ckpts=meta["n_ckpts"],
            epochs=meta.get("epochs", 1), batch_size=meta.get("batch_size", 0),
            lr=meta.get("lr", 0.0), max_seq=meta.get("max_seq", 0),
            backend=meta.get("backend", "nccl"), extra_args=meta.get("extra_args", ""),
            resume_from=resume_from, skip_steps=skip, wandb_id=meta.get("wandb_id", ""))
        json.dump({"last_spawn_ts": time.time(), "call": call.object_id,
                   "from_step": steps[-1] if steps else -1}, open(st_path, "w"))
        vol.commit()
        print(f"[supervisor] {run}: resume leg spawned {call.object_id}", flush=True)


@app.local_entrypoint()
def launch(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
           batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
           backend: str = "nccl", extra_args: str = ""):
    """Spawn one SFT run on the DEPLOYED app (run `modal deploy modal_sft.py` first). Returns
    immediately — invoke once per datamix to train many mixes in parallel."""
    call = modal.Function.from_name(APP_NAME, "train").spawn(
        run_name=run_name, data_dir=data_dir, n_ckpts=n_ckpts, epochs=epochs,
        batch_size=batch_size, lr=lr, max_seq=max_seq, backend=backend, extra_args=extra_args)
    print(f"spawned SFT run {run_name!r} on bank {data_dir}: {call.object_id}")
    print(f"logs:  modal app logs {APP_NAME}   |   ckpts: /data/sft_mix/{run_name}/step_*")


@app.local_entrypoint()
def run_train(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
              batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
              backend: str = "nccl", extra_args: str = ""):
    """Attached single run (live logs; use `modal run --detach` to survive disconnect)."""
    train.remote(run_name=run_name, data_dir=data_dir, n_ckpts=n_ckpts, epochs=epochs,
                 batch_size=batch_size, lr=lr, max_seq=max_seq, backend=backend,
                 extra_args=extra_args)


@app.local_entrypoint()
def run_smoke(data_dir: str, n_records: int = 256, batch_size: int = 8, extra_args: str = ""):
    smoke.remote(data_dir=data_dir, n_records=n_records, batch_size=batch_size, extra_args=extra_args)


@app.local_entrypoint()
def run_prewarm():
    prewarm.remote()


@app.local_entrypoint()
def run_fp8_eval(data_dir: str, n_records: int = 6400, extra_args: str = ""):
    fp8_eval.remote(data_dir=data_dir, n_records=n_records, extra_args=extra_args)


@app.local_entrypoint()
def run_env_check():
    env_check.remote()
