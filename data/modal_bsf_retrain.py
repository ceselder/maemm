"""Retrain the Qwen3.6-27B L42 block-sparse featurizer (SASA / BSF) on Modal, 1x B200.

Why: the original sasa.pt (Aug-21, EV->~0.8) lived on a since-stopped RunPod B300's container disk and
is not on any accessible storage. The Modal volume already holds a COMPLETE 27B L42 activation
collection in exactly the trainer's input format:
    /data/acts27b/{acts.f16 [50287,512,5120], toks.i32, meta.json, whiten_mu.npy}
    = 25.7M tokens, FineFineWeb (67 domains), BOS sink dropped   (2.5x the original ~10M tokens)
whiten_zca.npy is absent -> the trainer estimates it from 2048 random sequences at start.

Recipe = the original: G=32768 blocks x b=8 dims, k=32 active/token, ZCA-whitened + unit-norm inputs,
AdamW lr 1e-4, batch 8192, decoupled wd 1e-2, 20k steps (~40 step/min on a B300 -> ~8h).

Outputs -> /data/bsf27b_sasa/{sasa.pt, blocks_Q.pt, whiten_mu.npy, whiten_zca.npy, meta.json}
        -> HF  ceselder/qwen36-27b-bsf-l42  (auto-upload at the end so it can't get lost again)
        -> wandb project bsf-sasa-27b, run bsf-sasa-27b-v2-modal

Launch (MODAL_PROFILE=safety-sahan):
    modal run modal_bsf_retrain.py::smoke                 # ~5 min: 30 steps, batch 2048, no upload/wandb
    modal deploy modal_bsf_retrain.py
    python -c "import modal; modal.Function.from_name('maemm-bsf-retrain','train').spawn()"
Needs Modal secrets `maemm-hf` (HF_TOKEN) and `maemm-wandb` (WANDB_API_KEY) — same as the RL launcher.
"""
import os, subprocess, threading, time, json
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parent.parent   # repo root (this launcher lives one level down)
app = modal.App("maemm-bsf-retrain")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "numpy<2.3", "wandb==0.28.2", "huggingface_hub>=0.34")
    .add_local_file(REPO / "data" / "train_sasa.py", "/pmx/bsf/train_sasa.py")
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=True)

ACTS_DIR = "/data/acts27b"
OUT_DIR = "/data/bsf27b_sasa"
HF_REPO = "ceselder/qwen36-27b-bsf-l42"
WANDB_PROJECT = "bsf-sasa-27b"

README = """---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
tags: [interpretability, sparse-autoencoder, subspace, sasa, block-sparse, qwen3.6-27b, layer-42]
---
# Qwen3.6-27B layer-42 block-sparse subspace featurizer (SASA / "BSF")

A group-top-k autoencoder over the **residual stream after layer 42 of Qwen/Qwen3.6-27B** (d=5120):
**G={G} blocks x b={b} dims, k={k} blocks active per token**. Each block is a {b}-dim *subspace* feature
(SASA, arXiv 2606.06333). Inputs are ZCA-whitened and unit-normed; the factored dictionary is trained with
decoupled weight decay (nuclear-norm rank adaptation), no decoder renorm.

Trained on {ntok_m:.1f}M tokens of FineFineWeb (67 domains) activations, {steps} steps x batch {batch},
AdamW lr {lr}, wd {wd}. Final: EV {ev:.3f}, alive blocks {alive}/{G}.

## Files
- `sasa.pt`      — {{E [d,G*b], D [G*b,d], bias [d], G, b, d, k}}
- `blocks_Q.pt`  — per-block orthonormal basis Q [G,b,d] (project activations through a block's subspace)
- `whiten_mu.npy`, `whiten_zca.npy` — whitening applied before encoding: `y = normalize((x - mu) @ zca)`
- `meta.json`

## Encode
```python
import torch, numpy as np
s = torch.load("sasa.pt"); mu = torch.tensor(np.load("whiten_mu.npy")); zca = torch.tensor(np.load("whiten_zca.npy"))
y = torch.nn.functional.normalize((x - mu) @ zca, dim=-1)          # x: [n, 5120] layer-42 resid_post (BOS dropped)
z = (y @ s["E"]).view(-1, s["G"], s["b"]); gn = z.norm(dim=-1)      # gn: [n, G] block activations
top = gn.topk(s["k"], dim=-1)                                        # active blocks per token
```
Training code: `bsf/train_sasa.py` in github.com/ceselder/maemm. wandb: {wandb_url}
"""


def _commit_loop(stop, every=600):
    while not stop.wait(every):
        try:
            vol.commit()
            print("[modal] volume committed", flush=True)
        except Exception as e:  # noqa
            print(f"[modal] commit failed: {e}", flush=True)


def _run(args, env):
    print("[modal] launching:", " ".join(args), flush=True)
    p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_ev = None
    for line in p.stdout:
        print(line, end="", flush=True)
        if " EV " in line:
            try:
                last_ev = float(line.split(" EV ")[1].split()[0])
            except Exception:  # noqa
                pass
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"train_sasa.py exited {rc}")
    return last_ev


@app.function(image=image, gpu="B200:1", cpu=8, memory=98304, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb")],
              timeout=43200)
def train(steps: int = 20000, batch: int = 8192, G: int = 32768, b: int = 8, k: int = 32,
          lr: float = 1e-4, wd: float = 1e-2, pool_seqs: int = 4096, refresh_seqs: int = 8,
          out_dir: str = OUT_DIR, hf_repo: str = HF_REPO, run_name: str = "bsf-sasa-27b-v2-modal"):
    """Full retrain (~8h) + HF upload. Resumable only by restarting (no optimizer ckpt) — it's cheap."""
    vol.reload()
    assert os.path.exists(f"{ACTS_DIR}/meta.json"), f"{ACTS_DIR} missing on volume"
    meta = json.load(open(f"{ACTS_DIR}/meta.json"))
    print(f"[modal] acts: n_seq={meta['n_seq']} T={meta['seq_len']} d={meta.get('d', meta.get('d_model'))} "
          f"tokens={meta.get('n_tokens')} dataset={meta.get('dataset')}", flush=True)
    os.makedirs(out_dir, exist_ok=True); os.makedirs("/tmp/wandb", exist_ok=True)
    env = dict(os.environ, WANDB_DIR="/tmp/wandb", PYTHONUNBUFFERED="1")
    stop = threading.Event(); threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    try:
        ev = _run(["python", "/pmx/bsf/train_sasa.py", "--acts-dir", ACTS_DIR, "--out-dir", out_dir,
                   "--G", str(G), "--b", str(b), "--k", str(k), "--batch", str(batch), "--steps", str(steps),
                   "--lr", str(lr), "--wd", str(wd), "--loader", "buffer", "--pool-seqs", str(pool_seqs),
                   "--refresh-seqs", str(refresh_seqs), "--wandb", WANDB_PROJECT, "--run-name", run_name], env)
    finally:
        stop.set(); vol.commit()
    # ---- upload to HF so this can never be lost again ----
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(hf_repo, exist_ok=True, private=False)
    m = json.load(open(f"{out_dir}/meta.json"))
    wandb_url = f"https://wandb.ai/octahedral-systems/{WANDB_PROJECT}"
    Path(f"{out_dir}/README.md").write_text(README.format(
        G=G, b=b, k=k, ntok_m=meta.get("n_tokens", 0) / 1e6, steps=steps, batch=batch, lr=lr, wd=wd,
        ev=ev if ev is not None else float("nan"), alive=m.get("alive", "?"), wandb_url=wandb_url))
    for f in ["sasa.pt", "blocks_Q.pt", "whiten_mu.npy", "whiten_zca.npy", "meta.json", "README.md"]:
        print(f"[modal] uploading {f} ...", flush=True)
        api.upload_file(path_or_fileobj=f"{out_dir}/{f}", path_in_repo=f, repo_id=hf_repo)
    vol.commit()
    print(f"[modal] DONE. EV~{ev} -> https://huggingface.co/{hf_repo}", flush=True)
    return {"ev": ev, "hf": f"https://huggingface.co/{hf_repo}", "out_dir": out_dir}


@app.function(image=image, gpu="B200:1", cpu=8, memory=32768, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb")],
              timeout=3600)
def smoke():
    """~5-10 min validation: zca estimate from 256 seqs, tiny buffer, 30 steps, batch 2048, no wandb, /tmp out.
    Also writes whiten_zca.npy next to the acts so the real run skips the estimate."""
    vol.reload()
    os.makedirs("/tmp/bsf_smoke", exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.time()
    ev = _run(["python", "/pmx/bsf/train_sasa.py", "--acts-dir", ACTS_DIR, "--out-dir", "/tmp/bsf_smoke",
               "--steps", "30", "--batch", "2048", "--log-every", "10", "--loader", "buffer",
               "--pool-seqs", "256", "--refresh-seqs", "4", "--zca-seqs", "256", "--wandb", "", "--save-every", "10000"], env)
    vol.commit()
    print(f"[smoke] OK in {time.time()-t0:.0f}s, last EV {ev}; zca on volume: "
          f"{os.path.exists(ACTS_DIR + '/whiten_zca.npy')}", flush=True)
    return {"ev": ev, "seconds": time.time() - t0}
