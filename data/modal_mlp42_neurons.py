"""Modal app: layer-42 MLP neurons of Qwen/Qwen3.6-27B -> do the sparsely-activating ones write structured directions?

One GPU, forward-only + short generations. Logic lives in data/mlp42_neurons_worker.py (mounted at /pmx/helpers).
Outputs land on the `maemm-data` volume under /data/mlp42/ (nothing needs recomputing afterwards):
    neuron_stats.npz     per-neuron moments, extremes, sign-resolved log-histograms of |a|, top-32 contexts, presence cosines
    down_proj_cols.f16   fp16 [d_ff, d_model] == down_proj.weight.T (neuron i's residual write direction = row i)
    sel_windows.npz      the 4000 x 256 token windows used (train rows of /data/acts27b) -> decode contexts locally
    sae_match.npz        nearest SAE feature by direction cosine (dec/enc) + activation Pearson correlation, with controls
    peak_context.npz     at each neuron's peak token: cos(h - mu, dir), write-norm share, SAE features active
    dirs_analysis.npz    BSF top-block energy fraction, cluster-probe NN cosine, neuron-neuron NN (+ random / SAE-dec controls)
    verbalize_<tag>.*    inverter generations (RL-A adapter) + clean-base scoring incl. neuron fire-back

Run (profile safety-sahan) — deployed app + spawn survives the launching client:
    modal deploy data/modal_mlp42_neurons.py
    python -c "import modal; print(modal.Function.from_name('maemm-mlp42-neurons', 'stats_and_dirs').spawn().object_id)"
    python -c "import modal; print(modal.Function.from_name('maemm-mlp42-neurons', 'verbalize').spawn(
        sparse_ids=[...], dense_ids=[...], tag='rlA').object_id)"
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = "maemm-mlp42-neurons"
app = modal.App(APP_NAME)

# same pins as eval/modal_eval.py / data/modal_bank_everything.py (one environment across the suite)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "peft==0.20.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "data" / "mlp42_neurons_worker.py", "/pmx/helpers/mlp42_neurons_worker.py")
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
GPUS = ["B200", "H200"]          # forward-only work: whichever schedules first
ADAPTER_DEFAULT = "/data/ckpts_rl_A_randctx/final"


def _env():
    import os
    import sys
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, "/pmx/helpers")


def _load_base(dev):
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mxf.config import MODEL
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": dev})
    base.eval()
    print(f"[mlp42] base {type(base).__name__} loaded in {time.time() - t0:.0f}s", flush=True)
    return tok, base


@app.function(image=image, gpu=GPUS, cpu=8, memory=98304, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=4 * 3600)
def stats_and_dirs(n_windows: int = 4000, win_len: int = 256, batch: int = 16, seed: int = 0, skip_dirs: bool = False):
    _env()
    import torch
    import mlp42_neurons_worker as W
    from mxf.sae import load_sae
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda:0"
    vol.reload()
    tok, base = _load_base(dev)
    sae = load_sae(path=W.SAE_PT, device=dev, dtype=torch.float32)
    W.log(f"SAE loaded: d_sae={sae.d_sae}")
    res = W.run_stats(base, tok, sae, n_windows=n_windows, win_len=win_len, batch=batch, seed=seed, dev=dev)
    vol.commit()
    W.log("stats committed to volume")
    if skip_dirs:
        return "stats done"
    del sae
    torch.cuda.empty_cache()
    W.run_dirs(res["signed"], res["R"], res["Wd_sample"], dev=dev)
    vol.commit()
    W.log("dirs committed to volume")
    return "stats+dirs done"


@app.function(image=image, gpu=GPUS, cpu=8, memory=65536, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=3 * 3600)
def verbalize(sparse_ids: list, dense_ids: list, tag: str = "rlA", adapter: str = ADAPTER_DEFAULT, n_sae: int = 128,
              n_random: int = 64, bo: int = 4, temp: float = 1.0, max_new: int = 48, min_new: int = 16, seed: int = 0):
    """Inverter verbalization of neuron directions (polarity-signed unit down_proj columns) vs SAE-feature and random
    controls, all with the SAME adapter / sampling / scoring. Neuron ids come from the local analysis of neuron_stats.npz."""
    _env()
    import numpy as np
    import torch
    import torch.nn.functional as F
    import mlp42_neurons_worker as W
    from mxf.config import D_MODEL
    from mxf.sae import load_sae
    dev = "cuda:0"
    vol.reload()
    st = np.load(f"{W.OUT}/neuron_stats.npz")
    N = int(st["N"])
    cols = np.fromfile(f"{W.OUT}/down_proj_cols.f16", np.float16).reshape(N, D_MODEL).astype(np.float32)
    pol = st["polarity"].astype(np.float32); max_abs = st["max_abs"].astype(np.float32)
    U = F.normalize(torch.from_numpy(cols), dim=1) * torch.from_numpy(pol)[:, None]
    sets = []
    for name, ids in (("neuron_sparse", sparse_ids), ("neuron_dense", dense_ids)):
        ids = [int(i) for i in ids]
        sets.append({"name": name, "ids": ids, "dirs": U[ids], "neuron": ids, "polarity": [float(pol[i]) for i in ids],
                     "ref_max": [float(max_abs[i]) for i in ids]})
    tok, base = _load_base(dev)
    sae = load_sae(path=W.SAE_PT, device=dev, dtype=torch.float32)
    ma = torch.load(W.MAXACTS_PT, map_location="cpu", weights_only=False)
    peak = ma["max_acts"].reshape(sae.d_sae, -1).max(1).values.float().numpy()
    del ma
    rng = np.random.default_rng(seed)
    feats = np.sort(rng.choice(np.flatnonzero(peak > 0), n_sae, replace=False)).tolist()
    sets.append({"name": "sae", "ids": feats, "dirs": sae.enc_dirs(feats).float().cpu(), "sae_feat": feats,
                 "ref_max": [float(peak[f]) for f in feats]})
    g = torch.Generator().manual_seed(seed)
    sets.append({"name": "random", "ids": list(range(n_random)), "dirs": F.normalize(torch.randn(n_random, D_MODEL, generator=g), dim=1)})
    W.log("sets: " + ", ".join(f"{s['name']}={len(s['ids'])}" for s in sets))
    summ = W.run_verbalize(base, tok, adapter, sets, sae, tag, dev=dev, bo=bo, temp=temp, max_new=max_new, min_new=min_new)
    vol.commit()
    return summ


@app.local_entrypoint()
def main(stage: str = "stats"):
    if stage == "stats":
        print(stats_and_dirs.remote())
    else:
        raise SystemExit("verbalize needs neuron id lists: spawn it via modal.Function.from_name(...).spawn(...)")
