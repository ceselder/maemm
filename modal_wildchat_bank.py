"""Modal launcher for the ONE-TIME WildChat fire-prediction bank (eval/wildchat_bank.py): one B200,
the maemm-data volume (base-model HF cache + SAE + autointerp testbed), HF online for the
allenai/WildChat-1M stream (ungated). Writes /data/eval_wildchat/windows.json, which
train/inline_extra_evals.py reads at trainer start for the inline `extra/wildchat/*` metrics.

    MODAL_PROFILE=safety-sahan modal run modal_wildchat_bank.py            # ~10 min, one B200
    MODAL_PROFILE=safety-sahan modal volume get maemm-data eval_wildchat/windows.json .
"""
from pathlib import Path

import modal

REPO = Path(__file__).parent

app = modal.App("maemm-wildchat-bank")

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
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "datasets",
        "hf_xet",
    )
    .add_local_dir(REPO / "eval", "/pmx/eval", ignore=["__pycache__", "out"])
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "train" / "inline_extra_evals.py", "/pmx/RL/inline_extra_evals.py")
    .add_local_file(REPO / "MAEMMBench" / "snippet_locality.py", "/pmx/eval/snippet_locality.py")
    .add_local_file(REPO / "eval" / "autointerp_detection.py", "/pmx/eval/autointerp_detection.py")
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=False)


@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=2 * 3600,
)
def build(testbed: str = "/data/eval_autointerp/testbed_v2.json", sae_path: str = "/data/sae/ae.pt",
          out: str = "/data/eval_wildchat/windows.json", n_windows: int = 40_000, win: int = 64,
          max_per_conv: int = 4, seed: int = 0):
    import os
    import sys

    os.environ["HF_HOME"] = "/data/hf_cache"          # base model cached there; WildChat streams online
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path[:0] = ["/pmx/helpers", "/pmx/eval", "/pmx/RL"]
    sys.argv = ["wildchat_bank.py", "--testbed", testbed, "--sae-path", sae_path, "--out", out,
                "--n-windows", str(n_windows), "--win", str(win), "--max-per-conv", str(max_per_conv), "--seed", str(seed)]
    import wildchat_bank
    wildchat_bank.main()
    vol.commit()
    print(f"[modal] committed {out} to maemm-data", flush=True)


@app.local_entrypoint()
def main(n_windows: int = 40_000, out: str = "/data/eval_wildchat/windows.json"):
    build.remote(n_windows=n_windows, out=out)
