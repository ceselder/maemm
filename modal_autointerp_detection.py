"""Modal launcher for the autointerp-detection eval's GPU stage (eval/autointerp_detection.py
build): one B200, the maemm-data volume (base model HF cache + SAE + maxacts + held-out eval
cache), HF online for the adapter (ceselder/qwen36-27b-maemm-inverter) and the Ultra-FineWeb
negatives stream. One-shot, not a daemon.

Run (from this box, profile safety-sahan; keep the client attached or use --detach):
    MODAL_PROFILE=safety-sahan modal run modal_autointerp_detection.py
Then pull the testbed locally:
    MODAL_PROFILE=safety-sahan modal volume get maemm-data eval_autointerp/testbed.json .
and continue with the local judge/score stages of eval/autointerp_detection.py.

Adapter-vs-adapter comparison (e.g. last-5 ckpt vs baseline; ON-POLICY rollouts per adapter,
same seed => identical held-out features/positives/negatives, only the rollouts differ):
    MODAL_PROFILE=safety-sahan modal run modal_autointerp_detection.py::compare \
        --adapters last5_step75=/data/ckpts_last5/step_75,v2_step225=/data/ckpts_v2/step_225
Outputs land at /data/eval_autointerp/testbed_<name>.json; then per testbed run the local
judge + score stages, and merge with eval/autointerp_compare.py.
"""
from pathlib import Path

import modal

REPO = Path(__file__).parent

app = modal.App("maemm-autointerp-detection")

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
    .add_local_dir(REPO / "eval", "/pmx/eval", ignore=["__pycache__"])
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=False)


@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=4 * 3600,
)
def build(adapter: str = "ceselder/qwen36-27b-maemm-inverter",
          n_features: int = 64, n_desc: int = 8, n_pos: int = 10, n_neg: int = 10,
          n_max: int = 8, seed: int = 0,
          out: str = "/data/eval_autointerp/testbed.json"):
    import os
    import sys

    os.environ["HF_HOME"] = "/data/hf_cache"     # base model cached; adapter+corpus need online
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path[:0] = ["/pmx/helpers", "/pmx/eval"]

    import autointerp_detection as AD

    a = AD.build_parser().parse_args([
        "build", "--adapter", adapter, "--n-features", str(n_features),
        "--n-desc", str(n_desc), "--n-pos", str(n_pos), "--n-neg", str(n_neg),
        "--n-max", str(n_max), "--seed", str(seed), "--out", out,
    ])
    a.fn(a)
    vol.commit()
    print(f"[modal] committed {out} to maemm-data", flush=True)


@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=2 * 3600,
)
def augment(testbed: str = "/data/eval_autointerp/testbed.json",
            out: str = "/data/eval_autointerp/testbed_v2.json",
            mine_pool: int = 8192, mine_skip_docs: int = 102_000, seed: int = 0):
    """Hard-negatives mining pass (no adapter, no generation): near-miss + embedding-NN pools
    appended to the existing testbed verbatim. See eval/autointerp_detection.py cmd_augment."""
    import os
    import sys

    os.environ["HF_HOME"] = "/data/hf_cache"     # base cached; corpus + bge-small need online
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path[:0] = ["/pmx/helpers", "/pmx/eval"]

    import autointerp_detection as AD

    a = AD.build_parser().parse_args([
        "augment", "--testbed", testbed, "--out", out, "--mine-pool", str(mine_pool),
        "--mine-skip-docs", str(mine_skip_docs), "--seed", str(seed),
    ])
    a.fn(a)
    vol.commit()
    print(f"[modal] committed {out} to maemm-data", flush=True)


@app.local_entrypoint()
def main(n_features: int = 64):
    build.remote(n_features=n_features)


@app.local_entrypoint()
def compare(adapters: str = ("last5_step75=/data/ckpts_last5/step_75,"
                             "v2_step225=/data/ckpts_v2/step_225"),
            n_features: int = 64, seed: int = 0):
    """Spawn one on-policy build per name=adapter_dir spec (parallel, one B200 each) and wait.
    Same seed for every arm => the testbeds share features/positives/negatives verbatim (the
    numpy RNG never sees the adapter; rollouts use a forked torch RNG) — paired by construction.
    Outputs land at /data/eval_autointerp/testbed_<name>.json on the maemm-data volume."""
    calls = []
    for spec in adapters.split(","):
        name, path = spec.split("=", 1)
        out = f"/data/eval_autointerp/testbed_{name}.json"
        calls.append((name, out, build.spawn(adapter=path, n_features=n_features,
                                             seed=seed, out=out)))
        print(f"[compare] spawned {name}: adapter={path} -> {out}")
    for name, out, c in calls:
        c.get()
        print(f"[compare] DONE {name} -> {out}")


@app.local_entrypoint()
def run_augment(mine_pool: int = 8192):
    augment.remote(mine_pool=mine_pool)
