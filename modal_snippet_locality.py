"""Modal launcher for the snippet-locality eval's GPU stage (MAEMMBench/snippet_locality.py
build): one B200, the maemm-data volume (base-model HF cache + SAE + augmented autointerp
testbed + RL adapter checkpoints). One-shot.

Two modes:
  * no --adapter: clean-base scoring of the testbed's STORED rollouts (the original run).
  * --adapter <dir>: regenerate the rollouts ON-POLICY for that adapter first (locality is an
    on-policy metric), same feature set + real examples + gen seed -> adapter-vs-adapter
    comparisons are apples-to-apples.

Run (from this box, profile safety-sahan):
    MODAL_PROFILE=safety-sahan modal run modal_snippet_locality.py \
        --adapter /data/ckpts_last5/step_75 --out /data/eval_autointerp/locality_last5_75.json
    # or both arms of the last-5-vs-baseline comparison in parallel (one B200 each):
    MODAL_PROFILE=safety-sahan modal run modal_snippet_locality.py::compare \
        --adapters last5_step75=/data/ckpts_last5/step_75,v2_step225=/data/ckpts_v2/step_225
Then pull the profiles locally and continue with the local score stage:
    MODAL_PROFILE=safety-sahan modal volume get maemm-data eval_autointerp/locality.json .
    python MAEMMBench/snippet_locality.py score --locality locality.json \
        --autointerp-results eval/out/results.json --testbed eval/out/testbed_v2.json \
        --out locality_results.json
"""
from pathlib import Path

import modal

REPO = Path(__file__).parent

app = modal.App("maemm-snippet-locality")

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
    .add_local_dir(REPO / "MAEMMBench", "/pmx/MAEMMBench", ignore=["__pycache__", "analysis"])
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=False)


@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maemm-hf")],
    timeout=2 * 3600,
)
def build(testbed: str = "/data/eval_autointerp/testbed_v2.json",
          out: str = "/data/eval_autointerp/locality.json",
          adapter: str = ""):
    import os
    import sys

    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path[:0] = ["/pmx/helpers", "/pmx/MAEMMBench"]

    import snippet_locality as SL

    argv = ["build", "--testbed", testbed, "--out", out]
    if adapter:
        argv += ["--adapter", adapter]
    a = SL.build_parser().parse_args(argv)
    a.fn(a)
    vol.commit()
    print(f"[modal] committed {out} to maemm-data", flush=True)


@app.local_entrypoint()
def main(testbed: str = "/data/eval_autointerp/testbed_v2.json",
         out: str = "/data/eval_autointerp/locality.json",
         adapter: str = ""):
    build.remote(testbed=testbed, out=out, adapter=adapter)


@app.local_entrypoint()
def compare(adapters: str = ("last5_step75=/data/ckpts_last5/step_75,"
                             "v2_step225=/data/ckpts_v2/step_225"),
            testbed: str = "/data/eval_autointerp/testbed_v2.json"):
    """Spawn one on-policy build per name=adapter_dir spec (parallel, one B200 each) and wait.
    Outputs land at /data/eval_autointerp/locality_<name>.json on the maemm-data volume."""
    calls = []
    for spec in adapters.split(","):
        name, path = spec.split("=", 1)
        out = f"/data/eval_autointerp/locality_{name}.json"
        calls.append((name, out, build.spawn(testbed=testbed, out=out, adapter=path)))
        print(f"[compare] spawned {name}: adapter={path} -> {out}")
    for name, out, c in calls:
        c.get()
        print(f"[compare] DONE {name} -> {out}")
