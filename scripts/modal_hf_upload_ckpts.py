"""Upload LoRA checkpoints from the Modal volume `maemm-data` to HuggingFace model repos (one repo per run, one subfolder per
checkpoint), in parallel, from inside Modal (datacenter bandwidth). The HF token is passed at call time, never stored here.

    modal deploy scripts/modal_hf_upload_ckpts.py
    python3 - <<'EOF'
    import json, modal
    manifest = json.load(open("hf_upload_manifest.json"))          # [{"repo": ..., "readme": ..., "items": [{"src": "/data/...", "dst": "step_280"}, ...]}, ...]
    calls = [modal.Function.from_name("maemm-hf-upload", "upload_repo").spawn(m, hf_token=TOKEN) for m in manifest]
    EOF
Each item copies adapter_model.safetensors + adapter_config.json (+ run_meta/wandb_id if present); optim.pt is skipped.
"""
import os

import modal

app = modal.App("maemm-hf-upload")
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]==0.34.4").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
KEEP = ("adapter_model.safetensors", "adapter_config.json", "wandb_id.txt", "run_meta.json", "README.md")


@app.function(image=image, cpu=4, memory=16384, volumes={"/data": vol}, timeout=6 * 3600)
def upload_repo(spec: dict, hf_token: str, private: bool = False) -> dict:
    import shutil
    import tempfile
    import time
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    vol.reload()
    repo = spec["repo"]
    api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
    out = {"repo": repo, "uploaded": [], "missing": [], "seconds": 0.0}
    t0 = time.time()
    for it in spec["items"]:
        src, dst = it["src"], it["dst"]
        if not os.path.exists(f"{src}/adapter_model.safetensors"):
            out["missing"].append(src)
            continue
        stage = tempfile.mkdtemp()
        for f in KEEP:
            if os.path.exists(f"{src}/{f}") and f != "README.md":
                shutil.copyfile(f"{src}/{f}", f"{stage}/{f}")
        api.upload_folder(folder_path=stage, repo_id=repo, path_in_repo=dst, commit_message=f"add {dst} from {src}")
        shutil.rmtree(stage, ignore_errors=True)
        out["uploaded"].append(dst)
    if spec.get("readme"):
        api.upload_file(path_or_fileobj=spec["readme"].encode(), path_in_repo="README.md", repo_id=repo, commit_message="model card")
    out["seconds"] = round(time.time() - t0, 1)
    print(f"[hf-upload] {repo}: uploaded {out['uploaded']} missing {out['missing']} in {out['seconds']}s", flush=True)
    return out
