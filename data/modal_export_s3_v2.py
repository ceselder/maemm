"""Modal app (CPU): export the MAEMM data bundle to the owner's S3 bucket with presigned URLs — the held-out bundle staged by
data/modal_hf_heldout_upload.py PLUS the training data of the simple2m chain and of the earlier 5M-mix chain (records as parquet per
family + directions as float16 .npy), the SAE max-acts files, the 2M feature split and the document registry.

Credentials are CALL ARGUMENTS (never stored in code or on the volume). Run these two commands yourself (the assistant's sandbox
refuses to launch data exports):

    cd ~/maemm-pub-simple2m && MODAL_PROFILE=safety-sahan modal deploy data/modal_export_s3_v2.py
    source ~/modal_venv/bin/activate && MODAL_PROFILE=safety-sahan python3 scripts/launchers/spawn_export_s3_v2.py      # reads ~/.aws/credentials [default]

The spawn helper records the call id in ~/shared/overnight/simple2m/ids.json["s3_export"]; when it finishes, urls.txt + manifest.json
are in s3://<bucket>/<prefix>/ and the call's return value carries every presigned URL (7 days).
"""
import os

import modal

from modal_hf_heldout_upload import image as _image, publish, vol  # same image + volume; publish.local(dry=True) stages the held-out bundle

app = modal.App(os.environ.get("S3_EXPORT_APP", "maemm-export-s3-v2"))
image = _image.pip_install("boto3").add_local_python_source("modal_hf_heldout_upload")

TRAIN_BANKS = [("simple2m/sft_mix", "/data/banks/mix_simple2m_sft"), ("simple2m/rl_pool", "/data/banks/mix_simple2m_rl"),
               ("legacy_5m_chain/sft_midtrain_mix_5m_sft", "/data/banks/mix_5m_sft"), ("legacy_5m_chain/rl_pool_mix_eq_1p45m", "/data/banks/mix_eq_1p45m")]
EXTRA_FILES = ["/data/sae/maxacts.pt", "/data/sae/maxacts_fresh.pt", "/data/sae2m/maxacts_top5.pt", "/data/sae2m/maxacts_top5.summary.json",
               "/data/sae2m/feature_split.npz", "/data/sae2m/feature_split.json", "/data/simple2m/doc_registry.json"]
TOP_DIRS = ("simple2m", "legacy_5m_chain", "extra", "README.md", "manifest.json", "urls.txt")


def _export_bank(tag, bank, out, log, chunk=200_000):
    """records.jsonl (+vecs.f32|f16) -> <out>/<tag>/<family>/{records.parquet, dirs_f16.npy}; dirs row i == records row i (sorted by vec_idx)."""
    import json
    import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    dst = f"{out}/{tag}"; os.makedirs(dst, exist_ok=True)
    vf, dt = (f"{bank}/vecs.f32", np.float32) if os.path.exists(f"{bank}/vecs.f32") else (f"{bank}/vecs.f16", np.float16)
    n = os.path.getsize(vf) // (5120 * np.dtype(dt).itemsize)
    vecs = np.memmap(vf, dt, "r", shape=(n, 5120))
    fams = {}
    with open(f"{bank}/records.jsonl") as f:
        for line in f:
            r = json.loads(line); fams.setdefault(r["family"], []).append(r)
    log(f"{tag}: {n} vecs, families { {k: len(v) for k, v in fams.items()} }")
    files = {}
    for fam, recs in fams.items():
        recs.sort(key=lambda r: r["vec_idx"])
        d = f"{dst}/{fam}"; os.makedirs(d, exist_ok=True)
        df = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in recs])
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f"{d}/records.parquet", compression="zstd")
        idx = np.asarray([r["vec_idx"] for r in recs], np.int64)
        arr = np.lib.format.open_memmap(f"{d}/dirs_f16.npy", mode="w+", dtype=np.float16, shape=(len(idx), 5120))
        for i0 in range(0, len(idx), chunk):
            arr[i0:i0 + chunk] = np.asarray(vecs[idx[i0:i0 + chunk]]).astype(np.float16)
        arr.flush(); del arr
        files[fam] = {"rows": len(recs), "records_bytes": os.path.getsize(f"{d}/records.parquet"), "dirs_bytes": os.path.getsize(f"{d}/dirs_f16.npy")}
        log(f"  {fam}: {len(recs)} rows -> parquet {files[fam]['records_bytes'] / 2**20:.0f} MB + dirs {files[fam]['dirs_bytes'] / 2**30:.1f} GB")
    for fn in ("build_stats.json", "meta.json"):
        if os.path.exists(f"{bank}/{fn}"):
            json.dump(json.load(open(f"{bank}/{fn}")), open(f"{dst}/{fn}", "w"), indent=1, default=str)
    json.dump({"bank": bank, "n_vecs": int(n), "dtype_on_volume": str(np.dtype(dt)), "families": files,
               "layout": "dirs_f16.npy row i == records.parquet row i (sorted by vec_idx); unit rows, raw layer-42 residual space"},
              open(f"{dst}/export_meta.json", "w"), indent=1)
    return files


@app.function(image=image, cpu=16, memory=160 * 1024, ephemeral_disk=1024 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=8 * 3600)
def export_s3(bucket: str, prefix: str, aws_key: str, aws_secret: str, region: str = "us-east-1", include_training: bool = True,
              include_legacy: bool = True, presign_days: int = 7):
    import json, shutil, time
    import boto3
    from boto3.s3.transfer import TransferConfig
    t0 = time.time(); vol.reload()
    log = lambda m: print(f"[s3-export +{time.time() - t0:5.0f}s] {m}", flush=True)
    out = "/root/hf_heldout"
    staged = publish.local(dry=True)                                   # held-out bundle (parquet) -> /root/hf_heldout
    manifest = {"heldout": staged["files"], "training": {}, "extra": []}
    if include_training:
        for tag, bank in TRAIN_BANKS:
            if not include_legacy and tag.startswith("legacy"):
                continue
            if os.path.exists(f"{bank}/build_stats.json"):
                manifest["training"][tag] = _export_bank(tag, bank, out, log)
            else:
                log(f"skip {tag}: {bank} not finalized")
    os.makedirs(f"{out}/extra", exist_ok=True)
    for f in EXTRA_FILES:
        if os.path.exists(f):
            shutil.copy(f, f"{out}/extra/{os.path.basename(f)}"); manifest["extra"].append(os.path.basename(f))
    open(f"{out}/README.md", "a").write(
        "\n\n## S3 bundle layout\n* `heldout/` — everything in the table above (the held-out data).\n"
        "* `simple2m/sft_mix/<family>/{records.parquet, dirs_f16.npy}` — the 8M-row SFT mix (realact = 4M activations with 8-64-token contexts and full-context targets; "
        "sae2m / sae2m_dec = 2M encoder / 2M decoder rows of the 2M SAE). `simple2m/rl_pool/...` — the 941k-row RL pool (realact_ctx64_2048, sae2m, sae2m_dec).\n"
        "* `legacy_5m_chain/...` — the earlier chain's 2.75M-row midtrain mix (realact, sae, sae_dec, mlp, mlp_pair, mlp_triple, bsf, cluster) and its 1.45M-row RL pool "
        "(realact, realact_long, sae, bsf, cluster, mlp).\n* `extra/` — SAE max-acts files (131k SAE: maxacts.pt = Ultra-FineWeb scan, maxacts_fresh.pt = FineFineWeb scan; 2M SAE: maxacts_top5.pt), "
        "the 2M feature split, the document registry.\n\nDirections: `dirs_f16.npy` row i == `records.parquet` row i, unit rows in the raw layer-42 residual space (float16).\n")
    s3 = boto3.client("s3", region_name=region, aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:  # noqa
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
        log(f"created bucket {bucket}")
    cfg = TransferConfig(multipart_threshold=64 * 2**20, multipart_chunksize=64 * 2**20, max_concurrency=16)
    urls, total = [], 0
    for root, _, files in os.walk(out):
        for fn in sorted(files):
            p = os.path.join(root, fn); rel = os.path.relpath(p, out)
            key = f"{prefix}/{rel if rel.split('/')[0] in TOP_DIRS else 'heldout/' + rel}"
            s3.upload_file(p, bucket, key, Config=cfg); size = os.path.getsize(p); total += size
            urls.append((key, size, s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=presign_days * 86400)))
            if size > 2**30:
                log(f"uploaded {key} ({size / 2**30:.1f} GB, total {total / 2**30:.1f} GB)")
    manifest.update({"bucket": bucket, "prefix": prefix, "region": region, "n_objects": len(urls), "bytes": total, "presign_days": presign_days,
                     "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_s": time.time() - t0})
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1, default=str)
    open(f"{out}/urls.txt", "w").write("\n".join(f"{k}\t{b}\t{u}" for k, b, u in urls))
    for fn in ("manifest.json", "urls.txt"):
        s3.upload_file(f"{out}/{fn}", bucket, f"{prefix}/{fn}")
    log(f"DONE: {len(urls)} objects, {total / 2**30:.1f} GB -> s3://{bucket}/{prefix}/")
    return {"bucket": bucket, "prefix": prefix, "n_objects": len(urls), "gb": round(total / 2**30, 2), "wall_s": round(time.time() - t0), "urls": urls,
            "manifest_url": s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": f"{prefix}/manifest.json"}, ExpiresIn=presign_days * 86400)}
