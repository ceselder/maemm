"""Modal app (CPU): stage the MAEMM held-out data (Qwen3.6-27B layer 42) as parquet and push it to a PRIVATE HuggingFace dataset repo of
the owner's account (flip visibility on the Hub yourself if you want it public).

    ceselder/maemm-27b-heldout   (private)
      README.md
      feature_split.parquet                       2,097,152 rows: feature_id, split in {eval, rl, sft} of the 2M SAE (seed 2026)
      eval_2m_features_512.parquet                the 512 held-out 2M-SAE eval features: enc_dir/dec_dir (5120 fp32), b_enc, corpus_peak, gate
      eval_2m_features_100k_windows.parquet       every eval-split feature x its top-5 max-act windows (doc_idx, position, act, text)
      eval_directions_v3/<family>.parquet         the frozen eval directions (512 per family) + per-row provenance / pool record fields
      pool_heldout/<family>.parquet               the full held-out pool (91,311 rows: bsf, realact, sae, jlens, cluster) with directions
      doc_registry.json, doc_ids/<source>.parquet the Ultra-FineWeb document registry of the simple2m chain
    MODAL_PROFILE=safety-sahan modal deploy data/modal_hf_heldout_upload.py
    python -c "import modal; print(modal.Function.from_name('maemm-hf-heldout-upload-s2m','publish').remote())"
"""
import os

import modal

app = modal.App(os.environ.get("HF_HELDOUT_APP", "maemm-hf-heldout-upload-s2m"))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("numpy==2.4.6", "pyarrow", "pandas", "transformers==5.15.0", "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet"))
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
REPO = "ceselder/maemm-27b-heldout"
V3 = "/data/eval_universal_ho/eval_sets_heldout_v3.pt"
POOL = "/data/pool_heldout"
SPLIT = "/data/sae2m/feature_split.npz"
MAXACTS = "/data/sae2m/maxacts_top5.pt"
REG = "/data/simple2m/doc_registry.json"
MODEL = "Qwen/Qwen3.6-27B"


@app.function(image=image, cpu=16, memory=128 * 1024, ephemeral_disk=524288, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=4 * 3600)
def publish(repo: str = REPO, private: bool = True, dry: bool = False):
    import json, time
    import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, torch
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    t0 = time.time(); vol.reload()
    out = "/root/hf_heldout"; os.makedirs(out, exist_ok=True)
    os.environ["HF_HOME"] = "/data/hf_cache"
    log = lambda m: print(f"[hf-heldout +{time.time() - t0:5.0f}s] {m}", flush=True)
    manifest = {}

    def write(name, df_or_table):
        p = f"{out}/{name}"; os.makedirs(os.path.dirname(p), exist_ok=True)
        t = df_or_table if isinstance(df_or_table, pa.Table) else pa.Table.from_pandas(df_or_table, preserve_index=False)
        pq.write_table(t, p, compression="zstd"); manifest[name] = {"rows": t.num_rows, "bytes": os.path.getsize(p), "columns": t.column_names}
        log(f"{name}: {t.num_rows} rows, {os.path.getsize(p) / 2**20:.1f} MB")

    # 1. feature split
    sp = np.load(SPLIT); n_f = int(sum(len(sp[k]) for k in sp.files))
    fid = np.concatenate([np.asarray(sp[k], np.int32) for k in sp.files]); lab = np.concatenate([np.full(len(sp[k]), k) for k in sp.files])
    o = np.argsort(fid); write("feature_split.parquet", pd.DataFrame({"feature_id": fid[o], "split": lab[o]}))

    # 2. eval directions v3 (+ provenance)
    es = torch.load(V3, map_location="cpu", weights_only=False); meta = es["meta"]
    recs = [json.loads(l) for l in open(f"{POOL}/records.jsonl")]
    fams = [k[:-5] for k in es if k.endswith("_dirs") and torch.is_tensor(es[k])]
    rows_meta = meta.get("rows") or {}
    for fam in fams:
        dirs = es[f"{fam}_dirs"].float().numpy(); n = len(dirs)
        cols = {"row": np.arange(n), "family": [fam] * n, "direction": [d.tolist() for d in dirs]}
        for k, v in es.items():
            if k.startswith(fam + "_") and k != f"{fam}_dirs" and (torch.is_tensor(v) or isinstance(v, (list, tuple))):
                arr = v.numpy() if torch.is_tensor(v) else np.asarray(v, dtype=object)
                if getattr(arr, "ndim", 0) >= 1 and len(arr) == n:
                    cols[k[len(fam) + 1:]] = [x.tolist() if hasattr(x, "tolist") else x for x in arr]
        if fam == "sae":
            cols["feature_id"] = list(es["sae_feats"]); cols["corpus_peak"] = es["corpus_peak"].float().numpy().tolist()
        if fam in rows_meta:
            vidx = list(rows_meta[fam]); cols["pool_vec_idx"] = vidx
            for fld in ("target_text", "feature", "cluster", "block", "seq", "pos", "token_id", "token_str", "act", "act_norm", "val_auc"):
                vals = [recs[i].get(fld) for i in vidx]
                if any(x is not None for x in vals): cols["pool_" + fld] = [json.dumps(x) if isinstance(x, (list, dict)) else x for x in vals]
        write(f"eval_directions_v3/{fam}.parquet", pd.DataFrame(cols))
    sl = es["sae_slice"]
    write("eval_2m_features_512.parquet", pd.DataFrame({"feature_id": [int(f) for f in es["sae2m_enc_feats"]], "enc_dir": [d.tolist() for d in es["sae2m_enc_dirs"].float().numpy()],
                                                          "dec_dir": [d.tolist() for d in es["sae2m_dec_dirs"].float().numpy()], "b_enc": sl["b_enc"].float().numpy().tolist(),
                                                          "corpus_peak": es["sae2m_enc_corpus_peak"].float().numpy().tolist(), "gate": [float(sl["threshold"])] * 512}))
    json.dump({"cos_families": meta.get("cos_families"), "extra_families": meta.get("extra_families"), "sae_slice_families": meta.get("sae_slice_families"), "n": meta.get("n"), "seed": meta.get("seed"),
               "heldout_pool": meta.get("heldout_pool"), "d_sae_131k": meta.get("d_sae"), "v3": meta.get("v3"), "sae_slice": {"F": sl["F"], "threshold": float(sl["threshold"]), "ae": sl.get("ae"), "b_dec_in": "eval_2m_b_dec.json"}},
              open(f"{out}/eval_directions_v3/meta.json", "w"), indent=1, default=str)
    json.dump({"b_dec": sl["b_dec"].float().numpy().tolist()}, open(f"{out}/eval_2m_b_dec.json", "w"))

    # 3. eval-split features x top-5 windows (text decoded)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ma = torch.load(MAXACTS, map_location="cpu", weights_only=False)
    ev = np.asarray(sp["eval"], np.int64); ft = torch.as_tensor(ev)
    mt, ln, da, pos, acts = ma["max_tokens"][ft].numpy(), ma["lengths"][ft].numpy().astype(int), ma["doc_ids"][ft].numpy().astype(np.int64), ma["positions"][ft].numpy().astype(int), ma["max_acts"][ft].float().numpy()
    N = mt.shape[1]; rows = {"feature_id": [], "rank": [], "doc_idx": [], "position": [], "act": [], "n_tok": [], "text": []}
    ids_all, idx_all = [], []
    for i in range(len(ev)):
        for r in range(N):
            if acts[i, r] < 0: continue
            L = int(ln[i, r]); ids_all.append(mt[i, r, -L:].tolist() if L > 0 else []); idx_all.append((i, r))
    texts = tok.batch_decode(ids_all, skip_special_tokens=True)
    for (i, r), txt, ids_ in zip(idx_all, texts, ids_all):
        rows["feature_id"].append(int(ev[i])); rows["rank"].append(r); rows["doc_idx"].append(int(da[i, r]) + 100_000 if da[i, r] >= 0 else -1); rows["position"].append(int(pos[i, r]))
        rows["act"].append(float(acts[i, r])); rows["n_tok"].append(len(ids_)); rows["text"].append(txt)
    write("eval_2m_features_100k_windows.parquet", pd.DataFrame(rows))

    # 4. full held-out pool with directions
    vecs = np.memmap(f"{POOL}/vecs.f32", np.float32, "r", shape=(len(recs), 5120))
    for fam in sorted({r["family"] for r in recs}):
        idx = [i for i, r in enumerate(recs) if r["family"] == fam]
        df = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in recs[i].items()} for i in idx])
        df["direction"] = [np.asarray(vecs[recs[i]["vec_idx"]]).tolist() for i in idx]
        write(f"pool_heldout/{fam}.parquet", df)
    json.dump(json.load(open(f"{POOL}/build_stats.json")), open(f"{out}/pool_heldout/build_stats.json", "w"), indent=1, default=str)

    # 5. document registry
    if os.path.exists(REG):
        reg = json.load(open(REG)); json.dump(reg, open(f"{out}/doc_registry.json", "w"), indent=1)
        for name, s in reg["sources"].items():
            f = s.get("file")
            if f and os.path.exists(f):
                ids_ = json.load(open(f))["doc_idx"]; write(f"doc_ids/{name}.parquet", pd.DataFrame({"source": [name] * len(ids_), "doc_idx": np.asarray(ids_, np.int64)}))
    n_pool = {}
    for r in recs: n_pool[r["family"]] = n_pool.get(r["family"], 0) + 1
    readme = f"""---
license: mit
pretty_name: MAEMM held-out data (Qwen3.6-27B, layer 42)
---
# MAEMM held-out data — Qwen3.6-27B layer-42 activation-to-text inversion

Everything the MAEMM inverter is **evaluated** on and **never trained on**. All directions live in the raw layer-42 residual space of
`Qwen/Qwen3.6-27B` (d = 5120, unit rows). Corpus: `openbmb/Ultra-FineWeb` (split `en`); `doc_idx` is the 0-based document index in the
ordered single stream of that split. Uploaded {time.strftime('%Y-%m-%d %H:%MZ', time.gmtime())}.

| file | rows | what |
|---|---|---|
| `feature_split.parquet` | {n_f:,} | the 2,097,152-feature layer-42 BatchTopK SAE (k=64, gate {float(sl['threshold']):.4f}) split by seed 2026 into `eval` (100,000) / `rl` (150,000) / `sft` (1,847,152). No training bank of the simple2m chain contains an `eval` feature. |
| `eval_2m_features_512.parquet` | 512 | the standard-eval subset of the eval split: unit encoder column (`enc_dir`), unit decoder row (`dec_dir`), encoder bias, corpus peak (max activation over a 1.0B-token scan), gate. `eval_2m_b_dec.json` holds the shared decoder bias. |
| `eval_2m_features_100k_windows.parquet` | {len(rows['feature_id']):,} | every eval-split feature x its top-5 max-activating 32-token windows from the SAE's 1.0B-token training stream: doc_idx, position of the peak token, activation, decoded text (window ENDS at the peak token). |
| `eval_directions_v3/<family>.parquet` | 512 each | the frozen eval directions of eval cache v3: cosine families {meta.get('cos_families')}, the 131k-SAE family `sae` (feature ids + corpus peaks), MLP extras {meta.get('extra_families')}, and the 2M-SAE slice families {meta.get('sae_slice_families')}. Pool-derived families carry `pool_vec_idx` + the pool record fields (target text, feature / cluster ids ...). `random` is a synthetic Gaussian control. |
| `pool_heldout/<family>.parquet` | {n_pool} | the full held-out pool the eval directions are sampled from (train-disjoint per family): `realact` (real activations, Ultra-FineWeb stream head), `bsf` (block-sparse featurizer subspace directions), `sae` (131k-SAE encoder columns, 13,107 held-out features), `jlens` (J-lens token directions), `cluster` (probe clusters). Each row: the pool record + `direction`. |
| `doc_registry.json`, `doc_ids/<source>.parquet` | | which Ultra-FineWeb documents every training / eval source of the simple2m chain used, with pairwise overlaps (train vs eval = 0). |

## Held-out definitions
* **SAE features**: split by feature id (seed 2026). The 512 standard-eval features are a seeded sample of the 100,000 eval features.
* **Real activations**: the eval pool's activations come from the head of the Ultra-FineWeb stream (documents 0-100k), which no SAE training and no activation collection ever streamed; the simple2m training banks use documents from index 5.5M (SFT) and 9.5M (RL); the 2M SAE trained on 100k-1.78M.
* **131k-SAE features** (`sae`, `pool_heldout/sae`): 13,107 features excluded from every training bank of the earlier chains; the 512 eval features are a subset.

## Metrics we report on these
mean_all = mean over the 10 cosine families (random excluded) of the best-of-4 max-token cosine between the frozen base model's layer-42
activation of the generated text and the injected direction. SAE families: `fired` = best-of-4 max-token activation above the SAE's learned
gate; `norm_act` = that activation over the feature's corpus peak; `beat_corpus`. Generation: T=1, top-p 1, 16-64 new tokens, best of 4.

## Loading
```python
import pandas as pd
fs = pd.read_parquet("hf://datasets/{repo}/feature_split.parquet")
ev = pd.read_parquet("hf://datasets/{repo}/eval_directions_v3/realact.parquet")   # direction: list[5120] float32
```
"""
    open(f"{out}/README.md", "w").write(readme)
    json.dump({"repo": repo, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": manifest}, open(f"{out}/manifest.json", "w"), indent=1)
    total = sum(v["bytes"] for v in manifest.values()); log(f"staged {len(manifest)} parquet files, {total / 2**30:.2f} GB")
    if dry:
        return {"dry": True, "files": manifest}
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=repo, repo_type="dataset",
                      commit_message="MAEMM held-out data: feature split, eval directions v3, held-out pool, 2M eval features + windows, document registry")
    log(f"uploaded -> https://huggingface.co/datasets/{repo} (private={private})")
    return {"repo": f"https://huggingface.co/datasets/{repo}", "private": private, "files": {k: v["rows"] for k, v in manifest.items()}, "gb": round(total / 2**30, 2), "wall_s": time.time() - t0}
