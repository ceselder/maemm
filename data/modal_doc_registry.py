"""Modal app (CPU): the Ultra-FineWeb DOCUMENT REGISTRY of the simple2m chain -- which documents every training / eval source used,
and proof that the splits are document-disjoint.

    extract(bank_dir, field, offset)   -> <bank>/doc_ids_used.json from records.jsonl (banks whose builder did not write it:
                                          the 2M-SAE banks carry doc_id = single-stream index - 100_000 -> offset 100000)
    registry(sources_json)             -> /data/simple2m/doc_registry.json: per source {n_docs, min, max, n_rows} + pairwise
                                          overlaps of the document sets + the SAE dictionary's training span + eval sources

    MODAL_PROFILE=safety-sahan modal deploy data/modal_doc_registry.py
"""
import os

import modal

app = modal.App(os.environ.get("DOC_REGISTRY_APP", "maemm-doc-registry-s2m"))
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.4.6")
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
OUT = "/data/simple2m/doc_registry.json"


@app.function(image=image, cpu=8, memory=64 * 1024, volumes={"/data": vol}, timeout=3 * 3600)
def extract(bank_dir: str, field: str = "doc_id", offset: int = 0, out_name: str = "doc_ids_used.json", overwrite: bool = False):
    """records.jsonl -> sorted unique (field + offset) -> <bank_dir>/<out_name>. Streaming parse (no full load)."""
    import json, time
    vol.reload()
    out = f"{bank_dir}/{out_name}"
    if os.path.exists(out) and not overwrite:
        return json.load(open(out)) | {"cached": True}
    t0 = time.time(); ids = set(); n_rows = 0; n_missing = 0; per_family = {}
    needle = f'"{field}": '
    with open(f"{bank_dir}/records.jsonl") as f:
        for line in f:
            n_rows += 1
            i = line.find(needle)
            if i < 0:
                n_missing += 1; continue
            j = i + len(needle); k = j
            while k < len(line) and (line[k].isdigit() or line[k] == "-"):
                k += 1
            v = int(line[j:k])
            if v >= 0:
                ids.add(v + offset)
            fi = line.find('"family": "')
            if fi >= 0:
                fam = line[fi + 11: line.find('"', fi + 11)]; per_family[fam] = per_family.get(fam, 0) + 1
    doc_ids = sorted(ids)
    reg = {"bank": bank_dir, "field": field, "offset": offset, "n_rows": n_rows, "rows_without_field": n_missing, "families": per_family,
           "n_docs_used": len(doc_ids), "doc_range": [doc_ids[0], doc_ids[-1]] if doc_ids else None,
           "doc_idx_meaning": "absolute document index in the ordered single stream of openbmb/Ultra-FineWeb (split en), 0-based",
           "doc_idx": doc_ids, "wall_s": time.time() - t0}
    json.dump(reg, open(out, "w")); vol.commit()
    return {k: v for k, v in reg.items() if k != "doc_idx"}


@app.function(image=image, cpu=8, memory=64 * 1024, volumes={"/data": vol}, timeout=3600)
def registry(sources_json: str, out: str = OUT):
    """sources_json: JSON list of {"name", "role" (sft|rl|eval|sae_dict), "ids_file" (doc_ids_used.json path) OR "range" [lo, hi)}.
    Writes the registry with per-source counts and the pairwise document overlaps (train vs eval MUST be 0)."""
    import json, time
    import numpy as np
    vol.reload()
    srcs = json.loads(sources_json); sets = {}; summary = {}
    for s in srcs:
        if s.get("ids_file"):
            r = json.load(open(s["ids_file"])); ids = np.asarray(r["doc_idx"], np.int64)
            summary[s["name"]] = {"role": s["role"], "kind": "exact", "n_docs": int(len(ids)), "doc_range": [int(ids.min()), int(ids.max())] if len(ids) else None,
                                  "n_rows": r.get("n_rows"), "file": s["ids_file"], "bank": r.get("bank")}
        else:
            lo, hi = s["range"]; ids = None
            summary[s["name"]] = {"role": s["role"], "kind": "range", "doc_range": [lo, hi], "n_docs": hi - lo, "note": s.get("note")}
        sets[s["name"]] = (ids, s.get("range"))
    names = list(sets); overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia, ra = sets[a]; ib, rb = sets[b]
            if ia is not None and ib is not None:
                ov = int(np.intersect1d(ia, ib).size)
            elif ia is not None:
                ov = int(((ia >= rb[0]) & (ia < rb[1])).sum())
            elif ib is not None:
                ov = int(((ib >= ra[0]) & (ib < ra[1])).sum())
            else:
                ov = max(0, min(ra[1], rb[1]) - max(ra[0], rb[0]))
            overlaps[f"{a} & {b}"] = ov
    roles = {n: summary[n]["role"] for n in names}
    train_eval_overlaps = {k: v for k, v in overlaps.items() if v and {roles[k.split(" & ")[0]], roles[k.split(" & ")[1]]} & {"eval"}}
    reg = {"built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset": "openbmb/Ultra-FineWeb (split en), ordered single-stream document index",
           "sources": summary, "pairwise_doc_overlaps": overlaps, "train_eval_overlaps_nonzero": train_eval_overlaps,
           "disjoint_train_eval": not train_eval_overlaps}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(reg, open(out, "w"), indent=1); vol.commit()
    return reg
