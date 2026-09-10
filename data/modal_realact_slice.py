"""Contiguous 1M-row slice of one realact_short part -> a small single-family bank the compositor reads on its DENSE path
(sequential streaming) instead of 91k random preads per 9M-row part, which crawled (106 s -> 211 s -> 1939 s per part) at 00:19Z Sep 10."""
import json, os, time
import modal
app = modal.App("maemm-realact-slice")
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.4.6")
D = 5120

@app.function(image=image, cpu=8, memory=32768, ephemeral_disk=524288, volumes={"/data": vol}, timeout=3 * 3600)
def slice_part(src: str = "/data/banks/realact_short_20m_p", n: int = 1_000_000, out: str = "/data/banks/realact_short_1m_p") -> dict:
    import numpy as np
    vol.reload(); t0 = time.time()
    st = json.load(open(f"{src}/build_stats.json")); assert list(st["families"]) == ["realact"], st["families"]
    os.makedirs(out, exist_ok=True)
    vf = f"{src}/vecs.f16"; assert os.path.getsize(vf) == st["n_examples"] * D * 2, (os.path.getsize(vf), st["n_examples"])
    # vectors: rows [0, n) are a contiguous 10 GB read
    with open(vf, "rb") as fh, open(f"{out}/vecs.f16.tmp", "wb") as fo:
        left = n * D * 2
        while left:
            b = fh.read(min(256 << 20, left)); assert b; fo.write(b); left -= len(b)
    os.replace(f"{out}/vecs.f16.tmp", f"{out}/vecs.f16"); t1 = time.time()
    # records: lines are in shuffled order; keep vec_idx < n, write sorted by vec_idx so line i == vec_idx i
    keep = [None] * n; got = 0
    with open(f"{src}/records.jsonl") as fh:
        for line in fh:
            assert line.startswith('{"vec_idx": '), line[:40]
            vi = int(line[12:line.index(",", 12)])
            if vi < n:
                keep[vi] = line if line.endswith("\n") else line + "\n"; got += 1
    assert got == n and all(k is not None for k in keep), got
    with open(f"{out}/records.jsonl.tmp", "w") as fo:
        fo.writelines(keep)
    os.replace(f"{out}/records.jsonl.tmp", f"{out}/records.jsonl")
    meta = {k: st[k] for k in ("kind", "model", "layer", "d", "dataset", "seed", "seq_len", "per_window") if k in st}
    meta.update({"n_examples": n, "families": {"realact": n}, "source": src, "slice": f"vec_idx [0, {n}) of {src} (contiguous), records re-sorted so line i == vec_idx i",
                 "created": time.time(), "wall_s": time.time() - t0})
    json.dump(meta, open(f"{out}/build_stats.json", "w"), indent=1); json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    vol.commit()
    r = {"out": out, "n": n, "vec_copy_s": round(t1 - t0, 1), "records_s": round(time.time() - t1, 1)}; print(r, flush=True); return r
