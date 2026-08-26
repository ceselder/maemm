"""Collect layer-42 residuals from Ultra-FineWeb for (a) BSF/SASA training and (b) SFT firing-context
spans. Takes the first `seq_len` tokens of each doc (context-aware, per the reward design), forwards
Qwen3.6-27B, reads L42 via the shared read_resid hook (full forward — safe on the hybrid arch, no layer
truncation footgun). Saves:
  acts.f16   [n_seq, seq_len, d]  layer-42 residual per token
  toks.i32   [n_seq, seq_len]     token ids (for pulling 16-64 tok context spans later)
  whiten_mu.npy [d], whiten_zca.npy [d,d]   ZCA whitening from a sample (the key 8B fix)
Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/pmx/helpers python bsf/collect_acts.py --n-seq 20000
"""
import argparse, os, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from mxf.config import MODEL, READ_LAYER, D_MODEL
from mxf.inject import read_resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=20000)      # 20k * 512 = 10.2M tokens
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-dir", default="/root/pmx/bsf27b/acts")
    ap.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    ap.add_argument("--config", default="default")
    ap.add_argument("--split", default="en")
    a = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(a.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": dev})
    model.eval()
    print(f"[collect] model loaded; target {a.n_seq} seqs x {a.seq_len} tok = {a.n_seq*a.seq_len/1e6:.1f}M", flush=True)

    ds = load_dataset(a.dataset, a.config, split=a.split, streaming=True)
    acts = np.memmap(f"{a.out_dir}/acts.f16", dtype=np.float16, mode="w+", shape=(a.n_seq, a.seq_len, D_MODEL))
    toks = np.memmap(f"{a.out_dir}/toks.i32", dtype=np.int32, mode="w+", shape=(a.n_seq, a.seq_len))

    it = iter(ds)
    n = 0
    while n < a.n_seq:
        batch = []
        while len(batch) < a.batch and n + len(batch) < a.n_seq:
            try:
                doc = next(it)
            except StopIteration:
                break
            text = doc.get("content") or doc.get("text") or ""
            ids = tok(text, add_special_tokens=False, truncation=True, max_length=a.seq_len).input_ids
            if len(ids) < a.seq_len:      # need a full context window
                continue
            batch.append(ids[:a.seq_len])
        if not batch:
            break
        ids = torch.tensor(batch, device=dev)
        with torch.no_grad():
            h, _ = read_resid(model, READ_LAYER, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, pool="all")
        b = len(batch)
        acts[n:n + b] = h.float().cpu().numpy().astype(np.float16)
        toks[n:n + b] = np.asarray(batch, dtype=np.int32)
        n += b
        if n % (a.batch * 20) == 0:
            print(f"[collect] {n}/{a.n_seq} seqs", flush=True)
    acts.flush(); toks.flush()

    # ---- ZCA whitening from a sample (flattens the variance spectrum; the decisive 8B fix) ----
    ns = min(500, n)
    samp = np.asarray(acts[:ns]).reshape(-1, D_MODEL).astype(np.float32)
    mu = samp.mean(0)
    X = samp - mu
    cov = (X.T @ X) / len(X)
    U, S, _ = np.linalg.svd(cov)
    zca = (U / np.sqrt(S + 1e-5)) @ U.T
    np.save(f"{a.out_dir}/whiten_mu.npy", mu.astype(np.float32))
    np.save(f"{a.out_dir}/whiten_zca.npy", zca.astype(np.float32))
    meta = {"n_seq": int(n), "seq_len": a.seq_len, "d_model": D_MODEL, "read_layer": READ_LAYER,
            "dataset": a.dataset, "whiten_sample_tokens": int(len(samp))}
    import json
    json.dump(meta, open(f"{a.out_dir}/meta.json", "w"), indent=1)
    print(f"[collect] DONE {n} seqs -> {a.out_dir} | whiten from {len(samp)} tok", flush=True)


if __name__ == "__main__":
    main()
