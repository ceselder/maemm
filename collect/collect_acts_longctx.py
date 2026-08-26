"""Collect LONG-CONTEXT layer-42 activations as an RL-ONLY data source (fjiahai's suggestion: long
contexts have genuinely different features; use them for RL, NOT SFT). Streams real docs, tokenizes up to
--seq-len (8192), forwards Qwen3.6-27B, reads L42, samples --pos-per-doc positions in [--pos-min, len)
(i.e. OUTSIDE the SFT 16-512 range), saves the activation h_p (the RL inject-direction source) + the
target span ending at p. Shardable across GPUs.

Usage (one per GPU): CUDA_VISIBLE_DEVICES=N PYTHONPATH=/root/pmx/helpers python bsf/collect_acts_longctx.py --shard N --n-shards 5
"""
import argparse, os, json, numpy as np, torch, random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from mxf.config import MODEL, READ_LAYER, D_MODEL
from mxf.inject import read_resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--min-len", type=int, default=1024)   # skip docs shorter than this (need room >512)
    ap.add_argument("--pos-min", type=int, default=512)    # sample positions in [pos-min, len) -> OUTSIDE SFT's 16-512
    ap.add_argument("--pos-per-doc", type=int, default=4)
    ap.add_argument("--n-docs", type=int, default=6000)    # per shard; 5 shards * 6000 * 4 = 120k long-ctx dirs
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=5)
    ap.add_argument("--out-dir", default="/root/pmx/bsf27b/acts_long")
    ap.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    ap.add_argument("--config", default="default"); ap.add_argument("--split", default="en")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda:0"; rng = random.Random(a.seed + 1000 * a.shard)
    os.makedirs(a.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": dev}).eval()
    print(f"[longctx s{a.shard}] model loaded; target {a.n_docs} docs x {a.pos_per_doc} pos", flush=True)
    ds = load_dataset(a.dataset, a.config, split=a.split, streaming=True)
    it = iter(ds)
    cap = a.n_docs * a.pos_per_doc
    H = np.memmap(f"{a.out_dir}/acts_long_s{a.shard}.f16", dtype=np.float16, mode="w+", shape=(cap, D_MODEL))
    recs = open(f"{a.out_dir}/records_s{a.shard}.jsonl", "w")
    doc_i = -1; kept = 0; emitted = 0
    while kept < a.n_docs:
        try: doc = next(it)
        except StopIteration: break
        doc_i += 1
        if doc_i % a.n_shards != a.shard: continue          # shard by doc index
        text = doc.get("content") or doc.get("text") or ""
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=a.seq_len).input_ids
        if len(ids) < a.min_len: continue
        t = torch.tensor([ids], device=dev)
        with torch.no_grad():
            h, _ = read_resid(model, READ_LAYER, {"input_ids": t, "attention_mask": torch.ones_like(t)}, pool="all")
        h = h[0]                                            # [seq, d]
        L_seq = len(ids)
        for _ in range(a.pos_per_doc):
            p = rng.randint(1, L_seq - 1)                   # uniform 1..8192 (capped by doc len / --seq-len)
            Lspan = rng.randint(16, 64)
            txt = tok.decode(ids[max(0, p - Lspan + 1): p + 1])
            if len(txt.strip()) < 3: continue
            H[emitted] = h[p].float().cpu().numpy().astype(np.float16)
            recs.write(json.dumps({"idx": emitted, "pos": p, "doc_len": L_seq,
                                   "target_text": txt, "family": "realact_long"}) + "\n")
            emitted += 1
        kept += 1
        if kept % 50 == 0: print(f"[longctx s{a.shard}] {kept}/{a.n_docs} docs, {emitted} acts", flush=True)
    H.flush(); recs.close()
    json.dump({"shard": a.shard, "n_docs": kept, "n_acts": emitted, "seq_len": a.seq_len,
               "pos_min": a.pos_min, "min_len": a.min_len}, open(f"{a.out_dir}/meta_s{a.shard}.json", "w"))
    print(f"[longctx s{a.shard}] DONE {kept} docs {emitted} acts", flush=True)


if __name__ == "__main__":
    main()
