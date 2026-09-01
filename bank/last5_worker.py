"""Per-GPU worker for the peak-in-last-5 SFT bank (launched by modal_last5_bank.py).

Scoring protocol (matches train/rl.py's reward scorer): forward [sink=bos_id] + span
token ids (right-padded, attention-masked) through the BASE 27B, read layer-42
resid_post (mxf.inject.read_resid), drop position 0 (attention sink), per-token score
= cos(h_t, v) on UNCENTERED residuals — rl.py normalizes h and dots the unit
direction with no mu subtraction, so the peak is defined exactly like the reward.

--mode probes   Re-anchor cluster/probe target spans so the direction's per-token
                cosine peaks within the LAST 5 tokens. Causal trick: truncating a
                span AFTER the argmax token leaves every kept activation unchanged
                (h_t depends only on the prefix), so one forward pass suffices and
                the re-anchored peak is exact, no re-forward needed:
                    peak already in last 5  -> keep the span verbatim
                    else                    -> truncate to tokens [0 .. t*+2]
                                               (peak = 3rd-from-last; 2-token margin
                                               against decode->retokenize drift)
                Drops only degenerate rows (empty tokenization / <4 final tokens /
                ~zero direction).
--mode verify   Re-score sampled rows of the FINISHED bank from records.jsonl +
                vecs.f32 (re-tokenizing target_text exactly like the SFT harness
                will) and report the peak-in-last-5 hit rate per family
                (uncentered primary; centered cos(h_t - mu, v) alongside).
                Writes <bank>/verify.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/pmx/helpers")

MAX_TOK = 256          # spans are 16-64 tokens; hard cap for safety
TOK_BUDGET = 49152     # max batch tokens (B * (Lmax+1))
MAX_B = 512            # max rows per batch


def log(tag, *a):
    print(f"[{tag}]", *a, flush=True)


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mxf.config import D_MODEL, MODEL

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, device_map={"": "cuda:0"}).eval()
    assert model.config.hidden_size == D_MODEL, model.config.hidden_size
    return tok, model


def make_batches(items, budget=TOK_BUDGET, max_b=MAX_B):
    """items: list of (key, ids). Length-sorted greedy packing under a token budget."""
    items = sorted(items, key=lambda x: -len(x[1]))
    out, cur = [], []
    for it in items:
        if cur:
            lmax = len(cur[0][1]) + 1
            if (len(cur) + 1) * lmax > budget or len(cur) >= max_b:
                out.append(cur)
                cur = []
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def score_batches(model, batches, get_dir, bos, mu=None, tag="score", log_every=50):
    """Yield (key, scores_uncentered, scores_centered|None) per row. get_dir(key) -> np [d]."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mxf.config import READ_LAYER
    from mxf.inject import read_resid

    t0, rows_done = time.time(), 0
    for bi, batch in enumerate(batches):
        B = len(batch)
        Lm = max(len(ids) for _, ids in batch)
        inp = torch.full((B, Lm + 1), int(bos), dtype=torch.long)
        att = torch.zeros((B, Lm + 1), dtype=torch.long)
        for j, (_, ids) in enumerate(batch):
            inp[j, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
            att[j, :1 + len(ids)] = 1
        dirs = torch.from_numpy(
            np.stack([get_dir(k).astype(np.float32) for k, _ in batch])).cuda()
        dirs = F.normalize(dirs, dim=-1)
        with torch.no_grad():
            h, _ = read_resid(model, READ_LAYER,
                              {"input_ids": inp.cuda(), "attention_mask": att.cuda()},
                              pool="all")                       # fp32 [B, Lm+1, d]
            h = h[:, 1:, :]                                     # drop sink position 0
            su = torch.einsum("btd,bd->bt", F.normalize(h, dim=-1), dirs).cpu().numpy()
            sc = None
            if mu is not None:
                mu_t = torch.from_numpy(mu.astype(np.float32)).cuda()
                sc = torch.einsum("btd,bd->bt",
                                  F.normalize(h - mu_t, dim=-1), dirs).cpu().numpy()
        for j, (k, ids) in enumerate(batch):
            L = len(ids)
            yield k, su[j, :L], (sc[j, :L] if sc is not None else None)
        rows_done += B
        if (bi + 1) % log_every == 0:
            el = time.time() - t0
            log(tag, f"batch {bi + 1}/{len(batches)} rows {rows_done} "
                     f"({rows_done / max(el, 1):.0f} rows/s)")


def run_probes(a):
    import numpy as np
    from mxf.config import D_MODEL

    rows = []
    with open(a.in_jsonl) as f:
        for line in f:
            o = json.loads(line)
            if o["i"] % a.world == a.rank:
                rows.append(o)
    log(f"r{a.rank}", f"{len(rows)} probe rows (world {a.world})")
    dirs = np.memmap(a.dirs, np.float32, "r").reshape(-1, D_MODEL)

    tok, model = load_model()
    enc = tok([o["text"] for o in rows], add_special_tokens=False)["input_ids"]
    outs, items, info = {}, [], {}
    for o, ids in zip(rows, enc):
        i = o["i"]
        ids = ids[:MAX_TOK]
        if len(ids) < 1:
            outs[i] = {"i": i, "src_vec_idx": o["src_vec_idx"], "keep": False,
                       "reason": "empty"}
        elif float(np.linalg.norm(dirs[i])) < 1e-6:
            outs[i] = {"i": i, "src_vec_idx": o["src_vec_idx"], "keep": False,
                       "reason": "zero_dir"}
        else:
            items.append((i, ids))
            info[i] = (o["src_vec_idx"], o["text"], ids)

    for i, su, _ in score_batches(model, make_batches(items), lambda k: dirs[k],
                                  a.bos_id, tag=f"r{a.rank}"):
        sv, text, ids = info[i]
        L = len(ids)
        t = int(np.argmax(su))
        if t >= L - 5:                                   # already peaks in the last 5
            final_ids, out_text, trunc = ids, text, False
        else:                                            # causal prefix-truncate at t*+2
            final_ids = ids[:t + 3]
            out_text, trunc = tok.decode(final_ids), True
        if len(final_ids) < 4:
            outs[i] = {"i": i, "src_vec_idx": sv, "keep": False, "reason": "too_short",
                       "peak_idx": t, "n_tok_orig": L}
        else:
            outs[i] = {"i": i, "src_vec_idx": sv, "keep": True, "target_text": out_text,
                       "n_tok_orig": L, "n_tok_final": len(final_ids), "peak_idx": t,
                       "cos_peak": round(float(su[t]), 4), "truncated": trunc}

    with open(a.out + ".tmp", "w") as f:
        for i in sorted(outs):
            f.write(json.dumps(outs[i]) + "\n")
    os.replace(a.out + ".tmp", a.out)
    kept = sum(1 for o in outs.values() if o["keep"])
    trunc = sum(1 for o in outs.values() if o.get("truncated"))
    log(f"r{a.rank}", f"DONE {kept}/{len(outs)} kept ({trunc} truncated)")


def run_verify(a):
    import numpy as np
    from mxf.config import D_MODEL

    recs = [json.loads(l) for l in open(f"{a.bank}/records.jsonl")]
    byfam = {}
    for r in recs:
        byfam.setdefault(r.get("family", "?"), []).append(r)
    rng = np.random.default_rng(a.seed)
    mu = np.load(a.mu).astype(np.float32)
    vfd = os.open(f"{a.bank}/vecs.f32", os.O_RDONLY)
    row_b = D_MODEL * 4

    def get_dir(vec_idx):
        buf = b""
        while len(buf) < row_b:
            chunk = os.pread(vfd, row_b - len(buf), vec_idx * row_b + len(buf))
            assert chunk, f"short read at vec {vec_idx}"
            buf += chunk
        return np.frombuffer(buf, np.float32)

    tok, model = load_model()
    result = {"n_requested_per_family": a.n, "seed": a.seed,
              "protocol": "fwd [sink]+retokenized(target_text); drop pos0; "
                          "uncentered cos(h_t,v) primary (rl.py reward metric); "
                          "centered cos(h_t-mu,v) secondary; hit = argmax in last 5",
              "families": {}}
    for fam, frecs in sorted(byfam.items()):
        pick = [frecs[j] for j in rng.permutation(len(frecs))[:a.n]]
        enc = tok([r["target_text"] for r in pick], add_special_tokens=False)["input_ids"]
        items, meta = [], {}
        n_bad = 0
        for r, ids in zip(pick, enc):
            ids = ids[:MAX_TOK]
            if len(ids) < 1:
                n_bad += 1
                continue
            items.append((r["vec_idx"], ids))
            meta[r["vec_idx"]] = len(ids)
        hits_u = hits_c = 0
        peaks_u, lens, from_end = [], [], []
        for k, su, sc in score_batches(model, make_batches(items),
                                       lambda v: get_dir(v), a.bos_id, mu=mu,
                                       tag=f"verify-{fam}"):
            L = meta[k]
            tu, tc = int(np.argmax(su)), int(np.argmax(sc))
            hits_u += tu >= L - 5
            hits_c += tc >= L - 5
            peaks_u.append(float(su[tu]))
            lens.append(L)
            from_end.append(L - 1 - tu)
        n = len(items)
        result["families"][fam] = {
            "n": n, "n_untokenizable": n_bad,
            "hit_last5_uncentered": round(hits_u / max(n, 1), 4),
            "hit_last5_centered": round(hits_c / max(n, 1), 4),
            "peak_cos_mean": round(float(np.mean(peaks_u)), 4),
            "peak_cos_median": round(float(np.median(peaks_u)), 4),
            "len_tok_mean": round(float(np.mean(lens)), 1),
            "peak_from_end_median": float(np.median(from_end)),
            "peak_from_end_p90": float(np.percentile(from_end, 90)),
        }
        log("verify", fam, json.dumps(result["families"][fam]))
    with open(f"{a.bank}/verify.json.tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(f"{a.bank}/verify.json.tmp", f"{a.bank}/verify.json")
    log("verify", "DONE ->", f"{a.bank}/verify.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("probes", "verify"), required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--in-jsonl", help="probes: driver-written {i, src_vec_idx, text} lines")
    ap.add_argument("--dirs", help="probes: f32 [n_probe, d] direction rows, row i = line i")
    ap.add_argument("--out", help="probes: output jsonl path")
    ap.add_argument("--bank", help="verify: finished bank dir")
    ap.add_argument("--n", type=int, default=1024, help="verify: rows per family")
    ap.add_argument("--mu", help="verify: whiten_mu.npy path (centered variant)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bos-id", type=int, default=248044)
    a = ap.parse_args()
    if a.mode == "probes":
        assert a.in_jsonl and a.dirs and a.out
        run_probes(a)
    else:
        assert a.bank and a.mu
        run_verify(a)


if __name__ == "__main__":
    main()
