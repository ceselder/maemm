"""ONE-TIME bank for the inline WildChat fire-prediction eval (train/inline_extra_evals.py):
64-token windows of real WildChat-1M (English, non-toxic) conversations, scored with OUR Qwen3.6-27B
layer-42 SAE on the clean base, and for every testbed feature 4 FIRING + 4 NON-FIRING windows.

Read path == the shared clean-base protocol (BOS sink prepended + skipped, 10x-median norm filter,
ReLU SAE encode of the target feature, max over content tokens). Firing: act >= max(--fire-frac x
corpus_peak, fire) (0.25 x corpus peak, the Neel-eval-2 criterion); when a feature has fewer than
--n-pos such windows the threshold relaxes to the testbed fire threshold (flagged `relaxed`); still
short -> whatever is available (the inline eval requires >= 2 of each side). Non-firing: act <
--neg-max-act (0.01, as the testbed's verified negatives). Selection is a seeded random draw among
the qualifying windows (not the top-k) so positives are representative, not extreme.

    python eval/wildchat_bank.py --testbed /data/eval_autointerp/testbed_v2.json \
        --sae-path /data/sae/ae.pt --out /data/eval_wildchat/windows.json      # GPU (modal_wildchat_bank.py)

Output {"config": {...}, "features": [{"feature", "corpus_peak", "fire_thresh", "relaxed",
"n_fire_available", "n_zero_available", "windows": [{"text", "act", "fires", "peak_tok"}]}]}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "train"), "/pmx/RL", "/pmx/helpers"):
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

NORM_FILTER_MULT = 10.0


def iter_windows(rows, tok, win, max_per_conv, rng, english_only=True, skip_toxic=True):
    """Yield (token_ids[win], conversation_hash) — non-overlapping `win`-token windows over the
    concatenated turns of each conversation, at most max_per_conv (seeded random pick) per conversation."""
    for row in rows:
        if english_only and row.get("language") != "English":
            continue
        if skip_toxic and row.get("toxic"):
            continue
        text = "\n\n".join((m.get("content") or "") for m in (row.get("conversation") or []) if m.get("content"))
        if not text.strip():
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n = len(ids) // win
        if n == 0:
            continue
        starts = list(range(0, n * win, win))
        if len(starts) > max_per_conv:
            starts = sorted(int(s) for s in rng.choice(starts, max_per_conv, replace=False))
        for s in starts:
            yield ids[s:s + win], row.get("conversation_hash")


def select_windows(acts, feats, corpus_peak, fire, fire_frac, neg_max_act, n_pos, n_neg, rng):
    """acts [n_windows, n_feats] per-window max act. -> per-feature dict of selection metadata +
    (pos_idx, neg_idx). Pure numpy (unit-testable)."""
    out = {}
    for j, f in enumerate(feats):
        av = acts[:, j]
        strict = max(fire_frac * float(corpus_peak[f]), fire)
        pos_all = np.where(av >= strict)[0]
        relaxed = False
        if len(pos_all) < n_pos:
            pos_all = np.where(av >= fire)[0]
            relaxed = True
        neg_all = np.where(av < neg_max_act)[0]
        pos = np.sort(rng.choice(pos_all, min(n_pos, len(pos_all)), replace=False)) if len(pos_all) else np.array([], int)
        neg = np.sort(rng.choice(neg_all, min(n_neg, len(neg_all)), replace=False)) if len(neg_all) else np.array([], int)
        out[f] = {"fire_thresh": float(fire if relaxed else strict), "relaxed": relaxed,
                  "n_fire_available": int(len(pos_all)), "n_zero_available": int(len(neg_all)),
                  "pos": pos.tolist(), "neg": neg.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--testbed", default="/data/eval_autointerp/testbed_v2.json")
    ap.add_argument("--sae-path", default=None, help="default: the testbed config's sae_path")
    ap.add_argument("--out", default="/data/eval_wildchat/windows.json")
    ap.add_argument("--dataset", default="allenai/WildChat-1M")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-windows", type=int, default=40_000)
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--max-per-conv", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fire-frac", type=float, default=0.25, help="firing = act >= this x corpus peak (and > fire)")
    ap.add_argument("--neg-max-act", type=float, default=0.01)
    ap.add_argument("--n-pos", type=int, default=4)
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from inline_extra_evals import SubSAE
    from mxf.config import MODEL, READ_LAYER
    from mxf.inject import read_resid

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    tb = json.load(open(a.testbed))
    cfg = tb["config"]
    feats = [int(r["feature"]) for r in tb["features"]]
    corpus_peak = {int(r["feature"]): float(r["corpus_peak"]) for r in tb["features"]}
    fire = float(cfg.get("fire", 1.0))
    sae_path = a.sae_path or cfg["sae_path"]
    dev = a.device

    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset(a.dataset, split=a.split, streaming=True)
    windows, hashes = [], []
    for ids, h in iter_windows(ds, tok, a.win, a.max_per_conv, rng):
        windows.append(ids)
        hashes.append(h)
        if len(windows) >= a.n_windows:
            break
        if len(windows) % 5000 == 0:
            print(f"[wildchat] {len(windows)} windows from {len(set(hashes))} conversations ({time.time() - t0:.0f}s)", flush=True)
    n_conv = len(set(hashes))
    print(f"[wildchat] {len(windows)} x {a.win}-token windows from {n_conv} English non-toxic conversations "
          f"({time.time() - t0:.0f}s)", flush=True)

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                device_map={"": dev})
    base.eval()
    subsae = SubSAE.from_file(sae_path, feats, dev)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    print(f"[wildchat] base + {len(feats)}-feature sub-SAE ready ({time.time() - t0:.0f}s)", flush=True)

    acts = np.zeros((len(windows), len(feats)), dtype=np.float32)
    peak_pos = np.zeros((len(windows), len(feats)), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(windows), a.batch):
            w = torch.tensor(windows[s:s + a.batch], device=dev)
            ids = torch.cat([torch.full((w.shape[0], 1), sink, device=dev, dtype=w.dtype), w], 1)
            am = torch.ones_like(ids)
            h, mask = read_resid(base, READ_LAYER, {"input_ids": ids, "attention_mask": am}, pool="all")
            keep = mask.clone()
            keep[:, 0] = False
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            keep = keep & (nrm <= NORM_FILTER_MULT * med)
            per = subsae.encode_features(h, feats).masked_fill(~keep.unsqueeze(-1), 0.0)     # [B, T, k]
            mx, arg = per.max(1)
            acts[s:s + w.shape[0]] = mx.float().cpu().numpy()
            peak_pos[s:s + w.shape[0]] = arg.cpu().numpy()
            if (s // a.batch) % 100 == 0:
                print(f"[wildchat] scored {min(s + a.batch, len(windows))}/{len(windows)} ({time.time() - t0:.0f}s)", flush=True)

    sel = select_windows(acts, feats, corpus_peak, fire, a.fire_frac, a.neg_max_act, a.n_pos, a.n_neg, rng)
    recs = []
    for j, f in enumerate(feats):
        s = sel[f]
        rows = []
        for i in s["pos"] + s["neg"]:
            pk = int(peak_pos[i, j]) - 1                                                  # sink offset
            rows.append({"text": tok.decode(windows[i], skip_special_tokens=True), "act": float(acts[i, j]),
                         "fires": int(i in s["pos"]),
                         "peak_tok": tok.decode(windows[i][pk:pk + 1]) if 0 <= pk < a.win else None,
                         "conversation_hash": hashes[i]})
        recs.append({"feature": f, "corpus_peak": corpus_peak[f], "fire_thresh": s["fire_thresh"], "relaxed": s["relaxed"],
                     "n_fire_available": s["n_fire_available"], "n_zero_available": s["n_zero_available"], "windows": rows})
    n_full = sum(1 for r in recs if sum(w["fires"] for w in r["windows"]) >= a.n_pos and
                 sum(1 - w["fires"] for w in r["windows"]) >= a.n_neg)
    n_strict = sum(1 for r in recs if not r["relaxed"] and sum(w["fires"] for w in r["windows"]) >= a.n_pos)
    n_usable = sum(1 for r in recs if sum(w["fires"] for w in r["windows"]) >= 2 and sum(1 - w["fires"] for w in r["windows"]) >= 2)
    out = {"config": {"dataset": a.dataset, "split": a.split, "win": a.win, "n_windows": len(windows), "n_convs": n_conv,
                      "max_per_conv": a.max_per_conv, "fire_frac_of_peak": a.fire_frac, "fire": fire,
                      "neg_max_act": a.neg_max_act, "n_pos": a.n_pos, "n_neg": a.n_neg, "seed": a.seed,
                      "testbed": os.path.abspath(a.testbed), "sae_path": sae_path, "model": MODEL, "read_layer": READ_LAYER,
                      "read_path": "bos-sink-skip + 10x-median norm filter, clean base, ReLU SAE encode, max over tokens",
                      "english_only": True, "skip_toxic": True,
                      "summary": {"features": len(recs), "full_4pos_4neg": n_full, "strict_thresh_4pos": n_strict,
                                  "usable_2pos_2neg": n_usable,
                                  "frac_windows_zero_any_feat": float((acts < a.neg_max_act).mean())}},
           "features": recs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wildchat] features: {len(recs)} | 4pos+4neg: {n_full} | strict-threshold 4pos: {n_strict} | usable (>=2/>=2): "
          f"{n_usable} | mean fire-available {np.mean([r['n_fire_available'] for r in recs]):.1f} windows/feature", flush=True)
    print(f"WILDCHAT_BANK_DONE {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
