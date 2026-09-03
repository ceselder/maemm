"""Snippet-locality eval: within a text, does the target SAE feature fire on a LOCALIZED short
snippet, or is the activation smeared across the whole text?

A good max-activating example fires on a crisp, interpretable snippet. This eval asks whether
MAEMM inverter rollouts reproduce that property — i.e. whether the peak-token scoring used
everywhere else in MAEMMBench reflects genuine localized evocation — by comparing the per-token
activation PROFILE of each rollout against the profiles of the feature's own top max-activating
corpus examples, under the exact shared clean-base read path (BOS sink prepended + skipped,
right-padding masked, 10x-median norm filter, ReLU SAE encode of the target feature).

Locality metrics per text (profile a_1..a_T >= 0 over kept content tokens):
  peak_share   max_t a_t / sum_t a_t                (1.0 = all mass on one token)
  win3_share   best contiguous 3-token window's fraction of total positive mass
  win5_share   best contiguous 5-token window's fraction of total positive mass
  gini         Gini coefficient of the profile      (1.0 = maximally concentrated)
  spread_half  number of tokens with a_t >= 0.5 * max_t a_t (lower = more localized)
Locality is only meaningful where the feature actually FIRES: metrics are computed for every
text but aggregated only over texts with peak act > the fire threshold (testbed config's
`fire`); the firing fraction per arm is reported alongside.

Length caveat: rollouts (<=64 new tokens) are ~2x longer than the 32-token max-act windows, and
peak_share / gini are length-sensitive. The fixed-k window shares (win3/win5) are the most
length-robust metrics — read those first. Per-text token counts ship in the JSON.

Two stages:
  # GPU (Modal: ../modal_snippet_locality.py) — profiles + per-text metrics -> locality.json
  python eval/snippet_locality.py build \
      --testbed /data/eval_autointerp/testbed_v2.json --out /data/eval_autointerp/locality.json
  # local — aggregates, PAIRED maemm-vs-real test, autointerp cross-links -> results json
  python eval/snippet_locality.py score --locality locality.json \
      --autointerp-results results.json --out locality_results.json
"""
import argparse
import json
import os
import time

import numpy as np

GEN_SEED = 20260829    # same forked-RNG seed as eval/autointerp_detection.py — deterministic,
                       # and identical across adapters so A-vs-B comparisons are apples-to-apples

METRICS = ["peak_share", "win3_share", "win5_share", "gini", "spread_half"]
# for every metric, the direction that means MORE localized (for readable paired-diff signs)
MORE_LOCAL_IS = {"peak_share": +1, "win3_share": +1, "win5_share": +1, "gini": +1,
                 "spread_half": -1}


def profile_metrics(a):
    """a: 1-D np.float64 ReLU activation profile over kept content tokens.
    Returns dict (Nones when the profile carries no positive mass)."""
    a = np.asarray(a, dtype=np.float64)
    S = float(a.sum())
    n = len(a)
    if n == 0 or S <= 0:
        return {m: None for m in METRICS}
    peak = float(a.max())
    csum = np.concatenate([[0.0], np.cumsum(a)])

    def win_share(k):
        if n <= k:
            return 1.0
        return float((csum[k:] - csum[:-k]).max() / S)

    srt = np.sort(a)
    gini = float((2.0 * np.sum(np.arange(1, n + 1) * srt) / (n * S)) - (n + 1) / n)
    return {"peak_share": peak / S, "win3_share": win_share(3), "win5_share": win_share(5),
            "gini": gini, "spread_half": int((a >= 0.5 * peak).sum())}


# ===============================================================================================
# stage 1: build (GPU) — per-token clean-base feature profiles for rollouts + real examples
# ===============================================================================================

def _gen_rollouts(a, tok, base, sae, cfg, feats, dev):
    """ON-POLICY rollouts for `feats` under the given PEFT adapter — the exact SFT/RL/autointerp
    inject recipe (unit(W_enc[:,f]) norm-matched at INJECT_LAYER on the trailing ' ?' marker),
    gen params (temp/max_new/min_new/n_max) taken from the testbed config so only the adapter
    differs. Returns (greedy [F][1], sampled [F][n_max]); the adapter is unloaded afterwards so
    `base` is a clean base again for the profile read path."""
    import torch
    from peft import PeftModel

    from mxf.config import INJECT_LAYER, STEER_COEFF
    from mxf.inject import get_layer, hooked, make_inject_hook
    from mxf.prompts import build_prompt_ids

    actor = PeftModel.from_pretrained(base, a.adapter, is_trainable=False)
    actor.eval()
    sub = get_layer(actor, INJECT_LAYER)
    prompt_ids, mpos = build_prompt_ids(tok)
    marker = mpos[0]
    dirs = sae.enc_dirs(feats).float().cpu()

    @torch.no_grad()
    def gen(reps, do_sample):
        rows_all = [i for i in range(len(feats)) for _ in range(reps)]
        out = [[] for _ in feats]
        for s in range(0, len(rows_all), a.gen_chunk):
            rows = rows_all[s:s + a.gen_chunk]
            hook = make_inject_hook([dirs[i:i + 1] for i in rows], [[marker]] * len(rows),
                                    STEER_COEFF, dev, torch.bfloat16, mode="add")
            ids = torch.tensor([list(prompt_ids)] * len(rows), device=dev)
            with hooked(sub, hook):
                g = actor.generate(ids, do_sample=do_sample, max_new_tokens=cfg["max_new"],
                                   min_new_tokens=cfg["min_new"], pad_token_id=tok.pad_token_id,
                                   **(dict(temperature=cfg["temp"], top_p=1.0, top_k=0,
                                           min_p=0.0) if do_sample else {}))
            for i, t in zip(rows, tok.batch_decode(g[:, len(prompt_ids):],
                                                   skip_special_tokens=True)):
                out[i].append(t.strip() or " ")
            print(f"  [gen {'temp' if do_sample else 'greedy'}] "
                  f"{min(s + a.gen_chunk, len(rows_all))}/{len(rows_all)}", flush=True)
        return out

    with torch.random.fork_rng(devices=[dev] if str(dev).startswith("cuda") else []):
        torch.manual_seed(GEN_SEED)
        greedy = gen(1, False)
        sampled = gen(cfg["n_max"], True)
    actor.unload()                       # strip LoRA modules in place -> clean base restored
    return greedy, sampled


def cmd_build(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mxf.config import MODEL, READ_LAYER
    from mxf.inject import read_resid
    from mxf.sae import load_sae

    dev = a.device
    tb = json.load(open(a.testbed))
    cfg = tb["config"]
    fire = cfg["fire"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    base.eval()
    sae = load_sae(path=a.sae_path or cfg["sae_path"], device=dev, dtype=torch.float32)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    print(f"[locality] base+sae ready ({time.time() - t0:.0f}s)", flush=True)

    if a.adapter:                        # regenerate the rollouts ON-POLICY for this adapter
        t1 = time.time()
        feats_all = [r["feature"] for r in tb["features"]]
        greedy, sampled = _gen_rollouts(a, tok, base, sae, cfg, feats_all, dev)
        for fi, r in enumerate(tb["features"]):
            r["rollout_greedy"] = greedy[fi][0]
            r["rollouts_temp"] = sampled[fi]
        print(f"[locality] on-policy rollouts regenerated for {len(feats_all)} features "
              f"with adapter {a.adapter} ({time.time() - t1:.0f}s)", flush=True)

    @torch.no_grad()
    def profiles(texts, feat):
        """Exact MAEMMBench shared read path: BOS sink prepended + skipped, right padding
        masked, 10x-median norm filter, ReLU encode of `feat` on the clean base.
        Returns list of (profile list[float], tokens list[str])."""
        prev = tok.padding_side
        tok.padding_side = "right"
        try:
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=95, add_special_tokens=False).to(dev)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), bos, device=dev,
                                        dtype=enc["input_ids"].dtype), enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=dev,
                                       dtype=enc["attention_mask"].dtype),
                            enc["attention_mask"]], 1)
            h, mask = read_resid(base, READ_LAYER,
                                 {"input_ids": ids, "attention_mask": am}, pool="all")
            keep = mask.clone()
            keep[:, 0] = False                                       # attention-sink guard
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            keep = keep & (nrm <= 10.0 * med)                        # norm filter (shared proto)
            per = sae.encode_features(h, [feat])[:, :, 0]            # [B, T] ReLU acts
            out = []
            for b in range(B):
                kb = keep[b].cpu().numpy()
                prof = per[b].float().cpu().numpy()[kb]
                toks = [tok.convert_ids_to_tokens(int(t))
                        for t, k in zip(ids[b].cpu().tolist(), kb.tolist()) if k]
                out.append((prof.tolist(), toks))
            return out
        finally:
            tok.padding_side = prev

    recs = []
    for fi, r in enumerate(tb["features"]):
        f = r["feature"]
        texts, kinds = [], []
        texts.append(r["rollout_greedy"]); kinds.append("greedy")
        for t in r["rollouts_temp"]:
            texts.append(t); kinds.append("temp")
        for e in r["desc_examples"]:
            texts.append(e["text"]); kinds.append("real")
        rows = []
        for (prof, toks), kind, text in zip(profiles(texts, f), kinds, texts):
            pa = np.asarray(prof, dtype=np.float64)
            peak = float(pa.max()) if len(pa) else 0.0
            rows.append({"kind": kind, "text": text, "n_tokens": len(prof),
                         "peak": peak, "fired": bool(peak > fire),
                         "metrics": profile_metrics(pa),
                         "profile": [round(float(x), 4) for x in prof], "tokens": toks})
        recs.append({"feature": f, "texts": rows})
        if fi % 16 == 0:
            print(f"[locality] {fi + 1}/{len(tb['features'])} features "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = {"config": {"testbed": os.path.abspath(a.testbed), "model": MODEL,
                      "read_layer": READ_LAYER, "sae_path": a.sae_path or cfg["sae_path"],
                      "adapter": a.adapter or cfg.get("adapter"),
                      "rollout_source": ("on-policy regen" if a.adapter else "testbed"),
                      "gen_seed": GEN_SEED if a.adapter else cfg.get("gen_seed"),
                      "fire": fire, "n_features": len(recs),
                      "read_path": "bos-sink-skip + right-pad-mask + 10x-median norm filter, "
                                   "clean base, ReLU SAE encode",
                      "metrics": METRICS},
           "features": recs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    fired = {k: [] for k in ("greedy", "temp", "real")}
    for r in recs:
        for t in r["texts"]:
            fired[t["kind"]].append(t["fired"])
    print("[locality] fired frac: " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in fired.items()),
          flush=True)
    print(f"LOCALITY_BUILD_DONE {a.out}", flush=True)


# ===============================================================================================
# stage 2: score (local) — aggregates, paired maemm-vs-real test, autointerp cross-links
# ===============================================================================================

def _mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def crop_to_best_window(a, k):
    """Highest-total-mass contiguous k-token sub-profile (length control: judge a long rollout
    the way maxacts windows the corpus — as its best k-token window)."""
    a = np.asarray(a, dtype=np.float64)
    if len(a) <= k:
        return a
    csum = np.concatenate([[0.0], np.cumsum(a)])
    s = int(np.argmax(csum[k:] - csum[:-k]))
    return a[s:s + k]


def cmd_score(a):
    L = json.load(open(a.locality))
    fire = L["config"]["fire"]

    def arm_of(kind):
        return "real" if kind == "real" else "maemm"

    # ---- pooled distributions (firing texts only) + firing fractions + token counts
    pooled = {arm: {m: [] for m in METRICS} for arm in ("maemm", "real")}
    pooled_peak = {arm: [] for arm in ("maemm", "real")}
    fired_n = {arm: [0, 0] for arm in ("maemm", "real")}         # [fired, total]
    ntok = {arm: [] for arm in ("maemm", "real")}
    per_feat = []
    for r in L["features"]:
        fmeans = {"feature": r["feature"], "maemm": {}, "real": {}, "maemm_crop": {},
                  "fired_frac": {}, "n_fired": {}}
        by_arm = {"maemm": [], "real": []}
        by_crop = []
        for t in r["texts"]:
            arm = arm_of(t["kind"])
            fired_n[arm][1] += 1
            ntok[arm].append(t["n_tokens"])
            if t["fired"]:
                fired_n[arm][0] += 1
                by_arm[arm].append(t["metrics"])
                pooled_peak[arm].append(t["peak"])
                for m in METRICS:
                    pooled[arm][m].append(t["metrics"][m])
                if arm == "maemm":
                    # LENGTH CONTROL: rollouts are ~2x longer than the 32-token max-act
                    # windows, which mechanically deflates share metrics (uniform-null win_k
                    # share = k/T). Re-judge each rollout as its best contiguous
                    # crop-length window, like maxacts windows the corpus.
                    by_crop.append(profile_metrics(
                        crop_to_best_window(t["profile"], a.crop_len)))
        for arm in ("maemm", "real"):
            fmeans["n_fired"][arm] = len(by_arm[arm])
            fmeans["fired_frac"][arm] = (len(by_arm[arm])
                                         / sum(1 for t in r["texts"] if arm_of(t["kind"]) == arm))
            fmeans[arm] = ({m: float(np.mean([x[m] for x in by_arm[arm]])) for m in METRICS}
                           if by_arm[arm] else None)
        fmeans["maemm_crop"] = ({m: float(np.mean([x[m] for x in by_crop])) for m in METRICS}
                                if by_crop else None)
        per_feat.append(fmeans)

    # ---- paired tests over features with >=1 firing text in BOTH arms
    both = [pf for pf in per_feat if pf["maemm"] and pf["real"]]

    def paired_block(key):
        out = {}
        for m in METRICS:
            d = np.array([pf[key][m] - pf["real"][m] for pf in both])
            mu, sem = _mean_sem(d)
            out[m] = {"diff_mean": mu, "diff_sem": sem,
                      "ci95": [mu - 1.96 * sem, mu + 1.96 * sem],
                      "n": len(d), "more_local_is": MORE_LOCAL_IS[m],
                      "maemm_mean": float(np.mean([pf[key][m] for pf in both])),
                      "real_mean": float(np.mean([pf["real"][m] for pf in both]))}
        return out

    paired = paired_block("maemm")
    paired_crop = paired_block("maemm_crop")

    # ---- cross-links: per-feature rollout locality vs autointerp AUC + rollout fire-rate
    cross = {}
    if a.autointerp_results and os.path.exists(a.autointerp_results):
        R = json.load(open(a.autointerp_results))
        auc = {f["feature"]: f["auc"].get(a.auc_key) for f in R["per_feature"]}
        tb = json.load(open(a.testbed)) if a.testbed and os.path.exists(a.testbed) else None
        frate = ({f["feature"]: float(np.mean([x > fire for x in
                                               f["rollout_self_acts"]["temp"]]))
                  for f in tb["features"]} if tb else {})
        rows = [(pf["feature"], pf["maemm"], auc.get(pf["feature"]),
                 frate.get(pf["feature"])) for pf in per_feat if pf["maemm"]]
        for m in METRICS:
            x = np.array([r[1][m] for r in rows if r[2] is not None])
            y = np.array([r[2] for r in rows if r[2] is not None])
            cross[f"r({m}, auc_{a.auc_key})"] = (float(np.corrcoef(x, y)[0, 1])
                                                 if len(x) > 2 else None)
            if frate:
                xf = np.array([r[1][m] for r in rows if r[3] is not None])
                yf = np.array([r[3] for r in rows if r[3] is not None])
                cross[f"r({m}, fire_rate)"] = (float(np.corrcoef(xf, yf)[0, 1])
                                               if len(xf) > 2 else None)
        cross["_rows"] = [{"feature": r[0], **{m: r[1][m] for m in METRICS},
                           "auc": r[2], "fire_rate": r[3]} for r in rows]

    out = {"config": L["config"],
           "aggregate": {
               "fired_frac": {arm: fired_n[arm][0] / fired_n[arm][1]
                              for arm in ("maemm", "real")},
               "n_texts": {arm: fired_n[arm][1] for arm in ("maemm", "real")},
               "mean_n_tokens": {arm: float(np.mean(ntok[arm])) for arm in ("maemm", "real")},
               "mean_peak_act_fired": {arm: float(np.mean(pooled_peak[arm]))
                                       for arm in ("maemm", "real")},
               "pooled": {arm: {m: {"mean": _mean_sem(pooled[arm][m])[0],
                                    "sem": _mean_sem(pooled[arm][m])[1],
                                    "median": float(np.median(pooled[arm][m])),
                                    "n": len(pooled[arm][m])}
                                for m in METRICS} for arm in ("maemm", "real")},
               "paired": paired, "paired_crop": paired_crop, "crop_len": a.crop_len},
           "pooled_values": {arm: pooled[arm] for arm in ("maemm", "real")},
           "per_feature": per_feat, "crosslinks": cross}
    json.dump(out, open(a.out, "w"), indent=1)
    print("=== SNIPPET LOCALITY (firing texts; maemm vs real max-act examples) ===", flush=True)
    print(f"  fired frac: maemm {out['aggregate']['fired_frac']['maemm']:.3f} "
          f"real {out['aggregate']['fired_frac']['real']:.3f} | mean tokens "
          f"maemm {out['aggregate']['mean_n_tokens']['maemm']:.1f} "
          f"real {out['aggregate']['mean_n_tokens']['real']:.1f}", flush=True)
    for m in METRICS:
        p, pc = paired[m], paired_crop[m]
        arrow = "MORE local" if p["diff_mean"] * MORE_LOCAL_IS[m] > 0 else "LESS local"
        print(f"  {m:>12}  maemm {p['maemm_mean']:.4f}  real {p['real_mean']:.4f}  "
              f"paired diff {p['diff_mean']:+.4f} ±{p['diff_sem']:.4f} (n={p['n']}) "
              f"-> rollouts {arrow} | crop{a.crop_len}: maemm {pc['maemm_mean']:.4f} "
              f"diff {pc['diff_mean']:+.4f} ±{pc['diff_sem']:.4f}", flush=True)
    for k, v in cross.items():
        if not k.startswith("_") and v is not None:
            print(f"  {k} = {v:.3f}", flush=True)
    print(f"LOCALITY_SCORE_DONE {a.out}", flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build", help="GPU: per-token clean-base feature profiles + metrics")
    b.add_argument("--testbed", default="/data/eval_autointerp/testbed_v2.json")
    b.add_argument("--sae-path", default=None, help="default: the testbed config's sae_path")
    b.add_argument("--adapter", default=None,
                   help="PEFT adapter dir/repo: REGENERATE the testbed's rollouts on-policy for "
                        "this adapter before profiling (locality is an on-policy metric — the "
                        "testbed's stored rollouts belong to the adapter that built it). Real "
                        "examples + feature set + gen seed are reused verbatim, so runs with "
                        "different adapters are directly comparable. Default: score the "
                        "testbed's stored rollouts as-is")
    b.add_argument("--gen-chunk", type=int, default=128)
    b.add_argument("--out", default="/data/eval_autointerp/locality.json")
    b.add_argument("--device", default="cuda:0")
    b.set_defaults(fn=cmd_build)

    s = sp.add_parser("score", help="local: aggregates + paired test + autointerp cross-links")
    s.add_argument("--locality", required=True)
    s.add_argument("--autointerp-results", default=None,
                   help="autointerp results.json for the AUC cross-link")
    s.add_argument("--auc-key", default="maemm_N8")
    s.add_argument("--testbed", default=None,
                   help="testbed json (rollout_self_acts) for the fire-rate cross-link")
    s.add_argument("--crop-len", type=int, default=32,
                   help="length control: re-judge each rollout as its best contiguous window "
                        "of this many tokens (= the max-act corpus window length)")
    s.add_argument("--out", default="locality_results.json")
    s.set_defaults(fn=cmd_score)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
