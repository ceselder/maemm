"""Autointerp DETECTION eval for the MAEMM inverter: are N inverter rollouts a good
*explanation* of a held-out SAE feature, measured as detection AUC by an LLM judge?

Standard autointerp detection methodology (Bills et al. / EleutherAI "detection" scoring),
with the feature DESCRIPTION built from MAEMM inverter rollouts instead of a natural-language
summary:

  1. For each held-out SAE feature f (never trained on), generate rollouts with the EXACT
     SFT/RL inject recipe (unit(W_enc[:,f]) norm-matched at INJECT_LAYER on the trailing ' ?'
     marker): 1 GREEDY rollout + --n-max temperature samples.
  2. Per-feature test set: --n-pos POSITIVES = real corpus windows where f fires strongly
     (maxacts ranks AFTER the top --n-desc, which are reserved for the baseline description;
     deduped, peak act > --fire) + --n-neg NEGATIVES = random corpus windows from a DISJOINT
     slice of the same corpus, VERIFIED near-zero for f (clean-base L42 -> SAE encode,
     max act < --neg-max-act). Ground truth = fires / doesn't, from the SAE itself.
  3. Judge (default claude-opus-5, Anthropic Message Batches API): per (feature, description
     variant, test snippet), the judge sees the description snippets + the test snippet and
     rates 0-100 how likely the feature fires on it. NO embedding model anywhere.
  4. Metric: detection AUC (Mann-Whitney, ties = 0.5) of the judge scores vs ground truth,
     per feature, averaged. Balanced accuracy at score >= 50 as a secondary.
  5. DIVERSITY curve: description = N in --n-list rollouts (N=1 greedy, N>1 the first N
     temperature samples, nested). More/diverse rollouts -> higher AUC = diversity is real
     autointerp value.
  6. BASELINE arm: identical judging with the feature's top-N max-act corpus EXAMPLES as the
     description (matched N, nested top-N; disjoint from the test positives by construction).

Three stages (build = GPU; judge/score = any box with the Anthropic key):

  # on Modal / a GPU box with the base model + SAE + maxacts (see modal_autointerp_detection.py)
  python autointerp_detection.py build --out /data/eval_autointerp/testbed.json

  # local: submit the judge batch (reads ANTHROPIC_API_KEY_BATCH or --api-key)
  python autointerp_detection.py judge --testbed testbed.json --state batch_state.json

  # local: poll + score
  python autointerp_detection.py score --state batch_state.json --out results.json
"""
import argparse
import json
import os
import re
import time

import numpy as np

GEN_SEED = 20260829            # forked-RNG sampling seed for rollouts (deterministic reruns)
JUDGE_SYSTEM = (
    "You are scoring an automated-interpretability explanation of a feature (a 'neuron') "
    "inside a language model. You will be given:\n"
    "1. DESCRIPTION SNIPPETS - short text snippets intended to illustrate what makes the "
    "feature activate strongly. They may be noisy or redundant; infer the underlying "
    "concept/pattern they share.\n"
    "2. TEST SNIPPET - a snippet of web text.\n\n"
    "Judge how likely it is that the feature activates strongly somewhere in the TEST "
    "snippet, i.e. whether the test snippet contains the concept/pattern that the "
    "description snippets illustrate.\n"
    "Respond with ONLY an integer from 0 to 100 (the probability in percent that the "
    "feature activates on the test snippet). No other text."
)


# ===============================================================================================
# stage 1: build (GPU) — rollouts + positives + verified negatives -> testbed.json
# ===============================================================================================

def cmd_build(a):
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mxf.config import MODEL, INJECT_LAYER, READ_LAYER, STEER_COEFF
    from mxf.inject import get_layer, make_inject_hook, hooked, read_resid
    from mxf.prompts import build_prompt_ids
    from mxf.sae import load_sae, load_max_acts

    dev = a.device
    rng = np.random.default_rng(a.seed)
    need = a.n_desc + a.n_pos          # deduped live max-act windows required per feature

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- feature selection: held-out sae feats with >= need deduped live (act > fire) windows
    es = torch.load(a.heldout_cache, map_location="cpu", weights_only=False)
    cand = np.asarray(es["sae_feats"], dtype=np.int64)
    rng.shuffle(cand)
    ma = load_max_acts(path=a.maxacts_path)
    mtok, mact = ma["max_tokens"], ma["max_acts"]                    # [F, topM, L]
    L = mtok.shape[2]
    feats, windows = [], {}
    for f in cand.tolist():
        peaks = mact[f].amax(-1)                                     # [topM] (already sorted desc)
        order = torch.argsort(peaks, descending=True).tolist()
        rows, seen = [], set()
        for i in order:
            p = float(peaks[i])
            if p <= a.fire:
                break
            text = tok.decode(mtok[f, i].tolist(), skip_special_tokens=True).strip()
            key = " ".join(text.lower().split())
            if not text or key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "act": p})
            if len(rows) >= need:
                break
        if len(rows) >= need:
            feats.append(int(f))
            windows[int(f)] = rows
        if len(feats) >= a.n_features:
            break
    if len(feats) < a.n_features:
        raise ValueError(f"only {len(feats)} held-out features have >= {need} deduped live "
                         f"max-act windows (wanted {a.n_features})")
    print(f"[build] {len(feats)} features selected (need {need} live windows each)", flush=True)

    # ---- negatives pool: random L-token windows from a DISJOINT corpus slice
    ds = load_dataset(a.corpus, split=a.corpus_split, streaming=True).skip(a.corpus_skip_docs)
    pool = []
    for doc in ds:
        text = doc.get("content") or doc.get("text")
        if not text:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < L:
            continue
        s = int(rng.integers(0, len(ids) - L + 1))
        w = ids[s:s + L]
        if tok.decode(w, skip_special_tokens=True).strip():
            pool.append(w)
        if len(pool) >= a.neg_pool:
            break
    print(f"[build] negatives pool: {len(pool)} random {L}-token windows "
          f"(docs skipped: {a.corpus_skip_docs})", flush=True)

    # ---- ground-truth acts of every eval feature on the pool (clean base, BOS-prepend, L42)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    base.eval()
    sae = load_sae(path=a.sae_path, device=dev, dtype=torch.float32)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    acts = np.zeros((len(pool), len(feats)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(pool), 64):
            win = torch.tensor(pool[s:s + 64], device=dev)
            inp = torch.cat([torch.full((win.shape[0], 1), bos, device=dev, dtype=win.dtype),
                             win], 1)
            h, _ = read_resid(base, READ_LAYER,
                              {"input_ids": inp, "attention_mask": torch.ones_like(inp)},
                              pool="all")
            acts[s:s + 64] = sae.encode_features(h[:, 1:], feats).amax(1).float().cpu().numpy()
    frac0 = float((acts < a.neg_max_act).mean())
    print(f"[build] base+sae ready, pool acts done ({time.time() - t0:.0f}s) | "
          f"frac(act<{a.neg_max_act})={frac0:.4f}", flush=True)

    # ---- rollouts with the adapter (exact train inject recipe)
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
                g = actor.generate(ids, do_sample=do_sample, max_new_tokens=a.max_new,
                                   min_new_tokens=a.min_new, pad_token_id=tok.pad_token_id,
                                   **(dict(temperature=a.temp, top_p=1.0, top_k=0, min_p=0.0)
                                      if do_sample else {}))
            for i, t in zip(rows, tok.batch_decode(g[:, len(prompt_ids):],
                                                   skip_special_tokens=True)):
                out[i].append(t.strip() or " ")
            print(f"  [gen {'temp' if do_sample else 'greedy'}] "
                  f"{min(s + a.gen_chunk, len(rows_all))}/{len(rows_all)}", flush=True)
        return out

    with torch.random.fork_rng(devices=[dev] if str(dev).startswith("cuda") else []):
        torch.manual_seed(GEN_SEED)
        greedy = gen(1, False)
        sampled = gen(a.n_max, True)

    # ---- diagnostic: does each rollout actually fire its own feature? (clean base re-encode)
    @torch.no_grad()
    def self_acts(texts, fi):
        prev = tok.padding_side
        tok.padding_side = "right"
        try:
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=95,
                      add_special_tokens=False).to(dev)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), bos, device=dev,
                                        dtype=enc["input_ids"].dtype), enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=dev,
                                       dtype=enc["attention_mask"].dtype),
                            enc["attention_mask"]], 1)
            with actor.disable_adapter():
                h, mask = read_resid(actor, READ_LAYER,
                                     {"input_ids": ids, "attention_mask": am}, pool="all")
            per = sae.encode_features(h, [feats[fi]])[:, :, 0]      # [B, T]
            keep = mask.clone()
            keep[:, 0] = False
            return per.masked_fill(~keep, 0.0).amax(1).float().cpu().tolist()
        finally:
            tok.padding_side = prev

    # ---- assemble testbed
    recs = []
    for fi, f in enumerate(feats):
        okn = [w for w in np.where(acts[:, fi] < a.neg_max_act)[0].tolist()]
        assert len(okn) >= a.n_neg, f"feature {f}: only {len(okn)} zero-act pool windows"
        sel = rng.choice(okn, a.n_neg, replace=False)
        negs = [{"text": tok.decode(pool[w], skip_special_tokens=True).strip(),
                 "act": float(acts[w, fi])} for w in sel]
        rolls = greedy[fi] + sampled[fi]
        sa = self_acts(rolls, fi)
        recs.append({
            "feature": f, "corpus_peak": windows[f][0]["act"],
            "desc_examples": windows[f][:a.n_desc],                 # baseline description
            "positives": windows[f][a.n_desc:a.n_desc + a.n_pos],   # test positives (disjoint)
            "negatives": negs,
            "rollout_greedy": greedy[fi][0], "rollouts_temp": sampled[fi],
            "rollout_self_acts": {"greedy": sa[0], "temp": sa[1:]},
        })
    out = {"config": {"adapter": a.adapter, "model": MODEL, "sae_path": a.sae_path,
                      "maxacts_path": a.maxacts_path, "heldout_cache": a.heldout_cache,
                      "n_features": len(feats), "n_desc": a.n_desc, "n_pos": a.n_pos,
                      "n_neg": a.n_neg, "n_max": a.n_max, "fire": a.fire,
                      "neg_max_act": a.neg_max_act, "neg_pool": len(pool),
                      "corpus": a.corpus, "corpus_split": a.corpus_split,
                      "corpus_skip_docs": a.corpus_skip_docs, "ctx_len": L,
                      "temp": a.temp, "max_new": a.max_new, "min_new": a.min_new,
                      "seed": a.seed, "gen_seed": GEN_SEED,
                      "inject_layer": INJECT_LAYER, "read_layer": READ_LAYER,
                      "steer_coeff": STEER_COEFF},
           "features": recs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    fired = np.mean([[x > a.fire for x in r["rollout_self_acts"]["temp"]] for r in recs])
    print(f"[build] DONE -> {a.out} | {len(recs)} features | "
          f"temp-rollout fired-frac {fired:.3f}", flush=True)
    print(f"TESTBED_DONE {a.out}", flush=True)


# ===============================================================================================
# stage 2: judge — submit one Anthropic Message Batch (opus-5), save state
# ===============================================================================================

def _variants(rec, n_list):
    """Yield (arm, N, desc_snippets). N=1 MAEMM = greedy; N>1 = first N temp samples (nested).
    Baseline 'examples' arm = top-N max-act corpus examples (nested, disjoint from positives)."""
    for N in n_list:
        yield "maemm", N, ([rec["rollout_greedy"]] if N == 1 else rec["rollouts_temp"][:N])
        yield "examples", N, [e["text"] for e in rec["desc_examples"][:N]]


def cmd_judge(a):
    import anthropic

    tb = json.load(open(a.testbed))
    n_list = sorted(int(x) for x in a.n_list.split(","))
    assert max(n_list) <= tb["config"]["n_max"] and max(n_list) <= tb["config"]["n_desc"]
    reqs, labels = [], {}
    for rec in tb["features"]:
        tests = ([(i, p["text"], 1) for i, p in enumerate(rec["positives"])]
                 + [(len(rec["positives"]) + i, n["text"], 0)
                    for i, n in enumerate(rec["negatives"])])
        for arm, N, desc in _variants(rec, n_list):
            block = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(desc))
            for ti, text, label in tests:
                cid = f"f{rec['feature']}-{arm}-N{N}-t{ti}"
                labels[cid] = label
                reqs.append({
                    "custom_id": cid,
                    "params": {
                        # opus-5 API notes: `temperature` is removed (400 if sent) and thinking
                        # is ON by default (it would silently eat a small max_tokens) -> disable.
                        "model": a.judge_model, "max_tokens": 8,
                        "thinking": {"type": "disabled"},
                        "system": [{"type": "text", "text": JUDGE_SYSTEM,
                                    "cache_control": {"type": "ephemeral"}}],
                        "messages": [{"role": "user", "content":
                                      f"DESCRIPTION SNIPPETS:\n{block}\n\nTEST SNIPPET:\n{text}"}],
                    }})
    print(f"[judge] {len(reqs)} requests ({len(tb['features'])} features x "
          f"{len(n_list)} N x 2 arms x {len(tb['features'][0]['positives']) + len(tb['features'][0]['negatives'])} tests)", flush=True)

    client = anthropic.Anthropic(api_key=a.api_key or os.environ["ANTHROPIC_API_KEY_BATCH"])
    batch = client.messages.batches.create(requests=reqs)
    state = {"batch_id": batch.id, "judge_model": a.judge_model, "n_list": n_list,
             "testbed": os.path.abspath(a.testbed), "labels": labels,
             "created_at": str(batch.created_at)}
    json.dump(state, open(a.state, "w"))
    print(f"[judge] submitted batch {batch.id} -> state {a.state}", flush=True)


# ===============================================================================================
# stage 3: score — poll the batch, parse 0-100 scores, detection AUC per (feature, arm, N)
# ===============================================================================================

def _auc(pos, neg):
    """Mann-Whitney AUC with ties counted 0.5."""
    p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def cmd_score(a):
    import anthropic

    state = json.load(open(a.state))
    client = anthropic.Anthropic(api_key=a.api_key or os.environ["ANTHROPIC_API_KEY_BATCH"])
    while True:
        b = client.messages.batches.retrieve(state["batch_id"])
        print(f"[score] batch {b.id}: {b.processing_status} | counts "
              f"{b.request_counts}", flush=True)
        if b.processing_status == "ended":
            break
        if a.no_wait:
            print("[score] --no-wait: batch not finished, exiting", flush=True)
            return
        time.sleep(a.poll_s)

    scores, fails = {}, []
    for r in client.messages.batches.results(state["batch_id"]):
        if r.result.type != "succeeded":
            fails.append((r.custom_id, r.result.type))
            continue
        txt = "".join(blk.text for blk in r.result.message.content
                      if getattr(blk, "type", "") == "text")
        m = re.search(r"\d+", txt)
        if not m:
            fails.append((r.custom_id, f"unparseable: {txt!r}"))
            continue
        scores[r.custom_id] = min(100, max(0, int(m.group())))
    print(f"[score] parsed {len(scores)} scores, {len(fails)} failures", flush=True)

    tb = json.load(open(state["testbed"] if os.path.exists(state["testbed"]) else a.testbed))
    n_list = state["n_list"]
    n_pos = tb["config"]["n_pos"]
    n_tests = n_pos + tb["config"]["n_neg"]
    per_feat, missing = [], 0
    for rec in tb["features"]:
        f = rec["feature"]
        row = {"feature": f, "auc": {}, "bal_acc": {}, "scores": {}}
        for arm in ("maemm", "examples"):
            for N in n_list:
                s = [scores.get(f"f{f}-{arm}-N{N}-t{ti}") for ti in range(n_tests)]
                missing += sum(x is None for x in s)
                s = [50 if x is None else x for x in s]              # failures -> uninformative
                pos, neg = s[:n_pos], s[n_pos:]
                row["auc"][f"{arm}_N{N}"] = _auc(pos, neg)
                row["bal_acc"][f"{arm}_N{N}"] = 0.5 * (np.mean(np.array(pos) >= 50)
                                                       + np.mean(np.array(neg) < 50))
                row["scores"][f"{arm}_N{N}"] = s
        per_feat.append(row)

    agg = {}
    for arm in ("maemm", "examples"):
        for metr in ("auc", "bal_acc"):
            for N in n_list:
                v = np.array([r[metr][f"{arm}_N{N}"] for r in per_feat])
                agg[f"{metr}/{arm}/N{N}"] = {"mean": float(v.mean()),
                                             "sem": float(v.std(ddof=1) / np.sqrt(len(v)))}
    out = {"state": {k: state[k] for k in ("batch_id", "judge_model", "n_list", "testbed")},
           "config": tb["config"], "aggregate": agg, "per_feature": per_feat,
           "n_parse_failures": len(fails), "n_missing_scores": missing,
           "failures": fails[:50]}
    json.dump(out, open(a.out, "w"), indent=1)
    print("=== AUTOINTERP DETECTION (judge = " + state["judge_model"] + ") ===", flush=True)
    for N in n_list:
        mm, ex = agg[f"auc/maemm/N{N}"], agg[f"auc/examples/N{N}"]
        print(f"  N={N:>2}  MAEMM-rollout AUC {mm['mean']:.4f} ±{mm['sem']:.4f}   "
              f"max-act-example AUC {ex['mean']:.4f} ±{ex['sem']:.4f}", flush=True)
    print(f"SCORE_DONE {a.out}", flush=True)


# ===============================================================================================

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build", help="GPU stage: rollouts + positives + verified negatives")
    b.add_argument("--adapter", default="ceselder/qwen36-27b-maemm-inverter")
    b.add_argument("--sae-path", default="/data/sae/ae.pt")
    b.add_argument("--maxacts-path", default="/data/sae/maxacts.pt")
    b.add_argument("--heldout-cache", default="/data/eval_universal_ho/eval_sets_heldout.pt")
    b.add_argument("--n-features", type=int, default=64)
    b.add_argument("--n-desc", type=int, default=8,
                   help="top max-act examples reserved for the baseline description")
    b.add_argument("--n-pos", type=int, default=10)
    b.add_argument("--n-neg", type=int, default=10)
    b.add_argument("--n-max", type=int, default=8, help="temperature rollouts per feature")
    b.add_argument("--fire", type=float, default=1.0, help="act threshold: window counts as firing")
    b.add_argument("--neg-max-act", type=float, default=0.01,
                   help="negatives must have clean-base max act below this")
    b.add_argument("--neg-pool", type=int, default=1024)
    b.add_argument("--corpus", default="openbmb/Ultra-FineWeb")
    b.add_argument("--corpus-split", default="en")
    b.add_argument("--corpus-skip-docs", type=int, default=100_000,
                   help="skip the maxacts scan slice so negatives are corpus-disjoint")
    b.add_argument("--temp", type=float, default=1.0)
    b.add_argument("--max-new", type=int, default=64)
    b.add_argument("--min-new", type=int, default=16)
    b.add_argument("--gen-chunk", type=int, default=128)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--out", default="/data/eval_autointerp/testbed.json")
    b.set_defaults(fn=cmd_build)

    j = sp.add_parser("judge", help="submit the Opus judge batch (Anthropic Batches API)")
    j.add_argument("--testbed", required=True)
    j.add_argument("--state", default="batch_state.json")
    j.add_argument("--judge-model", default="claude-opus-5")
    j.add_argument("--n-list", default="1,2,4,8")
    j.add_argument("--api-key", default=None, help="default: $ANTHROPIC_API_KEY_BATCH")
    j.set_defaults(fn=cmd_judge)

    s = sp.add_parser("score", help="poll the batch + compute detection AUC")
    s.add_argument("--state", default="batch_state.json")
    s.add_argument("--testbed", default=None, help="fallback if the state's testbed path moved")
    s.add_argument("--out", default="results.json")
    s.add_argument("--poll-s", type=int, default=60)
    s.add_argument("--no-wait", action="store_true")
    s.add_argument("--api-key", default=None)
    s.set_defaults(fn=cmd_score)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
