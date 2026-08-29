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
# stage 1b: augment (GPU) — mine HARD negatives into an existing testbed (judge-comparability-
# preserving: features, positives, rollouts and the random-negative pool are reused VERBATIM)
# ===============================================================================================

def cmd_augment(a):
    """Two hard-negative pools per feature, mined from a FURTHER-disjoint corpus slice:
      negatives_nearmiss  highest clean-base peak act still safely below fire
                          (act in [--nearmiss-lo, --nearmiss-hi-frac * fire)), top-n_neg by act
      negatives_embnn     nearest to the feature's description examples in a small text-embedding
                          space (BGE CLS+L2, MINING ONLY — the metric stays the Opus judge) among
                          non-firing candidates (act < neg_max_act)
    The original random pool is kept as negatives_random (and legacy key `negatives`)."""
    import torch
    import torch.nn.functional as TF
    from datasets import load_dataset
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from mxf.config import MODEL, READ_LAYER
    from mxf.inject import read_resid
    from mxf.sae import load_sae

    dev = a.device
    rng = np.random.default_rng(a.seed + 7)   # fresh stream — original testbed is NOT rebuilt
    tb = json.load(open(a.testbed))
    cfg = tb["config"]
    feats = [r["feature"] for r in tb["features"]]
    L, n_neg, fire, nma = cfg["ctx_len"], cfg["n_neg"], cfg["fire"], cfg["neg_max_act"]
    lo_band, hi_band = a.nearmiss_lo, a.nearmiss_hi_frac * fire
    # doc-range disjointness from the original random pool (which consumed ~neg_pool docs
    # starting at corpus_skip_docs) => span-level disjointness
    assert a.mine_skip_docs >= cfg["corpus_skip_docs"] + cfg["neg_pool"] + 500, \
        "mine slice overlaps the original random-negatives doc range"

    def norm_key(t):
        return " ".join(t.lower().split())

    reserved = set()   # candidate spans may not collide with any judged text
    for r in tb["features"]:
        for k in ("desc_examples", "positives", "negatives"):
            for e in r[k]:
                reserved.add(norm_key(e["text"]))

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset(cfg["corpus"], split=cfg["corpus_split"],
                      streaming=True).skip(a.mine_skip_docs)
    pool_ids, pool_texts, seen_pool = [], [], set()
    for doc in ds:
        text = doc.get("content") or doc.get("text")
        if not text:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < L:
            continue
        s = int(rng.integers(0, len(ids) - L + 1))
        w = ids[s:s + L]
        t = tok.decode(w, skip_special_tokens=True).strip()
        k = norm_key(t)
        if not t or k in reserved or k in seen_pool:
            continue
        seen_pool.add(k)
        pool_ids.append(w)
        pool_texts.append(t)
        if len(pool_ids) >= a.mine_pool:
            break
    print(f"[augment] candidate pool {len(pool_ids)} windows "
          f"(docs skipped {a.mine_skip_docs}; reserved-text collisions filtered)", flush=True)

    # ---- clean-base peak acts of every eval feature on every candidate (same recipe as build)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    base.eval()
    sae = load_sae(path=a.sae_path or cfg["sae_path"], device=dev, dtype=torch.float32)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    acts = np.zeros((len(pool_ids), len(feats)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(pool_ids), 64):
            win = torch.tensor(pool_ids[s:s + 64], device=dev)
            inp = torch.cat([torch.full((win.shape[0], 1), bos, device=dev, dtype=win.dtype),
                             win], 1)
            h, _ = read_resid(base, READ_LAYER,
                              {"input_ids": inp, "attention_mask": torch.ones_like(inp)},
                              pool="all")
            acts[s:s + 64] = sae.encode_features(h[:, 1:], feats).amax(1).float().cpu().numpy()
    del base
    torch.cuda.empty_cache()
    print(f"[augment] pool acts done ({time.time() - t0:.0f}s) | "
          f"frac(act<{nma})={float((acts < nma).mean()):.4f} "
          f"frac(in nearmiss band)={float(((acts >= lo_band) & (acts < hi_band)).mean()):.4f}",
          flush=True)

    # ---- small text embedder, mining only (BGE convention: CLS pooling + L2 normalize)
    etok = AutoTokenizer.from_pretrained(a.embed_model)
    emb_model = AutoModel.from_pretrained(a.embed_model).to(dev).eval()

    @torch.no_grad()
    def embed(texts, bs=256):
        out = []
        for s in range(0, len(texts), bs):
            enc = etok(texts[s:s + bs], padding=True, truncation=True, max_length=128,
                       return_tensors="pt").to(dev)
            h = emb_model(**enc).last_hidden_state[:, 0]
            out.append(TF.normalize(h, dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float64)

    cand_emb = embed(pool_texts)

    shortfalls = []
    for fi, r in enumerate(tb["features"]):
        av = acts[:, fi]
        # near-miss: activates the feature, safely below firing
        band = np.where((av >= lo_band) & (av < hi_band))[0]
        nm = band[np.argsort(-av[band], kind="stable")][:n_neg]
        if len(nm) < n_neg:
            shortfalls.append((r["feature"], "nearmiss", int(len(nm))))
        r["negatives_nearmiss"] = [{"text": pool_texts[i], "act": float(av[i])} for i in nm]
        # embedding-NN: topically closest non-firing candidates to the desc-example centroid
        de = embed([e["text"] for e in r["desc_examples"]])
        c = de.mean(0)
        c /= np.linalg.norm(c)
        cos = cand_emb @ c
        ok = np.where(av < nma)[0]
        en = ok[np.argsort(-cos[ok], kind="stable")][:n_neg]
        if len(en) < n_neg:
            shortfalls.append((r["feature"], "embnn", int(len(en))))
        r["negatives_embnn"] = [{"text": pool_texts[i], "act": float(av[i]),
                                 "desc_cos": float(cos[i])} for i in en]
        r["negatives_random"] = r["negatives"]        # explicit alias, kept verbatim

    # ---- sanity guards
    for r in tb["features"]:
        assert all(e["act"] < fire for e in r["negatives_nearmiss"]), "near-miss span fires!"
        assert all(e["act"] < nma for e in r["negatives_embnn"]), "embnn span not near-zero!"
        judged = {norm_key(e["text"]) for k in ("positives", "negatives") for e in r[k]}
        for k in ("negatives_nearmiss", "negatives_embnn"):
            assert not judged & {norm_key(e["text"]) for e in r[k]}, f"{k} overlaps test set"
    nm_sizes = [len(r["negatives_nearmiss"]) for r in tb["features"]]
    en_sizes = [len(r["negatives_embnn"]) for r in tb["features"]]
    nm_act = float(np.mean([e["act"] for r in tb["features"] for e in r["negatives_nearmiss"]]))
    en_cos = float(np.mean([e["desc_cos"] for r in tb["features"] for e in r["negatives_embnn"]]))
    print(f"[augment] pool sizes: nearmiss min/med {min(nm_sizes)}/{int(np.median(nm_sizes))} "
          f"(mean act {nm_act:.3f}) | embnn min/med {min(en_sizes)}/{int(np.median(en_sizes))} "
          f"(mean desc-cos {en_cos:.3f}) | shortfalls: {shortfalls or 'none'}", flush=True)

    tb["config"].update({"mine_pool": len(pool_ids), "mine_skip_docs": a.mine_skip_docs,
                         "nearmiss_band": [lo_band, hi_band], "mine_embed_model": a.embed_model,
                         "augment_seed": a.seed + 7, "hardneg_shortfalls": shortfalls})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(tb, open(a.out, "w"), indent=1)
    print(f"AUGMENT_DONE {a.out}", flush=True)


# ===============================================================================================
# stage 2: judge — submit one Anthropic Message Batch (opus-5), save state
# ===============================================================================================

def _variants(rec, n_list):
    """Yield (arm, N, desc_snippets). N=1 MAEMM = greedy; N>1 = first N temp samples (nested).
    Baseline 'examples' arm = top-N max-act corpus examples (nested, disjoint from positives)."""
    for N in n_list:
        yield "maemm", N, ([rec["rollout_greedy"]] if N == 1 else rec["rollouts_temp"][:N])
        yield "examples", N, [e["text"] for e in rec["desc_examples"][:N]]


def _variants_marginal(rec, n_list, n_base):
    """MARGINAL mode: does APPENDING N rollouts to a fixed base of top-n_base real max-act
    examples add detection value? base{n_base} (N=0) = the reference; base{n_base}roll_N =
    base + N rollouts (N=1 greedy, N>1 first-N temp — same reuse as the standard arms)."""
    base = [e["text"] for e in rec["desc_examples"][:n_base]]
    yield f"base{n_base}", 0, base
    for N in n_list:
        rolls = [rec["rollout_greedy"]] if N == 1 else rec["rollouts_temp"][:N]
        yield f"base{n_base}roll", N, base + rolls


def cmd_judge(a):
    import anthropic

    import itertools

    tb = json.load(open(a.testbed))
    n_list = sorted(int(x) for x in a.n_list.split(","))
    assert max(n_list) <= tb["config"]["n_max"] and max(n_list) <= tb["config"]["n_desc"]
    if a.variant == "marginal":
        assert a.n_base <= tb["config"]["n_desc"]
        make = lambda rec: _variants_marginal(rec, n_list, a.n_base)
    elif a.variant == "hardneg":
        # ALL variant keys from both prior modes x ONLY the two new hard-neg pools; positive +
        # random-negative scores are reused from the prior batches (--prior), whose prompts for
        # those tests are byte-identical.
        assert "negatives_nearmiss" in tb["features"][0], "testbed not augmented — run augment"
        assert a.prior, "--prior required for hardneg (comma list of prior results jsons)"
        make = lambda rec: itertools.chain(_variants(rec, n_list),
                                           _variants_marginal(rec, n_list, a.n_base))
    else:
        make = lambda rec: _variants(rec, n_list)

    def tests_for(rec):
        np_, nn = len(rec["positives"]), len(rec["negatives"])
        if a.variant == "hardneg":   # ti offsets: nearmiss after random, embnn after nearmiss
            return ([(np_ + nn + i, x["text"], 0)
                     for i, x in enumerate(rec["negatives_nearmiss"])]
                    + [(np_ + 2 * nn + i, x["text"], 0)
                       for i, x in enumerate(rec["negatives_embnn"])])
        return ([(i, p["text"], 1) for i, p in enumerate(rec["positives"])]
                + [(np_ + i, n["text"], 0) for i, n in enumerate(rec["negatives"])])

    reqs, labels, variant_keys = [], {}, []
    for rec in tb["features"]:
        tests = tests_for(rec)
        for arm, N, desc in make(rec):
            if f"{arm}_N{N}" not in variant_keys:
                variant_keys.append(f"{arm}_N{N}")
            block = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(desc))
            for ti, text, label in tests:
                cid = f"f{rec['feature']}-{arm}-N{N}-t{ti}"
                labels[cid] = label
                # delimit the test snippet + end on the question: with thinking disabled the
                # model otherwise sometimes CONTINUES the raw snippet instead of rating it
                prompt = (f"DESCRIPTION SNIPPETS:\n{block}\n\nTEST SNIPPET:\n<<<\n{text}\n>>>"
                          "\n\nProbability (integer 0-100) that the feature activates on the "
                          "test snippet:")
                reqs.append({
                    "custom_id": cid,
                    "params": {
                        # opus-5 API notes: `temperature` is removed (400 if sent) and thinking
                        # is ON by default (it would silently eat a small max_tokens) -> disable.
                        "model": a.judge_model, "max_tokens": 8,
                        "thinking": {"type": "disabled"},
                        "system": [{"type": "text", "text": JUDGE_SYSTEM,
                                    "cache_control": {"type": "ephemeral"}}],
                        "messages": [{"role": "user", "content": prompt}],
                    }})
    print(f"[judge] {len(reqs)} requests ({len(tb['features'])} features x "
          f"{len(variant_keys)} variants {variant_keys} x "
          f"{len(tb['features'][0]['positives']) + len(tb['features'][0]['negatives'])} tests)",
          flush=True)

    client = anthropic.Anthropic(api_key=a.api_key or os.environ["ANTHROPIC_API_KEY_BATCH"])
    batch = client.messages.batches.create(requests=reqs)
    state = {"batch_id": batch.id, "judge_model": a.judge_model, "n_list": n_list,
             "variant": a.variant, "variants": variant_keys, "n_base": a.n_base,
             "prior_results": [os.path.abspath(p) for p in a.prior.split(",")] if a.prior else [],
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


def _score_hardneg(a, state, tb, scores, fails):
    """Merge the hard-negative batch with prior batches' positive/random scores (byte-identical
    prompts) -> AUC per (variant key, negative type). Writes a.out."""
    n_pos, n_neg = tb["config"]["n_pos"], tb["config"]["n_neg"]
    priors = {}
    for p in state["prior_results"]:
        Rp = json.load(open(p))
        for f in Rp["per_feature"]:
            priors.setdefault(f["feature"], {}).update(f["scores"])
    per_feat, missing = [], 0
    for rec in tb["features"]:
        f = rec["feature"]
        row = {"feature": f, "auc": {}, "scores": {}}
        for vk in state["variants"]:
            arm, Ns = vk.rsplit("_N", 1)
            prior = priors[f][vk]
            pos, rand = prior[:n_pos], prior[n_pos:n_pos + n_neg]

            def grab(base_off, pool):
                nonlocal missing
                out = []
                for i in range(len(rec[pool])):
                    x = scores.get(f"f{f}-{arm}-N{Ns}-t{base_off + i}")
                    missing += x is None
                    out.append(50 if x is None else x)
                return out

            nm = grab(n_pos + n_neg, "negatives_nearmiss")
            en = grab(n_pos + 2 * n_neg, "negatives_embnn")
            row["auc"][vk] = {"random": _auc(pos, rand),
                              "nearmiss": _auc(pos, nm) if nm else None,
                              "embnn": _auc(pos, en) if en else None}
            row["scores"][vk] = {"nearmiss": nm, "embnn": en}
        per_feat.append(row)

    agg = {}
    for vk in state["variants"]:
        arm, Ns = vk.rsplit("_N", 1)
        for nt in ("random", "nearmiss", "embnn"):
            v = np.array([r["auc"][vk][nt] for r in per_feat
                          if r["auc"][vk][nt] is not None])
            agg[f"auc/{arm}/N{Ns}/{nt}"] = {"mean": float(v.mean()),
                                            "sem": float(v.std(ddof=1) / np.sqrt(len(v))),
                                            "n": int(len(v))}
    out = {"state": {k: state[k] for k in ("batch_id", "judge_model", "n_list", "variants",
                                           "variant", "testbed", "prior_results", "n_base")},
           "config": tb["config"], "aggregate": agg, "per_feature": per_feat,
           "n_parse_failures": len(fails), "n_missing_scores": missing,
           "failures": fails[:50]}
    json.dump(out, open(a.out, "w"), indent=1)
    print("=== HARD-NEGATIVE AUC (judge = " + state["judge_model"] + ") ===", flush=True)
    for vk in state["variants"]:
        arm, Ns = vk.rsplit("_N", 1)
        r = {nt: agg[f"auc/{arm}/N{Ns}/{nt}"]["mean"] for nt in ("random", "nearmiss", "embnn")}
        print(f"  {vk:>16}  random {r['random']:.4f}  nearmiss {r['nearmiss']:.4f}  "
              f"embnn {r['embnn']:.4f}", flush=True)
    print(f"SCORE_DONE {a.out}", flush=True)


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
    if state.get("variant") == "hardneg":
        return _score_hardneg(a, state, tb, scores, fails)
    n_list = state["n_list"]
    # variant keys "{arm}_N{N}": stored by newer judge runs; legacy states get the standard set
    variants = state.get("variants") or [f"{arm}_N{N}" for arm in ("maemm", "examples")
                                         for N in n_list]
    n_pos = tb["config"]["n_pos"]
    n_tests = n_pos + tb["config"]["n_neg"]
    per_feat, missing = [], 0
    for rec in tb["features"]:
        f = rec["feature"]
        row = {"feature": f, "auc": {}, "bal_acc": {}, "scores": {}}
        for vk in variants:
            arm, Ns = vk.rsplit("_N", 1)
            s = [scores.get(f"f{f}-{arm}-N{Ns}-t{ti}") for ti in range(n_tests)]
            missing += sum(x is None for x in s)
            s = [50 if x is None else x for x in s]              # failures -> uninformative
            pos, neg = s[:n_pos], s[n_pos:]
            row["auc"][vk] = _auc(pos, neg)
            row["bal_acc"][vk] = 0.5 * (np.mean(np.array(pos) >= 50)
                                        + np.mean(np.array(neg) < 50))
            row["scores"][vk] = s
        per_feat.append(row)

    agg = {}
    for vk in variants:
        arm, Ns = vk.rsplit("_N", 1)
        for metr in ("auc", "bal_acc"):
            v = np.array([r[metr][vk] for r in per_feat])
            agg[f"{metr}/{arm}/N{Ns}"] = {"mean": float(v.mean()),
                                          "sem": float(v.std(ddof=1) / np.sqrt(len(v)))}
    out = {"state": {k: state[k] for k in ("batch_id", "judge_model", "n_list", "testbed")
                     if k in state} | {"variants": variants,
                                       "variant": state.get("variant", "standard")},
           "config": tb["config"], "aggregate": agg, "per_feature": per_feat,
           "n_parse_failures": len(fails), "n_missing_scores": missing,
           "failures": fails[:50]}
    json.dump(out, open(a.out, "w"), indent=1)
    print("=== AUTOINTERP DETECTION (judge = " + state["judge_model"] + ") ===", flush=True)
    for vk in variants:
        arm, Ns = vk.rsplit("_N", 1)
        m = agg[f"auc/{arm}/N{Ns}"]
        print(f"  {vk:>16}  AUC {m['mean']:.4f} ±{m['sem']:.4f}", flush=True)
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

    g = sp.add_parser("augment", help="GPU stage: mine hard negatives into an existing testbed")
    g.add_argument("--testbed", required=True, help="existing testbed.json (reused verbatim)")
    g.add_argument("--out", default="/data/eval_autointerp/testbed_v2.json")
    g.add_argument("--sae-path", default=None, help="default: the testbed config's sae_path")
    g.add_argument("--mine-pool", type=int, default=8192, help="candidate windows to mine from")
    g.add_argument("--mine-skip-docs", type=int, default=102_000,
                   help="corpus docs to skip — must clear the original random pool's doc range")
    g.add_argument("--nearmiss-lo", type=float, default=0.2)
    g.add_argument("--nearmiss-hi-frac", type=float, default=0.9,
                   help="near-miss band upper bound as a fraction of the fire threshold")
    g.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5",
                   help="mining-only text embedder (CLS+L2); the metric stays the Opus judge")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cuda:0")
    g.set_defaults(fn=cmd_augment)

    j = sp.add_parser("judge", help="submit the Opus judge batch (Anthropic Batches API)")
    j.add_argument("--testbed", required=True)
    j.add_argument("--state", default="batch_state.json")
    j.add_argument("--judge-model", default="claude-opus-5")
    j.add_argument("--n-list", default="1,2,4,8")
    j.add_argument("--prior", default=None,
                   help="hardneg: comma list of prior results jsons whose positive/random "
                        "scores are reused (their prompts are byte-identical)")
    j.add_argument("--variant", choices=("standard", "marginal", "hardneg"), default="standard",
                   help="standard: maemm-alone vs examples-alone at each N. marginal: "
                        "top-n_base examples alone (N=0 reference) vs the same base + N "
                        "appended rollouts — the marginal value of rollouts on top of real "
                        "examples. Judge-only; reuses the cached testbed")
    j.add_argument("--n-base", type=int, default=4,
                   help="marginal mode: real max-act examples in the fixed base description")
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
