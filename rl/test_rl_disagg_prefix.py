"""B200 parity + timing harness for rl_disagg's trainer speed knobs (--prefix-cache, --score-length-bucket): TRAINER ONLY, no
vLLM engine, one GPU, in the SAME image modal_rl_disagg.py builds (with the transformers fork), so the fork/vLLM
coexistence (`import vllm`, `import vllm_lens`) is checked in the same container.

    source ~/modal_venv/bin/activate; export MODAL_PROFILE=safety-sahan
    modal run rl/test_rl_disagg_prefix.py::parity --n-groups 8 --group-size 8 --n-time 1024

Phases (all on the real Qwen3.6-27B + the RL init adapter, real HF-generated rollouts with the injection hook):
  parity : ONE batch of rollouts -> update_disagg() three times without stepping the optimizer: full-sequence path (reference),
           full-sequence path again (bf16 run-to-run noise floor), --prefix-cache path. Reports |d loss|, max|d grad|, rel-L2
           and cosine of the flat LoRA gradient, grad norms, and the per-step stats side by side. Also score() plain vs
           --score-length-bucket (reorder) on the same texts: max|d reward|.
  timing : the rollouts tiled to --n-time sequences (realistic length distribution) -> micro-batch probe + update + score
           timings for both paths on one GPU (update_s, ref_s, fwd_bwd_s, score_s plain/bucketed, peak GB, micro-batch).
Results: printed + written locally to rl/results/rl_prefix_parity.json.
"""
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("DISAGG_APP", "maemm-rl-prefix-test")
import modal_rl_disagg as M  # noqa: E402  (image with DISAGG_TRANSFORMERS, volume, env)

os.environ.setdefault("DISAGG_TRANSFORMERS", M.PREFIX_CACHE_TRANSFORMERS)
if "git+" not in os.environ["DISAGG_TRANSFORMERS"]:
    raise SystemExit("this harness needs the prefix-cache transformers fork: unset DISAGG_TRANSFORMERS or point it at the fork")
if M._TRANSFORMERS != os.environ["DISAGG_TRANSFORMERS"]:   # module was imported before the default was set -> rebuild its image
    import importlib
    M = importlib.reload(M)

app = modal.App("maemm-rl-prefix-test")
GPU = os.environ.get("PFX_GPU", "B200:1")
# this module is re-imported inside the container (Modal ships only the entrypoint file): mount its sibling import next to it
image = M.image.add_local_file(HERE / "modal_rl_disagg.py", "/root/modal_rl_disagg.py")

ARGS = ["--role", "trainer", "--n-rollout", "1", "--n-trainer", "1", "--data-dir", M.POOL_DIR, "--bank-file", "vecs.f32",
        "--init-adapter", M.SFT_INIT, "--lr", "1e-5", "--reward-metric", "cosine", "--reward-scale", "1", "--len-penalty-start", "8",
        "--len-penalty-per-tok", "0.00025", "--no-gates", "--kl-coef", "0.01", "--adv-mode", "group", "--reward-window-last", "5",
        "--reward-topk", "1", "--min-new-tokens", "8", "--max-new-tokens", "96", "--score-batch", "128", "--ref-micro-batch", "32",
        "--loss", "cispo", "--cispo-eps-max", "5", "--loss-agg", "prompt", "--zero-var-filter", "--fp32-head", "--autocast-bf16",
        "--rollout-chunk", "64", "--logp-chunk", "16", "--no-wandb"]


def _plain(x):
    out = json.loads(json.dumps(x, default=str))
    print("RESULTS_JSON " + json.dumps(out), flush=True)
    return out


class _GradRecorder:
    """Stands in for AdamW inside update_disagg: zero_grad + a step() that SNAPSHOTS the flat LoRA gradient (params untouched).
    Pair with --max-grad-norm 1e9 so clip_grad_norm_ multiplies by exactly 1.0 before step()."""

    def __init__(self, params):
        self.params, self.grads = params, None
        self.param_groups = [{"lr": 0.0}]

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def step(self):
        import torch
        self.grads = torch.cat([p.grad.detach().flatten().float() for p in self.params if p.grad is not None])


@app.function(image=image, gpu=GPU, volumes={"/data": M.vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=3 * 3600)
def parity_remote(n_groups: int = 8, group_size: int = 8, seed: int = 0, micro_batch: int = 16, n_time: int = 1024,
                  extra_args: str = ""):
    import time
    for k, v in M._env().items():
        os.environ[k] = v
    for p in ("/pmx/helpers", "/pmx/eval", "/pmx/RL"):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["DISAGG_RANK"] = "0"; os.environ["DISAGG_WORLD"] = "1"
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    import peft
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, cache_utils
    import rl_disagg as D
    import rl_hf as R
    from mxf.config import INJECT_LAYER, MODEL
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids

    # ---- environment / coexistence report ----
    env = {"gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__), "transformers": str(transformers.__version__),
           "transformers_path": str(os.path.dirname(transformers.__file__)), "peft": str(peft.__version__),
           "transformers_fork": bool(hasattr(cache_utils, "_write_cached_state") and hasattr(cache_utils.LinearAttentionLayer, "batch_repeat_interleave"))}
    try:
        import fla, triton
        env["fla"], env["triton"] = str(fla.__version__), str(triton.__version__)
    except Exception as e:  # noqa
        env["fla"] = f"ABSENT ({e})"
    try:
        t0 = time.time()
        import vllm, vllm_lens  # noqa
        env["vllm"], env["vllm_import_s"] = str(vllm.__version__), round(time.time() - t0, 1)
        import importlib.metadata as md
        env["vllm_requires_transformers"] = [r for r in md.requires("vllm") if r.startswith("transformers")]
    except Exception as e:  # noqa
        env["vllm"] = f"IMPORT FAILED: {type(e).__name__}: {e}"
    print("[env]", json.dumps(env), flush=True)
    assert env["transformers_fork"], "image does not carry the prefix-cache fork"

    a = D.parse_args(ARGS + ["--groups-per-step", str(n_groups), "--group-size", str(group_size), "--micro-batch", str(micro_batch),
                             "--max-grad-norm", "1e9"] + (extra_args.split() if extra_args else []))
    device = "cuda:0"
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    t0 = time.time()
    actor = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    actor = PeftModel.from_pretrained(actor, a.init_adapter, is_trainable=True)
    actor.train()
    if a.fp32_head:
        D.install_fp32_head(actor)
    submodule = get_layer(actor, INJECT_LAYER)
    if a.kl_coef > 0:
        actor.load_adapter(a.ref_adapter or a.init_adapter, adapter_name="ref"); actor.set_adapter("default")
    params = [p for p in actor.parameters() if p.requires_grad]
    opt = _GradRecorder(params)
    print(f"[load] actor+adapters in {time.time() - t0:.0f}s | trainable {sum(p.numel() for p in params) / 1e6:.0f}M | "
          f"prompt {p_len} marker {marker} | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
    res = {"env": env, "args": {"n_groups": n_groups, "group_size": group_size, "seed": seed, "micro_batch": micro_batch,
                               "n_time": n_time, "extra_args": extra_args, "prompt_len": p_len, "marker": marker}}

    # ---- rollouts: HF generate with the injection hook (real lengths, on-policy logprobs) ----
    bank, n_vecs, eval_rows = D._bank_open(a)
    rng = np.random.default_rng(seed)
    idx = D._sample_block_idx(rng, eval_rows, n_vecs, n_groups)
    dirs = F.normalize(torch.from_numpy(np.asarray(bank[idx], dtype=np.float32)), dim=-1)
    t0 = time.time()
    with torch.no_grad():
        texts, gen_ids, old_lps = R.rollout(actor, submodule, tok, prompt_ids, marker, dirs, a, device)
    lens = [len(g) for g in gen_ids]
    print(f"[rollout] {len(gen_ids)} seqs in {time.time() - t0:.0f}s | len mean {np.mean(lens):.1f} min {min(lens)} max {max(lens)} | "
          f"sample: {texts[0][:90]!r}", flush=True)
    res["rollout"] = {"n": len(gen_ids), "len_mean": float(np.mean(lens)), "len_min": min(lens), "len_max": max(lens)}
    G = group_size
    adv_mode = a.adv_mode or "none"

    def prep(texts, gen_ids, old_lps, dirs, Bl):
        """run_trainer's reward + shaping + padding, verbatim; returns everything update_disagg needs (+ score timings)."""
        dirs_rep = dirs.repeat_interleave(G, 0).to(device)
        ll = [len(g) for g in gen_ids]
        torch.cuda.synchronize(); t = time.time()
        r = R.score(texts, dirs_rep, actor, tok, device, a)
        torch.cuda.synchronize(); t_plain = time.time() - t
        t = time.time()
        rb = D.score_bucketed(R, texts, dirs_rep, actor, tok, device, a, ll)
        torch.cuda.synchronize(); t_bucket = time.time() - t
        # padding the scoring pass runs in arrival order vs length-sorted (score() tokenizes the decoded text, truncation 95)
        enc = [min(len(x), 95) + 1 for x in tok(texts, add_special_tokens=False)["input_ids"]]

        def pad_frac(order):
            tot = real = 0
            for s0 in range(0, len(order), a.score_batch):
                b = [enc[i] for i in order[s0 : s0 + a.score_batch]]
                tot += len(b) * max(b); real += sum(b)
            return 1.0 - real / max(tot, 1)
        sc = {"score_plain_s": t_plain, "score_bucketed_s": t_bucket, "reward_max_abs_diff": float((r - rb).abs().max()),
              "reward_mean_abs_diff": float((r - rb).abs().mean()), "reward_bitwise_equal": bool(torch.equal(r, rb)),
              "reward_mean": float(r.mean()), "score_pad_frac_plain": pad_frac(list(range(len(texts)))),
              "score_pad_frac_bucketed": pad_frac(sorted(range(len(texts)), key=lambda i: ll[i]))}
        r = r * a.reward_scale
        if a.len_penalty_start is not None:
            over = torch.tensor([max(0, len(g) - a.len_penalty_start) for g in gen_ids], dtype=torch.float32)
            r = r - a.len_penalty_per_tok * over
        adv, keep = D.compute_advantages_disagg(r, Bl, G, adv_mode, a.zero_var_eps, a.zero_var_filter)
        L = p_len + max(len(g) for g in gen_ids)
        ids = torch.full((Bl * G, L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((Bl * G, L), dtype=torch.long)
        old_lp = torch.zeros((Bl * G, L - p_len))
        known = torch.zeros((Bl * G, L - p_len), dtype=torch.bool)
        pt = torch.tensor(prompt_ids, dtype=torch.long)
        for i, (g, lp) in enumerate(zip(gen_ids, old_lps)):
            ids[i, :p_len] = pt
            ids[i, p_len : p_len + len(g)] = torch.tensor(g)
            attn[i, : p_len + len(g)] = 1
            for j, v in enumerate(lp.tolist()):
                old_lp[i, j] = float(v); known[i, j] = True
        return dict(ids=ids, attn=attn, old_lp=old_lp, known=known, adv=adv, keep=keep, dirs_rep=dirs_rep), sc

    def run_update(batch, mb, pfx, label):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        st = D.update_disagg(actor, opt, submodule, batch["ids"], batch["attn"], p_len, marker, batch["old_lp"], batch["known"],
                             batch["adv"], batch["dirs_rep"], a, device, mb, keep=batch["keep"], pfx=pfx)
        torch.cuda.synchronize(); t = time.time() - t
        peak = torch.cuda.max_memory_allocated() / 2**30
        g = opt.grads.clone(); opt.zero_grad()
        row = {"loss": st["loss"], "grad_norm": st["grad_norm"], "entropy": st["entropy"], "kl": st["kl"], "ratio_mean": st["ratio_mean"],
               "clipfrac": st["clipfrac"], "sampler_abs_dlogp": st["sampler_abs_dlogp"], "is_weight_mean": st["is_weight_mean"],
               "sync_w": st["sync_w"], "update_s": t, "ref_s": st["t_ref"], "fwd_bwd_s": st["t_fb"], "peak_gb": peak, "micro_batch": mb,
               "n_rollouts": int(batch["ids"].shape[0]), "pad_frac": st["pad_frac"], "body_tok_per_rollout": st["body_tok_per_rollout"],
               "real_tok_per_rollout": st["real_tok_per_rollout"]}
        print(f"[{label}] loss {st['loss']:.6f} gnorm {st['grad_norm']:.5f} ent {st['entropy']:.4f} kl {st['kl']:.6f} ratio {st['ratio_mean']:.5f} "
              f"| {t:.1f}s (ref {st['t_ref']:.1f} fb {st['t_fb']:.1f}) mb {mb} peak {peak:.0f} GB | pad {st['pad_frac']:.1%} "
              f"body tok/rollout {st['body_tok_per_rollout']:.1f} (real {st['real_tok_per_rollout']:.1f})", flush=True)
        return row, g

    def cmp(g_ref, g):
        return {"max_abs": float((g_ref - g).abs().max()), "rel_l2": float((g_ref - g).norm() / g_ref.norm()),
                "cosine": float(F.cosine_similarity(g_ref[None], g[None])), "norm_ref": float(g_ref.norm()), "norm_other": float(g.norm())}

    # ---- parity on ONE batch ----
    from mxf.config import STEER_COEFF
    batch, sc = prep(texts, gen_ids, old_lps, dirs, n_groups)
    res["score_parity"] = sc
    print(f"[score] plain {sc['score_plain_s']:.2f}s bucketed {sc['score_bucketed_s']:.2f}s | max|d r| {sc['reward_max_abs_diff']:.2e} "
          f"bitwise {sc['reward_bitwise_equal']} | mean r {sc['reward_mean']:.4f} | pad frac plain {sc['score_pad_frac_plain']:.1%} "
          f"bucketed {sc['score_pad_frac_bucketed']:.1%}", flush=True)
    gen_mask = batch["attn"][:, p_len:].bool()
    w1, sw1 = D.loss_weights(gen_mask, a.loss_agg, G, batch["keep"])
    res["loss_weights"] = {"sync_w": sw1, "sum": float(w1.sum()), "n_rows_weight0": int((w1.sum(1) == 0).sum()),
                           "note": "computed once from gen_mask before chunking and indexed per micro-batch (w_all[ix]); micro-batch composition cannot change them"}
    pfx = D.PrefixRunner(actor, prompt_ids, marker, device)

    # (a) per-token completion logprobs, no grad: the full-sequence path at two micro-batch sizes (= two paddings/shapes ->
    # the bf16 noise floor between two batchings of the SAME computation) vs the prefix-cached path
    def token_logps(mb, use_pfx):
        n, L = batch["ids"].shape
        lens_ = gen_mask.sum(1); order = torch.argsort(lens_)
        out = torch.zeros(n, L - p_len)
        with torch.no_grad():
            cache = None
            if use_pfx:
                with D._policy_precision(actor, a.autocast_bf16):
                    cache = pfx.run_prefix()
            for s0 in range(0, n, mb):
                ix = order[s0 : s0 + mb]
                Lc = p_len + min(L - p_len, -(-int(lens_[ix].max()) // 16) * 16); Tc = Lc - p_len
                hook = D._hook_outside_autocast(R.make_inject_hook([batch["dirs_rep"][i : i + 1] for i in ix.tolist()], [[0 if use_pfx else marker]] * len(ix),
                                                                   STEER_COEFF, device, torch.bfloat16), a.autocast_bf16)
                with R.hooked(submodule, hook), D._policy_precision(actor, a.autocast_bf16):
                    if use_pfx:
                        lg = pfx.suffix_logits(cache, batch["ids"][ix, marker:Lc].to(device), batch["attn"][ix, marker:Lc].to(device))[:, :-1]
                    else:
                        lg = actor(input_ids=batch["ids"][ix, :Lc].to(device), attention_mask=batch["attn"][ix, :Lc].to(device), use_cache=False,
                                   logits_to_keep=Tc + 1).logits[:, :-1]
                lp, _ = D._chunked_logp(lg, batch["ids"][ix, p_len:Lc].to(device), a.vocab_chunk, False)
                out[ix, :Tc] = lp.float().cpu()
                del lg, lp
            del cache
        return out

    def lp_cmp(ref, other):
        d = (ref - other).abs()[gen_mask]
        seq = ((ref - other) * gen_mask).sum(1).abs()
        return {"token_mean_abs": float(d.mean()), "token_max_abs": float(d.max()), "token_p99_abs": float(d.quantile(0.99)),
                "seq_logp_mean_abs": float(seq.mean()), "seq_logp_max_abs": float(seq.max()), "n_tokens": int(gen_mask.sum()),
                "ref_mean_logp": float(ref[gen_mask].mean())}
    lp_full = token_logps(micro_batch, False)
    lp_full8 = token_logps(max(1, micro_batch // 2), False)
    lp_pfx = token_logps(micro_batch, True)
    res["token_logp"] = {"prefix_vs_full": lp_cmp(lp_full, lp_pfx), "full_mb_half_vs_full (noise floor)": lp_cmp(lp_full, lp_full8),
                         "old_lp_vs_full (rollout-time HF logp)": lp_cmp(lp_full, batch["old_lp"])}
    for k, v in res["token_logp"].items():
        print(f"[logp] {k}: " + json.dumps({kk: round(vv, 6) for kk, vv in v.items()}), flush=True)
    del lp_full, lp_full8, lp_pfx

    # (b) the real update: loss + flat LoRA gradient
    rows, grads = {}, {}

    def try_run(label, mb, use_pfx):
        try:
            rows[label], grads[label] = run_update(batch, mb, pfx if use_pfx else None, label)
        except torch.OutOfMemoryError as e:  # noqa
            rows[label] = {"error": "OOM: " + str(e)[:160], "micro_batch": mb}
            opt.zero_grad(); import gc; gc.collect(); torch.cuda.empty_cache()
            print(f"[{label}] OOM at mb {mb}", flush=True)
    try_run("full", micro_batch, False)
    try_run("full_again", micro_batch, False)
    try_run("full_mb_half", max(1, micro_batch // 2), False)
    try_run("prefix", micro_batch, True)
    try_run("prefix_mb_2x", 2 * micro_batch, True)
    par = {"runs": rows, "loss_abs_diff": {}, "grad": {}}
    for k in ("prefix", "prefix_mb_2x", "full_again", "full_mb_half"):
        if k in grads and "full" in grads:
            tagk = k + (" (noise floor)" if k.startswith("full") else "") + "_vs_full"
            par["loss_abs_diff"][tagk] = abs(rows[k]["loss"] - rows["full"]["loss"])
            par["grad"][tagk] = cmp(grads["full"], grads[k])
            print(f"[parity] {tagk}: loss |d| {par['loss_abs_diff'][tagk]:.3e} | grad " + json.dumps({kk: round(vv, 6) for kk, vv in par["grad"][tagk].items()}), flush=True)
    res["parity"] = par
    grads.clear(); import gc; gc.collect(); torch.cuda.empty_cache()

    # ---- timing on --n-time rollouts (the parity batch tiled: realistic length distribution) ----
    if n_time > 0:
        rep = max(1, n_time // len(gen_ids))
        batch_t, sc_t = prep(texts * rep, gen_ids * rep, old_lps * rep, dirs.repeat(rep, 1), n_groups * rep)
        tim = {"n_rollouts": int(batch_t["ids"].shape[0]), "score": sc_t}
        print(f"[timing] {tim['n_rollouts']} rollouts | score plain {sc_t['score_plain_s']:.2f}s bucketed {sc_t['score_bucketed_s']:.2f}s "
              f"(max|d r| {sc_t['reward_max_abs_diff']:.2e})", flush=True)
        for label, use_pfx in (("full", False), ("prefix", True)):
            a.prefix_cache = use_pfx
            mb, mbres = D.find_micro_batch(actor, opt, submodule, prompt_ids, marker, a, device, D._mb_candidates(a), f"probe-{label}",
                                           pfx=pfx if use_pfx else None)
            opt.zero_grad()
            row_w, _ = run_update(batch_t, mb, pfx if use_pfx else None, f"time-{label}-warm")   # Triton autotune warm-up
            row, _ = run_update(batch_t, mb, pfx if use_pfx else None, f"time-{label}")
            row["mb_probe"] = {str(k): v for k, v in mbres.items()}
            row["warm_update_s"] = row_w["update_s"]
            tim[label] = row
        tim["speedup_update"] = tim["full"]["update_s"] / tim["prefix"]["update_s"]
        tim["speedup_fwd_bwd"] = tim["full"]["fwd_bwd_s"] / tim["prefix"]["fwd_bwd_s"]
        tim["speedup_ref"] = tim["full"]["ref_s"] / max(tim["prefix"]["ref_s"], 1e-9)
        tim["speedup_score"] = sc_t["score_plain_s"] / sc_t["score_bucketed_s"]
        res["timing"] = tim
        print(f"[timing] update {tim['full']['update_s']:.1f}s -> {tim['prefix']['update_s']:.1f}s (x{tim['speedup_update']:.2f}) | "
              f"fwd/bwd x{tim['speedup_fwd_bwd']:.2f} ref x{tim['speedup_ref']:.2f} score x{tim['speedup_score']:.2f} | "
              f"mb {tim['full']['micro_batch']} -> {tim['prefix']['micro_batch']} | peak {tim['full']['peak_gb']:.0f} -> {tim['prefix']['peak_gb']:.0f} GB", flush=True)
    return _plain(res)


@app.local_entrypoint()
def parity(n_groups: int = 8, group_size: int = 8, seed: int = 0, micro_batch: int = 16, n_time: int = 1024, extra_args: str = "",
           out: str = ""):
    res = parity_remote.remote(n_groups=n_groups, group_size=group_size, seed=seed, micro_batch=micro_batch, n_time=n_time,
                               extra_args=extra_args)
    out = out or str(HERE / "results" / "rl_prefix_parity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")
