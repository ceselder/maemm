#!/usr/bin/env python3
"""Best-of-N harvest worker (ONE GPU). For every target direction of a shard: sample N rollouts from a FULL-model inverter
checkpoint (vLLM + the RL fast injection hook, rl_disagg._generate_block), score every rollout with the RL reward (rl.py score():
clean-base layer-42 max-cosine over the whole span, then the RL length penalty), also score the target's ORIGINAL text, keep the
top-k rollouts per target. Output = one JSONL line per target (rejection-sampling distillation data + best-of-N statistics).

Protocol = eval_ckpt_daemon --full-model: HF clean base FIRST (truncated to layers [0, READ_LAYER+1) to save memory), then the
engine serving the checkpoint dir, marker-norm from the engine, numeric injection proof, then generate/score chunks.

    python harvest_bon_worker.py --ckpt /data/ckpts_rl_simple2m_8x2048_anywin/step_250 --targets targets_00.jsonl --dirs dirs_00.npy \
        --out shard.jsonl --n-samples 32 --temperature 1.0 --top-k 4 --cuda-graphs [--probe]

--probe additionally stores every sample's cosine and length in generation order (prefix-max curves = best-of-n for n <= N).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace


def log(m):
    print(f"[harvest] {m}", flush=True)


def _vllm_marker_norm(llm, prompt_ids, marker, inject_layer):
    """||h|| at the marker of the SERVED model's INJECT_LAYER residual (vllm_lens capture, clean prompt, greedy 1 token).
    Copied from eval/eval_ckpt_daemon.py so the worker does not import the daemon module."""
    from vllm import SamplingParams
    out = llm.generate([{"prompt_token_ids": list(prompt_ids)}],
                       [SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [inject_layer]})],
                       use_tqdm=False)[0]
    act = getattr(out, "activations", None)
    assert act is not None and "residual_stream" in act, "vllm_lens capture returned nothing -- plugin not active?"
    return act["residual_stream"][0].float()[marker].norm().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="full-model checkpoint dir (SAVE_DONE) served by vLLM, or the base MODEL id")
    ap.add_argument("--targets", required=True, help="jsonl: one target per line {bank_id, vec_idx, family, target_text, ...}")
    ap.add_argument("--dirs", required=True, help=".npy [n_targets, 5120] (f16/f32) directions, row i == targets line i")
    ap.add_argument("--out", required=True, help="output jsonl (appended; --resume skips targets already present)")
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96, help="rl.py score() re-tokenizes at most 95 tokens anyway")
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--reward-window-last", type=int, default=0, help="selection window: 0 = whole span (the anywin recipe), 16 = last-16 recipe")
    ap.add_argument("--also-window", type=int, default=-1, help="also score every rollout under this window (stored as cos_w<k>; -1 = off; the probe used 16)")
    ap.add_argument("--len-penalty-per-tok", type=float, default=0.00025)
    ap.add_argument("--len-penalty-start", type=int, default=8)
    ap.add_argument("--dirs-per-call", type=int, default=0, help="directions per generate() call; 0 = queue_mult * max_num_seqs // n_samples")
    ap.add_argument("--queue-mult", type=int, default=1, help="sequences queued per generate() call = queue_mult x max_num_seqs (>1 backfills but forces eager mixed steps: slower)")
    ap.add_argument("--quant", default="", help="vLLM quantization for the SAMPLER weights (e.g. fp8 = online per-tensor fp8); scorer stays bf16")
    ap.add_argument("--max-num-seqs", type=int, default=512, help="Qwen3.6 GDN recurrent state is allocated per sequence: 2048 OOMs next to the resident scorer")
    ap.add_argument("--max-num-batched-tokens", type=int, default=24576)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.5)
    ap.add_argument("--cuda-graphs", action="store_true")
    ap.add_argument("--score-batch", type=int, default=256)
    ap.add_argument("--scorer-layers", type=int, default=43, help="keep decoder layers [0, n) of the clean base (0 = all 64)")
    ap.add_argument("--probe", action="store_true", help="store every sample's cosine + length (best-of-n curves)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="process only the first n targets (smoke test)")
    ap.add_argument("--manifest", default="", help="write a done manifest json here when finished")
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import rl_hf as R
    import rl_disagg as DG
    from mxf.config import INJECT_LAYER, READ_LAYER, MODEL
    from mxf.prompts import build_prompt_ids

    t_start = time.time()
    device = "cuda:0"
    torch.cuda.set_device(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)

    targets = [json.loads(l) for l in open(a.targets)]
    dirs = torch.from_numpy(np.load(a.dirs).astype(np.float32))
    assert dirs.shape == (len(targets), 5120), f"dirs {tuple(dirs.shape)} vs {len(targets)} targets"
    dirs = F.normalize(dirs, dim=1)
    done = set()
    if a.resume and os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(int(json.loads(l)["t"]))
            except Exception:  # noqa - a torn last line
                pass
    todo = [i for i in range(len(targets)) if i not in done]
    if a.limit:
        todo = todo[: a.limit]
    N = a.n_samples
    dpc = a.dirs_per_call or max(1, a.queue_mult * a.max_num_seqs // N)
    log(f"{len(targets)} targets ({len(done)} already done, {len(todo)} to do) | N={N} T={a.temperature} top-k={a.top_k} "
        f"dirs/call={dpc} ({dpc * N} seqs queued per call, {a.max_num_seqs} concurrent) max_new={a.max_new_tokens} window_last={a.reward_window_last} | ckpt {a.ckpt}")

    # ---- HF clean scorer FIRST (vllm's import clobbers transformers' AutoConfig for this model) ----
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    if a.scorer_layers > 0:
        n_before, head_dropped = DG._truncate_scorer(base, a.scorer_layers)
        log(f"scorer truncated to layers [0,{a.scorer_layers}) of {n_before} (lm_head dropped: {head_dropped})")
    scorer = DG.BaseActor(base)
    scorer.eval()
    eos_ids = R._eos_ids(tok, DG._GenCfgStub(GenerationConfig.from_pretrained(MODEL)))
    log(f"scorer ready in {time.time() - t_start:.0f}s | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB | prompt {p_len} toks, "
        f"marker @{marker} | eos {sorted(eos_ids)}")

    # ---- engine serving the checkpoint (fast steering hook + CUDA graphs), then the numeric injection proof ----
    ea = SimpleNamespace(stock_lens_hook=False, vllm_gpu_mem=a.vllm_gpu_mem, max_new_tokens=a.max_new_tokens, min_new_tokens=a.min_new_tokens,
                         seed=a.seed, gdn_prefill_backend="triton", engine_model=a.ckpt, engine_lora=False, policy_base=None,
                         max_num_batched_tokens=a.max_num_batched_tokens, group_size=N, temperature=a.temperature,
                         score_batch=a.score_batch, reward_metric="cosine", reward_window_last=a.reward_window_last,
                         reward_pos_penalty=0.0, reward_topk=1, log_reward=False)
    if a.quant:   # _build_engine does `from vllm import LLM` at call time -> inject the quantization kwarg through the module attribute
        import vllm as _vllm
        _LLM = _vllm.LLM
        _vllm.LLM = lambda **kw: _LLM(quantization=a.quant, **kw)
        log(f"sampler weights served with quantization={a.quant}")
    llm = DG._build_engine(ea, 0, p_len, a.max_num_seqs, a.cuda_graphs, "harvest")
    hnorm = _vllm_marker_norm(llm, prompt_ids, marker, INJECT_LAYER)
    chk = DG._verify_injection(llm, prompt_ids, marker, hnorm, "harvest", seed=a.seed)
    log(f"injection check: cos {chk['cos']:.4f} | magnitude ratio {chk['norm_ratio']:.3f} | marker ||h|| served {hnorm:.1f} -> "
        f"{'OK' if chk['ok'] else 'FAIL'}")
    assert chk["ok"], f"injection proof failed: {chk}"

    # ---- generate / score / select. Sequential on purpose: (1) queuing more than max_num_seqs per call makes vLLM backfill, and every
    #      mixed prefill+decode step runs EAGER in the hooked engine (the marker must be prefilled in a hooked pass) -> ~2x slower decode;
    #      (2) scoring in a thread next to LLM.generate() starves both on the GIL. So: exactly max_num_seqs sequences per call, then score.
    def _score(idx, d, texts, lens):
        t1 = time.time()
        with torch.inference_mode():
            dirs_rep = d.repeat_interleave(N, 0).to(device)
            cos = DG.score_bucketed(R, texts, dirs_rep, scorer, tok, device, ea, lens).float().cpu()
            orig_texts = [targets[i]["target_text"] for i in idx]
            ocos = R.score(orig_texts, d.to(device), scorer, tok, device, ea).float().cpu()
            cos2 = ocos2 = None
            if a.also_window >= 0 and a.also_window != a.reward_window_last:
                ea.reward_window_last = a.also_window
                cos2 = DG.score_bucketed(R, texts, dirs_rep, scorer, tok, device, ea, lens).float().cpu()
                ocos2 = R.score(orig_texts, d.to(device), scorer, tok, device, ea).float().cpu()
                ea.reward_window_last = a.reward_window_last
        return cos, ocos, cos2, ocos2, time.time() - t1

    fout = open(a.out, "a")
    agg = {"targets": 0, "rollouts": 0, "gen_s": 0.0, "score_s": 0.0, "cos_sum": 0.0, "best_sum": 0.0, "orig_sum": 0.0, "beat": 0, "len_sum": 0}
    t_loop = time.time()
    chunks = list(range(0, len(todo), dpc))
    for ci, c0 in enumerate(chunks):
        idx = todo[c0: c0 + dpc]
        d = dirs[idx]
        gen_ids, _lps, _app, gen_s = DG._generate_block(llm, ea, tok, prompt_ids, marker, d, hnorm, None, eos_ids, f"h{c0}")
        texts = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
        lens = [len(g) for g in gen_ids]
        cos, ocos, cos2, ocos2, score_s = _score(idx, d, texts, lens)
        pen = torch.tensor([max(0, L - a.len_penalty_start) for L in lens], dtype=torch.float32) * a.len_penalty_per_tok
        rew = cos - pen
        for j, i in enumerate(idx):
            sl = slice(j * N, (j + 1) * N)
            c, r = cos[sl], rew[sl]
            order = torch.argsort(r, descending=True).tolist()
            top = [{"text": texts[j * N + k], "cos": round(float(c[k]), 5), "reward": round(float(r[k]), 5), "n_tok": lens[j * N + k],
                    **({f"cos_w{a.also_window}": round(float(cos2[j * N + k]), 5)} if cos2 is not None else {})} for k in order[: a.top_k]]
            tg = targets[i]
            row = {"t": i, "bank_id": tg["bank_id"], "vec_idx": tg["vec_idx"], "family": tg["family"],
                   "orig_cos": round(float(ocos[j]), 5), "cos_mean": round(float(c.mean()), 5), "cos_std": round(float(c.std()), 5),
                   "cos_max": round(float(c.max()), 5), "len_mean": round(float(np.mean(lens[sl])), 2),
                   "best": top[0], "topk": top, "n": N, "temperature": a.temperature, "window_last": a.reward_window_last, "ckpt": a.ckpt}
            if cos2 is not None:
                row[f"orig_cos_w{a.also_window}"] = round(float(ocos2[j]), 5); row[f"cos_max_w{a.also_window}"] = round(float(cos2[sl].max()), 5)
                row[f"cos_mean_w{a.also_window}"] = round(float(cos2[sl].mean()), 5)
            if a.probe:
                row["cos_all"] = [round(float(x), 4) for x in c.tolist()]
                row["len_all"] = lens[sl]
                if cos2 is not None:
                    row[f"cos_all_w{a.also_window}"] = [round(float(x), 4) for x in cos2[sl].tolist()]
            fout.write(json.dumps(row) + "\n")
            agg["targets"] += 1; agg["best_sum"] += float(c.max()); agg["orig_sum"] += float(ocos[j]); agg["beat"] += int(float(c.max()) > float(ocos[j]))
        fout.flush()
        agg["rollouts"] += len(texts); agg["gen_s"] += gen_s; agg["score_s"] += score_s
        agg["cos_sum"] += float(cos.sum()); agg["len_sum"] += sum(lens)
        if ci % 5 == 0 or ci + 1 == len(chunks):
            el = time.time() - t_loop
            log(f"{agg['targets']}/{len(todo)} targets | {agg['rollouts']} rollouts in {el:.0f}s ({agg['rollouts'] / max(el, 1e-6):.0f}/s; "
                f"gen {agg['gen_s']:.0f}s score {agg['score_s']:.0f}s) | cos mean {agg['cos_sum'] / max(agg['rollouts'], 1):.3f} "
                f"best-of-{N} {agg['best_sum'] / max(agg['targets'], 1):.3f} orig-text {agg['orig_sum'] / max(agg['targets'], 1):.3f} "
                f"beat-orig {agg['beat'] / max(agg['targets'], 1):.2f} | len {agg['len_sum'] / max(agg['rollouts'], 1):.0f}")
    fout.close()
    man = {"ckpt": a.ckpt, "n_targets_total": len(targets), "n_done": len(done) + agg["targets"], "n_samples": N, "temperature": a.temperature,
           "top_k": a.top_k, "max_new_tokens": a.max_new_tokens, "reward_window_last": a.reward_window_last, "also_window": a.also_window,
           "len_penalty": [a.len_penalty_start, a.len_penalty_per_tok], "hnorm_served": hnorm, "injection_check": chk,
           "rollouts": agg["rollouts"], "wall_s": round(time.time() - t_start), "loop_s": round(time.time() - t_loop),
           "mean_cos": agg["cos_sum"] / max(agg["rollouts"], 1), "mean_best": agg["best_sum"] / max(agg["targets"], 1),
           "mean_orig": agg["orig_sum"] / max(agg["targets"], 1), "beat_orig_frac": agg["beat"] / max(agg["targets"], 1),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.manifest:
        json.dump(man, open(a.manifest, "w"), indent=1)
    log(f"DONE {json.dumps({k: v for k, v in man.items() if k not in ('injection_check',)})}")
    print("HARVEST_DONE", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # skip interpreter teardown (vLLM/NCCL finalizer crash after the work is done)


if __name__ == "__main__":
    main()
