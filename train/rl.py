"""Stage 6: Dr. GRPO RL — vllm-lens rollouts, no KL, no /std, global-token normalizer.

Rollouts: ONE llm.generate() per step; per-request SteeringVector(norm_match=True) == our
norm-matched inject@INJECT_LAYER at the marker. old_logp comes from vLLM's generation logprobs
(valid behavior-policy logps at temperature 1.0 ONLY). new_logp is recomputed HF-side with the
same inject hook; TIS (ratio capped at cfg.tis_cap, upper only) absorbs the residual vLLM/HF
kernel mismatch; the LoRA-merged actor is pushed back into vLLM every --sync-every steps.

Reward: each generation re-tokenized STANDALONE through the CLEAN base model (adapter disabled,
no injection); reward = max over kept positions of x_t · unit(v) at READ_LAYER, position 0
skipped (attention-sink guard). No μ-centering: v is shared within a group, so μ·v is a constant
that cancels exactly in the Dr. GRPO advantage (r − group_mean).

Held-out eval: the first --n-eval-dirs UNIQUE bank directions are reserved (never sampled for
training); every --eval-every steps each gets ONE greedy (T=0) rollout, scored by the same clean-
base scorer → eval/greedy_act_{mean,max} (+ eval/greedy_beat_frac vs the corpus_max_proj baseline
when the pool's records.jsonl carries it).

    python scripts/rl.py --tp 8                                                    # full box (sbatch_rl.sh)
    python scripts/rl.py --groups-per-step 8 --group-size 4 --total-steps 3 --no-wandb   # 1-GPU smoke
"""
import argparse
import functools
import json
import math
import os
import time
import zlib
from collections import defaultdict

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # pickle for apply_model(partial)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from mxf.config import D_MODEL, INJECT_LAYER, MODEL, READ_LAYER, STEER_COEFF, RLConfig, TrainConfig
from mxf.inject import get_layer, hooked, make_inject_hook, read_resid
from mxf.prompts import build_prompt_ids


def _distinct_ngram_ratio(ids, n=3):
    """Fraction of distinct n-grams in a token sequence. 1.0 = no repetition; low = repetitive.
    Catches verbatim loops (e.g. 'Duration: 10 minutes' ×N) the distinct-TOKEN gate misses."""
    if len(ids) < n + 1:
        return 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def _compression_ratio(text):
    """zlib compressed/raw byte ratio of the decoded text. Catches TEMPLATED repetition (number/
    citation lists like 'Moore 2000; Moore 2002; …') that distinct-n-gram under-penalizes because
    the varying tokens keep n-grams nominally distinct. Natural text ~0.5-0.7; degenerate ~0.15-0.35."""
    b = text.encode("utf-8")
    return len(zlib.compress(b, 6)) / max(len(b), 1) if b else 1.0


def _load_chunk(model, chunk):
    """Module-level (picklable) target for llm.apply_model — runs on every TP worker."""
    model.load_weights(iter(chunk))


def sync_weights(actor=None, llm=None):
    """No-op stub. HF-generate rollouts use the LoRA actor DIRECTLY as the rollout engine, so there
    is no separate vLLM process to push merged weights into. Kept so main() call sites are unchanged."""
    return 0.0


@torch.no_grad()
def rollout(actor, submodule, tok, prompt_ids, marker, dirs, a, device):
    """B groups × G rollouts via HF generate() + the inject hook, chunked into rollout-chunk-sized
    mini-batches (no paged attn, so 1024 seqs won't fit one generate). dirs: [B, d]. Returns flat
    group-major lists (texts, gen_ids, old_logps) — rollout i belongs to group i // group_size.

    The inject hook fires only at PREFILL (h.shape[1] > 1) and adds unit(dir) at the marker, so the
    injected direction conditions the whole continuation; decode steps (h.shape[1] == 1) are skipped.
    Sampling is pure temperature-T FULL softmax (top_p=1, top_k off, rep-penalty 1) so the behavior
    policy == the T=1 policy old_logp measures. old_logp is then recomputed with the SAME inject via a
    fresh no-grad forward over prompt+gen (log_softmax of RAW logits gathered at the sampled tokens, NO
    temperature scaling) — identical to how update() computes new_logp, so the importance ratio is
    exact (mean ratio == 1 at step 0, before any optimizer step)."""
    G = a.group_size
    all_dirs = dirs.repeat_interleave(G, 0).to(device)          # [B*G, d] group-major
    N = all_dirs.shape[0]
    p_len = len(prompt_ids)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    # generate() stops on any of these; a padded row's first stop token == its true end.
    eos_ids = set()
    def _add_eos(e):
        if isinstance(e, (list, tuple)):
            for x in e:
                eos_ids.add(int(x))
        elif e is not None:
            eos_ids.add(int(e))
    _add_eos(tok.eos_token_id)
    try:
        _add_eos(actor.generation_config.eos_token_id)
    except Exception:
        pass
    texts, gen_ids, old_lps = [], [], []
    chunk = a.rollout_chunk
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        bsz = e - s
        inp = prompt.unsqueeze(0).expand(bsz, -1).contiguous()          # shared prompt, left-aligned
        attn = torch.ones_like(inp)
        vecs = [all_dirs[s + i : s + i + 1] for i in range(bsz)]        # one direction per row
        hook = make_inject_hook(vecs, [[marker]] * bsz, STEER_COEFF, device, torch.bfloat16)
        with hooked(submodule, hook):
            out = actor.generate(input_ids=inp, attention_mask=attn, do_sample=True,
                                 temperature=a.temperature, top_p=1.0, top_k=0, min_p=0.0,
                                 repetition_penalty=1.0, max_new_tokens=a.max_new_tokens,
                                 min_new_tokens=a.min_new_tokens, use_cache=True,
                                 pad_token_id=tok.pad_token_id)
        gen_full = out[:, p_len:].tolist()                              # generated ids (right-padded)
        row_gen = []
        for g in gen_full:
            trimmed = []
            for t in g:
                trimmed.append(t)
                if t in eos_ids:                                       # include the stop token, drop pad tail
                    break
            row_gen.append(trimmed if trimmed else g)                  # min_new_tokens keeps this non-empty
        # ---- recompute old_logp with the SAME inject hook (exactly how update() gets new_logp) ----
        gmax = max(len(g) for g in row_gen)
        Lc = p_len + gmax
        ids_f = torch.full((bsz, Lc), tok.pad_token_id, dtype=torch.long, device=device)
        attn_f = torch.zeros((bsz, Lc), dtype=torch.long, device=device)
        ids_f[:, :p_len] = prompt
        for i, g in enumerate(row_gen):
            ids_f[i, p_len : p_len + len(g)] = torch.tensor(g, dtype=torch.long, device=device)
            attn_f[i, : p_len + len(g)] = 1
        with hooked(submodule, hook):
            logits = actor(input_ids=ids_f, attention_mask=attn_f).logits[:, p_len - 1 : -1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids_f[:, p_len:, None]).squeeze(-1)  # [bsz, gmax]
        del logits
        for i, g in enumerate(row_gen):
            gen_ids.append(g)
            old_lps.append(lp[i, : len(g)].detach().float().cpu())
            texts.append(tok.decode(g, skip_special_tokens=True))
    return texts, gen_ids, old_lps


def _distinct_fraction(input_ids, attention_mask):
    """Vectorized distinct-token fraction, including special tokens like the original gate."""
    masked = input_ids.masked_fill(~attention_mask.bool(), -1)
    ordered = masked.sort(dim=1).values
    unique = torch.ones(len(input_ids), dtype=torch.long, device=input_ids.device)
    unique += (ordered[:, 1:] != ordered[:, :-1]).sum(1)
    unique -= (~attention_mask.bool()).any(1).long()  # remove the padding sentinel, if present
    return unique.float() / attention_mask.sum(1).clamp(min=1)


@torch.no_grad()
def score(texts, dirs_rep, actor, tok, device, a, with_fluency=False, return_act=False):
    """Score standalone generations through the clean base model.

    With fluency gates enabled, capture the layer-READ_LAYER residual during the full fluency
    forward. This produces the same reward and gate values in one clean-model pass instead of the
    previous full fluency pass plus a redundant early-exit reward pass.
    """
    r = torch.zeros(len(texts))
    meanact = torch.zeros(len(texts), D_MODEL)
    logp = torch.full((len(texts),), -20.0) if with_fluency else None
    dis = torch.zeros(len(texts)) if with_fluency else None
    valid = [i for i, t in enumerate(texts) if t.strip()]
    prev = tok.padding_side
    tok.padding_side = "right"  # position 0 must be the first real token
    try:
        for s in range(0, len(valid), a.score_batch):
            idxs = valid[s : s + a.score_batch]
            enc = tok([texts[i] for i in idxs], return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_new_tokens + 32, add_special_tokens=True).to(device)
            if with_fluency:
                captured = {}

                def capture(_module, _inputs, output):
                    captured["h"] = output[0] if isinstance(output, tuple) else output

                handle = get_layer(actor, READ_LAYER).register_forward_hook(capture)
                try:
                    with actor.disable_adapter():
                        logits = actor(**enc).logits[:, :-1].float()
                finally:
                    handle.remove()
                h = captured["h"].float()
                mask = enc["attention_mask"].bool()
            else:
                with actor.disable_adapter():
                    h, mask = read_resid(actor, READ_LAYER, dict(enc), pool="all")
            keep = mask.clone()
            keep[:, 0] = False  # attention-sink guard (old repo also norm-filtered; keep it simple)
            hh = F.normalize(h, dim=-1) if a.reward_metric == "cosine" else h  # cosine: kill norm-inflation
            proj = torch.einsum("btd,bd->bt", hh, dirs_rep[idxs])
            best = proj.masked_fill(~keep, torch.finfo(proj.dtype).min).max(1).values
            has = keep.any(1)
            r[idxs] = torch.where(has, best, 0).cpu()
            _filled = proj.masked_fill(~keep, torch.finfo(proj.dtype).min)
            _pstar = _filled.argmax(1)                       # peak (max-activation) token per rollout
            _tt = torch.arange(h.shape[1], device=h.device)
            _pmask = keep & (_tt.unsqueeze(0) <= _pstar.unsqueeze(1))   # kept tokens up to & incl peak (excl after)
            _msum = (h * _pmask.unsqueeze(-1)).sum(1); _mcnt = _pmask.sum(1, keepdim=True).clamp(min=1)
            meanact[idxs] = (_msum / _mcnt).float().cpu()
            if with_fluency and logits.shape[1]:
                targets = enc["input_ids"][:, 1:]
                token_lp = -F.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), reduction="none"
                ).view_as(targets)
                next_mask = mask[:, 1:]
                mean_lp = (token_lp * next_mask).sum(1) / next_mask.sum(1).clamp(min=1)
                # rows that retokenize to a single token have no next-token logprob → keep the -20.0
                # default so they FAIL the fluency floor (the degenerate collapse the gate exists to punish)
                logp[idxs] = torch.where(next_mask.any(1), mean_lp,
                                         torch.full_like(mean_lp, -20.0)).cpu()
                dis[idxs] = _distinct_fraction(enc["input_ids"], mask).cpu()
    finally:
        tok.padding_side = prev
    if return_act:
        return (r, logp, dis, meanact) if with_fluency else (r, meanact)
    return (r, logp, dis) if with_fluency else r


@torch.no_grad()
def greedy_eval(llm, prompt_ids, marker, eval_dirs, eval_base, actor, tok, device, a):
    """Greedy-decode (T=0) ONE rollout per held-out eval dir through the same vLLM+steering path,
    scored by the training scorer (clean base, max-over-kept-positions act·dir, sink skip).
    Returns (metrics, texts). Identical texts across distinct dirs mean steering silently isn't
    firing (the FLASHINFER failure mode) — crash loudly rather than log garbage."""
    from vllm import SamplingParams
    from vllm_lens import SteeringVector

    reqs, params = [], []
    for v in eval_dirs:
        sv = SteeringVector(activations=v.view(1, 1, -1).cpu().float(), layer_indices=[INJECT_LAYER],
                            scale=STEER_COEFF, norm_match=True, position_indices=[marker])
        reqs.append({"prompt_token_ids": list(prompt_ids)})
        params.append(SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens,
                                     min_tokens=a.min_new_tokens,
                                     extra_args={"apply_steering_vectors": [sv]}))
    outs = llm.generate(reqs, params)
    texts = [out.outputs[0].text for out in outs]
    assert len(texts) < 2 or len(set(texts)) > 1, (
        "greedy eval: all rollouts identical across distinct directions — steering is not firing "
        "(wrong attention backend? vllm-lens/vLLM version drift?)")
    r = score(texts, eval_dirs.to(device), actor, tok, device, a)  # max over positions, per dir
    m = {"eval/greedy_act_mean": r.mean().item(), "eval/greedy_act_max": r.max().item()}
    if eval_base is not None:  # corpus baseline: best candidate-span projection per direction
        m["eval/greedy_beat_frac"] = (r > eval_base).float().mean().item()
    return m, texts


@torch.no_grad()
def sae_score(texts, feats, sae, actor, tok, device, a):
    """max_t relu((x_t - b_dec)·W_enc[:,f] + b_enc[f]) at READ_LAYER, clean base, standalone re-tok.
    Row i is scored on ITS paired feature feats[i]. Mirrors SL/eval_sae.py exactly so the number is
    comparable to the pretrain cross-uplift curve: add_special_tokens=False + 10x-median norm-filter
    + pos-0 attention-sink drop (deliberately different from score()'s simpler dot path)."""
    NORM_FILTER_MULT = 10.0
    r = torch.zeros(len(texts))
    valid = [i for i, t in enumerate(texts) if t.strip()]
    prev = tok.padding_side
    tok.padding_side = "right"  # position 0 must be the first real token
    try:
        for s in range(0, len(valid), a.score_batch):
            idxs = valid[s : s + a.score_batch]
            enc = tok([texts[i] for i in idxs], return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_new_tokens + 32, add_special_tokens=False).to(device)
            with actor.disable_adapter():
                h, mask = read_resid(actor, READ_LAYER, dict(enc), pool="all")  # [b,T,d] fp32, [b,T]
            norms = h.norm(dim=-1)
            med = norms[mask].median() if mask.any() else norms.new_tensor(1.0)
            keep = mask & (norms <= NORM_FILTER_MULT * med)
            keep[:, 0] = False
            per = sae.encode_features(h, [feats[i] for i in idxs])   # [b,T,b]
            b = torch.arange(len(idxs), device=per.device)
            per = per[b, :, b]                                        # diagonal: row i on feature i -> [b,T]
            best = per.masked_fill(~keep, 0.0).max(1).values
            r[idxs] = best.float().cpu()
    finally:
        tok.padding_side = prev
    return r


@torch.no_grad()
def sae_eval(llm, prompt_ids, marker, sae_dirs, sae_feats, sae, dataset_max, actor, tok, device, a):
    """Zero-shot held-out SAE eval (greedy). Inject each held-out feature's unit encoder column,
    greedy-decode one rollout, score the true SAE feature activation, and compare to the feature's
    corpus peak. normalized_act / beat_frac match SL/eval_sae.py's greedy metrics."""
    from vllm import SamplingParams
    from vllm_lens import SteeringVector
    reqs, params = [], []
    for v in sae_dirs:
        sv = SteeringVector(activations=v.view(1, 1, -1).cpu().float(), layer_indices=[INJECT_LAYER],
                            scale=STEER_COEFF, norm_match=True, position_indices=[marker])
        reqs.append({"prompt_token_ids": list(prompt_ids)})
        params.append(SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens,
                                     min_tokens=a.min_new_tokens,
                                     extra_args={"apply_steering_vectors": [sv]}))
    texts = [out.outputs[0].text for out in llm.generate(reqs, params)]
    assert len(texts) < 2 or len(set(texts)) > 1, "sae eval: all greedy rollouts identical — steering not firing"
    act = sae_score(texts, sae_feats, sae, actor, tok, device, a)   # [N] max-pos post-ReLU enc act
    dm = dataset_max.detach().cpu().clamp(min=1e-6)
    dmax = dataset_max.detach().cpu()
    metrics = {"eval/sae_act_mean": act.mean().item(),
               "eval/sae_norm_act": (act / dm).mean().item(),        # canonical GREEDY cross-uplift metric
               "eval/sae_beat_frac": (act > dmax).float().mean().item()}
    if a.sae_eval_bo > 1:  # best-of-N sampling read: sampling is far more productive than greedy
        bo_reqs, bo_params = [], []                                  # (mirrors the SL Bo16 curve)
        for v in sae_dirs:
            sv = SteeringVector(activations=v.view(1, 1, -1).cpu().float(), layer_indices=[INJECT_LAYER],
                                scale=STEER_COEFF, norm_match=True, position_indices=[marker])
            bo_reqs.append({"prompt_token_ids": list(prompt_ids)})
            bo_params.append(SamplingParams(temperature=a.sae_eval_temp, top_p=1.0, n=a.sae_eval_bo,
                                            max_tokens=a.max_new_tokens, min_tokens=a.min_new_tokens,
                                            extra_args={"apply_steering_vectors": [sv]}))
        bo_texts, bo_feats = [], []
        for i, out in enumerate(llm.generate(bo_reqs, bo_params)):
            for o in out.outputs:                                   # N samples/feature, feature-major
                bo_texts.append(o.text); bo_feats.append(sae_feats[i])
        bo_act = sae_score(bo_texts, bo_feats, sae, actor, tok, device, a).view(len(sae_dirs), a.sae_eval_bo)
        bo_best = bo_act.max(1).values                              # per-feature best of the N samples
        k = a.sae_eval_bo
        metrics.update({f"eval/sae_bo{k}_act_mean": bo_best.mean().item(),
                        f"eval/sae_bo{k}_norm_act": (bo_best / dm).mean().item(),
                        f"eval/sae_bo{k}_beat_frac": (bo_best > dmax).float().mean().item()})
    return metrics, texts


@torch.no_grad()
def fluency(texts, actor, tok, device, a):
    """(mean clean-base logprob/token, distinct-token fraction) per standalone text — gate inputs.
    Adapter disabled so the policy can't inflate its own fluency score."""
    logp, dis = torch.full((len(texts),), -20.0), torch.zeros(len(texts))
    valid = [i for i, t in enumerate(texts) if t.strip()]
    prev = tok.padding_side
    tok.padding_side = "right"
    try:
        for s in range(0, len(valid), a.score_batch):
            idxs = valid[s : s + a.score_batch]
            enc = tok([texts[i] for i in idxs], return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_new_tokens + 32, add_special_tokens=True).to(device)
            if enc["input_ids"].shape[1] < 2:
                continue
            with actor.disable_adapter():
                logits = actor(**enc).logits[:, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, enc["input_ids"][:, 1:, None]).squeeze(-1)
            m = enc["attention_mask"][:, 1:].bool()
            for row, i in enumerate(idxs):
                n = int(m[row].sum())
                if n:
                    logp[i] = (lp[row][m[row]].sum() / n).item()
                ids = enc["input_ids"][row][enc["attention_mask"][row].bool()]
                dis[i] = len(set(ids.tolist())) / max(len(ids), 1)
    finally:
        tok.padding_side = prev
    return logp, dis


def _ddp_sync_grads(params, total_tok):
    """All-reduce the trainable (LoRA) grads across ranks — ONE flat CPU buffer over gloo (NCCL
    deadlocks on this box; gloo is fine because only the small LoRA grads move). Token-weighted
    average: each rank's grad is (sum of its token grads)/local_tok, so
    Σ_r grad_r·tok_r / Σ_r tok_r == EXACTLY the single-GPU gradient over the union batch (with
    equal per-rank token counts this reduces to the plain average). The trailing buffer slot
    carries local_tok so a single collective yields both the weighted sum and the global count."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.detach().reshape(-1).float() for g in grads]
                     + [torch.ones(1, device=grads[0].device)]).cpu()
    flat.mul_(float(total_tok))                    # token-weight this rank's contribution
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)    # gloo: CPU tensor, no NCCL anywhere
    flat.div_(flat[-1].item())                     # /= global completion-token count
    off = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[off : off + n].view_as(g))    # copy_ handles cpu->cuda + dtype cast
        off += n


def update(actor, opt, submodule, ids, attn, p_len, marker, old_lp, adv, dirs_rep, a, device):
    """ONE Dr. GRPO optimizer update. loss = Σ_tokens −min(ratio·A, clip(ratio)·A)·mask / TOTAL
    completion tokens in batch (GLOBAL constant normalizer — no per-sequence mean, no /std, no KL).
    ratio TIS-capped (upper only). new_logp forward runs with the SAME inject hook as rollout."""
    n = ids.shape[0]
    gen_mask = attn[:, p_len:].bool()
    total_tok = max(int(gen_mask.sum()), 1)
    lo, hi = 1 - a.clip_eps, 1 + a.clip_eps
    loss_sum, clipped_tok, ent_sum, kl_sum, ratio_sum = 0.0, 0, 0.0, 0.0, 0.0
    opt.zero_grad(set_to_none=True)
    for s in range(0, n, a.micro_batch):
        e = min(s + a.micro_batch, n)
        b_ids, b_attn = ids[s:e].to(device), attn[s:e].to(device)
        hook = make_inject_hook([dirs_rep[i : i + 1] for i in range(s, e)], [[marker]] * (e - s),
                                STEER_COEFF, device, torch.bfloat16)
        with hooked(submodule, hook):
            logits = actor(input_ids=b_ids, attention_mask=b_attn).logits[:, p_len - 1 : -1]
        logp_full = torch.log_softmax(logits.float(), -1)
        del logits
        new_lp = logp_full.gather(-1, b_ids[:, p_len:, None]).squeeze(-1)
        m = gen_mask[s:e].to(device)
        ratio = torch.exp(new_lp - old_lp[s:e].to(device)).clamp(max=a.tis_cap)  # TIS, upper only
        A = adv[s:e, None].to(device)
        loss = (-torch.minimum(ratio * A, ratio.clamp(lo, hi) * A) * m).sum() / total_tok
        # per-token entropy: ALWAYS computed for logging (logits already materialized, ~free);
        # bonus term added to the loss only when entropy_coef > 0 (Dr. GRPO stays KL/bonus-free by default).
        ent = -(logp_full.exp() * logp_full).sum(-1)
        ent_sum += float((ent.detach() * m).sum())
        if a.entropy_coef > 0:  # maximize r + β·H(π): keeps policy stochastic for Bo-N, no KL anchoring
            loss = loss - a.entropy_coef * (ent * m).sum() / total_tok
        if a.kl_coef > 0:  # capped KL-to-init anchor: ref logps from the FROZEN init adapter, same inject hook
            with torch.no_grad():
                actor.set_adapter("ref")
                with hooked(submodule, hook):
                    ref_logits = actor(input_ids=b_ids, attention_mask=b_attn).logits[:, p_len - 1 : -1]
                ref_lp = torch.log_softmax(ref_logits.float(), -1).gather(-1, b_ids[:, p_len:, None]).squeeze(-1)
                del ref_logits
                actor.set_adapter("default")  # MUST restore before next micro-batch / sync_weights merge
            delta = ref_lp - new_lp                                    # log(π_ref / π)
            kl = (torch.exp(delta) - delta - 1).clamp(0.0, a.kl_cap)   # k3 estimator, capped per token
            loss = loss + a.kl_coef * (kl * m).sum() / total_tok
            kl_sum += float((kl.detach() * m).sum())
        del logp_full
        loss.backward()  # micro-losses share the global normalizer → grads sum correctly
        loss_sum += loss.item()
        clipped_tok += int((((ratio < lo) | (ratio > hi)) & m).sum())
        ratio_sum += float((ratio.detach() * m).sum())
    params = [p for p in actor.parameters() if p.requires_grad]
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        _ddp_sync_grads(params, total_tok)  # after this, grads (hence clip + step) match on all ranks
    gn = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm))
    if math.isfinite(gn):
        opt.step()
    else:  # stepping Adam on nan/inf grads corrupts moments AND weights
        opt.zero_grad(set_to_none=True)
        print(f"[update] non-finite grad norm ({gn}) — skipping step", flush=True)
    return {"loss": loss_sum, "grad_norm": gn, "clipfrac": clipped_tok / total_tok,
            "entropy": ent_sum / total_tok, "kl": kl_sum / total_tok,
            "ratio_mean": ratio_sum / total_tok}


def main():
    cfg, tr = RLConfig(), TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pretrain")
    ap.add_argument("--init-adapter", default=cfg.init_adapter)
    ap.add_argument("--save-dir", default=cfg.save_dir)
    ap.add_argument("--run-name", default=cfg.run_name)
    ap.add_argument("--direction-source", default=cfg.direction_source)
    ap.add_argument("--groups-per-step", type=int, default=cfg.groups_per_step)
    ap.add_argument("--group-size", type=int, default=cfg.group_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--clip-eps", type=float, default=cfg.clip_eps)
    ap.add_argument("--tis-cap", type=float, default=cfg.tis_cap)
    ap.add_argument("--max-new-tokens", type=int, default=cfg.max_new_tokens)
    ap.add_argument("--min-new-tokens", type=int, default=cfg.min_new_tokens)
    ap.add_argument("--temperature", type=float, default=cfg.temperature)
    ap.add_argument("--total-steps", type=int, default=cfg.total_steps)
    ap.add_argument("--sync-every", type=int, default=cfg.sync_every)
    ap.add_argument("--fluency-floor", type=float, default=cfg.fluency_floor)
    ap.add_argument("--distinct-floor", type=float, default=cfg.distinct_floor)
    ap.add_argument("--gate-penalty", type=float, default=cfg.gate_penalty)
    ap.add_argument("--len-penalty-start", type=int, default=cfg.len_penalty_start)
    ap.add_argument("--len-penalty-per-tok", type=float, default=cfg.len_penalty_per_tok)
    ap.add_argument("--div-coef", type=float, default=0.0,
                    help="within-group activation-orthogonal diversity bonus (0=off)")
    ap.add_argument("--no-gates", action="store_true", help="disable fluency/distinct/len shaping")
    ap.add_argument("--entropy-coef", type=float, default=cfg.entropy_coef,
                    help="β for maximize r + β·H(π): direct diversity pressure (Bo-N depends on it)")
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="mild KL-to-init anchor (0=off): loss += kl_coef * capped_KL(pi || pi_init). "
                         "reference = base + the --init-adapter LoRA (the SL policy RL starts from)")
    ap.add_argument("--kl-cap", type=float, default=10.0,
                    help="per-token KL clamp in nats — a single low-ref-prob token can't blow up the penalty")
    ap.add_argument("--tri-floor", type=float, default=0.0,
                    help="min distinct-trigram ratio; below it a rollout fails the gate (0=off)")
    ap.add_argument("--comp-floor", type=float, default=0.0,
                    help="min zlib compression ratio of the text; catches templated repetition (0=off)")
    ap.add_argument("--reward-metric", choices=("proj", "cosine"), default="proj",
                    help="proj: max_t <h_t, unit(v)> (raw residual dot unit dir; norm-sensitive = the "
                         "v1/v2 objective). cosine: max_t cos(h_t, v) — scale-invariant, kills the "
                         "residual-norm inflation hack (pumping ||h|| buys nothing).")
    ap.add_argument("--reward-scale", type=float, default=1.0,
                    help="multiply raw reward before shaping; use ~1000 with --reward-metric cosine to keep "
                         "the ~proj magnitude so LR/gate/KL coefs stay valid under Dr.GRPO unnormalized adv.")
    ap.add_argument("--firsttok-coef", type=float, default=0.0,
                    help="reward shaping (0=off): += coef * fraction of THIS direction's max-act target's "
                         "first-k token ids present in the rollout. Direction-SPECIFIC anchor the "
                         "direction-agnostic hack can't collect; needs records.jsonl target_text.")
    ap.add_argument("--firsttok-k", type=int, default=4,
                    help="how many of the target's leading token ids to anchor on")
    ap.add_argument("--tp", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.35)
    ap.add_argument("--attn-backend", default="TRITON_ATTN",
                    help="vLLM attention backend; TRITON_ATTN is the only one verified to expose "
                         "the metadata vllm-lens needs (FLASHINFER silently breaks injection)")
    ap.add_argument("--vllm-max-len", type=int, default=1024)
    ap.add_argument("--rollout-chunk", type=int, default=32,
                    help="HF-generate mini-batch size — sequences per generate() call; lower it if OOM")
    ap.add_argument("--bank-file", default="vecs.f32",
                    help="direction-bank filename under --data-dir (e.g. probes.f32 for the RL pool)")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--score-batch", type=int, default=64)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=25,
                    help="greedy-decode the held-out eval dirs every N steps (0 disables)")
    ap.add_argument("--n-eval-dirs", type=int, default=64,
                    help="UNIQUE directions reserved from the FRONT of the bank as eval-only; "
                         "training samples strictly past their rows, so they are never RL'd on")
    # ---- zero-shot held-out SAE eval: inject a held-out SAE feature's unit encoder column, greedy-
    # decode, score the TRUE post-ReLU feature activation vs the feature's corpus-peak baseline.
    # SAE features are NEVER trained on (RL is on cluster probes) — this is a pure cross-basis
    # generalization probe, mirroring SL/eval_sae.py so numbers match the pretrain cross-uplift curve.
    ap.add_argument("--sae-eval-every", type=int, default=0,
                    help="every N steps, zero-shot eval on held-out SAE features (0 disables)")
    ap.add_argument("--n-sae-eval-feats", type=int, default=128,
                    help="held-out SAE features (prefix of the split 'eval' list) for the in-loop eval")
    ap.add_argument("--sae-path", default=None, help="local ae.pt (offline env); None -> HF cache")
    ap.add_argument("--maxacts-path", default=None,
                    help="local max_acts .pt for per-feature corpus-peak baselines; None -> HF cache")
    ap.add_argument("--sae-split", default=None,
                    help="split.json carrying an 'eval' list of held-out SAE feature ids")
    ap.add_argument("--sae-eval-bo", type=int, default=1,
                    help="best-of-N SAE eval: also sample N/feature and report Bo-N max (1=greedy only)")
    ap.add_argument("--sae-eval-temp", type=float, default=0.75,
                    help="sampling temperature for the best-of-N SAE eval (greedy eval stays T=0)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.no_gates:
        a.fluency_floor = a.distinct_floor = a.len_penalty_start = None
    # vLLM generation logprobs equal the sampling distribution's ONLY at T=1 (raw_logprobs).
    assert a.temperature == 1.0, "sampling temp must be 1.0 so behavior == the T=1 policy old_logp measures"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    # ---- data-parallel over groups (torchrun sets WORLD_SIZE/RANK/LOCAL_RANK). Launch with
    # DDP_BACKEND=gloo on this box: NCCL deadlocks at the first collective (ranks spin 100% CPU,
    # 0% GPU — see SL/pretrain.py, fixed the same way). world=1 (no torchrun): every DDP branch
    # below is skipped and behavior is identical to the original single-GPU script. ----
    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0)); is_main = rank == 0
    if world > 1:
        assert a.groups_per_step % world == 0, (
            f"groups_per_step ({a.groups_per_step}) must divide by world ({world}): each rank "
            f"takes a contiguous slice of WHOLE groups so Dr.GRPO advantages stay intra-rank")
        dist.init_process_group(os.environ.get("DDP_BACKEND", "nccl"))
        torch.cuda.set_device(local)
    device = f"cuda:{local}"  # HF actor lives here; world=1 -> "cuda:0" exactly as before

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    assert p_len + a.max_new_tokens <= a.vllm_max_len

    # ---- direction bank ----
    if a.direction_source == "cluster":
        stats_p = f"{a.data_dir}/build_stats.json"
        n_vecs = (json.load(open(stats_p))["n_examples"] if os.path.exists(stats_p)
                  else os.path.getsize(f"{a.data_dir}/{a.bank_file}") // (4 * D_MODEL))
        bank = np.memmap(f"{a.data_dir}/{a.bank_file}", dtype=np.float32, mode="r", shape=(n_vecs, D_MODEL))
    elif a.direction_source == "random":
        # isotropic random unit directions sampled fresh every step — infinite data, no bank/targets.
        # tests whether the invert-a-direction skill generalizes to ARBITRARY (non-feature) directions.
        bank, n_vecs = None, None
    else:
        # TODO: "sae" = unit encoder columns of the L27 SAE, "mix" = interleave cluster+sae.
        # The SAE loader isn't in this repo yet — port from max-activating-examples/src/maxact/sae.py.
        raise NotImplementedError(f"direction_source={a.direction_source!r}: only 'cluster'/'random' so far")

    # ---- held-out greedy-eval reservation. build_data mints --targets consecutive rows per probe
    # direction, so unique directions are contiguous blocks; reserve the first n_eval_dirs UNIQUE
    # blocks as EVAL-ONLY and make training sample rows strictly past them. SAE features are a
    # separate zero-shot eval elsewhere — they are never RL'd on. ----
    eval_dirs, eval_base, eval_rows = None, None, 0
    if a.eval_every > 0 and a.n_eval_dirs > 0 and bank is not None:
        blocks, starts, i = [], [], 0
        while i < n_vecs and len(blocks) < a.n_eval_dirs:
            row = np.asarray(bank[i], dtype=np.float32)
            starts.append(i)
            blocks.append(row)
            i += 1
            while i < n_vecs and np.array_equal(np.asarray(bank[i]), row):
                i += 1
        assert len(blocks) == a.n_eval_dirs, f"bank has only {len(blocks)} unique dirs < {a.n_eval_dirs}"
        eval_rows = i
        eval_dirs = torch.nn.functional.normalize(torch.from_numpy(np.stack(blocks)), dim=-1)
        rec_p = f"{a.data_dir}/records.jsonl"
        if os.path.exists(rec_p):  # per-dir corpus baseline (only in pools built with corpus_max_proj)
            with open(rec_p) as f:
                recs = [json.loads(next(f)) for _ in range(eval_rows)]
            base = [recs[s].get("corpus_max_proj") for s in starts]
            if all(b is not None for b in base):
                eval_base = torch.tensor(base, dtype=torch.float32)
        if is_main:
            print(f"[eval] reserved rows [0, {eval_rows}) = {a.n_eval_dirs} eval-only dirs "
                  f"(corpus baseline: {'yes' if eval_base is not None else 'no'})", flush=True)
    tgt_texts = None
    if a.firsttok_coef > 0:
        tgt_texts = [json.loads(l).get("target_text", "") for l in open(f"{a.data_dir}/records.jsonl")]
        if is_main:
            print(f"[firsttok] loaded {len(tgt_texts)} target texts (first-{a.firsttok_k}-tok anchor, "
                  f"coef {a.firsttok_coef})", flush=True)
    assert a.direction_source == "random" or n_vecs - eval_rows >= a.groups_per_step

    # ---- actor (HF + LoRA, cuda:0). NO gradient checkpointing EVER: recompute happens after the
    # inject-hook context exits → silently wrong grads. ----
    actor = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": device})
    if a.init_adapter:
        actor = PeftModel.from_pretrained(actor, a.init_adapter, is_trainable=True)
    else:
        actor = get_peft_model(actor, LoraConfig(
            r=tr.lora_r, lora_alpha=tr.lora_alpha, lora_dropout=0.0, use_rslora=True,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    actor.train()
    opt = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    submodule = get_layer(actor, INJECT_LAYER)
    if a.kl_coef > 0:  # frozen reference = the init (SL) policy, for the capped KL-to-init anchor
        assert a.init_adapter, "--kl-coef needs --init-adapter (reference = base + init-LoRA)"
        actor.load_adapter(a.init_adapter, adapter_name="ref")  # loaded frozen (not trainable)
        actor.set_adapter("default")                            # keep the trainable adapter active
        if is_main:
            print(f"[kl] ref=init adapter loaded; kl_coef={a.kl_coef} cap={a.kl_cap} nats/tok", flush=True)

    # ---- HF-generate rollouts: the LoRA actor IS the rollout engine (no vLLM). ----
    llm = None
    if is_main:
        print(f"[hf-rollout] actor ready | {n_vecs} directions | prompt {p_len} toks, marker @{marker} "
              f"| rollout_chunk {a.rollout_chunk} | world {world}", flush=True)

    # ---- held-out SAE zero-shot eval setup (features NEVER RL'd on — a pure cross-basis
    # generalization probe; mirrors SL/eval_sae.py so numbers track the pretrain cross-uplift curve) ----
    sae_obj = sae_dirs = sae_feats = sae_dataset_max = None
    if a.sae_eval_every > 0 and is_main:  # in-loop evals are rank-0-only under DDP
        from mxf.sae import load_max_acts, load_sae
        assert a.sae_split, "--sae-eval-every set but --sae-split (split.json with 'eval' ids) missing"
        sae_obj = load_sae(path=a.sae_path, device=device, dtype=torch.float32)
        sae_feats = json.load(open(a.sae_split))["eval"][: a.n_sae_eval_feats]
        sae_dirs = sae_obj.enc_dirs(sae_feats)                        # [n,d] unit encoder cols, cuda:0
        ma = load_max_acts(path=a.maxacts_path)["max_acts"]           # [F,N,L] corpus peaks
        sae_dataset_max = ma[torch.as_tensor(sae_feats)].amax(dim=(1, 2)).to(device)  # [n]
        del ma
        print(f"[sae-eval] {len(sae_feats)} held-out SAE feats every {a.sae_eval_every} steps; "
              f"baseline median {sae_dataset_max.median().item():.1f}", flush=True)

    if not a.no_wandb and is_main:
        wandb.init(project="maxact-fast", name=a.run_name, config=vars(a))
    if is_main:
        os.makedirs(a.save_dir, exist_ok=True)
    B, G = a.groups_per_step, a.group_size   # B = GLOBAL groups/step (same meaning as world=1)
    Bl = B // world                          # groups THIS rank rolls out / scores / backprops

    for step in range(a.total_steps):
        ev = {}
        if eval_dirs is not None and is_main and step % a.eval_every == 0:
            te = time.time()
            ev, ev_texts = greedy_eval(llm, prompt_ids, marker, eval_dirs, eval_base, actor, tok, device, a)
            ev["time/eval_s"] = time.time() - te
            print(f"[eval] step {step:05d} | greedy act mean {ev['eval/greedy_act_mean']:.2f} "
                  f"max {ev['eval/greedy_act_max']:.2f}"
                  + (f" | beat {ev['eval/greedy_beat_frac']:.0%}" if "eval/greedy_beat_frac" in ev else "")
                  + f" | {ev_texts[0][:90]!r}", flush=True)
        if sae_obj is not None and step % a.sae_eval_every == 0:
            ts = time.time()
            sev, sev_texts = sae_eval(llm, prompt_ids, marker, sae_dirs, sae_feats, sae_obj,
                                      sae_dataset_max, actor, tok, device, a)
            sev["time/sae_eval_s"] = time.time() - ts
            ev.update(sev)
            print(f"[sae-eval] step {step:05d} | norm_act {sev['eval/sae_norm_act']:.3f} | "
                  f"beat {sev['eval/sae_beat_frac']:.0%} | act_mean {sev['eval/sae_act_mean']:.1f}"
                  + (f" | Bo{a.sae_eval_bo}@{a.sae_eval_temp} beat {sev[f'eval/sae_bo{a.sae_eval_bo}_beat_frac']:.0%} "
                     f"norm {sev[f'eval/sae_bo{a.sae_eval_bo}_norm_act']:.3f}" if a.sae_eval_bo > 1 else "")
                  + f" | {sev_texts[0][:60]!r}", flush=True)
        t0 = time.time()
        # B distinct vec_idx past the eval reservation (sorted: memmap-friendly). Under DDP every
        # rank draws the SAME B directions (identical seed -> identical rng stream), then takes its
        # contiguous slice of Bl WHOLE groups — group membership never crosses a rank boundary, so
        # the Dr.GRPO advantage (r - group_mean) stays intra-rank and exact.
        idx = None
        if a.direction_source == "random":                     # fresh isotropic unit dirs each step
            dirs = torch.nn.functional.normalize(torch.randn(B, D_MODEL, dtype=torch.float32), dim=-1)
        else:
            idx = eval_rows + np.sort(rng.choice(n_vecs - eval_rows, size=B, replace=False))
            dirs = torch.nn.functional.normalize(
                torch.from_numpy(np.asarray(bank[idx], dtype=np.float32)), dim=-1)
        if world > 1:
            dirs = dirs[rank * Bl : (rank + 1) * Bl]
            if idx is not None:
                idx = idx[rank * Bl : (rank + 1) * Bl]
        texts, gen_ids, old_lps = rollout(actor, submodule, tok, prompt_ids, marker, dirs, a, device)
        t_roll = time.time() - t0
        dirs_rep = dirs.repeat_interleave(G, 0).to(device)  # [B*G, d] rollout i's group direction

        use_fluency = a.fluency_floor is not None or a.distinct_floor is not None
        scored = score(texts, dirs_rep, actor, tok, device, a, with_fluency=use_fluency, return_act=(a.div_coef > 0))
        meanact = None
        if use_fluency and a.div_coef > 0:
            r, flu, dis, meanact = scored
        elif use_fluency:
            r, flu, dis = scored
        elif a.div_coef > 0:
            r, meanact = scored
        else:
            r = scored
        r = r * a.reward_scale                       # cosine runs: ~1000x to match proj magnitude
        raw_r, gate_frac = r.clone(), 1.0
        if use_fluency:
            gate = torch.ones(Bl * G, dtype=torch.bool)
            if a.fluency_floor is not None:
                gate &= flu >= a.fluency_floor
            if a.distinct_floor is not None:
                gate &= dis >= a.distinct_floor
            # repetition gates (catch the reward-hack templates the distinct-TOKEN floor misses):
            # trigram-distinctness kills verbatim loops; compression-ratio kills templated number/citation spam
            if a.tri_floor > 0 or a.comp_floor > 0:
                tri = torch.tensor([_distinct_ngram_ratio(g, 3) for g in gen_ids])
                comp = torch.tensor([_compression_ratio(t) for t in texts])
                if a.tri_floor > 0:
                    gate &= tri >= a.tri_floor
                if a.comp_floor > 0:
                    gate &= comp >= a.comp_floor
            # sign-safe subtract, NOT zero: zeroing would rank gated garbage above coherent
            # negative-dot rollouts
            r = r - a.gate_penalty * (~gate).float()
            gate_frac = gate.float().mean().item()
        if a.len_penalty_start is not None:
            over = torch.tensor([max(0, len(g) - a.len_penalty_start) for g in gen_ids],
                                dtype=torch.float32)
            r = r - a.len_penalty_per_tok * over
        div_mean = 0.0
        if a.div_coef > 0 and meanact is not None:
            ma = meanact.to(device).view(Bl, G, D_MODEL)
            vhat = dirs.to(device)
            dots = torch.einsum("bgd,bd->bg", ma, vhat)
            perp = F.normalize(ma - dots.unsqueeze(-1) * vhat.unsqueeze(1), dim=-1)
            sim = torch.einsum("bgd,bhd->bgh", perp, perp)
            div = (1.0 - (sim.sum(2) - 1.0) / max(G - 1, 1)).flatten().cpu()
            _gmask = gate.float() if use_fluency else torch.ones_like(div)
            r = r + a.div_coef * div * _gmask
            div_mean = div.mean().item()
        ftok_mean = 0.0
        if a.firsttok_coef > 0 and tgt_texts is not None:
            # per-group (=direction) target first-k token-id sets; rollout i's group is i//G.
            # overlap varies ACROSS a group's rollouts, so it does NOT cancel in adv = r - group_mean.
            tgt_sets = [frozenset(tok.encode(tgt_texts[j], add_special_tokens=False)[: a.firsttok_k])
                        for j in idx]
            ftok = torch.tensor([len(set(gen_ids[i]) & tgt_sets[i // G]) / max(len(tgt_sets[i // G]), 1)
                                 for i in range(Bl * G)], dtype=torch.float32)
            r = r + a.firsttok_coef * ftok
            ftok_mean = ftok.mean().item()
        adv = (r.view(Bl, G) - r.view(Bl, G).mean(1, keepdim=True)).flatten().detach()  # NO /std

        # pad the batch — prompt is shared, so p_len is constant across rows
        L = p_len + max(len(g) for g in gen_ids)
        ids = torch.full((Bl * G, L), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((Bl * G, L), dtype=torch.long)
        old_lp = torch.zeros((Bl * G, L - p_len))
        pt = torch.tensor(prompt_ids, dtype=torch.long)
        for i, (g, lp) in enumerate(zip(gen_ids, old_lps)):
            ids[i, :p_len] = pt
            ids[i, p_len : p_len + len(g)] = torch.tensor(g)
            attn[i, : p_len + len(g)] = 1
            old_lp[i, : len(g)] = lp
        stats = update(actor, opt, submodule, ids, attn, p_len, marker, old_lp, adv, dirs_rep, a, device)
        if os.environ.get("RL_DDP_CHECK"):  # per-rank LoRA weight checksum — MUST match across ranks
            with torch.no_grad():
                cs = torch.stack([p.detach().float().norm()
                                  for p in actor.parameters() if p.requires_grad]).norm().item()
            print(f"[ddp-check] rank {rank} step {step} lora_l2 {cs:.10f}", flush=True)

        sync_s = 0.0  # no-op: the HF actor IS the rollout engine (no vLLM to sync)
        secs = time.time() - t0
        n_gen = float(sum(len(g) for g in gen_ids))
        if world > 1:  # aggregate reward stats over ALL ranks so the logged numbers mean the
            gath = [torch.zeros_like(raw_r) for _ in range(world)]  # same thing as a world=1 run
            dist.all_gather(gath, raw_r)
            raw_r_all = torch.cat(gath)
            gath = [torch.zeros_like(r) for _ in range(world)]
            dist.all_gather(gath, r)
            r_all = torch.cat(gath)
            aux = torch.tensor([n_gen, gate_frac, div_mean, ftok_mean], dtype=torch.float64)
            dist.all_reduce(aux)                  # SUM; equal rollout counts/rank -> mean of per-
            n_gen = float(aux[0])                 # rank means is exact for the fraction metrics
            gate_frac, div_mean, ftok_mean = (aux[1:] / world).tolist()
        else:
            raw_r_all, r_all = raw_r, r
        log = {"reward/mean": raw_r_all.mean().item(), "reward/std": raw_r_all.std().item(),
               "reward/max": raw_r_all.max().item(), "reward/shaped_mean": r_all.mean().item(),
               "reward/div_mean": div_mean,
               "reward/firsttok_mean": ftok_mean,
               "reward/gate_frac": gate_frac, "ratio/clipfrac": stats["clipfrac"],
               "policy/entropy": stats["entropy"], "policy/kl_to_init": stats["kl"],
               "ratio/mean": stats["ratio_mean"],
               "loss": stats["loss"], "grad_norm": stats["grad_norm"],
               "rollout/mean_logp": torch.cat(old_lps).mean().item(),
               "rollout/len_mean": n_gen / (B * G), "tokens_per_sec": n_gen / secs,
               "time/rollout_s": t_roll, "time/sync_s": sync_s, "time/step_s": secs}
        log.update(ev)
        if is_main:
            print(f"step {step:05d} | r {log['reward/mean']:.2f} (max {log['reward/max']:.1f}) | "
                  f"gate {gate_frac:.0%} | ratio {log['ratio/mean']:.3f} clip {log['ratio/clipfrac']:.2%} | len {log['rollout/len_mean']:.0f} "
                  f"| {log['tokens_per_sec']:.0f} tok/s | {secs:.0f}s", flush=True)
            if step % 10 == 0:
                print(f"  sample r={raw_r[0]:.2f}: {texts[0][:110]!r}", flush=True)
            if not a.no_wandb:
                wandb.log(log, step=step)
            if a.save_every and step and step % a.save_every == 0:
                actor.save_pretrained(f"{a.save_dir}/step_{step}")
    if is_main:
        actor.save_pretrained(f"{a.save_dir}/final")
        print("RL_DONE", flush=True)
    if world > 1:
        dist.barrier()  # non-zero ranks wait for rank 0's final save before tearing down
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
