"""2-rank FSDP2 exactness smoke of the --full-ft speed knobs on a TINY random Qwen3.5 (no download, cheap GPUs).

    torchrun --standalone --nproc_per_node=2 SL/fullft_smoke.py [--dtype bf16|fp32] [--steps 3]

For every variant a fresh model is built from the same seed and ONE optimizer step's gradients (after clipping, before
the update; the sharded fp32 grads of this rank) are compared with the per-micro-batch reference:
    ref            prefix fwd+bwd per micro-batch, no checkpointing, HF loss           (today's production path)
    share          --prefix-share-step (one prefix fwd/bwd per step; PrefixGradientAccumulator under FSDP2)
    share+ckpt     + --suffix-ckpt (exact per-layer activation checkpointing with the cache snapshot)
    share+ckpt+head+ --prefix-head-on-labels --ce-chunk 5
    share+keep     share + --fsdp-keep-unsharded -1 (params stay gathered across the step)
    share+keep+pf2 + --fsdp-prefetch 2
Then a multi-step run (ref vs share+keep+ckpt) checks that the second step trains on UPDATED weights (params compared
after every optimizer step), and every --optim variant is stepped once (adamw-fp32states must reproduce adamw).
"""
import argparse
import contextlib
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fullft as FT  # noqa: E402
from prefix_cache import PrefixCache, PrefixGradientAccumulator, SuffixCheckpointer, install_prefix_label_head  # noqa: E402

from mxf.inject import get_layer  # noqa: E402

D = 64
VOCAB = 512
PROMPT = list(range(10, 22))   # 12 prompt tokens; marker = 6 -> 6 shared prefix tokens, suffix = [marker] + target
MARKER = 6


def log(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **k, flush=True)


def tiny_model(seed, device, keep=0, prefetch=0, param_dtype=torch.bfloat16, world=2):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=VOCAB, hidden_size=D, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
                            max_position_embeddings=256, pad_token_id=0, eos_token_id=1, tie_word_embeddings=False)
    try:
        cfg._attn_implementation = "sdpa"
    except Exception:  # noqa
        pass
    model = Qwen3_5ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)   # production loads bf16, then upcasts per layer
    model.train()
    return FT.shard_full_model(model, world, device, log=lambda *a, **k: None, keep_unsharded_layers=keep,
                               prefetch=prefetch, param_dtype=param_dtype)


def make_data(seed, rank, n_steps, G, B):
    g = torch.Generator().manual_seed(seed * 1000 + rank)
    steps = []
    for _ in range(n_steps):
        group = []
        for _ in range(G):
            tg = [torch.randint(2, VOCAB, (int(torch.randint(3, 8, (1,), generator=g)),), generator=g).tolist() + [1]
                  for _ in range(B)]
            vecs = torch.randn(B, D, generator=g)
            group.append((vecs / vecs.norm(dim=-1, keepdim=True), tg))
        steps.append(group)
    return steps


def run_step(model, pc, group, share, opt=None):
    ac = contextlib.nullcontext
    FT.set_persistent_unshard(model, True)
    shared = None
    loss_step = 0.0
    if share:
        shared = PrefixGradientAccumulator(pc.run_prefix(ac), extra_outputs=[pc.pop_prefix_logits()])
    for mi, (vecs, targets) in enumerate(group):
        last = mi == len(group) - 1
        if shared is not None:
            out = pc.forward(vecs, targets, autocast=ac, prefix_cache=shared.cache)
            (out.loss / len(group)).backward()
            shared.accumulate()
        else:
            out = pc.forward(vecs, targets, autocast=ac)
            if last:
                FT.set_persistent_unshard(model, False)
            (out.loss / len(group)).backward()
        loss_step += out.loss.item() / len(group)
        del out
    if shared is not None:
        FT.set_persistent_unshard(model, False)
        shared.backward()
    gn = FT.clip_grad_norm([p for p in model.parameters() if p.requires_grad], 1.0)
    if opt is not None:
        opt.step()
        opt.zero_grad()
    return loss_step, gn


def grads(model):
    out = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            out[n] = FT._local(p.grad).detach().float().clone()
    return out


def params(model):
    return {n: FT._local(p).detach().float().clone() for n, p in model.named_parameters()}


def compare(a, b):
    """(max |a-b|, max relative Frobenius diff, name of the worst param, n compared)"""
    worst, worst_name, mx = 0.0, "", 0.0
    for n in a:
        if n not in b:
            continue
        d = (a[n] - b[n])
        mx = max(mx, d.abs().max().item())
        rel = d.norm().item() / max(b[n].norm().item(), 1e-12)
        if rel > worst:
            worst, worst_name = rel, n
    return mx, worst, worst_name, len(a)


def build_variant(seed, device, spec, dtype):
    model = tiny_model(seed, device, keep=spec.get("keep", 0), prefetch=spec.get("prefetch", 0), param_dtype=dtype)
    pc = PrefixCache(model, PROMPT, MARKER, 0, get_layer(model, 1), 1.0, device, prefix_model=model, pad_multiple=1,
                     inject_mode="add_clone", keep_prefix_grad_path=True)
    if spec.get("ckpt"):
        pc.suffix_ckpt = SuffixCheckpointer(model)
    if spec.get("head"):
        install_prefix_label_head(model, spec.get("ce_chunk", 0))
    return model, pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32

    world = int(os.environ["WORLD_SIZE"]); rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl"); torch.cuda.set_device(local)
    device = f"cuda:{local}"
    assert world == 2, "run with --nproc_per_node=2"
    log(f"[smoke] torch {torch.__version__} | {torch.cuda.get_device_name(0)} | compute dtype {a.dtype}")
    data = make_data(a.seed, rank, a.steps, G=2, B=4)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-3   # relative Frobenius; paths differ by fp summation order only (fp32 run:
                                                       # worst 3e-4 on the 4-element A_log with |d| 2e-6, loss identical to 1e-6)
    failures = []

    variants = {
        "ref": dict(share=False),
        "share": dict(share=True),
        "share+ckpt": dict(share=True, ckpt=True),
        "share+ckpt+head": dict(share=True, ckpt=True, head=True, ce_chunk=5),
        "share+keep": dict(share=True, keep=-1),
        "share+keep+pf2": dict(share=True, keep=-1, prefetch=2),
        "ref+ckpt+head": dict(share=False, ckpt=True, head=True, ce_chunk=5),
        "ref+keep": dict(share=False, keep=-1),
    }
    ref = None
    log(f"[smoke] ---- one-step gradient equivalence vs 'ref' (tol rel {tol:g}) ----")
    for name, spec in variants.items():
        model, pc = build_variant(a.seed, device, spec, dtype)
        t0 = time.time()
        loss, gn = run_step(model, pc, data[0], spec["share"])
        g = grads(model)
        FT.kernel_backends(model) if name == "ref" and rank == 0 else None
        if ref is None:
            ref = (loss, gn, g)
            log(f"[smoke] {name:18s} loss {loss:.6f} gnorm {gn:.6f} ({len(g)} grads, {time.time() - t0:.1f}s) | "
                f"backends {FT.kernel_backends(model)}")
        else:
            mx, rel, worst, n = compare(g, ref[2])
            ok = rel < tol and n == len(ref[2]) and abs(loss - ref[0]) < (1e-2 if dtype == torch.bfloat16 else 1e-4)
            log(f"[smoke] {name:18s} loss {loss:.6f} (ref {ref[0]:.6f}, diff {loss - ref[0]:+.2e}) gnorm {gn:.6f} | "
                f"grads vs ref: max|d| {mx:.2e} worst rel {rel:.2e} ({worst}) over {n} params | {time.time() - t0:.1f}s -> "
                f"{'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(name)
        del model, pc
        torch.cuda.empty_cache()

    log(f"[smoke] ---- {a.steps}-step training: params after every step; ref+keep must be BIT-IDENTICAL to ref (stale gathered "
        f"weights would show up as a step-1 loss jump), share+keep+ckpt at fp-noise level ----")
    runs = {}
    for name, spec in (("ref", dict(share=False)), ("ref+keep", dict(share=False, keep=-1)),
                       ("share+keep+ckpt", dict(share=True, keep=-1, ckpt=True))):
        model, pc = build_variant(a.seed, device, spec, dtype)
        opt = FT.make_optimizer("adamw", model.parameters(), 1e-3)
        hist = []
        for s in range(a.steps):
            loss, gn = run_step(model, pc, data[s], spec["share"], opt=opt)
            hist.append((loss, gn, params(model)))
        runs[name] = hist
        del model, pc, opt
        torch.cuda.empty_cache()
    for s in range(a.steps):
        (l0, g0, p0) = runs["ref"][s]
        (lk, gk, pk) = runs["ref+keep"][s]
        mx, rel, worst, n = compare(pk, p0)
        ok = mx == 0.0 and lk == l0
        log(f"[smoke] step {s}: ref+keep vs ref: loss {l0:.6f} vs {lk:.6f} (diff {lk - l0:+.2e}) | params max|d| {mx:.2e} "
            f"({worst}) -> {'OK (bit-identical)' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"keep-multistep{s}")
        (l1, g1, p1) = runs["share+keep+ckpt"][s]
        mx, rel, worst, n = compare(p1, p0)
        # Adam turns fp-noise-level gradient differences on near-zero-gradient elements into O(lr) parameter differences
        # (update ~ lr*sign(g)); the LOSS is the meaningful check: a stale-weight bug would show ~|loss(theta0)-loss(theta1)|
        ok = abs(l1 - l0) < (2e-3 if dtype == torch.bfloat16 else 1e-4)
        log(f"[smoke] step {s}: share+keep+ckpt vs ref: loss {l0:.6f} vs {l1:.6f} (diff {l1 - l0:+.2e}) | params max|d| {mx:.2e} "
            f"worst rel {rel:.2e} ({worst}) -> {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"multistep{s}")

    log("[smoke] ---- optimizers (one step each; adamw-fp32states must reproduce adamw) ----")
    base_params = None
    for optim in FT.OPTIMS:
        model, pc = build_variant(a.seed, device, dict(share=True), dtype)
        before = params(model)
        try:
            opt = FT.make_optimizer(optim, model.parameters(), 1e-3)
            t0 = time.time()
            loss, gn = run_step(model, pc, data[0], True, opt=opt)
            after = params(model)
            moved = max((after[n] - before[n]).abs().max().item() for n in after)
            extra = ""
            if optim == "adamw":
                base_params = after
            elif optim == "adamw-fp32states" and base_params is not None:
                mx, rel, worst, _ = compare(after, base_params)
                extra = f" | vs adamw: max|d| {mx:.2e} rel {rel:.2e} -> {'OK' if rel < 1e-6 else 'FAIL'}"
                if rel >= 1e-6:
                    failures.append("adamw-fp32states")
            elif base_params is not None:
                mx, rel, worst, _ = compare(after, base_params)
                extra = f" | vs adamw: max|d| {mx:.2e} rel {rel:.2e} (low-bit states: expected small, nonzero)"
            log(f"[smoke] {optim:16s} OK: loss {loss:.6f}, max param move {moved:.2e}, state {FT.optimizer_state_gb(opt) * 1024:.2f} MB, "
                f"{time.time() - t0:.1f}s{extra}")
        except Exception as e:  # noqa
            import traceback
            log(f"[smoke] {optim:16s} FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            failures.append(f"optim:{optim}")
        del model, pc
        torch.cuda.empty_cache()

    log(f"[smoke] DONE: {'ALL OK' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    dist.barrier()
    dist.destroy_process_group()
    if failures and rank == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
