"""Trainer-only bench / hang-hunt of the --full-param update knobs on Y GPUs (no vLLM, synthetic rollouts, no optimizer state):
for every config in --configs, install/toggle the knobs on ONE FSDP2 policy and time rl_disagg.update_disagg on the same
synthetic batch (uneven shards across ranks so the dummy-micro-batch equalization runs). Rank 0 prints one JSON line per
config as soon as it finishes -- a hanging config is the first one without a line.

    torchrun --standalone --nproc_per_node=2 RL/fullparam_update_bench.py --policy-base <dir> --configs base,head,ckpt,ckpt+head,ckpt+head@24 \
        --rollouts 48 --uneven 8 [--fsdp-prefetch 2]
config grammar: knobs joined by '+' (base | head = --chunked-head | ckpt = --suffix-ckpt) optionally '@<micro-batch>' (default 8).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-base", required=True)
    ap.add_argument("--configs", default="base,head,ckpt,ckpt+head,ckpt+head@24")
    ap.add_argument("--rollouts", type=int, default=48, help="synthetic rollouts on every rank")
    ap.add_argument("--uneven", type=int, default=8, help="extra rollouts on rank 0 (dummy micro-batches on the others)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--fsdp-prefetch", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--vocab-chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    b = ap.parse_args()
    for p in ("/pmx/helpers", "/pmx/eval", "/pmx/RL"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    import rl_disagg as D
    import rl_fullparam as FP
    import rl_hf as R
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mxf.config import D_MODEL, INJECT_LAYER, MODEL
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    dist.init_process_group("cpu:gloo,cuda:nccl")
    tag = f"B{rank}"

    def log(m):
        print(f"[{tag}] {m}", flush=True)
    a = D.parse_args(["--role", "trainer", "--n-rollout", "1", "--n-trainer", str(world), "--full-param", "--init-adapter", "none",
                      "--policy-base", b.policy_base, "--prefix-cache", "--fp32-head", "--recipe", "scalerl", "--loss", "cispo", "--cispo-eps-max", "5",
                      "--loss-agg", "prompt", "--zero-var-filter", "--kl-coef", "0", "--group-size", "8", "--groups-per-step", str(8 * world),
                      "--max-new-tokens", str(b.max_new_tokens), "--vocab-chunk", str(b.vocab_chunk), "--fsdp-prefetch", str(b.fsdp_prefetch),
                      "--suffix-ckpt", "--chunked-head", "--no-wandb"])
    FT = FP.fullft_module()
    R.read_resid = FP.read_resid_noraise
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    t0 = time.time()
    actor = AutoModelForCausalLM.from_pretrained(b.policy_base, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    actor = FT.shard_full_model(actor, world, device, log=log if rank == 0 else (lambda *x, **k: None), prefetch=b.fsdp_prefetch, device_type="cuda")
    actor.train()
    fp = D.FullParamCtx(FP, FT, world, rank)
    submodule = get_layer(actor, INJECT_LAYER)
    pfx = D.PrefixRunner(actor, prompt_ids, marker, device)
    head = FP.ChunkedHead(actor)          # inactive unless fp.head is set for the config
    ckpt = FP.suffix_checkpointer(actor)  # inactive unless fp.ckpt is set for the config
    fp32_hook = [None]

    class NoOpt:   # no AdamW state on a 2-4 GPU bench: zero_grad + a step that leaves the masters alone
        param_groups = [{"lr": 0.0}]

        def zero_grad(self, set_to_none=True):
            for p in actor.parameters():
                p.grad = None

        def step(self):
            pass
    opt = NoOpt()
    log(f"policy sharded over {world} ranks in {time.time() - t0:.0f}s | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB | prefetch {b.fsdp_prefetch}")

    # synthetic rollouts: uneven shards -> the dummy-micro-batch equalization runs on the smaller ranks
    g = torch.Generator().manual_seed(b.seed + 17 * rank)
    n = b.rollouts + (b.uneven if rank == 0 else 0)
    G = a.group_size
    n = -(-n // G) * G
    lens = torch.randint(8, b.max_new_tokens + 1, (n,), generator=g)
    T = int(lens.max())
    ids = torch.full((n, p_len + T), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros_like(ids)
    for i in range(n):
        ids[i, :p_len] = torch.tensor(prompt_ids)
        ids[i, p_len : p_len + int(lens[i])] = torch.randint(1000, 100000, (int(lens[i]),), generator=g)
        attn[i, : p_len + int(lens[i])] = 1
    old_lp = torch.full((n, T), -2.5)
    known = attn[:, p_len:].bool()
    adv = torch.randn(n, generator=g)
    dirs_rep = torch.nn.functional.normalize(torch.randn(n // G, D_MODEL, generator=g), dim=-1).repeat_interleave(G, 0).to(device)
    log(f"synthetic batch: {n} rollouts (lens {int(lens.min())}..{T}, mean {float(lens.float().mean()):.0f})")

    results = {}
    for cfg in [c for c in b.configs.split(",") if c]:
        knobs, mb = (cfg.split("@") + ["8"])[:2]
        mb = int(mb)
        knobs = set(knobs.split("+")) - {"base"}
        fp.head = head if "head" in knobs else None
        fp.ckpt = ckpt if "ckpt" in knobs else None
        if fp.head is None and fp32_hook[0] is None:
            fp32_hook[0] = FP.install_fp32_head_trainable(actor)
        if fp.head is not None and fp32_hook[0] is not None:
            fp32_hook[0].remove(); fp32_hook[0] = None
        times, peaks, stats = [], [], None
        for rep in range(b.reps):
            dist.barrier()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            stats = D.update_disagg(actor, opt, submodule, ids, attn, p_len, marker, old_lp, known, adv, dirs_rep, a, device, mb, keep=None, pfx=pfx, fp=fp)
            torch.cuda.synchronize()
            times.append(time.time() - t1)
            peaks.append(torch.cuda.max_memory_allocated() / 2**30)
            log(f"{cfg}: rep {rep} {times[-1]:.1f}s peak {peaks[-1]:.1f} GB | fb {stats['t_fb']:.1f}s gnorm {stats['grad_norm']:.3f} dlogp {stats['sampler_abs_dlogp']:.4f}")
        row = {"config": cfg, "mb": mb, "knobs": sorted(knobs), "prefetch": b.fsdp_prefetch, "world": world, "n_rollouts_rank0": n,
               "update_s": times, "update_s_last": times[-1], "peak_gb": max(peaks), "grad_norm": stats["grad_norm"],
               "tok_per_s_rank": float(lens.sum()) / times[-1], "n_micro_batches": -(-n // mb)}
        results[cfg] = row
        if rank == 0:
            print("RESULT_JSON " + json.dumps(row), flush=True)
    if rank == 0:
        print("ALL_RESULTS_JSON " + json.dumps(results), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
