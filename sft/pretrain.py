"""Stage 4: pretrain the generator on the (direction, target_text) firehose.

Conditioning vector for each record is row `vec_idx` of the memmap vec bank (probe directions in
READ_LAYER residual space); injected at INJECT_LAYER at the marker. Teacher-force the target.

    torchrun --standalone --nproc_per_node=8 scripts/pretrain.py --data-dir data/pretrain --epochs 1
"""
import argparse
import contextlib
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from mxf.config import D_MODEL, INJECT_LAYER, MODEL, STEER_COEFF, TrainConfig
from mxf.inject import FixedPositionInjector, get_layer, hooked, make_inject_hook, make_packed_inject_hook
from mxf.mfu import mfu


@contextlib.contextmanager
def autocast_region(model, enabled):
    """--autocast-bf16: PEFT input-dtype casting off + torch.autocast(bf16) around the policy forward (mirrors
    rl_disagg._policy_precision). Off = no-op, byte-identical to the legacy path."""
    if not enabled:
        yield
        return
    try:
        from peft.helpers import disable_input_dtype_casting
        cm = disable_input_dtype_casting(model)
    except ImportError:
        cm = contextlib.nullcontext()
    with cm, torch.autocast("cuda", dtype=torch.bfloat16):
        yield
from mxf.prompts import build_prompt_ids, build_sft_ids


def pack_examples(toks, pack_len, seed=0):
    """Greedy-pack (ids, labels, marker_pos, vec_idx) tuples end-to-end into blocks of <= pack_len
    tokens (example order shuffled once; an example that would overflow starts the next block, so
    examples are never split). Each block: ids/labels concat + per-example seg_lens, absolute
    marker positions, vec idxs."""
    order = np.random.default_rng(seed).permutation(len(toks))
    blocks, cur = [], None
    for i in order:
        ids, labs, pos, vidx = toks[i]
        if len(ids) > pack_len:
            continue
        if cur is None or len(cur["ids"]) + len(ids) > pack_len:
            if cur is not None:
                blocks.append(cur)
            cur = {"ids": [], "labels": [], "seg_lens": [], "markers": [], "vec_idxs": []}
        cur["markers"].append(len(cur["ids"]) + pos[0])
        cur["ids"] += ids
        cur["labels"] += labs
        cur["seg_lens"].append(len(ids))
        cur["vec_idxs"].append(vidx)
    if cur is not None and cur["seg_lens"]:
        blocks.append(cur)
    return blocks


def pack_batch(bblocks, pack_len, pad_id):
    """CPU tensors for a batch of packed blocks. seg = example index per token; tail pads get a
    unique seg id each (self-attention only, never attended by real tokens, labels -100).
    position_ids restart at 0 for every example. Returns per-marker (rows, cols) for injection."""
    B = len(bblocks)
    input_ids = torch.full((B, pack_len), pad_id, dtype=torch.long)
    labels = torch.full((B, pack_len), -100, dtype=torch.long)
    pos_ids = torch.zeros((B, pack_len), dtype=torch.long)
    seg = torch.arange(pack_len, dtype=torch.long).repeat(B, 1) + 1_000_000  # pads: isolated
    rows, cols, n_real = [], [], 0
    for b, blk in enumerate(bblocks):
        n = len(blk["ids"])
        n_real += n
        input_ids[b, :n] = torch.tensor(blk["ids"])
        labels[b, :n] = torch.tensor(blk["labels"])
        s = 0
        for j, sl in enumerate(blk["seg_lens"]):
            seg[b, s : s + sl] = j
            pos_ids[b, s : s + sl] = torch.arange(sl)
            s += sl
        rows += [b] * len(blk["markers"])
        cols += blk["markers"]
    return input_ids, labels, pos_ids, seg, torch.tensor(rows), torch.tensor(cols), n_real


def packed_attn_mask(seg, causal, dtype):
    """Additive [B,1,L,L] mask: 0 where (same example ∧ causal), finfo.min elsewhere. transformers
    returns already-4D masks as-is, so this reaches sdpa untouched."""
    allowed = (seg[:, None, :, None] == seg[:, None, None, :]) & causal
    zero = torch.zeros((), dtype=dtype, device=seg.device)
    neg = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=seg.device)
    return torch.where(allowed, zero, neg)


def main():
    cfg = TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/pretrain")
    ap.add_argument("--init-adapter", default=cfg.init_adapter)
    ap.add_argument("--save-dir", default=cfg.save_dir)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--max-seq", type=int, default=cfg.max_seq)
    ap.add_argument("--pack-len", type=int, default=0,
                    help="0 = per-example padded batches + compile (validated 57%% MFU, the default). "
                         ">0 packs into fixed blocks but REGRESSES on Blackwell (no flash-attn → dense "
                         "attn mask wastes off-block compute); only use with a block-sparse attn backend.")
    ap.add_argument("--pack-blocks", type=int, default=8,
                    help="packed blocks per device micro-batch (tokens/step = pack-blocks * pack-len)")
    ap.add_argument("--run-name", default=cfg.run_name)
    ap.add_argument("--compile", action="store_true", help="torch.compile the policy (test injection still fires)")
    ap.add_argument("--grad-ckpt", type=int, default=1,
                    help="1 = gradient checkpointing (legacy default; +33%% recompute). 0 = off -- a 178 GB B200 holds batch 16 x "
                         "192 tokens without it (the RL trainer runs mb 12-16 at 295 tokens with no checkpointing).")
    ap.add_argument("--autocast-bf16", action="store_true",
                    help="bf16 LoRA matmuls/activations under torch.autocast with PEFT's fp32 input-dtype casting disabled "
                         "(fp32 LoRA masters unchanged; HF's loss still upcasts logits to fp32). Same region as rl_disagg.")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="compute the shared prompt prefix (tokens before the marker) ONCE per micro-batch and run only "
                         "[marker]+target per example on top of the expanded cache (sft/prefix_cache.py). Exact incl. "
                         "gradients; needs the transformers fork ceselder/transformers@maemm-prefix-cache, --grad-ckpt 0 "
                         "and the per-example path (no --pack-len). Lets --batch-size go to 64-128 on one B200.")
    ap.add_argument("--prefix-accum", type=int, default=1,
                    help="with --prefix-cache: split each --batch-size batch into N micro-batches that SHARE one prefix "
                         "forward (token-weighted losses => identical to one big-batch mean-loss step). Amortizes the "
                         "B=1 prefix fwd+bwd (~225 ms/step on B200) and lowers peak memory: e.g. --batch-size 128 "
                         "--prefix-accum 2 fits one B200 where a single 128 micro-batch OOMs.")
    ap.add_argument("--compile-mlp", action="store_true",
                    help="regional torch.compile of the 64 MLP blocks only (cache objects never enter a graph, so it is "
                         "safe with --prefix-cache, unlike --compile which recompiles endlessly on the cache path).")
    ap.add_argument("--log-steps", type=int, default=20,
                    help="synchronize and report MFU every N steps (1 for trustworthy microbenchmarks)")
    ap.add_argument("--save-examples", default="",
                    help="comma-separated global example counts for scaling-curve checkpoints")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--n-ckpts", type=int, default=0,
                    help=">0: save this many evenly-spaced checkpoints (every 100/N %% of training); else every 2000 steps")
    ap.add_argument("--skip-steps", type=int, default=0,
                    help="crash-resume: fast-forward N optimizer steps (scheduler + step counter advance, no "
                         "compute). Batch order is deterministic (seeded shuffle over identical data/world), so "
                         "pairing with --init-adapter <save-dir>/step_{N-1} resumes exactly; AdamW moments reset.")
    ap.add_argument("--wandb-id", default="", help="crash-resume: continue this wandb run id (resume='allow')")
    a = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0)); is_main = rank == 0
    if world > 1:
        dist.init_process_group(os.environ.get("DDP_BACKEND", "nccl")); torch.cuda.set_device(local)
    device = f"cuda:{local}"

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    records = [json.loads(l) for l in open(f"{a.data_dir}/records.jsonl")]
    n_vecs = max(r["vec_idx"] for r in records) + 1
    vecs = np.memmap(f"{a.data_dir}/vecs.f32", dtype=np.float32, mode="r", shape=(n_vecs, D_MODEL))
    records = records[rank::world][: len(records) // world]  # equal shards: unequal lengths deadlock DDP on the last batch
    if is_main:
        print(f"{len(records)*world} records, {n_vecs} vectors, world={world}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa",  # flash-attn has no sm_103 build
                                                 device_map={"": device})
    model.enable_input_require_grads()
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, use_rslora=True,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    # 27B/64-layer OOMs on the 178GB B200 without activation checkpointing (~176GB resident at any
    # batch). enable_input_require_grads() above is the prerequisite; recompute activations in the
    # backward to fit. use_reentrant=False is required for LoRA/frozen-base + checkpointing.
    # use_reentrant=True: the layer-1 injection forward-hook modifies activations, which breaks
    # non-reentrant checkpointing's forward-vs-recompute tensor-count determinism check. Reentrant
    # mode re-runs the forward without that check and tolerates the hook.
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.train()
    try:
        import fla  # noqa
        _fla = "v" + str(getattr(fla, "__version__", "?"))
    except Exception:  # noqa
        _fla = "ABSENT (torch GDN fallback -- slow)"
    if is_main:
        print(f"[pretrain] grad_ckpt={a.grad_ckpt} autocast_bf16={a.autocast_bf16} compile={a.compile} fla={_fla} "
              f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())  # ~8.19B; LoRA adds <0.2%, fine for MFU
    submodule = get_layer(model, INJECT_LAYER)
    persistent_injector = persistent_handle = None
    if a.compile and not a.pack_len:
        # Install once before Dynamo's first trace. The buffer address remains stable while values
        # are copied in per batch, so injection stays in-graph without recompiling on every vector.
        _, _, fixed_positions = build_sft_ids(tok, "compile marker probe")
        assert len(fixed_positions) == 1
        persistent_injector = FixedPositionInjector(
            # prefix-cache path: the marker is index 0 of the suffix forward, not its absolute prompt index
            a.batch_size, D_MODEL, 0 if a.prefix_cache else fixed_positions[0], STEER_COEFF, device, torch.bfloat16
        )
        persistent_handle = submodule.register_forward_hook(persistent_injector.hook)
    if a.compile:
        model.forward = torch.compile(model.forward, dynamic=True if a.prefix_cache else None)
    ddp = DDP(model, device_ids=[local]) if world > 1 else model
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    prefix_cache = None
    if a.prefix_cache:
        try:
            from sft.prefix_cache import PrefixCache  # repo layout (needs the transformers fork; checked inside)
        except ImportError:  # mounted next to this file (modal_sft.py puts both under /pmx/SL/)
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from prefix_cache import PrefixCache
        assert not a.pack_len, "--prefix-cache is the per-example path (no --pack-len)"
        assert not a.grad_ckpt, "--prefix-cache needs --grad-ckpt 0 (GradientCheckpointingLayer drops past_key_values)"
        prompt_ids, mpos = build_prompt_ids(tok)
        # suffix forward through `ddp` (primes DDP's reducer once per step), prefix forward through the bare model
        prefix_cache = PrefixCache(ddp, prompt_ids, mpos[0], tok.pad_token_id, submodule, STEER_COEFF, device,
                                   prefix_model=model, persistent_injector=persistent_injector)
        assert a.prefix_accum >= 1 and a.batch_size % a.prefix_accum == 0, "--prefix-accum must divide --batch-size"
        if is_main:
            print(f"[pretrain] prefix-cache ON: shared prefix {prefix_cache.prefix_len} tokens, suffix = "
                  f"{len(prefix_cache.suffix_prompt)} prompt token(s) + target, prefix shared by {a.prefix_accum} "
                  f"micro-batch(es) of {a.batch_size // a.prefix_accum}", flush=True)
    if a.compile_mlp:
        assert not a.compile, "--compile-mlp and --compile are exclusive"
        try:
            from sft.prefix_cache import compile_mlp_blocks
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from prefix_cache import compile_mlp_blocks
        compile_mlp_blocks(model)

    # pre-tokenize once. Packed path: shuffle-once greedy packing into fixed pack-len blocks (zero
    # intra-block padding, one static shape for compile). Legacy path: length-bucketed padded batches.
    toks_cache = []
    for r in records:
        ids, labs, pos = build_sft_ids(tok, r["target_text"])
        toks_cache.append((ids[: a.max_seq], labs[: a.max_seq], pos, r["vec_idx"]))
    if a.pack_len:
        blocks = pack_examples(toks_cache, a.pack_len, seed=0)
        if world > 1:  # equalize block count across ranks (packing yields ±1 per rank → DDP deadlock)
            t = torch.tensor([len(blocks)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            blocks = blocks[: int(t.item())]
        bper = len(blocks) // a.pack_blocks  # drop remainder batch: keeps a single static shape
        steps_total = bper * a.epochs
        causal = torch.tril(torch.ones(a.pack_len, a.pack_len, dtype=torch.bool, device=device))
        if is_main:
            fill = sum(len(b["ids"]) for b in blocks) / (len(blocks) * a.pack_len)
            print(f"packed: {len(blocks)} blocks of {a.pack_len} (fill {fill:.1%}), "
                  f"{bper} steps/epoch x {a.pack_blocks} blocks", flush=True)
    else:
        toks_cache.sort(key=lambda t: len(t[0]))
        steps_total = math.ceil(len(toks_cache) / a.batch_size) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps_total,
                                                pct_start=cfg.warmup_frac, anneal_strategy="linear")
    save_every = max(1, steps_total // a.n_ckpts) if a.n_ckpts else 2000
    if is_main:
        print(f"steps_total {steps_total}, checkpoint every {save_every} steps", flush=True)
    if is_main and not a.no_wandb:
        wandb.init(project="maxact-fast", name=a.run_name, config=vars(a),
                   id=a.wandb_id or None, resume="allow" if a.wandb_id else None)
    os.makedirs(a.save_dir, exist_ok=True)
    save_examples = sorted({int(x) for x in a.save_examples.split(",") if x.strip()})
    saved_examples = set()

    step = 0
    for ep in range(a.epochs):
        if a.pack_len:
            order = np.random.default_rng(ep).permutation(len(blocks))
            batches = [[blocks[i] for i in order[s : s + a.pack_blocks]]
                       for s in range(0, bper * a.pack_blocks, a.pack_blocks)]
        else:
            batches = [toks_cache[s : s + a.batch_size] for s in range(0, len(toks_cache), a.batch_size)]
            np.random.default_rng(ep).shuffle(batches)
        for batch in batches:
            if step < a.skip_steps:  # crash-resume fast-forward (see --skip-steps help)
                sched.step(); step += 1
                continue
            t0 = time.time()
            backward_done = False   # --prefix-accum runs its backward(s) inside the branch
            if a.pack_len:
                input_ids, labels, pos_ids, seg, rows, cols, n_real = pack_batch(
                    batch, a.pack_len, tok.pad_token_id)
                mask4 = packed_attn_mask(seg.to(device), causal, torch.bfloat16)
                vmat = torch.from_numpy(np.asarray(vecs[[v for blk in batch for v in blk["vec_idxs"]]]))
                hook = make_packed_inject_hook(vmat, rows, cols, STEER_COEFF, device, torch.bfloat16)
                with hooked(submodule, hook), autocast_region(model, a.autocast_bf16):
                    out = ddp(input_ids=input_ids.to(device), attention_mask=mask4,
                              position_ids=pos_ids.to(device), labels=labels.to(device),
                              use_cache=False)
            elif prefix_cache is not None:
                # toks_cache rows are prompt+target ids (truncated to max_seq): the target is everything after the
                # prompt. vecs: one memmap gather. n_real = tokens actually run (shared prefix once + suffixes).
                n_prompt = len(prompt_ids)
                targets = [t[0][n_prompt:] for t in batch]
                vmat = torch.from_numpy(np.asarray(vecs[[t[3] for t in batch]]))
                ac = lambda: autocast_region(model, a.autocast_bf16)  # noqa: E731
                if a.prefix_accum == 1:
                    out = prefix_cache.forward(vmat, targets, autocast=ac)
                    n_real = prefix_cache.prefix_len + int(out.suffix_mask.sum())
                else:
                    # one prefix forward, N micro-batches on copy-expanded caches; loss_k * (n_k / N_tokens) summed
                    # == the single mean-over-target-tokens loss, so the update is identical to one big micro-batch
                    mb = len(batch) // a.prefix_accum
                    n_tot = sum(len(t) for t in targets)
                    cache0 = prefix_cache.run_prefix(ac)
                    loss_sum = 0.0; n_real = prefix_cache.prefix_len
                    for k in range(a.prefix_accum):
                        sl = slice(k * mb, (k + 1) * mb)
                        last = k == a.prefix_accum - 1
                        # DDP: all-reduce only on the last micro-batch (no_sync must wrap the forward too)
                        with (ddp.no_sync() if (world > 1 and not last) else contextlib.nullcontext()):
                            o = prefix_cache.forward(vmat[sl], targets[sl], autocast=ac, prefix_cache=cache0)
                            (o.loss * (o.n_target_tokens / n_tot)).backward(retain_graph=not last)
                        loss_sum += o.loss.item() * o.n_target_tokens / n_tot
                        n_real += int(o.suffix_mask.sum())
                        del o
                    del cache0
                    out = SimpleNamespace(loss=torch.tensor(loss_sum))
                    backward_done = True
            else:
                L = max(len(t[0]) for t in batch)
                L = min(((L + 63) // 64) * 64, a.max_seq)  # round to mult-of-64 → ≤3 static shapes for compile
                input_ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
                labels = torch.full((len(batch), L), -100, dtype=torch.long)
                attn = torch.zeros((len(batch), L), dtype=torch.bool)
                pos = batch[0][2]
                for i, (ii, ll, _, _) in enumerate(batch):
                    input_ids[i, : len(ii)] = torch.tensor(ii)
                    labels[i, : len(ll)] = torch.tensor(ll)
                    attn[i, : len(ii)] = True
                n_real = int(attn.sum())
                if persistent_injector is not None:
                    # One memmap gather + H2D copy, versus one tiny transfer per row in the legacy
                    # hook. The registered hook reads this stable buffer inside the compiled graph.
                    vmat = torch.from_numpy(np.asarray(vecs[[t[3] for t in batch]])).to(device)
                    persistent_injector.set_vectors(vmat)
                    with autocast_region(model, a.autocast_bf16):
                        out = ddp(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                                  labels=labels.to(device))
                else:
                    vlist = [torch.from_numpy(np.asarray(vecs[t[3]])).unsqueeze(0) for t in batch]
                    hook = make_inject_hook(vlist, [pos] * len(batch), STEER_COEFF, device, torch.bfloat16)
                    with hooked(submodule, hook), autocast_region(model, a.autocast_bf16):
                        out = ddp(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                                  labels=labels.to(device))
            if not backward_done:
                out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if is_main and step % a.log_steps == 0:
                torch.cuda.synchronize()
                tfl, m = mfu(n_real, time.time() - t0, n_params, fwd_bwd=True)
                print(f"ep{ep} step {step}/{steps_total} loss {out.loss.item():.4f} | "
                      f"{tfl:.0f} TFLOP/s MFU {m:.0%}", flush=True)
                if not a.no_wandb:
                    wandb.log({"loss": out.loss.item(), "lr": sched.get_last_lr()[0],
                               "mfu": m, "tflops": tfl}, step=step)
            if is_main and step % save_every == 0 and step:
                model.save_pretrained(f"{a.save_dir}/step_{step}")
            global_seen = min((step + 1) * a.batch_size * world, len(records) * world)
            for requested in save_examples:
                if requested <= global_seen and requested not in saved_examples:
                    path = f"{a.save_dir}/examples_{requested}"
                    if is_main:
                        model.save_pretrained(path)
                        json.dump({"requested_examples": requested, "actual_examples": global_seen,
                                   "step": step + 1, "world_size": world,
                                   "batch_size_per_rank": a.batch_size},
                                  open(f"{path}/training_progress.json", "w"), indent=2)
                        print(f"SAVED_SCALING_POINT requested={requested} actual={global_seen} "
                              f"step={step + 1}", flush=True)
                    saved_examples.add(requested)
            step += 1
    if is_main:
        model.save_pretrained(f"{a.save_dir}/final")
        print("PRETRAIN_DONE", flush=True)
    if persistent_handle is not None:
        persistent_handle.remove()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
