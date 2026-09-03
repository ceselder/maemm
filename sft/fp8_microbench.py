"""fp8_microbench.py -- WHY is/isn't --fp8-base faster? Three levels on one GPU, no data bank needed.

  1. raw GEMMs at Qwen3.6-27B's linear shapes (M = tokens/step): bf16 torch.mm vs torch._scaled_mm with pre-cast fp8
     operands, tensorwise and rowwise scales, fast_accum on/off, in both the forward (x @ W^T) and grad_input (g @ W)
     layouts -> TFLOP/s per kernel. This is the ceiling fp8 can give.
  2. one frozen linear, fwd+bwd, nn.Linear(bf16) vs FrozenBaseFloat8Linear (sft/fp8.py), eager and torch.compile'd ->
     includes the per-step activation/grad_output casts that the raw GEMM number hides.
  3. --profile-model: the real 27B + LoRA + injector + compile, synthetic batch (B x L tokens), torch.profiler over 3 train
     steps in bf16 and then (after convert_frozen_base_to_fp8) fp8 -> top CUDA kernels + GEMM-vs-rest split of a step.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHAPES = [  # (K, N) = (in_features, out_features), count in the model
    (5120, 17408, 128), (17408, 5120, 64), (5120, 10240, 48), (5120, 6144, 48), (6144, 5120, 64),
    (5120, 12288, 16), (5120, 1024, 32),
]


def timeit(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def raw_gemms(M, out):
    e4m3 = torch.float8_e4m3fn
    rows = []
    for K, N, cnt in SHAPES:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        flops = 2.0 * M * K * N
        r = {"K": K, "N": N, "count": cnt, "M": M}
        # bf16 references (cuBLAS handles any layout)
        r["bf16_fwd_tflops"] = flops / timeit(lambda: x @ W.t()) / 1e12
        r["bf16_bwd_tflops"] = flops / timeit(lambda: g @ W) / 1e12
        # fp8 operands pre-cast (pure GEMM time). fwd: a=x[M,K] row-major, b=W^T[K,N] col-major (= W contiguous .t())
        x8 = x.to(e4m3); W8 = W.to(e4m3); g8 = g.to(e4m3); Wt8 = W.t().contiguous().to(e4m3)  # [K,N] contiguous
        one = torch.ones((), device="cuda")
        sx = torch.ones(M, 1, device="cuda"); sw_n = torch.ones(1, N, device="cuda")
        sg = torch.ones(M, 1, device="cuda"); sw_k = torch.ones(1, K, device="cuda")
        for tag, a, b, sa, sb in (("fwd", x8, W8.t(), sx, sw_n), ("bwd", g8, Wt8.t(), sg, sw_k)):
            for fa in (True, False):
                try:
                    t = timeit(lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16,
                                                        use_fast_accum=fa))
                    r[f"fp8_tensorwise_{tag}_fa{int(fa)}_tflops"] = flops / t / 1e12
                except Exception as e:  # noqa
                    r[f"fp8_tensorwise_{tag}_fa{int(fa)}_tflops"] = f"ERR {type(e).__name__}: {str(e)[:80]}"
                try:
                    t = timeit(lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16,
                                                        use_fast_accum=fa))
                    r[f"fp8_rowwise_{tag}_fa{int(fa)}_tflops"] = flops / t / 1e12
                except Exception as e:  # noqa
                    r[f"fp8_rowwise_{tag}_fa{int(fa)}_tflops"] = f"ERR {type(e).__name__}: {str(e)[:80]}"
        rows.append(r)
        fmt = lambda v: f"{v:7.0f}" if isinstance(v, float) else str(v)[:12]
        print(f"[gemm] M={M} K={K:5d} N={N:5d} x{cnt:3d} | bf16 fwd {fmt(r['bf16_fwd_tflops'])} bwd {fmt(r['bf16_bwd_tflops'])} | "
              f"fp8 tensorwise fwd fa1 {fmt(r['fp8_tensorwise_fwd_fa1_tflops'])} fa0 {fmt(r['fp8_tensorwise_fwd_fa0_tflops'])} "
              f"bwd fa0 {fmt(r['fp8_tensorwise_bwd_fa0_tflops'])} | rowwise fwd fa1 {fmt(r['fp8_rowwise_fwd_fa1_tflops'])} "
              f"fa0 {fmt(r['fp8_rowwise_fwd_fa0_tflops'])} bwd fa0 {fmt(r['fp8_rowwise_bwd_fa0_tflops'])} TFLOP/s", flush=True)
    out["raw_gemms"] = rows


def layer_level(M, out):
    import torch.nn as nn
    from torchao.float8 import Float8LinearConfig
    import fp8
    rows = []
    for K, N, cnt in SHAPES[:3]:
        lin = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
        lin.weight.requires_grad_(False)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        r = {"K": K, "N": N, "count": cnt}

        def fb(mod):
            def f():
                y = mod(x)
                y.backward(y)  # grad_output = y (any tensor of the right shape)
                x.grad = None
            return f
        r["bf16_eager_ms"] = timeit(fb(lin)) * 1e3
        r["bf16_compiled_ms"] = timeit(fb(torch.compile(lin))) * 1e3
        for recipe in ("rowwise", "tensorwise"):
            f8 = fp8.FrozenBaseFloat8Linear.from_frozen(lin, Float8LinearConfig.from_recipe_name(recipe))
            r[f"fp8_{recipe}_eager_ms"] = timeit(fb(f8)) * 1e3
            torch._dynamo.reset()
            r[f"fp8_{recipe}_compiled_ms"] = timeit(fb(torch.compile(f8))) * 1e3
        rows.append(r)
        print(f"[layer] M={M} K={K} N={N} x{cnt} fwd+bwd ms: bf16 eager {r['bf16_eager_ms']:.3f} compiled {r['bf16_compiled_ms']:.3f} | "
              f"fp8 rowwise eager {r['fp8_rowwise_eager_ms']:.3f} compiled {r['fp8_rowwise_compiled_ms']:.3f} | "
              f"tensorwise eager {r['fp8_tensorwise_eager_ms']:.3f} compiled {r['fp8_tensorwise_compiled_ms']:.3f}", flush=True)
    out["layer_level"] = rows


def profile_model(B, L, recipe, out, use_cache_false=False, recompile_limit=0, tag=""):
    """One model load: compiled bf16 step profile, then convert to fp8 `recipe`, compiled fp8 step profile.
    use_cache_false: pass use_cache=False (pretrain.py's padded path doesn't -> HF builds a DynamicCache per step).
    recompile_limit: torch._dynamo.config.recompile_limit for both phases (0 = torch default 8; fp8.py raises it to 64
    itself at conversion, so the fp8 phase always has >= 64)."""
    import gc

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mxf.config import D_MODEL, INJECT_LAYER, MODEL, STEER_COEFF, TrainConfig
    from mxf.inject import FixedPositionInjector, get_layer
    from mxf.prompts import build_sft_ids
    from pretrain import autocast_region
    import fp8

    torch._dynamo.config.recompile_limit = recompile_limit or 8
    torch._dynamo.config.cache_size_limit = recompile_limit or 8
    cfg = TrainConfig()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": "cuda:0"})
    model.enable_input_require_grads()
    torch.manual_seed(0)
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, use_rslora=True,
                                            target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.train()
    _, _, pos = build_sft_ids(tok, "compile marker probe")
    inj = FixedPositionInjector(B, D_MODEL, pos[0], STEER_COEFF, "cuda:0", torch.bfloat16)
    get_layer(model, INJECT_LAYER).register_forward_hook(inj.hook)
    inj.set_vectors(torch.randn(B, D_MODEL, device="cuda"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-5)
    ids = torch.randint(1000, 50000, (B, L), device="cuda")
    attn = torch.ones(B, L, dtype=torch.bool, device="cuda")
    orig_forward = model.forward

    kw = {"use_cache": False} if use_cache_false else {}

    def step():
        with autocast_region(model, True):
            o = model(input_ids=ids, attention_mask=attn, labels=ids, **kw)
        o.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad()

    def run(name):
        torch._dynamo.reset()
        model.forward = torch.compile(orig_forward)
        t0 = time.time()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        compile_s = time.time() - t0
        t = timeit(step, iters=5, warm=1) * 1e3
        torch.cuda.reset_peak_memory_stats()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        ka = prof.key_averages()
        total = sum(k.device_time_total for k in ka) / 3
        gemm_kw = ("gemm", "cutlass", "nvjet", "scaled_mm", "cublas", "Cijk", "xmma", "matmul")
        by_name = sorted(((k.device_time_total / 3, k.key) for k in ka), reverse=True)
        gemm = sum(t_ for t_, n in by_name if any(s in n.lower() for s in gemm_kw))
        fla = sum(t_ for t_, n in by_name if any(s in n for s in ("chunk_", "recompute_w_u", "prepare_wy", "causal_conv", "conv_depthwise")))
        triton = sum(t_ for t_, n in by_name if n.startswith("triton_"))
        aten_eager = sum(t_ for t_, n in by_name if "at::native" in n)
        top = [(n[:110], round(t_ / 1e3, 2)) for t_, n in by_name[:30]]
        r = {"ms_per_step_timed": t, "compile_plus_3_steps_s": compile_s, "peak_alloc_gb": peak,
             "profiled_cuda_ms_per_step": total / 1e3, "gemm_like_ms": gemm / 1e3, "gdn_conv_ms": fla / 1e3,
             "inductor_triton_ms": triton / 1e3, "aten_eager_ms": aten_eager / 1e3, "top_kernels_ms": top}
        print(f"[profile] {tag}{name}: {t:.0f} ms/step (timed) | CUDA {total/1e3:.0f} ms/step = GEMM {gemm/1e3:.0f} + "
              f"GDN/conv {fla/1e3:.0f} + inductor-triton {triton/1e3:.0f} + eager-ATen {aten_eager/1e3:.0f} + other "
              f"{(total-gemm-fla-triton-aten_eager)/1e3:.0f} ms | peak {peak:.1f} GB | compile+3 steps {compile_s:.0f}s", flush=True)
        for n, ms in top[:12]:
            print(f"    {ms:8.2f} ms  {n}")
        return r

    res = {"B": B, "L": L, "use_cache_false": use_cache_false, "recompile_limit": recompile_limit or 8,
           "bf16": run("bf16")}
    res["fp8_convert"] = fp8.convert_frozen_base_to_fp8(model, recipe=recipe)
    res[f"fp8_{recipe}"] = run(f"fp8_{recipe}")
    out.setdefault("profile", {})[tag or "default"] = res
    model = opt = trainable = None  # noqa: F841 -- drop the 27B (the closures above hold the only other refs) for the next variant
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=3072, help="M for the raw/layer benches (= batch x padded L)")
    ap.add_argument("--profile-model", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--recipe", default="rowwise")
    ap.add_argument("--variants", default="default",
                    help="comma list of <cache>[+limitN][:recipe]: e.g. default,nocache,nocache+limit64,nocache+limit64:tensorwise")
    ap.add_argument("--skip-gemm", action="store_true")
    ap.add_argument("--out", default="/tmp/fp8_microbench.json")
    a = ap.parse_args()
    out = {"args": vars(a), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    if not a.skip_gemm:
        raw_gemms(a.tokens, out)
        layer_level(a.tokens, out)
        json.dump(out, open(a.out, "w"), indent=1)
    if a.profile_model:
        for v in a.variants.split(","):
            spec, _, recipe = v.partition(":")
            lim = 0
            for part in spec.split("+"):
                if part.startswith("limit"):
                    lim = int(part[5:])
            profile_model(a.batch, a.seq, recipe or a.recipe, out, use_cache_false="nocache" in spec,
                          recompile_limit=lim, tag=v + " ")
            json.dump(out, open(a.out, "w"), indent=1)
    json.dump(out, open(a.out, "w"), indent=1)
    print("FP8_MICROBENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
