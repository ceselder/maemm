"""Equivalence + speed test for sft/prefix_cache.py on ONE GPU via Modal (ephemeral app `maemm-prefix-test`).

Image = sft/modal_sft.py's image with transformers replaced by the prefix-cache fork (PFX_TF_SPEC, default the
pinned fork commit) + flash-linear-attention==0.5.2. The real Qwen3.6-27B is loaded from the maemm-data volume's
HF cache with the LoRA adapter at /data/sft_mix/last5_rp/final (or a fresh rsLoRA with --fresh-lora).

    source ~/modal_venv/bin/activate; export MODAL_PROFILE=safety-sahan
    modal run sft/test_prefix_cache.py::equiv --batch 8 --seed 0          # naive vs prefix-cached, autocast off+on
    modal run sft/test_prefix_cache.py::bench --steps 10 --compile 1      # ex/s table naive mb16 vs cached 16/64/128
    PFX_GPU=H200:1 PFX_TRITON=3.7.1 modal run ...                           # Hopper fallback (fla GDN bwd needs triton>=3.7.1)

Results are printed and also written locally to sft/results/prefix_cache_{equiv,bench}.json by the entrypoints.
"""

import json
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
FORK_SHA = "e52940e567ab9a991a1c971c1094e340233baff3"
TF_SPEC = os.environ.get("PFX_TF_SPEC", f"transformers @ git+https://github.com/ceselder/transformers@{FORK_SHA}")

app = modal.App("maemm-prefix-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        TF_SPEC,
        "peft==0.20.0",
        "accelerate==1.14.0",
        "wandb==0.28.2",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    .pip_install("flash-linear-attention==0.5.2")
)
_TRITON = os.environ.get("PFX_TRITON", "")
if _TRITON:
    image = image.pip_install(f"triton=={_TRITON}")
image = (
    image.env({"HF_HOME": "/data/hf_cache", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONPATH": "/pmx/helpers:/pmx"})
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "sft" / "prefix_cache.py", "/pmx/sft/prefix_cache.py")
    .add_local_file(REPO / "sft" / "pretrain.py", "/pmx/sft/pretrain.py")
)

vol = modal.Volume.from_name("maemm-data")
GPU = os.environ.get("PFX_GPU", "B200:1")
ADAPTER = "/data/sft_mix/last5_rp/final"


# ----------------------------------------------------------------------------------------------------------------
# container-side helpers
# ----------------------------------------------------------------------------------------------------------------
def _load(adapter: str, fresh_lora: bool):
    import sys
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/pmx/helpers"); sys.path.insert(0, "/pmx")
    from mxf.config import MODEL, TrainConfig

    cfg = TrainConfig()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": "cuda:0"})
    model.enable_input_require_grads()
    if fresh_lora or not os.path.exists(adapter):
        print(f"[load] fresh rsLoRA r{cfg.lora_r}/a{cfg.lora_alpha} (adapter={'absent ' + adapter if not fresh_lora else 'not requested'})", flush=True)
        model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, use_rslora=True,
                                                 target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    else:
        print(f"[load] adapter {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    model.train()
    return tok, model


def _env_report():
    import torch, transformers, peft
    try:
        import fla
        fla_v = getattr(fla, "__version__", "?")
    except Exception as e:  # noqa
        fla_v = f"ABSENT ({e})"
    from transformers import cache_utils
    fork = hasattr(cache_utils, "_write_cached_state") and hasattr(cache_utils.LinearAttentionLayer, "batch_repeat_interleave")
    import triton
    # str(): torch.__version__ is a TorchVersion str-subclass, which the torch-less local client cannot unpickle
    rep = {"gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__), "transformers": str(transformers.__version__),
           "transformers_fork": bool(fork), "peft": str(peft.__version__), "fla": str(fla_v), "triton": str(triton.__version__),
           "tf_spec": TF_SPEC}
    print("[env]", json.dumps(rep), flush=True)
    return rep


def _plain(results):
    """Round-trip through JSON so the return value holds only builtins (the local client has no torch)."""
    out = json.loads(json.dumps(results, default=str))
    print("RESULTS_JSON " + json.dumps(out), flush=True)   # log fallback if the client-side deserialization fails
    return out


def _random_batch(gen, B, d_model, vocab, min_tgt, max_tgt, eos):
    import torch
    vecs = torch.nn.functional.normalize(torch.randn(B, d_model, generator=gen), dim=-1)
    lens = torch.randint(min_tgt, max_tgt + 1, (B,), generator=gen).tolist()
    targets = [torch.randint(1000, min(vocab, 150_000), (n,), generator=gen).tolist() + [eos] for n in lens]
    return vecs, targets


def _naive_forward(model, prompt_ids, marker, vecs, targets, submodule, autocast_on, device, max_seq=192, use_cache=None):
    """Exactly sft/pretrain.py's per-example padded path (L rounded to 64, <= max_seq; hook at the absolute marker)."""
    import torch
    from mxf.config import STEER_COEFF
    from mxf.inject import hooked, make_inject_hook
    from sft.pretrain import autocast_region

    rows = [prompt_ids + list(t) for t in targets]
    B = len(rows)
    L = max(len(r) for r in rows)
    L = min(((L + 63) // 64) * 64, max_seq)
    ids = torch.full((B, L), 0, dtype=torch.long); labels = torch.full((B, L), -100, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.bool)
    for i, r in enumerate(rows):
        r = r[:L]
        ids[i, : len(r)] = torch.tensor(r); attn[i, : len(r)] = True
        labels[i, len(prompt_ids) : len(r)] = torch.tensor(r[len(prompt_ids):])
    hook = make_inject_hook([v[None] for v in vecs], [[marker]] * B, STEER_COEFF, device, torch.bfloat16)
    kw = {} if use_cache is None else {"use_cache": use_cache}
    with hooked(submodule, hook), autocast_region(model, autocast_on):
        out = model(input_ids=ids.to(device), attention_mask=attn.to(device), labels=labels.to(device), **kw)
    return out, labels


def _flat_grads(params):
    import torch
    return torch.cat([p.grad.detach().flatten().float() for p in params if p.grad is not None])


def _cmp_logits(a, b):
    """a, b: [N, V] logits at matched positions. Returns dict of diffs (fp32 math)."""
    import torch
    a = a.float(); b = b.float()
    d = (a - b).abs()
    la = torch.log_softmax(a, -1); lb = torch.log_softmax(b, -1)
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "max_rel_to_scale": (d.max() / a.abs().max()).item(),
            "logprob_max_abs": (la - lb).abs().max().item(), "logprob_mean_abs": (la - lb).abs().mean().item(),
            "argmax_agree": (a.argmax(-1) == b.argmax(-1)).float().mean().item(), "n_positions": a.shape[0]}


def _cmp_grads(g1, g2):
    import torch
    cos = torch.nn.functional.cosine_similarity(g1[None], g2[None]).item()
    rel = ((g1 - g2).norm() / g1.norm()).item()
    return {"cosine": cos, "rel_l2": rel, "max_abs": (g1 - g2).abs().max().item(), "norm_ref": g1.norm().item(),
            "norm_other": g2.norm().item()}


# ----------------------------------------------------------------------------------------------------------------
# equivalence
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=5400)
def equiv_remote(adapter: str = ADAPTER, batch: int = 8, seed: int = 0, min_tgt: int = 8, max_tgt: int = 32,
                 fresh_lora: bool = False, stock_demo: bool = False):
    import sys
    import time
    import torch

    sys.path.insert(0, "/pmx/helpers"); sys.path.insert(0, "/pmx")
    env = _env_report()
    from mxf.config import D_MODEL, INJECT_LAYER, STEER_COEFF
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids
    from sft import prefix_cache as pcmod
    from sft.pretrain import autocast_region

    if stock_demo:  # show what stock transformers does on this path (image built with PFX_TF_SPEC=transformers==5.15.0)
        pcmod.check_transformers = lambda: None

    t0 = time.time()
    tok, model = _load(adapter, fresh_lora)
    print(f"[load] {time.time() - t0:.0f}s", flush=True)
    device = "cuda:0"
    prompt_ids, mpos = build_prompt_ids(tok)
    marker = mpos[0]
    print(f"[prompt] len={len(prompt_ids)} marker={marker} tail_after_marker={len(prompt_ids) - marker - 1}", flush=True)
    submodule = get_layer(model, INJECT_LAYER)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"[model] trainable params {n_train / 1e6:.1f}M", flush=True)

    gen = torch.Generator().manual_seed(seed)
    vecs, targets = _random_batch(gen, batch, D_MODEL, tok.vocab_size, min_tgt, max_tgt, tok.eos_token_id)
    print(f"[batch] B={batch} target lens {[len(t) for t in targets]}", flush=True)
    results = {"env": env, "prompt_len": len(prompt_ids), "marker": marker, "batch": batch, "seed": seed,
               "target_lens": [len(t) for t in targets], "trainable_params": n_train, "runs": {}}

    pc = pcmod.PrefixCache(model, prompt_ids, marker, tok.pad_token_id, submodule, STEER_COEFF, device)
    suffix_lens = [len(pc.suffix_prompt) + len(t) for t in targets]

    for autocast_on in (False, True):
        tag = f"autocast_bf16={'on' if autocast_on else 'off'}"
        r = {}
        # ---- naive (reference) ----
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.time()
        out_n, labels_n = _naive_forward(model, prompt_ids, marker, vecs, targets, submodule, autocast_on, device)
        out_n.loss.backward(); torch.cuda.synchronize()
        r["naive_time_s"] = time.time() - t0
        g_naive = _flat_grads(params)
        loss_naive = out_n.loss.item()
        # matched positions: naive absolute [marker, marker+suffix_len) <-> cached [0, suffix_len)
        naive_sel = torch.cat([out_n.logits[b, marker : marker + suffix_lens[b]] for b in range(batch)]).detach()
        del out_n

        # ---- prefix-cached ----
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        try:
            out_c = pc.forward(vecs, targets, autocast=lambda: autocast_region(model, autocast_on))
            out_c.loss.backward(); torch.cuda.synchronize()
        except Exception as e:  # noqa
            msg = f"{type(e).__name__}: {str(e)[:600]}"
            print(f"[{tag}] PREFIX-CACHED PATH FAILED: {msg}", flush=True)
            r["cached_error"] = msg
            results["runs"][tag] = r
            if stock_demo:
                continue
            raise
        r["cached_time_s"] = time.time() - t0
        g_cached = _flat_grads(params)
        loss_cached = out_c.loss.item()
        cached_sel = torch.cat([out_c.logits[b, : suffix_lens[b]] for b in range(batch)]).detach()
        assert cached_sel.shape == naive_sel.shape, (cached_sel.shape, naive_sel.shape)
        r["loss_naive"] = loss_naive; r["loss_cached"] = loss_cached; r["loss_abs_diff"] = abs(loss_naive - loss_cached)
        r["logits"] = _cmp_logits(naive_sel, cached_sel)
        r["grads"] = _cmp_grads(g_naive, g_cached)
        r["n_target_tokens"] = out_c.n_target_tokens; r["suffix_len_padded"] = out_c.suffix_len
        del out_c, cached_sel

        # ---- noise-floor control: the SAME naive computation split into two half-batches (different GEMM shapes
        # -> bf16 kernel/tiling noise), token-weighted so the total loss/grad equals the full-batch mean ----
        model.zero_grad(set_to_none=True)
        N = int((labels_n != -100).sum())
        half = batch // 2
        loss_ctrl = 0.0; ctrl_sel = []
        for lo, hi in ((0, half), (half, batch)):
            out_h, labels_h = _naive_forward(model, prompt_ids, marker, vecs[lo:hi], targets[lo:hi], submodule, autocast_on, device)
            n_h = int((labels_h != -100).sum())
            (out_h.loss * (n_h / N)).backward()
            loss_ctrl += out_h.loss.item() * n_h / N
            ctrl_sel.append(torch.cat([out_h.logits[b, marker : marker + suffix_lens[lo + b]] for b in range(hi - lo)]).detach())
            del out_h
        g_ctrl = _flat_grads(params)
        ctrl_sel = torch.cat(ctrl_sel)
        r["control_loss_abs_diff"] = abs(loss_naive - loss_ctrl)
        r["control_logits"] = _cmp_logits(naive_sel, ctrl_sel)
        r["control_grads"] = _cmp_grads(g_naive, g_ctrl)
        del ctrl_sel, g_ctrl, g_cached, g_naive, naive_sel
        model.zero_grad(set_to_none=True)
        results["runs"][tag] = r
        print(f"\n[{tag}] loss naive {loss_naive:.6f} cached {loss_cached:.6f} (|d|={r['loss_abs_diff']:.2e}; control |d|={r['control_loss_abs_diff']:.2e})")
        print(f"[{tag}] logits  cached-vs-naive: {json.dumps({k: round(v, 6) for k, v in r['logits'].items()})}")
        print(f"[{tag}] logits control(half-batches)-vs-naive: {json.dumps({k: round(v, 6) for k, v in r['control_logits'].items()})}")
        print(f"[{tag}] grads   cached-vs-naive: {json.dumps({k: round(v, 6) for k, v in r['grads'].items()})}")
        print(f"[{tag}] grads   control-vs-naive: {json.dumps({k: round(v, 6) for k, v in r['control_grads'].items()})}", flush=True)
    return _plain(results)


@app.local_entrypoint()
def equiv(adapter: str = ADAPTER, batch: int = 8, seed: int = 0, min_tgt: int = 8, max_tgt: int = 32,
          fresh_lora: bool = False, stock_demo: bool = False, out: str = ""):
    res = equiv_remote.remote(adapter=adapter, batch=batch, seed=seed, min_tgt=min_tgt, max_tgt=max_tgt,
                              fresh_lora=fresh_lora, stock_demo=stock_demo)
    path = Path(out) if out else REPO / "sft" / "results" / ("prefix_cache_equiv_stock.json" if stock_demo else "prefix_cache_equiv.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")


# ----------------------------------------------------------------------------------------------------------------
# speed
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=7200)
def bench_remote(adapter: str = ADAPTER, steps: int = 10, warmup: int = 3, seed: int = 0, min_tgt: int = 8,
                 max_tgt: int = 32, configs: str = "naive:16,naive_nocache:16,cached:16,cached:64,cached:128",
                 compile_: bool = False, fresh_lora: bool = False, autocast_on: bool = True):
    import sys
    import time
    import torch

    sys.path.insert(0, "/pmx/helpers"); sys.path.insert(0, "/pmx")
    env = _env_report()
    from mxf.config import D_MODEL, INJECT_LAYER, STEER_COEFF
    from mxf.inject import FixedPositionInjector, get_layer
    from mxf.prompts import build_prompt_ids
    from sft import prefix_cache as pcmod
    from sft.pretrain import autocast_region

    tok, model = _load(adapter, fresh_lora)
    device = "cuda:0"
    prompt_ids, mpos = build_prompt_ids(tok)
    marker = mpos[0]
    submodule = get_layer(model, INJECT_LAYER)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0, weight_decay=0.0)   # lr 0: identical weights for every config
    gen = torch.Generator().manual_seed(seed)
    results = {"env": env, "prompt_len": len(prompt_ids), "marker": marker, "steps": steps, "warmup": warmup,
               "autocast_bf16": autocast_on, "target_len_range": [min_tgt, max_tgt], "rows": []}

    def step_naive(B, use_cache):
        vecs, targets = _random_batch(gen, B, D_MODEL, tok.vocab_size, min_tgt, max_tgt, tok.eos_token_id)
        out, labels = _naive_forward(model, prompt_ids, marker, vecs, targets, submodule, autocast_on, device, use_cache=use_cache)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
        return B, int((labels != -100).sum()), out.logits.shape[1]

    def make_cached(B, use_compile):
        inj = None
        if use_compile:
            inj = FixedPositionInjector(B, D_MODEL, 0, STEER_COEFF, device, torch.bfloat16)
            handle = submodule.register_forward_hook(inj.hook)
            torch._dynamo.config.cache_size_limit = 64
            fwd = torch.compile(model.forward, dynamic=True)
        pc = pcmod.PrefixCache(model, prompt_ids, marker, tok.pad_token_id, submodule, STEER_COEFF, device, persistent_injector=inj)
        if use_compile:
            class _Wrap:  # route both forwards through the compiled function
                def __call__(self, **kw):
                    return fwd(**kw)
            pc.model = pc.prefix_model = _Wrap()

        def step(B=B):
            vecs, targets = _random_batch(gen, B, D_MODEL, tok.vocab_size, min_tgt, max_tgt, tok.eos_token_id)
            out = pc.forward(vecs, targets, autocast=lambda: autocast_region(model, autocast_on))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            return B, out.n_target_tokens, out.suffix_len

        def cleanup():
            if use_compile:
                handle.remove(); torch._dynamo.reset()
        return step, cleanup

    def run_case(name, B, fn, cleanup=None):
        row = {"config": name, "micro_batch": B}
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t_first = time.time(); fn(); torch.cuda.synchronize(); row["first_step_s"] = time.time() - t_first
            for _ in range(warmup - 1):
                fn()
            torch.cuda.synchronize(); t0 = time.time()
            n_ex = n_tok = 0; seq = []
            for _ in range(steps):
                b, nt, L = fn(); n_ex += b; n_tok += nt; seq.append(L)
            torch.cuda.synchronize(); dt = time.time() - t0
            row.update({"examples_per_s": n_ex / dt, "target_tokens_per_s": n_tok / dt, "ms_per_step": 1000 * dt / steps,
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "mean_seq_len": sum(seq) / len(seq)})
            print(f"[bench] {name:>14} mb={B:<4} {row['examples_per_s']:8.2f} ex/s  {row['target_tokens_per_s']:8.0f} tgt-tok/s  "
                  f"{row['ms_per_step']:8.1f} ms/step  peak {row['peak_mem_gb']:.1f} GB  L={row['mean_seq_len']:.0f}  first {row['first_step_s']:.1f}s", flush=True)
        except torch.OutOfMemoryError as e:
            row["error"] = f"OOM: {str(e)[:300]}"; print(f"[bench] {name} mb={B} OOM", flush=True)
            opt.zero_grad(); torch.cuda.empty_cache()
        except Exception as e:  # noqa
            row["error"] = f"{type(e).__name__}: {str(e)[:800]}"; print(f"[bench] {name} mb={B} FAILED {row['error']}", flush=True)
            opt.zero_grad(); torch.cuda.empty_cache()
        finally:
            if cleanup:
                cleanup()
        results["rows"].append(row)

    for spec in configs.split(","):
        kind, B = spec.split(":"); B = int(B)
        if kind == "naive":
            run_case("naive", B, lambda B=B: step_naive(B, None))
        elif kind == "naive_nocache":
            run_case("naive_nocache", B, lambda B=B: step_naive(B, False))
        elif kind == "cached":
            fn, cl = make_cached(B, False); run_case("cached", B, fn, cl)
        elif kind == "cached_compile":
            fn, cl = make_cached(B, True); run_case("cached_compile", B, fn, cl)
        else:
            raise ValueError(spec)
    if compile_:
        for B in (16, 64):
            fn, cl = make_cached(B, True); run_case("cached_compile", B, fn, cl)
    return _plain(results)


@app.local_entrypoint()
def bench(adapter: str = ADAPTER, steps: int = 10, warmup: int = 3, seed: int = 0, min_tgt: int = 8, max_tgt: int = 32,
          configs: str = "naive:16,naive_nocache:16,cached:16,cached:64,cached:128", compile: int = 0,
          fresh_lora: bool = False, out: str = ""):
    res = bench_remote.remote(adapter=adapter, steps=steps, warmup=warmup, seed=seed, min_tgt=min_tgt, max_tgt=max_tgt,
                              configs=configs, compile_=bool(compile), fresh_lora=fresh_lora)
    path = Path(out) if out else REPO / "sft" / "results" / "prefix_cache_bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path}")
