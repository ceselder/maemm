"""fp8_eval.py -- speed + fidelity harness for `--fp8-base` (sft/fp8.py) on ONE GPU.

Mirrors sft/pretrain.py's default path exactly (per-example padded batches rounded to multiples of 64, length-sorted +
seeded shuffle, FixedPositionInjector at INJECT_LAYER, torch.compile, no grad checkpointing, --autocast-bf16, AdamW +
OneCycleLR, grad-clip 1.0) and runs in ONE process, bf16 first (the fp8 conversion drops the bf16 masters -- one way):
  1. speed: --bench-steps train steps per --bench-batch-sizes (untimed warm pass for compile, then a timed pass):
     TFLOP/s by mxf.mfu's meter (6ND on real tokens), examples/s, real tokens/s, peak memory.
  2. fidelity at IDENTICAL weights on --kl-batches fixed batches: per-token KL(bf16 || fp8), top-1 agreement and the
     target-token NLL, over every position that predicts a target token; at the LoRA init (= base model + zero LoRA)
     and at the LoRA reached after the bf16 training run. A bf16-vs-bf16 repeat gives the noise floor.
  3. training: --train-steps steps bf16 vs fp8 from the same seeded LoRA init on the same batch order -> loss curves,
     mean over the last 100 steps, per-step gap.
Everything lands in --out (JSON) and a summary is printed. Run via modal_sft.py::run_fp8_eval.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from mxf.config import D_MODEL, INJECT_LAYER, MODEL, STEER_COEFF, TrainConfig
from mxf.inject import FixedPositionInjector, get_layer
from mxf.mfu import mfu
from mxf.prompts import build_sft_ids

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pretrain import autocast_region  # noqa: E402  (the trainer's exact autocast region)


def make_batches(toks, bs, seed):
    """pretrain.py's legacy batching: length-sorted, chunked, seeded shuffle; ragged tail dropped (static shapes)."""
    batches = [toks[s: s + bs] for s in range(0, len(toks), bs)]
    np.random.default_rng(seed).shuffle(batches)
    return [b for b in batches if len(b) == bs]


def main():
    cfg = TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="/tmp/fp8_eval.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=cfg.max_seq)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--bench-steps", type=int, default=30)
    ap.add_argument("--bench-batch-sizes", default="16")
    ap.add_argument("--kl-batches", type=int, default=2)
    ap.add_argument("--recipe", default=os.environ.get("MAEMM_FP8_RECIPE", "rowwise"))
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-autocast-bf16", action="store_true")
    ap.add_argument("--no-fp8", action="store_true", help="bf16 phase only")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    compile_on, autocast_on = not a.no_compile, not a.no_autocast_bf16
    bench_sizes = [int(x) for x in a.bench_batch_sizes.split(",") if x.strip()]
    device = "cuda:0"
    res = {"args": vars(a), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "bench": {"bf16": {}, "fp8": {}}, "fidelity": {}, "train": {}}
    try:
        import torchao
        res["torchao"] = torchao.__version__
    except Exception as e:  # noqa
        res["torchao"] = f"import failed: {e!r}"

    # ---- data (identical to pretrain.py) ----
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    records = [json.loads(l) for l in open(f"{a.data_dir}/records.jsonl")]
    n_vecs = max(r["vec_idx"] for r in records) + 1
    vecs = np.memmap(f"{a.data_dir}/vecs.f32", dtype=np.float32, mode="r", shape=(n_vecs, D_MODEL))
    toks = []
    for r in records:
        ids, labs, pos = build_sft_ids(tok, r["target_text"])
        toks.append((ids[: a.max_seq], labs[: a.max_seq], pos, r["vec_idx"]))
    toks.sort(key=lambda t: len(t[0]))
    main_batches = make_batches(toks, a.batch_size, seed=0)
    need = (0 if a.skip_train else a.train_steps) + a.kl_batches
    assert len(main_batches) >= need, f"bank too small: {len(main_batches)} batches of {a.batch_size} < {need}"
    train_batches = main_batches[: a.train_steps]
    kl_batches = main_batches[a.train_steps: a.train_steps + a.kl_batches]
    print(f"[fp8-eval] {len(records)} records -> {len(main_batches)} batches of {a.batch_size}; "
          f"train {len(train_batches)} / kl {len(kl_batches)} / bench {a.bench_steps} at {bench_sizes}", flush=True)

    def collate(batch):
        L = max(len(t[0]) for t in batch)
        L = min(((L + 63) // 64) * 64, a.max_seq)
        input_ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), L), dtype=torch.bool)
        for i, (ii, ll, _, _) in enumerate(batch):
            input_ids[i, : len(ii)] = torch.tensor(ii)
            labels[i, : len(ll)] = torch.tensor(ll)
            attn[i, : len(ii)] = True
        n_real = int(attn.sum())
        vmat = torch.from_numpy(np.asarray(vecs[[t[3] for t in batch]])).to(device)
        return input_ids.to(device), attn.to(device), labels.to(device), vmat, n_real

    # ---- model (identical to pretrain.py; LoRA init seeded so both arms start from the same adapter) ----
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": device})
    model.enable_input_require_grads()
    torch.manual_seed(a.seed)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, use_rslora=True,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    init_state = [p.detach().clone() for p in trainable]
    res["n_params"] = n_params
    res["n_trainable"] = sum(p.numel() for p in trainable)
    _, _, fixed_positions = build_sft_ids(tok, "compile marker probe")
    assert len(fixed_positions) == 1
    injector = FixedPositionInjector(max(bench_sizes + [a.batch_size]), D_MODEL, fixed_positions[0], STEER_COEFF,
                                     device, torch.bfloat16)
    get_layer(model, INJECT_LAYER).register_forward_hook(injector.hook)
    orig_forward = model.forward

    def recompile():
        torch._dynamo.reset()
        model.forward = torch.compile(orig_forward) if compile_on else orig_forward

    def restore(state):
        with torch.no_grad():
            for p, s in zip(trainable, state):
                p.copy_(s)

    def make_opt(steps):
        opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=cfg.warmup_frac,
                                                    anneal_strategy="linear")
        return opt, sched

    def forward(batch):
        input_ids, attn, labels, vmat, n_real = collate(batch)
        injector.set_vectors(vmat)
        with autocast_region(model, autocast_on):
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        return out, labels, n_real

    def train_step(opt, sched, batch):
        out, _, n_real = forward(batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        return out.loss, n_real

    def bench(tag, bs):
        batches = make_batches(toks, bs, seed=1)[: a.bench_steps]
        restore(init_state)
        opt, sched = make_opt(2 * len(batches))
        t0 = time.time()
        try:
            for b in batches:  # untimed warm pass: compiles every shape, warms the allocator
                train_step(opt, sched, b)
            torch.cuda.synchronize()
            warm = time.time() - t0
            torch.cuda.reset_peak_memory_stats()
            t0, n_tok = time.time(), 0
            for b in batches:
                _, n = train_step(opt, sched, b)
                n_tok += n
            torch.cuda.synchronize()
            dt = time.time() - t0
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.synchronize()
            opt.zero_grad()
            torch.cuda.empty_cache()
            r = {"batch_size": bs, "oom": True, "error": str(e).splitlines()[0][:300]}
            print(f"[fp8-eval] bench {tag} bs={bs}: OOM ({r['error']})", flush=True)
            return r
        tfl, m = mfu(n_tok, dt, n_params, fwd_bwd=True)
        r = {"batch_size": bs, "steps": len(batches), "warm_pass_s": warm, "s_per_step": dt / len(batches),
             "examples_per_s": bs * len(batches) / dt, "real_tokens_per_s": n_tok / dt, "tflops_meter": tfl,
             "mfu_vs_1500": m, "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
             "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
        print(f"[fp8-eval] bench {tag} bs={bs}: {tfl:.0f} TFLOP/s (meter) | {r['examples_per_s']:.1f} ex/s | "
              f"{r['s_per_step']*1e3:.0f} ms/step | peak {r['peak_alloc_gb']:.1f} GB alloc "
              f"/ {r['peak_reserved_gb']:.1f} GB reserved | warm pass {warm:.0f}s", flush=True)
        return r

    def target_logprobs(batches):
        """fp32 log-softmax at every position predicting a target token -> [(logp [n,V] cpu, targets [n] cpu)]."""
        outs = []
        for b in batches:
            out, labels, _ = forward(b)  # grad-enabled forward (same compiled graph as training), graph dropped below
            logits = out.logits.detach()
            del out
            mask = labels[:, 1:] != -100  # logits[:, t] predict labels[:, t+1]
            lg = logits[:, :-1][mask].float()
            outs.append((torch.log_softmax(lg, -1).cpu(), labels[:, 1:][mask].cpu()))
            del logits, lg
        return outs

    def kl_stats(ref, cur):
        kls, agree, n, nll_ref, nll_cur = [], 0, 0, 0.0, 0.0
        for (lr, tr), (lc, tc) in zip(ref, cur):
            assert torch.equal(tr, tc)
            lr, lc, tr = lr.to(device), lc.to(device), tr.to(device)
            kls.append((lr.exp() * (lr - lc)).sum(-1))
            agree += (lr.argmax(-1) == lc.argmax(-1)).sum().item()
            n += len(tr)
            nll_ref += -lr.gather(1, tr[:, None]).sum().item()
            nll_cur += -lc.gather(1, tr[:, None]).sum().item()
        kl = torch.cat(kls)
        return {"n_tokens": n, "kl_mean": kl.mean().item(), "kl_median": kl.median().item(),
                "kl_p99": kl.quantile(0.99).item(), "kl_max": kl.max().item(),
                "frac_kl_gt_1e-2": (kl > 1e-2).float().mean().item(),
                "frac_kl_gt_1e-1": (kl > 1e-1).float().mean().item(),
                "top1_agree": agree / n, "nll_ref": nll_ref / n, "nll_cur": nll_cur / n}

    def train_run(tag, batches):
        restore(init_state)
        opt, sched = make_opt(len(batches))
        torch.cuda.reset_peak_memory_stats()
        losses, times = [], []
        for i, b in enumerate(batches):
            t0 = time.time()
            loss, _ = train_step(opt, sched, b)
            losses.append(loss.item())  # syncs
            times.append(time.time() - t0)
            if i % 50 == 0 or i == len(batches) - 1:
                print(f"[fp8-eval] train {tag} step {i}/{len(batches)} loss {losses[-1]:.4f}", flush=True)
        k = min(100, len(losses))
        r = {"steps": len(losses), "losses": losses, "step_times": times,
             "mean_loss_all": float(np.mean(losses)), "mean_loss_last100": float(np.mean(losses[-k:])),
             "mean_loss_first50": float(np.mean(losses[:50])),
             "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30}
        print(f"[fp8-eval] train {tag}: mean loss last {k} = {r['mean_loss_last100']:.4f} "
              f"(all {r['mean_loss_all']:.4f}) peak {r['peak_alloc_gb']:.1f} GB", flush=True)
        return r

    # ================= bf16 phase =================
    recompile()
    for bs in bench_sizes:
        res["bench"]["bf16"][str(bs)] = bench("bf16", bs)
    ref_init = ref_trained = None
    if kl_batches:
        restore(init_state)
        ref_init = target_logprobs(kl_batches)
        res["fidelity"]["bf16_repeat_vs_bf16_init"] = kl_stats(ref_init, target_logprobs(kl_batches))
        print(f"[fp8-eval] noise floor (bf16 twice, init): {res['fidelity']['bf16_repeat_vs_bf16_init']}", flush=True)
    trained_state = None
    if not a.skip_train:
        res["train"]["bf16"] = train_run("bf16", train_batches)
        trained_state = [p.detach().clone() for p in trainable]
        if kl_batches:
            ref_trained = target_logprobs(kl_batches)
    json.dump(res, open(a.out, "w"), indent=1)

    # ================= fp8 phase =================
    if not a.no_fp8:
        from fp8 import convert_frozen_base_to_fp8
        t0 = time.time()
        res["fp8_convert"] = convert_frozen_base_to_fp8(model, recipe=a.recipe)
        res["fp8_convert"]["seconds"] = time.time() - t0
        res["fp8_convert"]["alloc_gb_after"] = torch.cuda.memory_allocated() / 2**30
        recompile()
        if kl_batches:
            restore(init_state)
            res["fidelity"]["fp8_vs_bf16_init"] = kl_stats(ref_init, target_logprobs(kl_batches))
            print(f"[fp8-eval] KL(bf16||fp8) @init: {res['fidelity']['fp8_vs_bf16_init']}", flush=True)
            if trained_state is not None:
                restore(trained_state)
                res["fidelity"]["fp8_vs_bf16_trained"] = kl_stats(ref_trained, target_logprobs(kl_batches))
                print(f"[fp8-eval] KL(bf16||fp8) @trained: {res['fidelity']['fp8_vs_bf16_trained']}", flush=True)
        for bs in bench_sizes:
            res["bench"]["fp8"][str(bs)] = bench("fp8", bs)
        if not a.skip_train:
            res["train"]["fp8"] = train_run("fp8", train_batches)
            lb, lf = np.array(res["train"]["bf16"]["losses"]), np.array(res["train"]["fp8"]["losses"])
            k = min(100, len(lb))
            res["train"]["gap"] = {
                "fp8_minus_bf16_last100": float(lf[-k:].mean() - lb[-k:].mean()),
                "fp8_minus_bf16_all": float((lf - lb).mean()),
                "abs_gap_mean_all": float(np.abs(lf - lb).mean()),
                "abs_gap_max_all": float(np.abs(lf - lb).max()),
                "bf16_step_to_step_noise": float(np.abs(np.diff(lb)).mean()),
            }
    json.dump(res, open(a.out, "w"), indent=1)

    # ---- summary ----
    print("\n================ fp8_eval SUMMARY ================")
    for mode in ("bf16", "fp8"):
        for bs, r in res["bench"][mode].items():
            if r.get("oom"):
                print(f"{mode:5s} bs={bs}: OOM")
            else:
                print(f"{mode:5s} bs={bs}: {r['tflops_meter']:.0f} TFLOP/s  {r['examples_per_s']:.1f} ex/s  "
                      f"{r['s_per_step']*1e3:.0f} ms/step  peak {r['peak_alloc_gb']:.1f} GB")
    for k, v in res["fidelity"].items():
        print(f"{k}: KL mean {v['kl_mean']:.2e} p99 {v['kl_p99']:.2e} max {v['kl_max']:.2e} | "
              f"top1 {v['top1_agree']:.4f} | nll ref {v['nll_ref']:.4f} cur {v['nll_cur']:.4f} | n={v['n_tokens']}")
    for mode in ("bf16", "fp8"):
        if mode in res["train"]:
            print(f"train {mode}: last100 {res['train'][mode]['mean_loss_last100']:.4f} "
                  f"all {res['train'][mode]['mean_loss_all']:.4f}")
    if "gap" in res["train"]:
        print(f"train gap: {res['train']['gap']}")
    print("FP8_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
