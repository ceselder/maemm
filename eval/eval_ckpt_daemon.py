"""Checkpoint eval daemon (ONE GPU): the FULL held-out protocol of rl.py's inline_eval (every family, 512/family,
Bo4, T=1, 16-64 new tokens, SAE norm_act + full-SAE rank metric) PLUS the extra evals of eval/inline_extra_evals.py
(snippet locality on the 64 testbed features, autointerp detection AUC random / emb-NN, WildChat fire-prediction AUC,
adversarial confirmation; Sonnet 5 judge via Anthropic native, OpenRouter fallback) for every checkpoint an RL run
saves -- decoupled from the trainer (rl/rl_disagg.py runs with --inline-eval-every 0).

How it is the SAME eval. rl.py's `inline_eval` and inline_extra_evals' `run_extra_evals_gpu` are called as-is:
they take (llm, actor, ...) and read the LoRA from /tmp/rl_lora/rank0/step<k>, which is where this daemon publishes
each checkpoint's adapter in vLLM key layout; steering uses the adapter-on marker norm they compute themselves from
the HF actor (rl._marker_norm). Generation: this process' own vLLM engine built by rl_disagg._build_engine (fast
steering hook + FULL_DECODE_ONLY CUDA graphs, injection verified with rl.verify_vllm_injection at start-up).
Scoring: the CLEAN HF base (adapter disabled) via eval_universal.score_probe_cos / score_sae_peaks / sae_rank_at_peaks,
exactly as inside the trainer. Memory: HF base bf16 54 GB + adapter + SAE encoder (~60 GB) and the engine at
--vllm-gpu-mem 0.5 of the card (89 GB on a 178 GB B200, 70 GB on a 141 GB H200; prefill budget 24k tokens keeps KV headroom).

Loop (eval/modal_eval_last5.py conventions): vol.reload -> newest un-evaled <ckpt_dir>/step_* (+ final as
--final-step, + the trainer's --save-examples points <ckpt_dir>/examples_<M>, logged at their optimizer step with an
extra `examples` field -- see examples_ckpt_meta) first -> load the adapter -> publish to vLLM layout -> inline_eval + run_extra_evals_gpu -> wandb.log
({..., "ckpt_step": k}, commit=True; define_metric makes ckpt_step the x-axis, so backfill lands out of order) ->
judge stage in the background (results polled and logged under their ckpt_step) -> state file on the volume.
One wandb run per RL run (name/id = <tag>), NEVER resuming the RL run itself (two writers race on _step).

RL adapters trained on a FULL fine-tuned policy base (rl_disagg --policy-base): <ckpt_dir>/run_meta.json carries
"policy_base"; the daemon picks it up automatically (--policy-base auto) and evaluates policy = policy_base + adapter
(the vLLM engine serves the full-FT dir with the LoRA slots; the adapter is renamed to the vLLM key layout from its
safetensors file, no PEFT actor) while the HF side is the clean ORIGINAL base for every score (same scoring code, same
eval cache). The adapter-on marker norm the steering scale needs is captured from the engine itself (vllm_lens residual
capture with the LoRA request; the LoRA protocol's HF _marker_norm agrees within the injection check's 3%).

    python eval/eval_ckpt_daemon.py --ckpt-dir /data/ckpts_last5_v15_g8 --tag last5_v15_g8 --once --only-step 90
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="RL run's checkpoint dir: <dir>/step_<k>/adapter_model.safetensors (+ final)")
    ap.add_argument("--tag", required=True, help="wandb run name/id + state-file key, one per RL run (e.g. last5_v15_g8)")
    ap.add_argument("--rl-run-id", default="", help="wandb id of the RL run being tracked (config cross-reference only)")
    ap.add_argument("--wandb-project", default="maxact-fast")
    ap.add_argument("--wandb-name", default="", help="wandb run name AND id (default eval_ckpt_<tag>); one run per RL run, never the RL run's own id")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--state", default="", help="evaled-ckpt state json (default /data/eval_state/evaled_ckpt_<tag>.json)")
    ap.add_argument("--out-dir", default="", help="per-ckpt metric json + judge artifacts (default /data/eval_ckpt/<tag>)")
    ap.add_argument("--final-step", type=int, default=1000, help="<ckpt_dir>/final is logged as this ckpt_step")
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument("--once", action="store_true", help="evaluate what is pending, then exit")
    ap.add_argument("--only-step", type=int, default=None, help="evaluate only this step (ignores the state file)")
    ap.add_argument("--list-only", action="store_true",
                    help="DRY RUN: print every complete checkpoint under --ckpt-dir with its ckpt_step (step_*, examples_* with the derived "
                         "step, final), which are already in the state file and what would be evaluated, then exit (no GPU, no model)")
    ap.add_argument("--eff-batch", type=int, default=0,
                    help="examples per optimizer step of the run (world x batch x grad_accum), ONLY the fallback for an examples_<M> dir "
                         "lacking training_progress.json / SAVE_DONE step metadata (ckpt_step = ceil(M / eff_batch) - 1); 0 = skip such dirs")
    ap.add_argument("--min-ckpt-mtime", type=float, default=0.0)
    ap.add_argument("--first-adapter", default="/data/sft_mix/last5_rp/final",
                    help="an adapter of the run's LoRA geometry to build the PEFT actor with before the engine (the SFT init)")
    ap.add_argument("--no-extra-evals", action="store_true")
    ap.add_argument("--policy-base", default="auto",
                    help="base the RL adapters were trained on (rl_disagg --policy-base, a full-FT checkpoint dir): the engine serves it + "
                         "the adapter, scoring stays the clean MODEL. 'auto' (default) = <ckpt_dir>/run_meta.json's policy_base when present "
                         "(MODEL or absent -> the plain LoRA-on-MODEL protocol, byte-identical to before); '' / 'none' = force MODEL")
    ap.add_argument("--full-model", action="store_true",
                    help="checkpoints are FULL HF models (dirs carrying SAVE_DONE, sft/fullft.py layout): the vLLM engine loads the "
                         "checkpoint ITSELF (no LoRA) while scoring stays on the clean base; the adapter-on marker norm becomes the "
                         "served model's own marker norm captured from the engine. One engine per process: requires --once --only-step k "
                         "(eval/modal_eval_ckpt.py fullmodel_daemon loops over checkpoints).")
    # held-out eval protocol (rl.py inline_eval flags; FULL 512/family by default)
    ap.add_argument("--eval-cache", default=os.environ.get("MAEMM_EVAL_CACHE", "/data/eval_universal_ho/eval_sets_heldout.pt"),
                    help="frozen eval-set cache (env MAEMM_EVAL_CACHE). eval_sets_heldout_v2.pt = the same 11 cos families + sae PLUS the "
                         "extra mlp / mlp_pair families (layer-42 MLP neuron cosine + fire-back; not in mean_all)")
    ap.add_argument("--eval-sae", default="/data/sae/ae.pt")
    ap.add_argument("--eval-n-per-family", type=int, default=0, help="0 = the whole cache (512/family)")
    ap.add_argument("--eval-bo", type=int, default=4)
    ap.add_argument("--eval-temp", type=float, default=1.0)
    ap.add_argument("--eval-max-new", type=int, default=64)
    ap.add_argument("--eval-min-new", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=96, help="engine max_model_len budget (rl.py's rollout cap)")
    # engine
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.5, help="engine share of the card next to the ~60 GB HF side (0.5 -> 89 GB on B200, 70 GB on H200)")
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--max-num-batched-tokens", type=int, default=24576,
                    help="prefill token budget per engine step; 512 x 207 = 106k made the profiling run leave only 11 GB of KV (238 concurrent seqs). "
                         "24k still prefills ~115 whole prompts per step; a prompt is only chunked when the budget is exhausted mid-prompt (rare at 512 seqs)")
    ap.add_argument("--no-cuda-graphs", action="store_true")
    ap.add_argument("--gdn-prefill-backend", choices=("triton", "flashinfer", "auto"), default="triton",
                    help="vLLM GDN prefill kernel (see rl_disagg): 'triton' runs on H200 and B200 alike; vLLM's 'auto' would JIT flashinfer on sm90 (needs nvcc)")
    ap.add_argument("--stock-lens-hook", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.state = a.state or f"/data/eval_state/evaled_ckpt_{a.tag}.json"
    a.out_dir = a.out_dir or f"/data/eval_ckpt/{a.tag}"
    a.save_dir = a.out_dir            # inline_extra_evals writes its artifacts under <save_dir>/extra_evals
    a.inline_eval_every = 1           # load_eval_assets prints it; every ckpt here is evaluated
    a.cuda_graphs = not a.no_cuda_graphs
    if a.full_model and not a.list_only:
        assert a.once and a.only_step is not None, "--full-model needs --once --only-step k (one engine per checkpoint)"
    return a


def _vllm_marker_norm(llm, prompt_ids, marker, inject_layer, lora_request=None):
    """||h|| at the marker of the served model's INJECT_LAYER residual (vllm_lens capture, clean prompt, greedy 1 token); with
    `lora_request` = the served policy WITH that adapter (the LoRA protocol's 'adapter-on' marker norm, taken from the engine)."""
    from vllm import SamplingParams
    kw = {"lora_request": lora_request} if lora_request is not None else {}
    out = llm.generate([{"prompt_token_ids": list(prompt_ids)}],
                       [SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [inject_layer]})],
                       use_tqdm=False, **kw)[0]
    act = getattr(out, "activations", None)
    assert act is not None and "residual_stream" in act, "vllm_lens capture returned nothing -- plugin not active?"
    return act["residual_stream"][0].float()[marker].norm().item()


def _save_adapter_for_vllm(actor, adapter_name, lora_dir):
    """rl.py _save_adapter_for_vllm for an arbitrary adapter name (rl.py's hardcodes 'default'): module names renamed to
    the Qwen3_5ForConditionalGeneration layout vLLM serves; bf16 (vLLM casts to the model dtype on load anyway)."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    os.makedirs(lora_dir, exist_ok=True)
    sd = get_peft_model_state_dict(actor, adapter_name=adapter_name)
    out = {}
    for k, v in sd.items():
        k2 = k if "language_model" in k else k.replace("model.layers.", "model.language_model.layers.", 1)
        out[k2] = v.detach().to(torch.bfloat16).to("cpu", copy=True).contiguous()
    save_file(out, f"{lora_dir}/adapter_model.safetensors", metadata={"format": "pt"})
    actor.peft_config[adapter_name].save_pretrained(lora_dir)
    return len(out)


def _convert_adapter_dir_for_vllm(src_dir, lora_dir):
    """A saved PEFT adapter dir -> the vLLM key layout, from its FILES (no PEFT actor; rl_disagg.run_bench_rollout does the same):
    `model.layers.` -> `model.language_model.layers.` (Qwen3_5ForConditionalGeneration), bf16, adapter_config.json copied."""
    import torch
    from safetensors.torch import load_file, save_file
    os.makedirs(lora_dir, exist_ok=True)
    sd = load_file(f"{src_dir}/adapter_model.safetensors")
    out = {}
    for k, v in sd.items():
        k2 = k if "language_model" in k else k.replace("model.layers.", "model.language_model.layers.", 1)
        out[k2] = v.to(torch.bfloat16).contiguous()
    save_file(out, f"{lora_dir}/adapter_model.safetensors", metadata={"format": "pt"})
    shutil.copy(f"{src_dir}/adapter_config.json", f"{lora_dir}/adapter_config.json")
    return len(out)


def resolve_policy_base(ckpt_dir, flag, model, log=print):
    """--policy-base -> the full-FT base dir the adapters under ckpt_dir were trained on, or '' for the plain LoRA-on-MODEL protocol."""
    pb = flag or ""
    if pb == "auto":
        pb = ""
        mp = f"{ckpt_dir}/run_meta.json"
        if os.path.exists(mp):
            try:
                pb = json.load(open(mp)).get("policy_base") or ""
            except Exception as e:  # noqa
                log(f"could not read {mp} ({type(e).__name__}: {e}); assuming policy base = {model}")
    if pb in ("none", "None", model):
        pb = ""
    return pb


def examples_ckpt_meta(path, eff_batch=0):
    """<ckpt_dir>/examples_<M> (sft/pretrain.py --save-examples) -> {"ckpt_step", "examples", "optimizer_updates"[, "examples_actual"]},
    or None when its optimizer step cannot be determined.

    Where the trainer puts it on the step_* axis: step_<k> is saved at the END of loop iteration k (k+1 optimizer updates applied)
    and logged here as ckpt_step k. examples_<M> is saved INSIDE the iteration i whose update first brings the examples seen to
    >= M ((i+1) x eff_batch >= M) and the trainer records step = i + 1 (training_progress.json / SAVE_DONE "step" = optimizer
    updates applied). The same weights would have been called step_i, so ckpt_step = recorded step - 1 = ceil(M / eff_batch) - 1
    (2,000,000 examples at eff_batch 4096 -> recorded 489, ckpt_step 488). Sources, in order: training_progress.json (LoRA + full
    FT; written right after the checkpoint dir, so a scan can race it), SAVE_DONE's "step" (full FT), ceil(M / eff_batch) - 1."""
    name = os.path.basename(path.rstrip("/"))
    try:
        m = int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    rec, actual = None, None
    tp = f"{path}/training_progress.json"
    if os.path.exists(tp):
        try:
            d = json.load(open(tp))
            rec, actual = int(d["step"]), d.get("actual_examples")
        except Exception:  # noqa — half-written; fall through to SAVE_DONE / eff_batch
            pass
    if rec is None and os.path.exists(f"{path}/SAVE_DONE"):
        try:
            rec = int(json.load(open(f"{path}/SAVE_DONE"))["step"])
        except Exception:  # noqa
            pass
    if rec is None and eff_batch > 0:
        rec = -(-m // eff_batch)   # ceil(M / eff_batch) = the recorded step
    if rec is None:
        return None
    out = {"ckpt_step": rec - 1, "examples": m, "optimizer_updates": rec}
    if actual is not None:
        out["examples_actual"] = int(actual)
    return out


_scan_warned = set()


def _scan(ckpt_dir, final_step, min_mtime, full_model=False, eff_batch=0, log=print):
    """{ckpt_step: dir} of complete checkpoints under ckpt_dir: step_<k> (ckpt_step k), examples_<M> (pretrain.py --save-examples;
    ckpt_step derived by examples_ckpt_meta) and final (ckpt_step final_step). Complete = LoRA: adapter files present; full model:
    SAVE_DONE present (written last). An examples_<M> and a step_<k> at the same ckpt_step are the same weights; the examples_ dir
    wins so its `examples` field gets logged."""
    def complete(p):
        w = f"{p}/SAVE_DONE" if full_model else f"{p}/adapter_model.safetensors"
        ok = os.path.exists(w) and (full_model or os.path.exists(f"{p}/adapter_config.json"))
        return ok and os.path.getmtime(w) >= min_mtime

    avail = {}
    for p in glob.glob(f"{ckpt_dir}/step_*"):
        try:
            k = int(p.rsplit("_", 1)[-1])
        except ValueError:   # step_<k>.tmp mid-save
            continue
        if complete(p):
            avail[k] = p
    for p in glob.glob(f"{ckpt_dir}/examples_*"):
        if not complete(p):
            continue
        meta = examples_ckpt_meta(p, eff_batch)
        if meta is None:
            if p not in _scan_warned and not p.endswith(".tmp"):
                _scan_warned.add(p)
                log(f"[eval-ckpt] {p}: complete but its optimizer step is unknown (no training_progress.json / SAVE_DONE step; "
                    f"pass --eff-batch to derive it) -> skipped")
            continue
        avail[meta["ckpt_step"]] = p
    if complete(f"{ckpt_dir}/final"):
        avail[final_step] = f"{ckpt_dir}/final"
    return avail


def _ckpt_extras(path, eff_batch=0):
    """Extra wandb / json fields of a checkpoint dir: the --save-examples metadata for examples_<M> ({} otherwise)."""
    if not os.path.basename(path.rstrip("/")).startswith("examples_"):
        return {}
    meta = examples_ckpt_meta(path, eff_batch) or {}
    return {k: v for k, v in meta.items() if k != "ckpt_step"}


def list_only(a, log=print):
    """--list-only: the discovery + state-file view, nothing loaded."""
    avail = _scan(a.ckpt_dir, a.final_step, a.min_ckpt_mtime, full_model=a.full_model, eff_batch=a.eff_batch, log=log)
    try:
        done = set(json.load(open(a.state))["done"])
    except Exception:  # noqa
        done = set()
    if a.only_step is not None:
        avail = {k: v for k, v in avail.items() if k == a.only_step}
    log(f"[eval-ckpt] --list-only {a.ckpt_dir} ({'full-model' if a.full_model else 'LoRA'} layout, final -> ckpt_step {a.final_step}) "
        f"| state {a.state} | previously evaled {sorted(done) or 'none'}")
    for k in sorted(avail):
        x = _ckpt_extras(avail[k], a.eff_batch)
        note = (f"  <- --save-examples point {x['examples']:,} (actual {x.get('examples_actual', '?')}, {x['optimizer_updates']} optimizer "
                f"updates -> ckpt_step {k})") if x else ""
        log(f"  ckpt_step {k:>6}  {'DONE   ' if k in done else 'pending'}  {avail[k]}{note}")
    pending = sorted(k for k in avail if k not in done)
    log(f"[eval-ckpt] would evaluate (latest first): {pending[::-1] or 'nothing'}")
    return {"avail": {k: avail[k] for k in sorted(avail)}, "done": sorted(done), "pending_latest_first": pending[::-1]}


def main():
    a = parse_args()
    if a.list_only:
        list_only(a)
        return
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import rl_hf as R
    import rl_disagg as DG
    from mxf.config import INJECT_LAYER, MODEL
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids

    def log(msg):
        print(f"[eval-ckpt] {msg}", flush=True)

    def vol_commit():
        try:
            import modal
            modal.Volume.from_name("maemm-data").commit()   # no-op outside Modal
        except Exception:  # noqa
            pass

    _reload_warned = [False]

    def vol_reload():   # best-effort; the launcher's parent process refreshes the mount every 45 s regardless
        try:
            import modal
            modal.Volume.from_name("maemm-data").reload()
        except Exception as e:  # noqa
            if not _reload_warned[0]:
                _reload_warned[0] = True
                log(f"in-process vol.reload unavailable ({type(e).__name__}: {str(e)[:80]}); relying on the launcher's reload thread")

    device = "cuda:0"
    torch.cuda.set_device(0)
    policy_base = resolve_policy_base(a.ckpt_dir, a.policy_base, MODEL, log)   # '' = adapters on MODEL (the protocol so far)
    fft_lora = bool(policy_base) and not a.full_model
    if fft_lora:
        if os.path.isdir(policy_base):
            assert os.path.exists(f"{policy_base}/SAVE_DONE"), f"policy base {policy_base} has no SAVE_DONE (incomplete full-FT checkpoint)"
        log(f"policy base {policy_base} (from {'run_meta.json' if a.policy_base == 'auto' else '--policy-base'}): the engine serves it + each "
            f"RL adapter; every score runs on the clean {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    t0 = time.time()
    # ---- HF actor FIRST (vllm's import clobbers transformers' AutoConfig for this model, see rl.py main) ----
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    if a.full_model or fft_lora:
        actor = DG.BaseActor(base)        # scoring side = the clean base; the policy (full model / full-FT base + LoRA) lives in the engine
    else:
        actor = PeftModel.from_pretrained(base, a.first_adapter, adapter_name="init", is_trainable=False)
    actor.eval()
    submodule = get_layer(actor, INJECT_LAYER)
    cur_name = "init"
    log(f"actor ready in {time.time() - t0:.0f}s | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB | prompt {p_len} toks marker @{marker}")
    EV = R.load_eval_assets(a, device, True)
    assert EV is not None, "eval assets failed to load (cache / SAE paths)"
    EX, IX = None, None
    if not a.no_extra_evals:
        import inline_extra_evals as IX
        EX = IX.prepare_extra_eval_assets(a, device, 0, 1, True, sae=EV["sae"])
        if EX is None:
            IX = None
    log(f"eval assets: {len(EV['fams'])} families x {len(EV['es'][EV['fams'][0] + '_dirs'])} dirs x Bo{a.eval_bo} + sae {len(EV['feats'])} "
        f"| extra evals {'ON' if EX is not None else 'OFF'} | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    # ---- engine (rl_disagg: fast steering hook + CUDA graphs), then the one-time numeric injection proof ----
    if a.full_model:
        avail0 = _scan(a.ckpt_dir, a.final_step, a.min_ckpt_mtime, full_model=True, eff_batch=a.eff_batch)
        assert a.only_step in avail0, f"--full-model: step {a.only_step} has no complete checkpoint (SAVE_DONE) under {a.ckpt_dir}: {sorted(avail0)}"
        a.engine_model, a.engine_lora = avail0[a.only_step], False
        log(f"full-model mode: engine serves {a.engine_model} (no LoRA); HF side = clean base for scoring")
    elif fft_lora:
        a.engine_model, a.engine_lora = policy_base, True
        log(f"policy-base mode: engine serves {policy_base} + LoRA slots; HF side = clean base for scoring")
    llm = DG._build_engine(a, 0, p_len, a.max_num_seqs, a.cuda_graphs, "eval-ckpt")
    _hn = {"v": None}   # policy-base mode: the served (policy base + adapter) marker norm, refreshed per checkpoint
    if a.full_model or fft_lora:
        # the 'adapter-on' marker norm of the LoRA protocol == the SERVED model's own marker norm here; take it from the engine
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        hnorm_ft = _vllm_marker_norm(llm, prompt_ids, marker, INJECT_LAYER)            # served base weights, no LoRA
        hnorm_base = R._marker_norm(actor, submodule, prompt_t, marker, device, adapter=False)
        chk = DG._verify_injection(llm, prompt_ids, marker, hnorm_ft, "eval-ckpt", seed=a.seed)
        chk.update({"hnorm_served": hnorm_ft, "hnorm_hf_base": hnorm_base, "hnorm_agree": hnorm_ft / max(hnorm_base, 1e-6),
                    "mode": "full_model" if a.full_model else "policy_base_lora", "policy_base": a.engine_model})
        if a.full_model:
            R._marker_norm = lambda *args, **kw: hnorm_ft            # inline_eval / extra evals ask the actor for it
            _orig_generate = llm.generate
            llm.generate = lambda *args, **kw: _orig_generate(*args, **{k: v for k, v in kw.items() if k != "lora_request"})
        else:
            _hn["v"] = hnorm_ft
            R._marker_norm = lambda *args, **kw: _hn["v"]           # set per checkpoint below (adapter ON, from the engine)
        log(f"injection check ({chk['mode']}): cos {chk['cos']:.4f} | magnitude ratio {chk['norm_ratio']:.3f} | marker ||h|| served {hnorm_ft:.1f} "
            f"vs clean base {hnorm_base:.1f} (x{chk['hnorm_agree']:.3f}) -> {'OK' if chk['ok'] else 'FAIL'}")
    else:
        chk = R.verify_vllm_injection(llm, actor, submodule, prompt_ids, marker, device, seed=a.seed)
        log(f"injection check: cos {chk['cos']:.4f} | magnitude ratio {chk['norm_ratio']:.3f} | ||h|| vllm/hf {chk['hnorm_agree']:.3f} -> {'OK' if chk['ok'] else 'FAIL'}")
    if not chk["ok"]:
        raise RuntimeError(f"vLLM steering does NOT match the HF inject hook: {chk}")
    eos_ids = R._eos_ids(tok, actor)

    if not a.no_wandb:
        wb_name = a.wandb_name or f"eval_ckpt_{a.tag}"
        wandb.init(project=a.wandb_project, name=wb_name, id=wb_name, resume="allow",
                   config={"ckpt_dir": a.ckpt_dir, "rl_run_id": a.rl_run_id, "families": EV["fams"], "n_per_family": len(EV["es"][EV["fams"][0] + "_dirs"]),
                           "bo": a.eval_bo, "temp": a.eval_temp, "max_new": a.eval_max_new, "min_new": a.eval_min_new, "cache": a.eval_cache,
                           "sae_rank_metric": True, "extra_evals": EX is not None, "engine": "vllm fast_lens_ext" + (" cudagraphs" if a.cuda_graphs else " eager"),
                           "schedule": "latest-first", "injection_check": chk, "policy_base": policy_base or MODEL, "full_model": a.full_model})
        wandb.define_metric("ckpt_step")
        wandb.define_metric("eval/*", step_metric="ckpt_step")
        wandb.define_metric("extra/*", step_metric="ckpt_step")
        wandb.define_metric("examples", step_metric="ckpt_step")            # --save-examples points: examples seen at that ckpt_step
    os.makedirs(a.out_dir, exist_ok=True)

    def load_state():
        try:
            return set(json.load(open(a.state))["done"])
        except Exception:  # noqa
            return set()

    def save_state(done):
        os.makedirs(os.path.dirname(a.state), exist_ok=True)
        json.dump({"done": sorted(done), "tag": a.tag, "ckpt_dir": a.ckpt_dir}, open(a.state, "w"))
        vol_commit()

    def flush_judge(final=False):
        if IX is None:
            return
        if final:
            IX.wait_for_judge_stages(1800)
        for cs, m in IX.poll_judge_results():
            keys = " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in m.items() if k.startswith("extra/") and ("auc" in k or "holds" in k))
            log(f"judge results for ckpt {cs}: {keys}")
            if not a.no_wandb:
                wandb.log({**m, "ckpt_step": cs}, commit=True)
            try:
                p = f"{a.out_dir}/ckpt_{cs}.json"
                d = json.load(open(p)) if os.path.exists(p) else {"ckpt_step": cs}
                d.setdefault("judge", {}).update(m)
                json.dump(d, open(p, "w"), indent=1)
            except Exception as e:  # noqa
                log(f"could not update ckpt_{cs}.json with judge results: {e}")

    done = set() if a.only_step is not None else load_state()
    log(f"previously evaled: {sorted(done) or 'none'} | ckpt_dir {a.ckpt_dir} | state {a.state}")
    while True:
        vol_reload()
        avail = _scan(a.ckpt_dir, a.final_step, a.min_ckpt_mtime, full_model=a.full_model, eff_batch=a.eff_batch)
        if a.only_step is not None:
            avail = {k: v for k, v in avail.items() if k == a.only_step}
        todo = sorted(k for k in avail if k not in done)
        if not todo:
            flush_judge(final=a.once)
            if a.once:
                log("--once: nothing pending, exiting")
                break
            time.sleep(a.poll_s)
            continue
        s = todo[-1]
        ck = avail[s]
        if len(todo) > 1:
            log(f"pending {todo} -> evaluating LATEST step {s} first")
        t1 = time.time()
        name = f"ck{s}"
        extras = _ckpt_extras(ck, a.eff_batch)   # examples_<M> dirs: {"examples": M, "optimizer_updates", "examples_actual"}
        log(f"evaluating step {s} ({ck})" + (f" = --save-examples point {extras['examples']:,} (actual {extras.get('examples_actual', '?')}, "
                                             f"{extras['optimizer_updates']} optimizer updates -> ckpt_step {s})" if extras else ""))
        # (the launcher pauses volume reloads while an eval is in progress)
        lora_dir = f"/tmp/rl_lora/rank0/step{s}"        # the path rl.py inline_eval / run_extra_evals_gpu read the LoRA from
        hnorm_on = None
        if a.full_model:
            n_t, t_load = 0, 0.0                        # the engine already IS this checkpoint
        elif fft_lora:
            try:
                n_t = _convert_adapter_dir_for_vllm(ck, lora_dir)
                from vllm.lora.request import LoRARequest   # same name/id/path as rl.py inline_eval's request -> one adapter load
                hnorm_on = _vllm_marker_norm(llm, prompt_ids, marker, INJECT_LAYER,
                                             lora_request=LoRARequest(lora_name=f"step{s}", lora_int_id=s + 1, lora_path=lora_dir))
            except Exception as e:  # noqa — a mid-save commit raced us: retry next poll
                log(f"step {s}: adapter load failed (convert / marker norm: {type(e).__name__}: {e}); will retry")
                shutil.rmtree(lora_dir, ignore_errors=True)
                time.sleep(a.poll_s)
                continue
            _hn["v"] = hnorm_on
            t_load = time.time() - t1
            log(f"step {s}: adapter -> vLLM layout ({n_t} tensors) | marker ||h|| served with the adapter ON {hnorm_on:.2f} "
                f"(policy base alone {hnorm_ft:.2f}, x{hnorm_on / max(hnorm_ft, 1e-6):.4f})")
        else:
            try:
                actor.load_adapter(ck, adapter_name=name)
                actor.set_adapter(name)
            except Exception as e:  # noqa — a mid-save commit raced us: retry next poll
                log(f"step {s}: adapter load failed ({type(e).__name__}: {e}); will retry")
                time.sleep(a.poll_s)
                continue
            if cur_name != name:
                try:
                    actor.delete_adapter(cur_name)
                except Exception as e:  # noqa
                    log(f"could not delete adapter {cur_name}: {e}")
                cur_name = name
            n_t = _save_adapter_for_vllm(actor, name, lora_dir)
            t_load = time.time() - t1
        ev = R.inline_eval(llm, actor, submodule, tok, prompt_ids, marker, a, device, s, s, 0, 1, EV)
        ex = {}
        if EX is not None:
            ex = IX.run_extra_evals_gpu(llm, actor, submodule, tok, prompt_ids, marker, a, device, s, s, 0, 1, EX,
                                        R._steer_vec, R._marker_norm, eos_ids, R._trim_at_stop)
        shutil.rmtree(lora_dir, ignore_errors=True)
        secs = time.time() - t1
        if "error" in ev:
            log(f"step {s}: inline_eval FAILED: {ev['error']}")
            time.sleep(min(a.poll_s, 30))
            continue
        if "error" in ex:
            log(f"step {s}: extra evals FAILED: {ex['error']}")
            ex = {}
        row = {**ev, **ex, **extras, "ckpt_step": s, "time/ckpt_eval_s": secs, "time/adapter_load_publish_s": t_load}
        if hnorm_on is not None:
            row["eval/marker_hnorm_adapter_on"] = hnorm_on
        if not a.no_wandb:
            wandb.log(row, commit=True)
        json.dump({"ckpt_step": s, "ckpt": ck, **extras, "metrics": row, "n_lora_tensors": n_t, "protocol": {
            "families": EV["fams"], "n_per_family": len(EV["es"][EV["fams"][0] + "_dirs"]), "bo": a.eval_bo, "temp": a.eval_temp,
            "min_new": a.eval_min_new, "max_new": a.eval_max_new, "eval_cache": a.eval_cache,
            "extra_families": {f: len(EV["es"][f + "_dirs"]) for f in EV.get("xfams", [])}, "full_model": a.full_model,
            "policy_base": policy_base or MODEL, "hnorm_adapter_on": hnorm_on,
            "injection_check": chk}}, open(f"{a.out_dir}/ckpt_{s}.json", "w"), indent=1)
        if EX is not None and "extra/locality/fire_frac" in ex:
            try:
                IX.launch_judge_stage(None, s, EX, a)
            except Exception as e:  # noqa
                log(f"judge launch failed: {type(e).__name__}: {e}")
        done.add(s)
        if a.only_step is None:
            save_state(done)
        vol_commit()
        log(f"step {s:>5} evaled in {secs:.0f}s (gen+score {ev['time/inline_eval_s']:.0f}s, extra {ex.get('time/extra_eval_gpu_s', 0):.0f}s, "
            f"adapter {t_load:.0f}s) | mean_all {ev['eval/mean_all']:.4f} | sae norm_act {ev['eval/sae/norm_act']:.4f} "
            f"rank1 {ev.get('eval/sae/rank1_frac', float('nan')):.3f} unverb {ev['eval/sae/unverbalized_frac']:.3f} "
            f"| realact {ev.get('eval/realact/cos', float('nan')):.4f} random {ev.get('eval/random/cos', float('nan')):.4f}"
            + "".join(f" | {f} cos {ev[f'eval/{f}/cos']:.4f} fireback {ev[f'eval/{f}/norm_act']:.3f} fired10 {ev[f'eval/{f}/fired10']:.3f}"
                      for f in EV.get("xfams", []) if f"eval/{f}/cos" in ev)
            + (f" | locality win5 {ex.get('extra/locality/win5_share', float('nan')):.3f} fire {ex.get('extra/locality/fire_frac', float('nan')):.3f}" if ex else ""))
        flush_judge()
    if not a.no_wandb:
        wandb.finish()
    print("EVAL_CKPT_DONE", flush=True)


if __name__ == "__main__":
    sys.path[:0] = [p for p in ("/pmx/helpers", "/pmx/eval", "/pmx/RL") if os.path.isdir(p) and p not in sys.path]
    main()
