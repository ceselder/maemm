"""Checkpoint eval daemon (ONE GPU): the FULL held-out protocol of rl.py's inline_eval (every family, 512/family,
Bo4, T=1, 16-64 new tokens, SAE norm_act + full-SAE rank metric) PLUS the extra evals of train/inline_extra_evals.py
(snippet locality on the 64 testbed features, autointerp detection AUC random / emb-NN, WildChat fire-prediction AUC,
adversarial confirmation; Sonnet 5 judge via Anthropic native, OpenRouter fallback) for every checkpoint an RL run
saves -- decoupled from the trainer (train/rl_disagg.py runs with --inline-eval-every 0).

How it is the SAME eval. rl.py's `inline_eval` and inline_extra_evals' `run_extra_evals_gpu` are called as-is:
they take (llm, actor, ...) and read the LoRA from /tmp/rl_lora/rank0/step<k>, which is where this daemon publishes
each checkpoint's adapter in vLLM key layout; steering uses the adapter-on marker norm they compute themselves from
the HF actor (rl._marker_norm). Generation: this process' own vLLM engine built by rl_disagg._build_engine (fast
steering hook + FULL_DECODE_ONLY CUDA graphs, injection verified with rl.verify_vllm_injection at start-up).
Scoring: the CLEAN HF base (adapter disabled) via eval_universal.score_probe_cos / score_sae_peaks / sae_rank_at_peaks,
exactly as inside the trainer. Memory: HF base bf16 54 GB + adapter + SAE encoder (~60 GB) and the engine at
--vllm-gpu-mem 0.45 of the card (80 GB on a 178 GB B200, 63 GB on a 141 GB H200).

Loop (MAEMMBench/modal_eval_last5.py conventions): vol.reload -> newest un-evaled <ckpt_dir>/step_* (+ final as
--final-step) first -> load the adapter -> publish to vLLM layout -> inline_eval + run_extra_evals_gpu -> wandb.log
({..., "ckpt_step": k}, commit=True; define_metric makes ckpt_step the x-axis, so backfill lands out of order) ->
judge stage in the background (results polled and logged under their ckpt_step) -> state file on the volume.
One wandb run per RL run (name/id = <tag>), NEVER resuming the RL run itself (two writers race on _step).

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
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--state", default="", help="evaled-ckpt state json (default /data/eval_state/evaled_ckpt_<tag>.json)")
    ap.add_argument("--out-dir", default="", help="per-ckpt metric json + judge artifacts (default /data/eval_ckpt/<tag>)")
    ap.add_argument("--final-step", type=int, default=1000, help="<ckpt_dir>/final is logged as this ckpt_step")
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument("--once", action="store_true", help="evaluate what is pending, then exit")
    ap.add_argument("--only-step", type=int, default=None, help="evaluate only this step (ignores the state file)")
    ap.add_argument("--min-ckpt-mtime", type=float, default=0.0)
    ap.add_argument("--first-adapter", default="/data/sft_mix/last5_rp/final",
                    help="an adapter of the run's LoRA geometry to build the PEFT actor with before the engine (the SFT init)")
    ap.add_argument("--no-extra-evals", action="store_true")
    # held-out eval protocol (rl.py inline_eval flags; FULL 512/family by default)
    ap.add_argument("--eval-cache", default="/data/eval_universal_ho/eval_sets_heldout.pt")
    ap.add_argument("--eval-sae", default="/data/sae/ae.pt")
    ap.add_argument("--eval-n-per-family", type=int, default=0, help="0 = the whole cache (512/family)")
    ap.add_argument("--eval-bo", type=int, default=4)
    ap.add_argument("--eval-temp", type=float, default=1.0)
    ap.add_argument("--eval-max-new", type=int, default=64)
    ap.add_argument("--eval-min-new", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=96, help="engine max_model_len budget (rl.py's rollout cap)")
    # engine
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.45)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--no-cuda-graphs", action="store_true")
    ap.add_argument("--stock-lens-hook", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.state = a.state or f"/data/eval_state/evaled_ckpt_{a.tag}.json"
    a.out_dir = a.out_dir or f"/data/eval_ckpt/{a.tag}"
    a.save_dir = a.out_dir            # inline_extra_evals writes its artifacts under <save_dir>/extra_evals
    a.inline_eval_every = 1           # load_eval_assets prints it; every ckpt here is evaluated
    a.cuda_graphs = not a.no_cuda_graphs
    return a


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


def _scan(ckpt_dir, final_step, min_mtime):
    avail = {}
    for p in glob.glob(f"{ckpt_dir}/step_*"):
        try:
            s = int(p.rsplit("_", 1)[-1])
        except ValueError:
            continue
        w = f"{p}/adapter_model.safetensors"
        if os.path.exists(w) and os.path.exists(f"{p}/adapter_config.json") and os.path.getmtime(w) >= min_mtime:
            avail[s] = p
    fw = f"{ckpt_dir}/final/adapter_model.safetensors"
    if os.path.exists(fw) and os.path.getmtime(fw) >= min_mtime:
        avail[final_step] = f"{ckpt_dir}/final"
    return avail


def main():
    a = parse_args()
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

    def vol_reload():
        try:
            import modal
            modal.Volume.from_name("maemm-data").reload()
        except Exception:  # noqa
            pass

    device = "cuda:0"
    torch.cuda.set_device(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker, p_len = mpos[0], len(prompt_ids)
    t0 = time.time()
    # ---- HF actor FIRST (vllm's import clobbers transformers' AutoConfig for this model, see rl.py main) ----
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
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
    llm = DG._build_engine(a, 0, p_len, a.max_num_seqs, a.cuda_graphs, "eval-ckpt")
    chk = R.verify_vllm_injection(llm, actor, submodule, prompt_ids, marker, device, seed=a.seed)
    log(f"injection check: cos {chk['cos']:.4f} | magnitude ratio {chk['norm_ratio']:.3f} | ||h|| vllm/hf {chk['hnorm_agree']:.3f} -> {'OK' if chk['ok'] else 'FAIL'}")
    if not chk["ok"]:
        raise RuntimeError(f"vLLM steering does NOT match the HF inject hook: {chk}")
    eos_ids = R._eos_ids(tok, actor)

    if not a.no_wandb:
        wandb.init(project=a.wandb_project, name=f"eval_ckpt_{a.tag}", id=f"eval_ckpt_{a.tag}", resume="allow",
                   config={"ckpt_dir": a.ckpt_dir, "rl_run_id": a.rl_run_id, "families": EV["fams"], "n_per_family": len(EV["es"][EV["fams"][0] + "_dirs"]),
                           "bo": a.eval_bo, "temp": a.eval_temp, "max_new": a.eval_max_new, "min_new": a.eval_min_new, "cache": a.eval_cache,
                           "sae_rank_metric": True, "extra_evals": EX is not None, "engine": "vllm fast_lens_ext" + (" cudagraphs" if a.cuda_graphs else " eager"),
                           "schedule": "latest-first", "injection_check": chk})
        wandb.define_metric("ckpt_step")
        wandb.define_metric("eval/*", step_metric="ckpt_step")
        wandb.define_metric("extra/*", step_metric="ckpt_step")
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
        avail = _scan(a.ckpt_dir, a.final_step, a.min_ckpt_mtime)
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
        lora_dir = f"/tmp/rl_lora/rank0/step{s}"        # the path rl.py inline_eval / run_extra_evals_gpu read the LoRA from
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
        row = {**ev, **ex, "ckpt_step": s, "time/ckpt_eval_s": secs, "time/adapter_load_publish_s": t_load}
        if not a.no_wandb:
            wandb.log(row, commit=True)
        json.dump({"ckpt_step": s, "ckpt": ck, "metrics": row, "n_lora_tensors": n_t, "protocol": {
            "families": EV["fams"], "n_per_family": len(EV["es"][EV["fams"][0] + "_dirs"]), "bo": a.eval_bo, "temp": a.eval_temp,
            "min_new": a.eval_min_new, "max_new": a.eval_max_new}}, open(f"{a.out_dir}/ckpt_{s}.json", "w"), indent=1)
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
            + (f" | locality win5 {ex.get('extra/locality/win5_share', float('nan')):.3f} fire {ex.get('extra/locality/fire_frac', float('nan')):.3f}" if ex else ""))
        flush_judge()
    if not a.no_wandb:
        wandb.finish()
    print("EVAL_CKPT_DONE", flush=True)


if __name__ == "__main__":
    sys.path[:0] = [p for p in ("/pmx/helpers", "/pmx/eval", "/pmx/RL") if os.path.isdir(p) and p not in sys.path]
    main()
