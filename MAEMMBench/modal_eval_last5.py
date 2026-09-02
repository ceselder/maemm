"""Modal eval daemon for the LAST-5 RL run (app maemm-rl-last5-8xb200, wandb or0215ub).

Variant of MAEMMBench/modal_eval.py (the paper-run daemon) with only the run-specific knobs
changed: watches /data/ckpts_last5 (--total-steps 400 --save-every 25 -> step_25..step_400 +
final), logs to project `maxact-fast` as run `last5_rp_rl_eval`. A SEPARATE wandb run — NOT a
resume of the RL run or0215ub — because that run is being actively written by the live trainer,
and two concurrent writers to one run id race on _step (rows overwrite; would corrupt the live
RL curves). The RL run id is recorded in this run's config for cross-referencing.

Runs on ONE B200, mounts the same `maemm-data` volume as training, and loops:
  vol.reload() -> pick the HIGHEST un-evaled /data/ckpts_last5/step_* (latest-first; when caught
  up on the newest ckpt this naturally backfills the backlog newest-first) -> load base(+adapter)
  -> fast held-out eval -> wandb.log (commit=True), with `ckpt_step` as the x-axis metric
  (define_metric), so out-of-order backfill points still land (wandb's own _step stays monotonic).

FAST eval = eval_universal's cos families (bsf, realact, jlens, cluster, random, ctx buckets,
indist_*) + a rank-free SAE family (best-of-bo target-feature act -> norm_act / fired /
beat_corpus / unverbalized_*). `mean_all` averages the higher-is-better cos families only
(the `random` control is EXCLUDED). The full-SAE rank-over-131k metric stays DROPPED.

The frozen cache is torch.load'ed DIRECTLY (not via build_eval_sets): the cache validates
meta["heldout_pool"] against the abspath it was built with (the dead box's path), which cannot
match inside the container. Evaled-ckpt state persists at /data/eval_state/evaled_last5.json.

Deploy + run (profile safety-sahan) — deployed app + spawn, NOT `modal run --detach`
(killing an ephemeral app's local client cancels the app):
    MODAL_PROFILE=safety-sahan modal deploy MAEMMBench/modal_eval_last5.py
    MODAL_PROFILE=safety-sahan python -c "import modal; \
        modal.Function.from_name('maemm-eval-last5-heldout', 'daemon').spawn()"
"""

from pathlib import Path

import modal

REPO = Path(__file__).parent.parent   # repo root (this file lives in MAEMMBench/)

app = modal.App("maemm-eval-last5-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.15.0",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "wandb==0.28.2",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    # MAEMMBench/ is mounted AT /pmx/eval so `import eval_universal` inside the container is
    # unchanged (eval/ at the repo root is now a shim dir that needs the repo on sys.path).
    .add_local_dir(REPO / "MAEMMBench", "/pmx/eval",
                   ignore=["__pycache__", "README.md", "analysis", "analysis/**"])
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

CACHE = "/data/eval_universal_ho/eval_sets_heldout.pt"
SAE_PATH = "/data/sae/ae.pt"
CKPT_DIR = "/data/ckpts_last5_v12"   # RL v12 (v11 recipe + KL 0.01)
STATE = "/data/eval_state/evaled_last5_v12.json"  # fresh state file for v11
WANDB_PROJECT = "maxact-fast"                  # same project as the RL run
WANDB_RUN = "last5_rp_rl_eval_v12"             # separate run (see module docstring for why
WANDB_RUN_ID = "last5_rp_rl_eval_v12"          # the RL run is NOT resumed); fresh id
RL_RUN_ID = "last5_rp_rl_v10"                  # the v10 RL run this eval tracks (config xref)
FINAL_STEP = 400   # <CKPT_DIR>/final is logged as this step (last-5 run: --total-steps 400)
CONTROL_FAMS = {"random"}  # lower-is-better controls: logged per-family, EXCLUDED from mean_all


@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maemm-hf"),
        modal.Secret.from_name("maemm-wandb"),
    ],
    timeout=86400,
)
def daemon(poll_s: int = 120, once: bool = False, bo: int = 4, temp: float = 1.0,
           max_new: int = 64, min_new: int = 16, gen_chunk: int = 128,
           min_ckpt_mtime: float = 0.0, tag: str = "v12", wandb_name: str = "", wandb_id: str = ""):
    # wandb_name / wandb_id: override the tag-derived wandb run name/id (e.g. after a run id was deleted,
    # or to keep an internal tag out of the run name)
    # tag: which /data/ckpts_last5_<tag> run to evaluate (state file + wandb run names follow it)
    # min_ckpt_mtime: ignore ckpt dirs whose adapter mtime predates this (stale artifacts from a
    # cancelled leg). When the resumed leg OVERWRITES such a dir, its mtime refreshes past the
    # cutoff and it gets evaled as new — no deletion needed, no stale evals, no skipped re-saves.
    import glob
    import json
    import os
    import sys
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    global CKPT_DIR, STATE, WANDB_RUN, WANDB_RUN_ID
    CKPT_DIR = f"/data/ckpts_last5_{tag}"
    STATE = f"/data/eval_state/evaled_last5_{tag}.json"
    WANDB_RUN = WANDB_RUN_ID = f"last5_rp_rl_eval_{tag}"
    if wandb_name:
        WANDB_RUN = wandb_name
    if wandb_id:
        WANDB_RUN_ID = wandb_id
    os.environ["HF_HUB_OFFLINE"] = "1"          # load purely from the volume cache
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DIR"] = "/tmp/wandb"
    os.makedirs("/tmp/wandb", exist_ok=True)
    sys.path[:0] = ["/pmx/helpers", "/pmx/eval"]

    import numpy as np
    import torch
    import wandb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import eval_universal as EU
    from mxf.config import INJECT_LAYER, MODEL
    from mxf.inject import get_layer
    from mxf.prompts import build_prompt_ids
    from mxf.sae import load_sae

    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt_ids, mpos = build_prompt_ids(tok)
    marker = mpos[0]
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    base.eval()
    sae = load_sae(path=SAE_PATH, device=dev, dtype=torch.float32)
    es = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert es["meta"]["d_sae"] == sae.d_sae, \
        f"cache d_sae {es['meta']['d_sae']} != SAE d_sae {sae.d_sae}"
    fams = es["meta"].get("cos_families", EU.COS_FAMILIES)
    feats = es["sae_feats"]
    cp = es["corpus_peak"].numpy().astype(np.float64)
    print(f"[eval-daemon] base+sae+cache ready ({time.time() - t0:.0f}s) | "
          f"ckpt_dir={CKPT_DIR} final_step={FINAL_STEP} | families {fams} "
          f"| n={es['meta'].get('n')} | bo={bo} temp={temp} max_new={max_new} "
          f"gen_chunk={gen_chunk} | LATEST-FIRST, rank metric OFF", flush=True)

    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN, id=WANDB_RUN_ID, resume="allow",
               config={"families": fams, "n": es["meta"].get("n"), "bo": bo, "temp": temp,
                       "max_new_tokens": max_new, "min_new_tokens": min_new,
                       "cache": CACHE, "ckpt_dir": CKPT_DIR, "gpu": "B200:1",
                       "gen_chunk": gen_chunk, "sae_rank_metric": False,
                       "schedule": "latest-first", "rl_run_id": RL_RUN_ID})
    wandb.define_metric("ckpt_step")
    wandb.define_metric("eval/*", step_metric="ckpt_step")

    @torch.no_grad()
    def run_eval_fast(actor, sub):
        """EU.run_eval minus the SAE rank/peak machinery, with per-family timings. Same frozen
        dirs, same forked GEN_SEED RNG -> per-ckpt deterministic and comparable across passes."""
        out, times = {}, {}
        actor.eval()
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(EU.GEN_SEED)
            gen_args = (actor, tok, prompt_ids, marker, sub, dev, bo, temp, max_new, min_new,
                        gen_chunk)
            for fam in fams:
                tf = time.time()
                best = EU.eval_cos_family(fam, es[f"{fam}_dirs"], *gen_args)
                out[f"eval/{fam}/cos"] = float(best.mean())
                times[fam] = time.time() - tf
            tf = time.time()
            best = np.full(len(feats), -1e9)
            for rows, texts in EU._gen_batches("sae", es["sae_dirs"], *gen_args):
                acts, _ = EU.score_sae_peaks(texts, [feats[i] for i in rows], sae, actor, tok, dev)
                np.maximum.at(best, rows, acts.numpy().astype(np.float64))
            times["sae"] = time.time() - tf
        na = best / np.maximum(cp, 1e-6)
        out["eval/sae/norm_act"] = float(na.mean())
        out["eval/sae/fired"] = float(np.mean(best > EU.SAE_FIRE))
        out["eval/sae/beat_corpus"] = float(np.mean(best > cp))
        out["eval/sae/unverbalized_frac"] = float(np.mean(best <= EU.SAE_FIRE))
        out["eval/sae/unverbalized_p10"] = float(np.mean(na < 0.10))
        # mean_all = mean over the HIGHER-IS-BETTER cos families only. `random` is the control
        # (should stay ~0.03, LOWER is better) — folding it in dragged the mean down and would
        # move mean_all the WRONG way if the control ever degraded. It stays logged separately.
        cos_keys = [k for k in out if k.startswith("eval/") and k.endswith("/cos")
                    and k.split("/")[1] not in CONTROL_FAMS]
        out["eval/mean_all"] = float(np.mean([out[k] for k in cos_keys]))
        # headline mirror group (matches EU.run_eval's eval/all/* panel, minus rank)
        for fam in fams:
            out[f"eval/all/{fam}_cos"] = out[f"eval/{fam}/cos"]
        out["eval/all/sae_norm_act"] = out["eval/sae/norm_act"]
        out["eval/all/sae_unverbalized"] = out["eval/sae/unverbalized_frac"]
        print("  [timing] " + " ".join(f"{k}={v:.0f}s" for k, v in times.items()), flush=True)
        return out

    def load_state():
        try:
            return set(json.load(open(STATE))["done"])
        except Exception:
            return set()

    def save_state(done):
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        json.dump({"done": sorted(done)}, open(STATE, "w"))
        vol.commit()

    done = load_state()
    print(f"[eval-daemon] previously evaled: {sorted(done) or 'none'}", flush=True)

    while True:
        try:
            vol.reload()   # pick up the training container's latest commits
        except Exception as e:
            print(f"[eval-daemon] vol.reload failed ({e}); retrying next poll", flush=True)
            time.sleep(poll_s)
            continue
        avail = {}
        for p in glob.glob(f"{CKPT_DIR}/step_*"):
            try:
                s = int(p.rsplit("_", 1)[-1])
            except ValueError:
                continue
            w = f"{p}/adapter_model.safetensors"
            if os.path.exists(w) and os.path.getmtime(w) >= min_ckpt_mtime:
                avail[s] = p
        fw = f"{CKPT_DIR}/final/adapter_model.safetensors"
        if os.path.exists(fw) and os.path.getmtime(fw) >= min_ckpt_mtime:
            avail[FINAL_STEP] = f"{CKPT_DIR}/final"
        todo = sorted(k for k in avail if k not in done)
        if not todo:
            if once:
                print("[eval-daemon] --once: nothing pending, exiting", flush=True)
                break
            time.sleep(poll_s)
            continue
        s = todo[-1]   # LATEST un-evaled ckpt; when the newest is done this backfills the backlog
        ck = avail[s]
        if len(todo) > 1:
            print(f"[eval-daemon] pending {todo} -> evaling LATEST step {s} first", flush=True)
        t0 = time.time()
        try:
            actor = PeftModel.from_pretrained(base, ck, is_trainable=False)
        except Exception as e:  # mid-save commit raced us — retry on the next poll
            print(f"[eval-daemon] step {s}: adapter load failed ({e}); will retry", flush=True)
            time.sleep(poll_s)
            continue
        sub = get_layer(actor, INJECT_LAYER)
        m = run_eval_fast(actor, sub)
        wandb.log({**m, "ckpt_step": s}, commit=True)
        base = actor.unload()   # strip adapter layers -> clean base for the next ckpt
        done.add(s)
        save_state(done)
        print(f"[eval-daemon] step {s:>5} evaled in {time.time() - t0:.0f}s | "
              f"mean_all {m['eval/mean_all']:.4f} | sae norm_act {m['eval/sae/norm_act']:.4f} "
              f"unverb {m['eval/sae/unverbalized_frac']:.3f} | realact {m.get('eval/realact/cos', float('nan')):.4f} "
              f"| random {m.get('eval/random/cos', float('nan')):.4f}", flush=True)
        # no sleep: immediately re-scan — a newer ckpt may have landed during this pass
    wandb.finish()


@app.local_entrypoint()
def main(poll_s: int = 120, once: bool = False):
    daemon.remote(poll_s=poll_s, once=once)
