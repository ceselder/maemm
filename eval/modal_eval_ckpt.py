"""Modal app `maemm-eval-ckpt`: ONE-GPU checkpoint eval daemon for the RL runs (eval/eval_ckpt_daemon.py).

Evaluates every <ckpt_dir>/step_* (+ final, + the SFT trainer's --save-examples points examples_<M>, logged at their
optimizer step with an extra `examples` field; see eval_ckpt_daemon.examples_ckpt_meta) -- rl.py's full inline_eval protocol
(512/family, Bo4, T=1, 16-64 tokens, SAE norm_act + rank) and the inline_extra_evals suite (locality, autointerp
AUC, WildChat AUC, adversarial holds; Sonnet 5 judge via Anthropic native with OpenRouter fallback) -- on one
B200 (or H200) hosting the HF base for scoring and a vLLM engine (fast steering hook + CUDA graphs) for generation.
Same image as modal_rl_disagg.py. Separate wandb run `eval_ckpt_<tag>` (x-axis ckpt_step).

Deploy + spawn (profile safety-sahan) -- a deployed function survives the local client:
    MODAL_PROFILE=safety-sahan EVAL_GPU=B200:1 modal deploy modal_eval_ckpt.py
    MODAL_PROFILE=safety-sahan python -c "import modal; modal.Function.from_name('maemm-eval-ckpt', 'daemon').spawn(
        ckpt_dir='/data/ckpts_last5_disagg_2x6', tag='last5_disagg_2x6', rl_run_id='<wandb id>',
        wandb_name='rl_everything_8x256_disagg_entropy2.0_last5win_eval')"
ONE GPU container at a time: the daemon holds its GPU while polling, so launch it only once the first checkpoint exists.
One-off (the step_90 protocol check):
    ... .spawn(ckpt_dir='/data/ckpts_last5_v15_g8', tag='last5_v15_g8', once=True, only_step=90)
Set EVAL_GPU (default B200:1; e.g. H200:1) at deploy time.
Dry run (CPU, no GPU) -- what a daemon would evaluate under a ckpt_dir right now (state file applied):
    ... modal.Function.from_name('maemm-eval-ckpt-fullft', 'list_pending').remote(ckpt_dir='/data/sft_mix/<run>', tag='sft_<run>',
        final_step=25391, full_model=True)      # or: modal run eval/modal_eval_ckpt.py --list-only --full-model --ckpt-dir ... --tag ...
RL adapters trained on a full-FT policy base (rl_disagg --policy-base): the SAME `daemon` -- eval_ckpt_daemon reads
<ckpt_dir>/run_meta.json's policy_base and serves base+adapter in vLLM while scoring on the clean MODEL (or pass policy_base=).
    EVAL_APP=maemm-eval-ckpt-fftbase modal deploy eval/modal_eval_ckpt.py
    ... modal.Function.from_name('maemm-eval-ckpt-fftbase', 'daemon').spawn(ckpt_dir='/data/ckpts_<run>', tag='<run>',
        extra_args='--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt')
Per-direction dump (one-off, its own app so the live daemons are untouched): extra_args='--dump-per-dir ...' makes
eval_ckpt_daemon ALSO write <out_dir>/perdir_ckpt_<k>.json (every direction's best-of-bo score + sae best texts; the metric json
is unchanged). A full-model checkpoint goes through `daemon` too: once=True, only_step=k, extra_args='--full-model ...'.
    EVAL_APP=maemm-eval-ckpt-perdir modal deploy eval/modal_eval_ckpt.py
    ... modal.Function.from_name('maemm-eval-ckpt-perdir', 'daemon').spawn(ckpt_dir='/data/sft_mix/<run>', tag='perdir_<run>', once=True,
        only_step=10107, extra_args='--dump-per-dir --no-extra-evals --no-wandb --eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt')
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent   # repo root (this launcher lives one level down)
app = modal.App(os.environ.get("EVAL_APP", "maemm-eval-ckpt"))   # EVAL_APP=maemm-eval-ckpt-h200 + EVAL_GPU=H200:1 = a second deployment for Hopper evaluators
GPU = os.environ.get("EVAL_GPU", "B200:1")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
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
    .pip_install("flash-linear-attention==0.5.2")
    .pip_install("anthropic")
    .add_local_file(REPO / "rl" / "rl.py", "/pmx/RL/rl_hf.py")
    .add_local_file(REPO / "rl" / "rl_disagg.py", "/pmx/RL/rl_disagg.py")               # _build_engine (fast hook + graphs)
    .add_local_file(REPO / "rl" / "fast_lens_ext.py", "/pmx/helpers/fast_lens_ext.py")
    .add_local_file(REPO / "eval" / "inline_extra_evals.py", "/pmx/RL/inline_extra_evals.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "eval" / "eval_universal.py", "/pmx/eval/eval_universal.py")
    .add_local_file(REPO / "eval" / "snippet_locality.py", "/pmx/eval/snippet_locality.py")
    .add_local_file(REPO / "eval" / "autointerp_detection.py", "/pmx/eval/autointerp_detection.py")
    .add_local_file(REPO / "eval" / "eval_ckpt_daemon.py", "/pmx/eval/eval_ckpt_daemon.py")
)

vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

# CPU-only image for the discovery dry run: eval_ckpt_daemon.py imports nothing but the stdlib at module level
list_image = modal.Image.debian_slim(python_version="3.12").add_local_file(REPO / "eval" / "eval_ckpt_daemon.py", "/pmx/eval/eval_ckpt_daemon.py")

_DAEMON_MOD = []


def _daemon_mod():
    """eval/eval_ckpt_daemon.py as a module (container: /pmx/eval; local: the repo) -- the launcher and the child share ONE
    checkpoint discovery (_scan / examples_ckpt_meta), so `--only-step k` always resolves to the dir the launcher meant."""
    if not _DAEMON_MOD:
        import importlib.util
        for cand in ("/pmx/eval/eval_ckpt_daemon.py", str(REPO / "eval" / "eval_ckpt_daemon.py")):
            if os.path.exists(cand):
                spec = importlib.util.spec_from_file_location("eval_ckpt_daemon", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _DAEMON_MOD.append(mod)
                break
        else:
            raise FileNotFoundError("eval_ckpt_daemon.py not found next to the launcher nor under /pmx/eval")
    return _DAEMON_MOD[0]


@app.function(image=image, gpu=GPU, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb"),
                       modal.Secret.from_name("maemm-anthropic"), modal.Secret.from_name("maemm-openrouter")],
              timeout=24 * 3600)
def daemon(ckpt_dir: str, tag: str, rl_run_id: str = "", poll_s: int = 120, once: bool = False, only_step: int = -1,
           final_step: int = 1000, vllm_gpu_mem: float = 0.5, wandb_name: str = "", extra_args: str = "", policy_base: str = ""):
    """LoRA checkpoints (one long-lived eval_ckpt_daemon.py process; it discovers step_*, examples_* and final itself, see _scan there).
    policy_base: the full-FT base the RL adapters under ckpt_dir were trained on (rl_disagg --policy-base). Default '' = auto:
    eval_ckpt_daemon reads <ckpt_dir>/run_meta.json (absent / MODEL -> the plain LoRA-on-MODEL protocol)."""
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers:/pmx/eval:/pmx/RL"
    env["HF_HOME"] = "/data/hf_cache"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs("/tmp/wandb", exist_ok=True)
    cmd = ["python", "/pmx/eval/eval_ckpt_daemon.py", "--ckpt-dir", ckpt_dir, "--tag", tag, "--rl-run-id", rl_run_id,
           "--poll-s", str(poll_s), "--final-step", str(final_step), "--vllm-gpu-mem", str(vllm_gpu_mem)]
    if wandb_name:
        cmd += ["--wandb-name", wandb_name]
    if once:
        cmd.append("--once")
    if only_step >= 0:
        cmd += ["--only-step", str(only_step)]
    if policy_base:
        cmd += ["--policy-base", policy_base]
    if extra_args:
        cmd += extra_args.split()
    print("[modal] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # The daemon polls the volume from a SUBPROCESS, where `modal.Volume.from_name(...).reload()` is not the mounted
    # handle (it silently no-ops) -> it would never see checkpoints committed after it started. Refresh the mount from
    # THIS process instead — but only while the daemon is idle: a reload during its model/asset load or a checkpoint
    # eval invalidates open file handles (that is how the first attempt died with LocalEntryNotFoundError).
    import threading
    _stop = threading.Event()
    _state = {"loaded": False, "busy": False}
    def _reload_loop():
        while not _stop.wait(30):
            if not _state["loaded"] or _state["busy"]:
                continue
            try:
                vol.reload()
            except Exception as e:  # noqa
                print(f"[modal] vol.reload failed: {e}", flush=True)
    threading.Thread(target=_reload_loop, daemon=True).start()
    for line in p.stdout:
        print(line, end="", flush=True)
        if "previously evaled:" in line:
            _state["loaded"] = True
        if "evaluating LATEST step" in line or "-> evaluating" in line or "] evaluating step" in line:
            _state["busy"] = True
        if "evaled in" in line or "adapter load failed" in line or "nothing pending" in line:
            _state["busy"] = False
    rc = p.wait()
    _stop.set()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"eval daemon exited rc={rc}")


def _scan_full(ckpt_dir, final_step):
    """Complete full-model checkpoints under ckpt_dir -> {ckpt_step: dir}: step_<k>, examples_<M> (pretrain.py --save-examples; ckpt_step =
    its recorded optimizer step - 1, i.e. on the step_* axis -- eval_ckpt_daemon.examples_ckpt_meta) and final (final_step). The child's
    own discovery, imported, so `--only-step k` resolves to the same dir."""
    return _daemon_mod()._scan(ckpt_dir, final_step, 0.0, full_model=True)


def _pending_listing(ckpt_dir, tag, final_step, full_model):
    """The discovery + state-file view every daemon here works from (list_pending / --list-only)."""
    import json
    D = _daemon_mod()
    avail = D._scan(ckpt_dir, final_step, 0.0, full_model=full_model)
    state = f"/data/eval_state/evaled_ckpt_{tag}.json"
    try:
        done = set(json.load(open(state))["done"])
    except Exception:  # noqa
        done = set()
    rows = []
    for k in sorted(avail):
        x = D._ckpt_extras(avail[k])
        rows.append({"ckpt_step": k, "dir": avail[k], "done": k in done, **x})
        note = f"  <- --save-examples point {x['examples']:,} ({x['optimizer_updates']} optimizer updates -> ckpt_step {k})" if x else ""
        print(f"[list] ckpt_step {k:>6}  {'DONE   ' if k in done else 'pending'}  {avail[k]}{note}", flush=True)
    pending = sorted(k for k in avail if k not in done)
    print(f"[list] {tag}: {'full-model' if full_model else 'LoRA'} layout under {ckpt_dir} | state {state} | previously evaled {sorted(done)} "
          f"| would evaluate (latest first): {pending[::-1] or 'nothing'}", flush=True)
    return {"tag": tag, "ckpt_dir": ckpt_dir, "full_model": full_model, "state": state, "done": sorted(done),
            "pending_latest_first": pending[::-1], "ckpts": rows}


@app.function(image=list_image, volumes={"/data": vol}, timeout=600)
def list_pending(ckpt_dir: str, tag: str, final_step: int = 1000, full_model: bool = True):
    """DRY RUN, CPU only: what fullmodel_daemon (full_model=True) / daemon (LoRA layout) would evaluate under ckpt_dir right now --
    every complete checkpoint by ckpt_step (step_*, examples_* with the derived step and their `examples`, final) minus the state file."""
    vol.reload()
    return _pending_listing(ckpt_dir, tag, final_step, full_model)


@app.function(image=image, gpu=GPU, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb"),
                       modal.Secret.from_name("maemm-anthropic"), modal.Secret.from_name("maemm-openrouter")],
              timeout=24 * 3600)
def fullmodel_daemon(ckpt_dir: str, tag: str, wandb_name: str = "", final_step: int = 1000, poll_s: int = 120,
                     vllm_gpu_mem: float = 0.5, extra_args: str = "", idle_exit_s: int = 6 * 3600, once_all: bool = False):
    """FULL-model checkpoints (sft/fullft.py layout): one eval_ckpt_daemon.py PROCESS per checkpoint (`--full-model --once
    --only-step k`, its vLLM engine loads that checkpoint), looping latest-first over every <ckpt_dir>/step_* + examples_<M>
    (--save-examples points, ckpt_step = recorded optimizer step - 1, logged with `examples` = M) + final that carries SAVE_DONE,
    until `final` (logged as final_step) is evaluated (or nothing new for idle_exit_s). State (a set of ckpt_steps, examples_* included):
    /data/eval_state/evaled_ckpt_<tag>.json. Same wandb run (<wandb_name>, id == name) across processes."""
    import json
    import subprocess
    import time
    env = os.environ.copy()
    env["PYTHONPATH"] = "/pmx/helpers:/pmx/eval:/pmx/RL"
    env["HF_HOME"] = "/data/hf_cache"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs("/tmp/wandb", exist_ok=True)
    state = f"/data/eval_state/evaled_ckpt_{tag}.json"
    os.makedirs(os.path.dirname(state), exist_ok=True)
    try:
        done = set(json.load(open(state))["done"])
    except Exception:  # noqa
        done = set()
    fails = {}
    last_new = time.time()
    print(f"[modal-full] tag {tag} ckpt_dir {ckpt_dir} previously evaled {sorted(done)}", flush=True)
    while True:
        try:
            vol.reload()
        except Exception as e:  # noqa
            print(f"[modal-full] vol.reload failed: {e}", flush=True)
        avail = _scan_full(ckpt_dir, final_step)
        todo = sorted(k for k in avail if k not in done and fails.get(k, 0) < 2)
        if not todo:
            if final_step in done or once_all or time.time() - last_new > idle_exit_s:
                print(f"[modal-full] done: evaled {sorted(done)} (final {'yes' if final_step in done else 'NO'}); exiting", flush=True)
                break
            time.sleep(poll_s)
            continue
        last_new = time.time()
        s = todo[-1]
        cmd = ["python", "/pmx/eval/eval_ckpt_daemon.py", "--ckpt-dir", ckpt_dir, "--tag", tag, "--final-step", str(final_step),
               "--vllm-gpu-mem", str(vllm_gpu_mem), "--full-model", "--once", "--only-step", str(s)]
        if wandb_name:
            cmd += ["--wandb-name", wandb_name]
        if extra_args:
            cmd += extra_args.split()
        print(f"[modal-full] pending {todo} -> step {s} = {avail[s]}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd="/pmx", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
        rc = p.wait()
        if rc == 0:
            done.add(s)
            json.dump({"done": sorted(done), "tag": tag, "ckpt_dir": ckpt_dir, "full_model": True}, open(state, "w"))
            vol.commit()
            print(f"[modal-full] step {s} DONE in {time.time() - t0:.0f}s", flush=True)
        else:
            fails[s] = fails.get(s, 0) + 1
            print(f"[modal-full] step {s} FAILED rc={rc} (attempt {fails[s]}); {'giving up on it' if fails[s] >= 2 else 'will retry'}", flush=True)
            time.sleep(60)


@app.local_entrypoint()
def main(ckpt_dir: str = "/data/ckpts_last5_v15_g8", tag: str = "last5_v15_g8", rl_run_id: str = "", once: bool = False,
         only_step: int = -1, final_step: int = 1000, extra_args: str = "", list_only: bool = False, full_model: bool = False):
    if list_only:   # dry run of the discovery (CPU container): what would be evaluated under ckpt_dir for this tag
        import json
        print(json.dumps(list_pending.remote(ckpt_dir=ckpt_dir, tag=tag, final_step=final_step, full_model=full_model), indent=1))
        return
    daemon.remote(ckpt_dir=ckpt_dir, tag=tag, rl_run_id=rl_run_id, once=once, only_step=only_step, final_step=final_step,
                  extra_args=extra_args)
