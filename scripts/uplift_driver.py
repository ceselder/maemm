#!/usr/bin/env python3
"""Driver of the CROSS-UPLIFT MATRIX experiment (six arms; see data/modal_uplift_banks.py for the banks).

Per arm, in order, on DEPLOYED Modal apps (Function.spawn; the driver only polls and never blocks a GPU):
  1. bank            /data/banks/uplift_<arm>                    (maemm-uplift-banks build, spawned once for all arms)
  2. midtrain SFT    /data/sft_mix/uplift_sft_<arm>/final        (maemm-sft-8xb200 train: 1 epoch over the 200k bank, lr 1e-4,
                                                                  b32/GPU x 8 = eff 256, max_seq 160, prefix-cache, init = the common
                                                                  23M realact-only SFT final; <= MAX_SFT_CONCURRENT at a time)
  3. SFT eval        /data/eval_ckpt/uplift_sft_<arm>/ckpt_0.json (maemm-eval-ckpt-mlp daemon --once on `final`, v2 cache = the 11
                                                                  cos families + sae + mlp/mlp_pair, no extra evals)
  4. RL              /data/ckpts_uplift_<arm>/{step_25,step_50,step_100,final}
                                                                 (maemm-rl-disagg-x4 train: 1 rollout + 3 trainer B200, 100 steps,
                                                                  10 warmup, 128 groups x 16, ScaleRL/CISPO recipe of RL-C/D, init =
                                                                  the arm's SFT final, pool = the arm's bank)
  5. RL eval         /data/eval_ckpt/uplift_rl_<arm>/ckpt_{25,50,100}.json
                                                                 (maemm-eval-ckpt-mlp daemon spawned once step_25 exists, cancelled by
                                                                  the driver after ckpt_100 is scored)
State (call ids, done flags) persists in --state so the driver can be restarted at any time. Live knobs are re-read every
loop from --ctl (json): {"rl_lr": "1e-5", "rl_hold": false, "max_sft_concurrent": 2, "poll_s": 60}.

    MODAL_PROFILE=safety-sahan nohup python scripts/uplift_driver.py run --state /tmp/uplift/state.json --ctl /tmp/uplift/ctl.json > /tmp/uplift/driver.log 2>&1 &
    python scripts/uplift_driver.py status --state /tmp/uplift/state.json
"""
import argparse
import json
import os
import posixpath
import sys
import time

import modal

ARMS = ["acts100", "acts_sae", "acts_bsf", "acts_cluster", "acts_realact_long", "acts_mlp"]
INIT_ADAPTER = "/data/sft_mix/realact20m_prefix_lr1e-4/final"
EVAL_CACHE_V2 = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
EVAL_APP = "maemm-eval-ckpt-mlp"          # eval/modal_eval_ckpt.py deployed from the mlp42-bank worktree (v2-cache-aware daemon)
SFT_APP = "maemm-sft-8xb200"
RL_APP = "maemm-rl-disagg-x4"
BANK_APP = "maemm-uplift-banks"
SFT_EXTRA = ("--prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 20 "
             f"--init-adapter {INIT_ADAPTER}")
RL_SAVES = "25,50,100"
RL_STEPS = 100


def rl_extra_args(arm, lr):
    return (f"--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter "
            f"--npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 8 --fp32-head --autocast-bf16 --length-control penalty "
            f"--kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 128 --group-size 16 --lr {lr} --warmup-steps 10 "
            f"--len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 "
            f"--init-adapter /data/sft_mix/uplift_sft_{arm}/final --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32 "
            f"--save-every 0 --save-steps {RL_SAVES} --transcript-every 5 "
            f"--run-name rl_uplift_{arm} --save-dir /data/ckpts_uplift_{arm}")


def eval_extra_args():
    return f"--eval-cache {EVAL_CACHE_V2} --no-extra-evals"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
VOL = None


def vol():
    global VOL
    if VOL is None:
        VOL = modal.Volume.from_name("maemm-data")
    return VOL


def vexists(path):
    """Does <volume>/<path> exist? (path relative to the volume root, no leading slash)"""
    path = path.strip("/")
    d, f = posixpath.split(path)
    try:
        for e in vol().listdir(d):
            if posixpath.basename(e.path.rstrip("/")) == f:
                return True
    except Exception:  # noqa — missing dir, transient grpc error: treat as absent this tick
        return False
    return False


def vread_json(path):
    try:
        return json.loads(b"".join(vol().read_file(path.strip("/"))).decode())
    except Exception:  # noqa
        return None


def call_state(call_id):
    """'pending' | 'done' | ('failed', message) for a spawned FunctionCall."""
    if not call_id:
        return "none"
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        return "done"
    except TimeoutError:
        return "pending"
    except Exception as e:  # noqa — the remote exception (or a cancelled call)
        return ("failed", f"{type(e).__name__}: {str(e)[:300]}")


def cancel(call_id):
    try:
        modal.FunctionCall.from_id(call_id).cancel()
        return True
    except Exception as e:  # noqa
        log(f"cancel {call_id} failed: {e}")
        return False


def log(msg):
    print(f"[uplift {time.strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}", flush=True)


def load(path, default):
    try:
        return json.load(open(path))
    except Exception:  # noqa
        return default


def save(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def spawn(app, fn, **kw):
    fc = modal.Function.from_name(app, fn).spawn(**kw)
    log(f"spawned {app}.{fn} -> {fc.object_id} {json.dumps({k: (v if len(str(v)) < 120 else str(v)[:117] + '...') for k, v in kw.items()})}")
    return fc.object_id


RETRYABLE = ("bank not found on volume", "missing /data/", "No such file", "LocalEntryNotFound", "nothing pending", "does not exist",
             "exited rc=")   # rl_disagg / torchrun non-zero exit: the launcher hides the cause; a stale-mount init adapter is the common one
MAX_RETRIES = 3


def retry(s, stage, msg, events, arm, delay=180):
    """Modal reuses WARM containers whose /data mount predates files written by other containers (the bank, the SFT final,
    the RL ckpt dir): the preflight asserts fail ("bank not found on volume", "missing /data/...") or a --once evaluator finds
    nothing. Clear the stage so it is re-spawned after `delay` s (idle containers scale down by then) on a fresh container.
    Returns True when a retry was scheduled (bounded by MAX_RETRIES and the RETRYABLE signatures)."""
    n = s.get(f"{stage}_retries", 0)
    if n >= MAX_RETRIES or not any(k in (msg or "") for k in RETRYABLE):
        return False
    s[f"{stage}_retries"] = n + 1
    s[f"{stage}_retry_after"] = time.time() + delay
    s[f"{stage}_failed_calls"] = s.get(f"{stage}_failed_calls", []) + [{"call": s.get(f"{stage}_call"), "error": msg, "ts": time.time()}]
    for k in (f"{stage}_call", f"{stage}_failed", f"{stage}_spawn_ts"):
        s.pop(k, None)
    events.append(f"{stage.upper()} {arm}: retry {n + 1}/{MAX_RETRIES} scheduled in {delay}s (stale-mount signature)")
    return True


def retry_ok(s, stage):
    return time.time() >= s.get(f"{stage}_retry_after", 0)


# ---------------------------------------------------------------------------------------------
# one driver tick
# ---------------------------------------------------------------------------------------------
def tick(st, ctl):
    arms = {a: st["arms"].setdefault(a, {}) for a in ARMS}
    events = []

    # 1. banks
    if not st.get("bank_call"):
        st["bank_call"] = spawn(BANK_APP, "build")
    for a, s in arms.items():
        if not s.get("bank_ready") and vexists(f"banks/uplift_{a}/build_stats.json"):
            s["bank_ready"] = True; events.append(f"bank {a} READY")
    if not all(s.get("bank_ready") for s in arms.values()):
        cs = call_state(st["bank_call"])
        if isinstance(cs, tuple):
            events.append(f"BANK BUILD FAILED: {cs[1]}")
            if not st.get("bank_failed_logged"):
                st["bank_failed_logged"] = True

    # 2. SFT (<= max concurrent)
    running_sft = 0
    for a, s in arms.items():
        if s.get("sft_call") and not s.get("sft_done"):
            if vexists(f"sft_mix/uplift_sft_{a}/final/adapter_model.safetensors"):
                s["sft_done"] = True; s["sft_done_ts"] = time.time(); events.append(f"SFT {a} DONE")
                meta = vread_json(f"sft_mix/uplift_sft_{a}/run_meta.json")
                if meta:
                    s["sft_wandb_id"] = meta.get("wandb_id")
            else:
                cs = call_state(s["sft_call"])
                if isinstance(cs, tuple) and not s.get("sft_failed"):
                    s["sft_failed"] = cs[1]; events.append(f"SFT {a} FAILED: {cs[1]}")
                    retry(s, "sft", cs[1], events, a, delay=int(ctl.get("retry_delay_s", 180)))
                if s.get("sft_call") and not s.get("sft_failed"):
                    running_sft += 1
    for a, s in arms.items():
        if s.get("bank_ready") and not s.get("sft_call") and running_sft < int(ctl.get("max_sft_concurrent", 2)) and retry_ok(s, "sft"):
            s["sft_call"] = spawn(SFT_APP, "train", run_name=f"uplift_sft_{a}", data_dir=f"/data/banks/uplift_{a}", n_ckpts=1, epochs=1,
                                  batch_size=32, lr=1e-4, max_seq=160, backend="nccl", extra_args=SFT_EXTRA)
            s["sft_spawn_ts"] = time.time(); running_sft += 1

    # 3. SFT eval (once, `final` as ckpt_step 0)
    for a, s in arms.items():
        if s.get("sft_done") and not s.get("sft_eval_call") and retry_ok(s, "sft_eval"):
            s["sft_eval_call"] = spawn(EVAL_APP, "daemon", ckpt_dir=f"/data/sft_mix/uplift_sft_{a}", tag=f"uplift_sft_{a}", rl_run_id="",
                                       wandb_name=f"uplift_sft_{a}_eval", final_step=0, once=True, only_step=0, extra_args=eval_extra_args())
        if s.get("sft_eval_call") and not s.get("sft_eval_done"):
            if vexists(f"eval_ckpt/uplift_sft_{a}/ckpt_0.json"):
                s["sft_eval_done"] = True; events.append(f"SFT-EVAL {a} DONE")
            else:
                cs = call_state(s["sft_eval_call"])
                if cs == "done" and not s.get("sft_eval_failed"):     # --once exited without scoring: stale mount saw no `final`
                    s["sft_eval_failed"] = "nothing pending (call returned without ckpt_0.json)"
                    events.append(f"SFT-EVAL {a} returned without a result (stale mount?)")
                    retry(s, "sft_eval", s["sft_eval_failed"], events, a, delay=int(ctl.get("retry_delay_s", 180)))
                elif isinstance(cs, tuple) and not s.get("sft_eval_failed"):
                    s["sft_eval_failed"] = cs[1]; events.append(f"SFT-EVAL {a} FAILED: {cs[1]}")
                    retry(s, "sft_eval", cs[1], events, a, delay=int(ctl.get("retry_delay_s", 180)))

    # 4. RL (each arm as soon as its own SFT final exists, unless held)
    for a, s in arms.items():
        if s.get("sft_done") and not s.get("rl_call") and not ctl.get("rl_hold", False) and retry_ok(s, "rl"):
            lr = str(ctl.get("rl_lr", "1e-5"))
            s["rl_lr"] = lr
            s["rl_call"] = spawn(RL_APP, "train", n_rollout=1, n_trainer=3, total_steps=RL_STEPS, extra_args=rl_extra_args(a, lr),
                                 pool_dir=f"/data/banks/uplift_{a}")
            s["rl_spawn_ts"] = time.time()
        if s.get("rl_call") and not s.get("rl_done"):
            if vexists(f"ckpts_uplift_{a}/final/adapter_model.safetensors"):
                s["rl_done"] = True; s["rl_done_ts"] = time.time(); events.append(f"RL {a} DONE")
            else:
                cs = call_state(s["rl_call"])
                if cs == "done":
                    s["rl_done"] = True; s["rl_done_ts"] = time.time(); events.append(f"RL {a} call returned (final not seen yet?)")
                elif isinstance(cs, tuple) and not s.get("rl_failed"):
                    s["rl_failed"] = cs[1]; events.append(f"RL {a} FAILED: {cs[1]}")
                    retry(s, "rl", cs[1], events, a, delay=int(ctl.get("retry_delay_s", 180)))
            if not s.get("rl_wandb_id"):
                wid = None
                try:
                    wid = b"".join(vol().read_file(f"ckpts_uplift_{a}/wandb_id.txt")).decode().strip()
                except Exception:  # noqa
                    pass
                if wid:
                    s["rl_wandb_id"] = wid
            for k in RL_SAVES.split(","):
                if not s.get(f"rl_step_{k}_ts") and vexists(f"ckpts_uplift_{a}/step_{k}/adapter_model.safetensors"):
                    s[f"rl_step_{k}_ts"] = time.time(); events.append(f"RL {a} step_{k} saved")

    # 5. RL eval daemon: start at the first checkpoint, stop after ckpt 100 is scored
    for a, s in arms.items():
        first = RL_SAVES.split(",")[0]
        if s.get("rl_call") and not s.get("rl_eval_call") and (s.get(f"rl_step_{first}_ts") or s.get("rl_done")) and retry_ok(s, "rl_eval"):
            s["rl_eval_call"] = spawn(EVAL_APP, "daemon", ckpt_dir=f"/data/ckpts_uplift_{a}", tag=f"uplift_rl_{a}", rl_run_id=s.get("rl_wandb_id", ""),
                                      wandb_name=f"uplift_rl_{a}_eval", final_step=RL_STEPS, extra_args=eval_extra_args())
        if s.get("rl_eval_call") and not s.get("rl_eval_done"):
            for k in RL_SAVES.split(","):
                if not s.get(f"rl_eval_{k}_done") and vexists(f"eval_ckpt/uplift_rl_{a}/ckpt_{k}.json"):
                    s[f"rl_eval_{k}_done"] = True; events.append(f"RL-EVAL {a} ckpt {k} DONE")
            if s.get(f"rl_eval_{RL_STEPS}_done") and s.get("rl_done"):
                all_k = all(s.get(f"rl_eval_{k}_done") for k in RL_SAVES.split(","))
                if all_k or time.time() - s.get("rl_done_ts", time.time()) > 2 * 3600:
                    if cancel(s["rl_eval_call"]):
                        s["rl_eval_done"] = True; events.append(f"RL-EVAL {a} complete ({'all ckpts' if all_k else 'ckpt 100; gave up on the rest'}) -> daemon cancelled")
            if not s.get("rl_eval_done"):
                cs = call_state(s["rl_eval_call"])
                if cs == "done" and not s.get("rl_eval_failed"):       # the daemon never exits by itself -> it died / was killed
                    s["rl_eval_failed"] = "daemon call returned early (missing /data/ ckpts?)"
                    events.append(f"RL-EVAL {a} daemon returned early")
                    retry(s, "rl_eval", s["rl_eval_failed"], events, a, delay=int(ctl.get("retry_delay_s", 180)))
                elif isinstance(cs, tuple) and not s.get("rl_eval_failed"):
                    s["rl_eval_failed"] = cs[1]; events.append(f"RL-EVAL {a} daemon FAILED: {cs[1]}")
                    retry(s, "rl_eval", cs[1], events, a, delay=int(ctl.get("retry_delay_s", 180)))

    # init eval (spawned by hand before the driver; tracked here for the report)
    if st.get("init_eval_call") and not st.get("init_eval_done") and vexists("eval_ckpt/uplift_init_realact23m/ckpt_0.json"):
        st["init_eval_done"] = True; events.append("INIT-EVAL DONE")
    return events


def summary(st):
    rows = []
    for a in ARMS:
        s = st["arms"].get(a, {})
        def f(k, ok="Y", no="."):
            return ok if s.get(k) else no
        rows.append(f"{a:18s} bank {f('bank_ready')} sft {f('sft_done', 'Y', 'run' if s.get('sft_call') else '.')} "
                    f"sft-eval {f('sft_eval_done', 'Y', 'run' if s.get('sft_eval_call') else '.')} "
                    f"rl {f('rl_done', 'Y', 'run' if s.get('rl_call') else '.')} "
                    f"[{''.join(k for k in ('25', '50', '100') if s.get(f'rl_step_{k}_ts')) or '-'}] "
                    f"rl-eval {f('rl_eval_done', 'Y', 'run' if s.get('rl_eval_call') else '.')} "
                    f"[{','.join(k for k in ('25', '50', '100') if s.get(f'rl_eval_{k}_done')) or '-'}] "
                    f"lr {s.get('rl_lr', '-')} wandb sft {s.get('sft_wandb_id', '-')} rl {s.get('rl_wandb_id', '-')}"
                    + (f" | FAILED: {[k for k in s if k.endswith('_failed')]}" if any(k.endswith('_failed') for k in s) else ""))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "tick"])
    ap.add_argument("--state", default="/tmp/uplift/state.json")
    ap.add_argument("--ctl", default="/tmp/uplift/ctl.json")
    a = ap.parse_args()
    st = load(a.state, {"arms": {}})
    if a.cmd == "status":
        print(summary(st)); print(json.dumps({k: v for k, v in st.items() if k != "arms"}, indent=1))
        return
    while True:
        ctl = load(a.ctl, {})
        try:
            events = tick(st, ctl)
        except Exception as e:  # noqa — transient Modal/grpc errors must not kill the driver
            events = [f"tick error: {type(e).__name__}: {str(e)[:300]}"]
        save(a.state, st)
        for e in events:
            log(e)
        done = all(st["arms"].get(x, {}).get("rl_eval_done") for x in ARMS)
        if events or int(time.time()) % 600 < int(ctl.get("poll_s", 60)):
            log("status:\n" + summary(st))
        if a.cmd == "tick":
            break
        if done:
            log("ALL ARMS COMPLETE (banks, SFT, SFT eval, RL, RL eval) -> exiting; build the report now")
            break
        time.sleep(int(ctl.get("poll_s", 60)))


if __name__ == "__main__":
    sys.exit(main())
