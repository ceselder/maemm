"""CHAIN v2 (re-cut bank): wait for the compose call -> verify /data/banks/mix_5m_sft -> spawn FFT midtrain (eff 4096, lr 1e-5, max_seq 192,
init = 104M FFT final) + fullmodel evaluator -> wait midtrain final -> spawn FULL-PARAM RL (spawn_rl_fullparam.py prod 1e-6 --split 2+6,
policy = midtrain final, run rl_fullparam_fft104m_mix5m_8x512; RL app from env RL_FULLPARAM_APP if the launcher supports it).
ids -> ~/shared/overnight/midtrain5m_ids.json ; log -> ~/shared/overnight/midtrain5m_driver.log"""
import json, math, os, subprocess, time, modal
IDS = "/home/celeste/shared/overnight/midtrain5m_ids.json"; LOG = "/home/celeste/shared/overnight/midtrain5m_driver.log"
ARM1 = "realact104m_fullft_b4096_lr1e-5"; RUN = "mix5msft_midtrain_fft_from_fft104m"; POOL = "/data/banks/mix_5m_sft"; EFF = 4096
COMPOSE = json.load(open("/tmp/bank5m/calls.json"))["compose_sft"]
def log(m):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
def wait_call(fc_id, what):
    fc = modal.FunctionCall.from_id(fc_id)
    while True:
        try:
            r = fc.get(timeout=0); log(f"{what} call finished: {str(r)[:300]}"); return r
        except TimeoutError:
            time.sleep(90)
        except Exception as e:
            log(f"{what} call FAILED: {type(e).__name__}: {str(e)[:400]}"); return None
def vol_ls(path):
    return subprocess.run(["modal", "volume", "ls", "maemm-data", path], capture_output=True, text=True).stdout
d = json.load(open(IDS)) if os.path.exists(IDS) else {}
# ---- 1. compose ----------------------------------------------------------------------------------------------------------
if "compose" not in d:
    r = wait_call(COMPOSE, "compose")
    ls = vol_ls("banks/mix_5m_sft")
    if r is None or "build_stats.json" not in ls or "vecs.f32" not in ls:
        subprocess.run(["notify-discord", "chain v2: compose of mix_5m_sft FAILED/incomplete — midtrain NOT launched"]); log(f"compose incomplete: {ls[:300]}"); raise SystemExit(1)
    subprocess.run(["modal", "volume", "get", "maemm-data", "banks/mix_5m_sft/build_stats.json", "/tmp/bank5m/mix5m_sft_build_stats.json", "--force"], capture_output=True)
    st = json.load(open("/tmp/bank5m/mix5m_sft_build_stats.json"))
    d["compose"] = {"call": COMPOSE, "bank": POOL, "n_examples": st["n_examples"], "families": st["families"]}; json.dump(d, open(IDS, "w"), indent=1)
    log(f"BANK READY: {st['n_examples']} rows {st['families']}")
    subprocess.run(["notify-discord", f"re-cut midtrain bank mix_5m_sft composed: {st['n_examples']} rows {st['families']} — launching FFT midtrain"])
N_ROWS = d["compose"]["n_examples"]; steps = math.ceil(N_ROWS / EFF)
# ---- 2. FFT midtrain ---------------------------------------------------------------------------------------------------
if "midtrain" not in d:
    extra = ("--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 16 --pad-multiple 1 --prefix-share-step "
             f"--fsdp-prefetch 2 --init-adapter /data/sft_mix/{ARM1}/final")
    t = modal.Function.from_name("maemm-sft-fullft", "train").spawn(run_name=RUN, data_dir=POOL, n_ckpts=4, epochs=1, batch_size=32, lr=1e-5,
                                                                     max_seq=192, backend="nccl", extra_args=extra)
    e = modal.Function.from_name("maemm-eval-ckpt-fullft", "fullmodel_daemon").spawn(ckpt_dir=f"/data/sft_mix/{RUN}", tag=f"sft_{RUN}",
                                                                                     wandb_name=f"{RUN}_eval", final_step=steps)
    d["midtrain"] = {"train": t.object_id, "eval": e.object_id, "run": RUN, "save": f"/data/sft_mix/{RUN}", "data": POOL, "eff_batch": EFF,
                     "micro_batch": 32, "grad_accum": 16, "lr": 1e-5, "max_seq": 192, "steps_expected": steps, "init": f"/data/sft_mix/{ARM1}/final",
                     "app": "maemm-sft-fullft", "eval_app": "maemm-eval-ckpt-fullft", "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "recipe": "FSDP2 full fine-tune midtrain on the re-cut bank (no realact_long; every target ends at the peak token; verbatim BSF; MLP fire-filtered), same eff batch/lr as the 104M pretrain"}
    json.dump(d, open(IDS, "w"), indent=1)
    log(f"MIDTRAIN spawned: train {t.object_id} eval {e.object_id} ({steps} steps expected)")
    subprocess.run(["notify-discord", f"FFT midtrain launched on mix_5m_sft ({RUN}, {N_ROWS} rows, {steps} steps, eff 4096, lr 1e-5); full-param RL follows automatically"])
# ---- 3. full-parameter RL ----------------------------------------------------------------------------------------------
r = wait_call(d["midtrain"]["train"], "midtrain")
ok = False
for _ in range(40):
    if "SAVE_DONE" in vol_ls(f"sft_mix/{RUN}/final"): ok = True; break
    time.sleep(60)
if not ok:
    subprocess.run(["notify-discord", "chain v2: midtrain final missing — full-param RL NOT launched"]); log("midtrain final missing"); raise SystemExit(1)
if "rl" not in d:
    name = "rl_fullparam_fft104m_mix5m_8x512"
    env = dict(os.environ); env.setdefault("RL_FULLPARAM_APP", "maemm-rl-disagg-fullparam2")
    out = subprocess.run(["python3", "/home/celeste/maemm-pub/scripts/launchers/spawn_rl_fullparam.py", "prod", "1e-6", "--split", "2+6",
                          "--policy-base", f"/data/sft_mix/{RUN}/final", "--name", name, "--ids-key", "fft104m_mix5m_fullparam"], capture_output=True, text=True, env=env)
    log("RL spawn: " + out.stdout.strip() + out.stderr.strip()[-400:])
    rec = json.load(open("/home/celeste/shared/overnight/rl_fullparam_ids.json")).get("fft104m_mix5m_fullparam", {})
    d["rl"] = rec; json.dump(d, open(IDS, "w"), indent=1)
    subprocess.run(["notify-discord", f"FFT midtrain DONE -> FULL-PARAM RL launched: {name} (lr 1e-6, 2+6, 300 steps) train {rec.get('train')} eval {rec.get('eval')}"])
log("chain complete")
