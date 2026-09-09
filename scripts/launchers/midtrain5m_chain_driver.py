"""CHAIN (user 21:00Z Sep 9: "do the midtrain with same batch size now and then run full ft RL"):
  1. wait for the 104M full fine-tune (arm 1) to finish -> verify /data/sft_mix/realact104m_fullft_b4096_lr1e-5/final SAVE_DONE
  2. spawn FFT midtrain mix5m_midtrain_fft_from_fft104m on maemm-sft-fullft: /data/banks/mix_5m (4,788,545 rows, end-anchored),
     eff batch 4096 (mb 32 x ga 16), lr 1e-5 OneCycle, 1 epoch (~1170 steps), max_seq 192, init = arm-1 final; fullmodel evaluator
  3. wait for the midtrain -> verify final SAVE_DONE -> spawn FULL-PARAMETER RL (spawn_rl_fullparam.py prod, lr 1e-6, split 2+6,
     pool mix_eq_1p45m = same RL recipe/pool as every other arm so only the init differs)
ids -> ~/shared/overnight/midtrain5m_ids.json ; log -> ~/shared/overnight/midtrain5m_driver.log"""
import json, os, subprocess, time, modal
IDS = "/home/celeste/shared/overnight/midtrain5m_ids.json"; LOG = "/home/celeste/shared/overnight/midtrain5m_driver.log"
ARM1 = "realact104m_fullft_b4096_lr1e-5"; RUN = "mix5m_midtrain_fft_from_fft104m"; POOL = "/data/banks/mix_5m"
N_ROWS, EFF = 4_788_545, 4096
def log(m):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
def wait_call(fc_id, what):
    fc = modal.FunctionCall.from_id(fc_id)
    while True:
        try:
            r = fc.get(timeout=0); log(f"{what} call finished: {str(r)[:200]}"); return True
        except TimeoutError:
            time.sleep(120)
        except Exception as e:
            log(f"{what} call FAILED: {type(e).__name__}: {str(e)[:300]}"); return False
def wait_final(path, tries=30):
    for _ in range(tries):
        ls = subprocess.run(["modal", "volume", "ls", "maemm-data", path], capture_output=True, text=True).stdout
        if "SAVE_DONE" in ls: return True
        time.sleep(60)
    log(f"final not found / no SAVE_DONE at {path}: {ls[:300]}"); return False
d = json.load(open(IDS)) if os.path.exists(IDS) else {}
# ---- 1. arm 1 ----------------------------------------------------------------------------------------------------------
arm1_call = json.load(open("/home/celeste/shared/overnight/sft_fullft_100m_ids.json"))[ARM1]["train"]
log(f"waiting for arm 1 train call {arm1_call}")
ok = wait_call(arm1_call, "arm1")
if not wait_final(f"sft_mix/{ARM1}/final"):
    subprocess.run(["notify-discord", "midtrain5m chain: arm-1 final missing after its call ended — midtrain NOT launched"]); raise SystemExit(1)
# ---- 2. FFT midtrain ---------------------------------------------------------------------------------------------------
if "midtrain" not in d:
    extra = ("--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 16 --pad-multiple 1 --prefix-share-step "
             f"--fsdp-prefetch 2 --init-adapter /data/sft_mix/{ARM1}/final")
    t = modal.Function.from_name("maemm-sft-fullft", "train").spawn(run_name=RUN, data_dir=POOL, n_ckpts=4, epochs=1, batch_size=32, lr=1e-5,
                                                                     max_seq=192, backend="nccl", extra_args=extra)
    steps = -(-N_ROWS // EFF)
    e = modal.Function.from_name("maemm-eval-ckpt-fullft", "fullmodel_daemon").spawn(ckpt_dir=f"/data/sft_mix/{RUN}", tag=f"sft_{RUN}",
                                                                                     wandb_name=f"{RUN}_eval", final_step=steps)
    d["midtrain"] = {"train": t.object_id, "eval": e.object_id, "run": RUN, "save": f"/data/sft_mix/{RUN}", "data": POOL, "eff_batch": EFF,
                     "micro_batch": 32, "grad_accum": 16, "lr": 1e-5, "max_seq": 192, "steps_expected": steps, "init": f"/data/sft_mix/{ARM1}/final",
                     "app": "maemm-sft-fullft", "eval_app": "maemm-eval-ckpt-fullft", "spawned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "recipe": "FSDP2 full fine-tune midtrain on the fresh 5M 9-family bank (end-anchored targets), same eff batch/lr as the 104M pretrain"}
    json.dump(d, open(IDS, "w"), indent=1)
    log(f"MIDTRAIN spawned: train {t.object_id} eval {e.object_id} ({steps} steps expected)")
    subprocess.run(["notify-discord", f"104M FFT pretrain DONE -> FFT midtrain on mix_5m launched ({RUN}, {steps} steps, eff 4096, lr 1e-5); full-param RL follows automatically"])
# ---- 3. full-parameter RL ----------------------------------------------------------------------------------------------
ok = wait_call(d["midtrain"]["train"], "midtrain")
if not wait_final(f"sft_mix/{RUN}/final"):
    subprocess.run(["notify-discord", "midtrain5m chain: midtrain final missing — full-param RL NOT launched"]); raise SystemExit(1)
if "rl" not in d:
    name = "rl_fullparam_fft104m_mix5m_8x512"
    out = subprocess.run(["python3", "/home/celeste/maemm-pub/scripts/launchers/spawn_rl_fullparam.py", "prod", "1e-6", "--split", "2+6",
                          "--policy-base", f"/data/sft_mix/{RUN}/final", "--name", name, "--ids-key", "fft104m_mix5m_fullparam"], capture_output=True, text=True)
    log("RL spawn: " + out.stdout.strip() + out.stderr.strip()[-400:])
    rec = json.load(open("/home/celeste/shared/overnight/rl_fullparam_ids.json")).get("fft104m_mix5m_fullparam", {})
    d["rl"] = rec; json.dump(d, open(IDS, "w"), indent=1)
    subprocess.run(["notify-discord", f"FFT midtrain DONE -> FULL-PARAM RL launched: {name} (lr 1e-6, 2+6, 300 steps) train {rec.get('train')} eval {rec.get('eval')}"])
log("chain complete")
