"""Wait for the midtrain-only SFT (mixeq_midtrain_only_from_base) to finish, then spawn its RL arm (initmid) via spawn_rl_ablation.py."""
import json, subprocess, time, modal
P = "/home/celeste/shared/overnight/rl_ablation_ids.json"; LOG = "/home/celeste/shared/overnight/abl_armA_driver.log"
def log(m):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
fc = modal.FunctionCall.from_id(json.load(open(P))["sft_initmid"]["train"])
while True:
    try:
        r = fc.get(timeout=0); log(f"SFT finished: {str(r)[:200]}"); break
    except TimeoutError:
        time.sleep(120); continue
    except Exception as e:
        log(f"SFT call FAILED: {type(e).__name__}: {str(e)[:300]} -> not launching arm initmid"); subprocess.run(["notify-discord", "SFT-init ablation: midtrain-only SFT failed; arm initmid not launched"]); raise SystemExit(1)
ls = subprocess.run(["modal", "volume", "ls", "maemm-data", "sft_mix/mixeq_midtrain_only_from_base/final"], capture_output=True, text=True).stdout
if "adapter_model" not in ls:
    log(f"final adapter not found on the volume: {ls[:300]}"); raise SystemExit(1)
out = subprocess.run(["python3", "/tmp/spawn_rl_ablation.py", "initmid"], capture_output=True, text=True); log(out.stdout.strip() + out.stderr.strip()[-300:])
