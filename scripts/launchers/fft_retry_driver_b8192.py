"""Respawn the 104M FFT leg whenever a launch attempt dies BEFORE training starts (staging throttled / startup error), landing on a
fresh host each time. Stops retrying once the run dir has run_meta.json (training started; the modal_sft supervisor owns resumes from
then on) or after MAX_TRIES. Log ~/shared/overnight/fft_retry_driver_b8192.log; ids ~/shared/overnight/sft_fullft_100m_ids.json."""
import json, os, subprocess, time
import modal
P = os.path.expanduser("~/shared/overnight/sft_fullft_100m_ids.json"); LOG = os.path.expanduser("~/shared/overnight/fft_retry_driver_b8192.log")
RUN = "realact104m_fullft_b8192_lr3e-5"; MAX_TRIES = 8
DATA = "/data/banks/realact_short_50m_all,/data/banks/realact_short_20m_h,/data/banks/realact_short_20m_i,/data/banks/realact_short_20m_j,/data/banks/realact_short_20m_k,/data/banks/realact_short_20m_l,/data/banks/realact_short_20m_m"
EXTRA = "--full-ft --prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 16 --pad-multiple 1 --save-examples 2000000,5000000,10000000,23000000,50000000 --prefix-share-step --fsdp-prefetch 2"
def log(m):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
def training_started():
    r = subprocess.run(["modal", "volume", "ls", "maemm-data", f"sft_mix/{RUN}"], capture_output=True, text=True)
    return "run_meta.json" in r.stdout
tries = 0
while tries < MAX_TRIES:
    d = json.load(open(P)); fc = modal.FunctionCall.from_id(d[RUN]["train"])
    try:
        fc.get(timeout=0); log(f"train call {d[RUN]['train']} RETURNED (leg finished) -> driver exits"); break
    except TimeoutError:
        time.sleep(60); continue
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        if training_started():
            log(f"train call failed AFTER training started ({msg}) -> the modal_sft supervisor owns the resume; driver exits"); break
        tries += 1
        log(f"attempt died before training started ({msg}); respawning (try {tries}/{MAX_TRIES})")
        t = modal.Function.from_name("maemm-sft-fullft", "train").spawn(run_name=RUN, data_dir=DATA, n_ckpts=14, epochs=1, batch_size=64, lr=3e-5, max_seq=160, backend="nccl", extra_args=EXTRA)
        d[RUN].setdefault("train_failed_legs", []).append(d[RUN]["train"]); d[RUN]["train"] = t.object_id; json.dump(d, open(P, "w"), indent=1)
        log(f"spawned {t.object_id}")
        time.sleep(120)
else:
    log("MAX_TRIES reached -- giving up"); subprocess.run(["notify-discord", f"FFT {RUN}: {MAX_TRIES} launch attempts died before training (staging throttled?) — needs a look"])
