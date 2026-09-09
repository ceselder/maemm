"""Modal harness for rl/fullparam_update_bench.py: the --full-param update knobs on 2-4 B200 (no vLLM, synthetic rollouts),
in the SAME image modal_rl_disagg.py builds (transformers fork, fla). Each torchrun invocation gets a wall-clock cap so a
hanging config is killed and reported instead of blocking the box; results (one JSON per config) are printed and copied to
/data/disagg_runs/fullparam_bench_<ts>.json.

    source ~/modal_venv/bin/activate; export MODAL_PROFILE=safety-sahan
    FPB_GPU=B200:2 modal run --detach rl/modal_fullparam_bench.py --configs base,head,ckpt,ckpt+head,ckpt+head@24 --prefetches 0,2
"""
import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("DISAGG_APP", "maemm-rl-fullparam-bench")
os.environ.setdefault("DISAGG_TRANSFORMERS", "transformers @ git+https://github.com/ceselder/transformers@e52940e567ab9a991a1c971c1094e340233baff3")
import modal_rl_disagg as M  # noqa: E402

app = modal.App("maemm-rl-fullparam-bench")
GPU = os.environ.get("FPB_GPU", "B200:2")
image = (M.image.add_local_file(HERE / "modal_rl_disagg.py", "/root/modal_rl_disagg.py")
                .add_local_file(HERE / "fullparam_update_bench.py", "/pmx/RL/fullparam_update_bench.py"))


@app.function(image=image, gpu=GPU, volumes={"/data": M.vol}, secrets=[modal.Secret.from_name("maemm-hf")], timeout=3 * 3600)
def bench(policy_base: str, configs: str, prefetches: str = "0", rollouts: int = 48, uneven: int = 8, reps: int = 2, cap_s: int = 1500):
    import re
    import signal
    import subprocess
    import time
    env = M._env()
    n_gpu = M._n_gpus() if hasattr(M, "_n_gpus") else int(subprocess.check_output(["nvidia-smi", "-L"], text=True).count("GPU "))
    out = {"gpu": GPU, "n_gpu": n_gpu, "policy_base": policy_base, "runs": []}
    for pf in [int(x) for x in prefetches.split(",") if x != ""]:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={n_gpu}", "/pmx/RL/fullparam_update_bench.py", "--policy-base", policy_base,
               "--configs", configs, "--rollouts", str(rollouts), "--uneven", str(uneven), "--reps", str(reps), "--fsdp-prefetch", str(pf)]
        print("[bench] " + " ".join(cmd), flush=True)
        p = subprocess.Popen(cmd, cwd="/pmx", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        rows, t0, hung, lines = [], time.time(), False, []

        def reader():
            for line in p.stdout:
                print(line, end="", flush=True)
                lines.append(line)
                if line.startswith("RESULT_JSON "):
                    rows.append(json.loads(line[len("RESULT_JSON "):]))
        import threading
        th = threading.Thread(target=reader, daemon=True); th.start()
        while p.poll() is None:
            if time.time() - t0 > cap_s:
                hung = True
                print(f"[bench] prefetch {pf}: exceeded {cap_s}s -> killing torchrun (configs done: {[r['config'] for r in rows]})", flush=True)
                os.killpg(p.pid, signal.SIGKILL)
                break
            time.sleep(5)
        th.join(timeout=30)
        rc = p.poll()
        done = [r["config"] for r in rows]
        planned = [c for c in configs.split(",") if c]
        out["runs"].append({"prefetch": pf, "rc": rc, "hung": hung, "results": rows, "configs_done": done,
                            "first_undone": next((c for c in planned if c not in done), None), "wall_s": time.time() - t0,
                            "tail": [re.sub(r"\s+", " ", l)[:300] for l in lines[-25:]]})
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("/data/disagg_runs", exist_ok=True)
    json.dump(out, open(f"/data/disagg_runs/fullparam_bench_{ts}.json", "w"), indent=1)
    M.vol.commit()
    print("BENCH_SUMMARY " + json.dumps({"file": f"/data/disagg_runs/fullparam_bench_{ts}.json",
                                        "runs": [{k: v for k, v in r.items() if k != "tail"} for r in out["runs"]]}), flush=True)
    return out


@app.local_entrypoint()
def main(policy_base: str = "/data/sft_mix/mixeq_midtrain_fft_from_fft23m_v2/final", configs: str = "base,head,ckpt,ckpt+head,ckpt+head@24",
         prefetches: str = "0", rollouts: int = 48, uneven: int = 8, reps: int = 2, cap_s: int = 1500):
    bench.remote(policy_base=policy_base, configs=configs, prefetches=prefetches, rollouts=rollouts, uneven=uneven, reps=reps, cap_s=cap_s)
