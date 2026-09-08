"""Scale the real-activation SFT corpus 23M -> 50M and train on it. Parts E/F/G (9M each, seeds 11/12/13, 8xH200) are spawned one at a time, each excluding the
FineFineWeb files of every earlier part (exclusion reads the other banks' assignment.json, so we wait for each assignment to exist before spawning the next).
Then merge -> /data/banks/realact_short_50m_all -> SFT realact50m_b16k_lr1e-4 (eff batch 16,384, lr 1e-4, same recipe as realact23m_b16k_lr1e-4) + evaluator.
Log: ~/shared/overnight/auto_collect_50m.log; ids: ~/shared/overnight/collect_50m_ids.json."""
import json, os, subprocess, sys, time
import modal
LOG = os.path.expanduser("~/shared/overnight/auto_collect_50m.log"); IDS = os.path.expanduser("~/shared/overnight/collect_50m_ids.json")
OLD = ["realact_short_20m", "realact_short_20m_b", "realact_short_20m_c", "realact_short_20m_d"]
NEW = [("realact_short_20m_e", 11), ("realact_short_20m_f", 12), ("realact_short_20m_g", 13)]
def log(m):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
def vol_has(path):
    r = subprocess.run(["modal", "volume", "ls", "maemm-data", os.path.dirname(path)], capture_output=True, text=True)
    return os.path.basename(path) in r.stdout
ids = json.load(open(IDS)) if os.path.exists(IDS) else {}
collect = modal.Function.from_name("maemm-collect-bank", "collect")
done = list(OLD)
for name, seed in NEW:
    if name not in ids:
        c = collect.spawn(n_examples=9_000_000, out_name=name, seed=seed, exclude_from=",".join(done))
        ids[name] = {"call": c.object_id, "seed": seed, "exclude_from": ",".join(done)}; json.dump(ids, open(IDS, "w"), indent=1)
        log(f"spawned {name} seed {seed} (9M, excludes {len(done)} banks) {c.object_id}")
    t0 = time.time()
    while not vol_has(f"banks/{name}/shards/assignment.json"):
        if time.time() - t0 > 3600: log(f"ABORT: {name} assignment.json not written after 1 h"); sys.exit(1)
        time.sleep(30)
    log(f"{name} assignment.json present -> next part may exclude it"); done.append(name)
while not all(vol_has(f"banks/{n}/build_stats.json") for n, _ in NEW):
    time.sleep(120)
log("all three parts FINALIZED")
if "merge" not in ids:
    parts = ",".join(OLD + [n for n, _ in NEW])
    m = modal.Function.from_name("maemm-collect-bank", "merge").spawn(out_name="realact_short_50m_all", parts=parts)
    ids["merge"] = {"call": m.object_id, "parts": parts}; json.dump(ids, open(IDS, "w"), indent=1); log(f"merge spawned {m.object_id}")
while not vol_has("banks/realact_short_50m_all/build_stats.json"):
    time.sleep(60)
log("merged bank realact_short_50m_all ready")
if "sft" not in ids:
    run = "realact50m_b16k_lr1e-4"
    extra = "--prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 1 --grad-accum 32 --pad-multiple 1 --save-examples 250000,500000,1000000,2000000,5000000,10000000,23000000"
    t = modal.Function.from_name("maemm-sft-8xb200", "train").spawn(run_name=run, data_dir="/data/banks/realact_short_50m_all", n_ckpts=14, epochs=1,
                                                                      batch_size=64, lr=1e-4, max_seq=160, backend="nccl", extra_args=extra)
    e = modal.Function.from_name("maemm-eval-ckpt", "daemon").spawn(ckpt_dir=f"/data/sft_mix/{run}", tag=f"sft_{run}", rl_run_id="", wandb_name=f"{run}_eval", final_step=3052)
    ids["sft"] = {"run": run, "train": t.object_id, "eval": e.object_id, "save": f"/data/sft_mix/{run}", "eff_batch": 16384, "lr": 1e-4, "steps_total_expected": 3052}
    json.dump(ids, open(IDS, "w"), indent=1); log(f"SFT {run} spawned train {t.object_id} eval {e.object_id}")
    subprocess.run(["notify-discord", f"50M real-activation bank merged; SFT {run} (eff batch 16k, lr 1e-4) launched on 8xB200 ({t.object_id}); evaluator {e.object_id}"])
log("driver done")
