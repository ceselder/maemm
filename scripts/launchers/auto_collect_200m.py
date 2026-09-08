"""Scale the real-activation corpus toward ~200M: 17 more parts (h..x, 9M each, seeds 14..30, 8xH200 via maemm-collect-bank `collect`), up to 4 running at once,
each excluding the FineFineWeb files of EVERY earlier bank (waits for the previous part's assignment.json before spawning the next). Does NOT merge or launch SFT
(the merge needs a streaming rewrite for 200M records; handled separately). Log ~/shared/overnight/auto_collect_200m.log; ids ~/shared/overnight/collect_200m_ids.json."""
import json, os, subprocess, sys, time
import modal
LOG = os.path.expanduser("~/shared/overnight/auto_collect_200m.log"); IDS = os.path.expanduser("~/shared/overnight/collect_200m_ids.json")
EXISTING = ["realact_short_20m", "realact_short_20m_b", "realact_short_20m_c", "realact_short_20m_d", "realact_short_20m_e", "realact_short_20m_f", "realact_short_20m_g"]
NEW = [(f"realact_short_20m_{c}", 14 + i) for i, c in enumerate("hijklmnopqrstuvwx")]   # 17 parts x 9M = 153M -> ~203M total
MAX_RUNNING = 4; PER_PART = 9_000_000
def log(m):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
def vol_has(path):
    r = subprocess.run(["modal", "volume", "ls", "maemm-data", os.path.dirname(path)], capture_output=True, text=True)
    return os.path.basename(path) in r.stdout
ids = json.load(open(IDS)) if os.path.exists(IDS) else {}
collect = modal.Function.from_name("maemm-collect-bank", "collect")
def finalized(name): return vol_has(f"banks/{name}/build_stats.json")
def assigned(name): return vol_has(f"banks/{name}/shards/assignment.json") or vol_has(f"banks/{name}/assignment.json") or vol_has(f"banks/{name}/build_stats.json")   # finalize moves shards/assignment.json to the bank root
done_names = list(EXISTING)
i = 0
while i < len(NEW) or any(not finalized(n) for n, _ in NEW if n in ids):
    running = [n for n, _ in NEW if n in ids and not finalized(n)]
    if i < len(NEW) and len(running) < MAX_RUNNING:
        name, seed = NEW[i]
        prev = NEW[i - 1][0] if i > 0 else None
        if prev is None or assigned(prev):
            if name not in ids:
                excl = EXISTING + [n for n, _ in NEW[:i]]
                c = collect.spawn(n_examples=PER_PART, out_name=name, seed=seed, exclude_from=",".join(excl))
                ids[name] = {"call": c.object_id, "seed": seed, "n_excluded_banks": len(excl), "spawned": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                json.dump(ids, open(IDS, "w"), indent=1); log(f"spawned {name} seed {seed} (9M, excludes {len(excl)} banks; {len(running)+1} running) {c.object_id}")
            i += 1
            continue
    newly = [n for n, _ in NEW if n in ids and finalized(n) and not ids[n].get("finalized")]
    for n in newly:
        ids[n]["finalized"] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); json.dump(ids, open(IDS, "w"), indent=1); log(f"{n} FINALIZED ({sum(1 for m,_ in NEW if ids.get(m,{}).get('finalized'))}/{len(NEW)})")
    time.sleep(60)
log("ALL 17 PARTS FINALIZED (~153M new examples; total with a..g ≈ 203M). Merge NOT started (needs the streaming merge).")
subprocess.run(["notify-discord", "Real-activation collection: all 17 extra parts (h..x, ~153M examples) finalized; total corpus ≈ 203M. Streaming merge + SFT next."])
