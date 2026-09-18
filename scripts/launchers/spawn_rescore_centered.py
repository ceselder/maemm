"""Re-score checkpoints with the CENTERED cosine scorer (fix 2026-09-18: eval_universal.center_mu) on the fresh app
maemm-eval-ckpt-s2m-c (deployed from this worktree; the legacy daemons on maemm-eval-ckpt-s2m keep running untouched).
Results land in /data/eval_ckpt/<tag>_c/ckpt_<k>.json with protocol.scorer.centered == true; call ids in
~/shared/overnight/simple2m/ids.json["rescore_centered"][tag].

    python scripts/launchers/spawn_rescore_centered.py once  <ckpt_dir> <tag> <final_step> <step,step,...>   # one B200 per step, in parallel
    python scripts/launchers/spawn_rescore_centered.py watch <ckpt_dir> <tag> <final_step>                    # fullmodel_daemon: backlog + new ckpts
"""
import json
import os
import sys
import time

import modal

os.environ.setdefault("MODAL_PROFILE", "safety-sahan")
APP = os.environ.get("RESCORE_APP", "maemm-eval-ckpt-s2m-c")
V3 = "/data/eval_universal_ho/eval_sets_heldout_v3.pt"
EXTRA = f"--eval-cache {V3} --no-extra-evals"
IDS = os.path.expanduser("~/shared/overnight/simple2m/ids.json")


def record(tag, rec):
    d = json.load(open(IDS)); d.setdefault("rescore_centered", {})
    d["rescore_centered"].setdefault(tag, {"app": APP, "calls": {}}); d["rescore_centered"][tag]["calls"].update(rec)
    d["rescore_centered"][tag]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); json.dump(d, open(IDS, "w"), indent=1)


def main():
    mode, ckpt_dir, tag, final_step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    ctag = tag if tag.endswith("_c") else tag + "_c"
    if mode == "once":
        steps = [int(x) for x in sys.argv[5].split(",")]
        rec = {}
        for s in steps:
            fc = modal.Function.from_name(APP, "daemon").spawn(ckpt_dir=ckpt_dir, tag=ctag, once=True, only_step=s, final_step=final_step,
                                                               extra_args=f"--full-model --no-wandb {EXTRA}")
            rec[f"step_{s}"] = fc.object_id; print(f"{ctag} step {s}: {fc.object_id}")
        record(ctag, rec)
    elif mode == "watch":
        fc = modal.Function.from_name(APP, "fullmodel_daemon").spawn(ckpt_dir=ckpt_dir, tag=ctag, wandb_name=f"{ctag}_eval", final_step=final_step,
                                                                     extra_args=EXTRA)
        print(f"{ctag} watcher: {fc.object_id}"); record(ctag, {"watcher": fc.object_id})
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
