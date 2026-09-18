"""Modal app: best-of-N rejection-sampling HARVEST for the activation->text inverter (Qwen3.6-27B), and the distilled-bank
composer. Stages:

  plan(out_name, spec_json, ...)            CPU   sample targets (per bank, per family quotas) from existing banks; gather their unit
                                                  directions; write /data/harvest/<out>/targets_XX.jsonl + dirs_XX.npy (f16) + plan.json
  harvest_shard(out_name, shard, tag, ...)  GPU   one B200: N rollouts per target from a full-model checkpoint (vLLM + fast injection
                                                  hook), RL reward on the clean base, top-k kept -> /data/harvest/<out>/<tag>/shard_XX.jsonl
  probe_report(out_name, tags)              CPU   best-of-n curves per family/tag (needs --probe shards), vs the original text
  finalize(out_name, tags, bank_out, ...)   CPU   distilled SFT bank /data/banks/<bank_out>: vecs.f32 + records.jsonl (target_text = best
                                                  rollout, provenance kept) + build_stats.json + meta.json  (pretrain.py / modal_sft /
                                                  rl_disagg._bank_open / mix composer compatible)

    cd ~/maemm-pub-simple2m && MODAL_PROFILE=safety-sahan modal deploy data/modal_harvest_bon.py
    python scripts/launchers/spawn_harvest_bon.py --probe        # plan + 5 sampler configs on 1 GPU each
"""
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
app = modal.App(os.environ.get("HARVEST_APP", "maemm-harvest-bon-s2m"))
GPU = os.environ.get("HARVEST_GPU", "B200:1")
D = 5120
HROOT = "/data/harvest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
    .pip_install("transformers==5.15.0", "peft==0.20.0", "accelerate==1.14.0", "wandb==0.28.2", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet")
    .pip_install("flash-linear-attention==0.5.2")
    .add_local_file(REPO / "rl" / "rl.py", "/pmx/RL/rl_hf.py")
    .add_local_file(REPO / "rl" / "rl_disagg.py", "/pmx/RL/rl_disagg.py")
    .add_local_file(REPO / "rl" / "fast_lens_ext.py", "/pmx/helpers/fast_lens_ext.py")
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_file(REPO / "eval" / "eval_universal.py", "/pmx/eval/eval_universal.py")
    .add_local_file(REPO / "data" / "harvest_bon_worker.py", "/pmx/harvest/harvest_bon_worker.py")
)
cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.4.6")
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)


def _env():
    env = dict(os.environ)
    env.update({"PYTHONPATH": "/pmx/helpers:/pmx/eval:/pmx/RL:/pmx/harvest", "HF_HOME": "/data/hf_cache", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    return env


# ----------------------------------------------------------------------------------------------------------------------- plan
@app.function(image=cpu_image, cpu=8, memory=96 * 1024, volumes={"/data": vol}, timeout=6 * 3600)
def plan(out_name: str, spec_json: str, seed: int = 2050, n_shards: int = 8, overwrite: bool = False):
    """spec = [{"bank": "/data/banks/mix_simple2m_sft", "families": {"realact": 400000, "sae2m": 150000}}, ...]. Banks must be composer
    output (records line i == vec_idx i, vecs.f32 unit rows) or any bank whose records carry vec_idx/family/target_text (vec_idx is used).
    Writes targets_XX.jsonl (bank_id, vec_idx, family, target_text, src fields) + dirs_XX.npy [n, 5120] f16, shard = i % n_shards after a
    seeded global shuffle."""
    import numpy as np
    vol.reload()
    out = f"{HROOT}/{out_name}"
    if os.path.exists(f"{out}/plan.json") and not overwrite:
        return json.load(open(f"{out}/plan.json"))
    os.makedirs(out, exist_ok=True)
    spec = json.loads(spec_json)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    log = lambda m: print(f"[plan +{time.time() - t0:5.0f}s] {m}", flush=True)
    chosen = []   # dicts
    per_bank = []
    for bid, s in enumerate(spec):
        bank = s["bank"]; quotas = s["families"]
        st = json.load(open(f"{bank}/build_stats.json"))
        n = int(st["n_examples"])
        fams = list(quotas)
        fam_id = {f: i for i, f in enumerate(fams)}
        lab = np.full(n, -1, dtype=np.int8)
        vidx = np.full(n, -1, dtype=np.int64)
        with open(f"{bank}/records.jsonl") as f:            # pass 1: family per line (cheap key sniffing before json.loads)
            for li, line in enumerate(f):
                for fam in fams:
                    if f'"family": "{fam}"' in line or f'"family":"{fam}"' in line:
                        r = json.loads(line); lab[li] = fam_id[r["family"]]; vidx[li] = int(r["vec_idx"]); break
        picks = {}
        for fam, q in quotas.items():
            cand = np.nonzero(lab == fam_id[fam])[0]
            k = min(int(q), len(cand))
            picks[fam] = np.sort(rng.choice(cand, size=k, replace=False)) if k < len(cand) else cand
            log(f"bank {bid} {bank}: {fam} {k}/{len(cand)} lines")
        want = {int(li): fam for fam, arr in picks.items() for li in arr.tolist()}
        rows = {}
        with open(f"{bank}/records.jsonl") as f:            # pass 2: the selected records in full
            for li, line in enumerate(f):
                if li in want:
                    r = json.loads(line)
                    rows[li] = {"bank_id": bid, "bank": bank, "line": li, "vec_idx": int(r["vec_idx"]), "family": r["family"], "target_text": r["target_text"],
                                "src": {k: r[k] for k in ("src_bank", "src_vec_idx", "doc_idx", "feature", "ctx_len", "W", "full_ctx", "full_forward", "window_rank") if k in r}}
        # gather vectors (sorted by vec_idx -> mostly sequential reads)
        vf, dt = (f"{bank}/vecs.f32", np.float32) if os.path.exists(f"{bank}/vecs.f32") else (f"{bank}/vecs.f16", np.float16)
        nv = os.path.getsize(vf) // (D * np.dtype(dt).itemsize)
        vecs = np.memmap(vf, dt, "r", shape=(nv, D))
        order = sorted(rows, key=lambda li: rows[li]["vec_idx"])
        vi = np.asarray([rows[li]["vec_idx"] for li in order], dtype=np.int64)
        assert vi.min() >= 0 and vi.max() < nv, f"vec_idx out of range for {bank}"
        got = np.empty((len(order), D), dtype=np.float16)
        for i0 in range(0, len(order), 20000):
            got[i0: i0 + 20000] = np.asarray(vecs[vi[i0: i0 + 20000]]).astype(np.float16)
        for k, li in enumerate(order):
            rows[li]["_vec"] = got[k]
        chosen.extend(rows[li] for li in order)
        per_bank.append({"bank": bank, "n_lines": n, "picked": {fam: int(len(arr)) for fam, arr in picks.items()}, "vec_file": vf, "vec_dtype": str(np.dtype(dt))})
        log(f"bank {bid}: gathered {len(order)} vectors")
    perm = rng.permutation(len(chosen))
    chosen = [chosen[i] for i in perm]
    fam_counts = {}
    for sh in range(n_shards):
        part = chosen[sh::n_shards]
        with open(f"{out}/targets_{sh:02d}.jsonl", "w") as f:
            for r in part:
                fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
                f.write(json.dumps({k: v for k, v in r.items() if k != "_vec"}) + "\n")
        np.save(f"{out}/dirs_{sh:02d}.npy", np.stack([r["_vec"] for r in part]) if part else np.zeros((0, D), np.float16))
        log(f"shard {sh}: {len(part)} targets")
    pl = {"out_name": out_name, "n_targets": len(chosen), "n_shards": n_shards, "seed": seed, "spec": spec, "banks": per_bank, "families": fam_counts,
          "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_s": round(time.time() - t0)}
    json.dump(pl, open(f"{out}/plan.json", "w"), indent=1)
    vol.commit()
    log(f"DONE {len(chosen)} targets in {n_shards} shards: {fam_counts}")
    return pl


# -------------------------------------------------------------------------------------------------------------- harvest shard
@app.function(image=image, gpu=GPU, cpu=8, memory=64 * 1024, volumes={"/data": vol}, secrets=[modal.Secret.from_name("maemm-hf")],
              timeout=24 * 3600)
def harvest_shard(out_name: str, shard: int, tag: str, ckpt: str, n_samples: int = 32, temperature: float = 1.0, top_k: int = 4,
                  max_new_tokens: int = 96, min_new_tokens: int = 8, reward_window_last: int = 0, dirs_per_call: int = 0,
                  max_num_seqs: int = 512, vllm_gpu_mem: float = 0.5, cuda_graphs: bool = True, probe: bool = False, seed: int = 0,
                  limit: int = 0, extra_args: str = "", partial_every_s: int = 600, also_window: int = -1, queue_mult: int = 1, quant: str = ""):
    """Runs the worker on this container's GPU for shard `shard` of plan `out_name`; sampler config named `tag` (e.g. s250_t1.0).
    Partial output is copied to the volume every `partial_every_s` (shard_XX.partial.jsonl) and resumed from on restart."""
    vol.reload()
    src = f"{HROOT}/{out_name}"
    dst = f"{src}/{tag}"
    os.makedirs(dst, exist_ok=True)
    assert os.path.exists(f"{src}/plan.json"), f"no plan at {src}"
    assert os.path.exists(f"{ckpt}/SAVE_DONE") or "/" not in ckpt, f"{ckpt} has no SAVE_DONE"
    loc = "/root/harvest"; os.makedirs(loc, exist_ok=True)
    shutil.copy(f"{src}/targets_{shard:02d}.jsonl", f"{loc}/targets.jsonl")
    shutil.copy(f"{src}/dirs_{shard:02d}.npy", f"{loc}/dirs.npy")
    out_local, man_local = f"{loc}/shard.jsonl", f"{loc}/manifest.json"
    final, partial, done_p = f"{dst}/shard_{shard:02d}.jsonl", f"{dst}/shard_{shard:02d}.partial.jsonl", f"{dst}/shard_{shard:02d}.done.json"
    if os.path.exists(done_p):
        print(f"[modal] shard {shard} of {tag} already done", flush=True)
        return json.load(open(done_p))
    resume = False
    if os.path.exists(partial):
        shutil.copy(partial, out_local); resume = True
        print(f"[modal] resuming from partial ({sum(1 for _ in open(out_local))} targets)", flush=True)
    cmd = ["python", "/pmx/harvest/harvest_bon_worker.py", "--ckpt", ckpt, "--targets", f"{loc}/targets.jsonl", "--dirs", f"{loc}/dirs.npy",
           "--out", out_local, "--manifest", man_local, "--n-samples", str(n_samples), "--temperature", str(temperature), "--top-k", str(top_k),
           "--max-new-tokens", str(max_new_tokens), "--min-new-tokens", str(min_new_tokens), "--reward-window-last", str(reward_window_last),
           "--dirs-per-call", str(dirs_per_call), "--max-num-seqs", str(max_num_seqs), "--vllm-gpu-mem", str(vllm_gpu_mem), "--seed", str(seed),
           "--also-window", str(also_window), "--queue-mult", str(queue_mult)] + (["--quant", quant] if quant else [])
    cmd += ["--cuda-graphs"] if cuda_graphs else []
    cmd += ["--probe"] if probe else []
    cmd += ["--resume"] if resume else []
    cmd += ["--limit", str(limit)] if limit else []
    cmd += extra_args.split()
    print("[modal] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/pmx", env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    stop = threading.Event()

    def _partial_loop():
        while not stop.wait(partial_every_s):
            try:
                if os.path.exists(out_local):
                    shutil.copy(out_local, partial); vol.commit()
            except Exception as e:  # noqa
                print(f"[modal] partial copy failed: {e}", flush=True)
    threading.Thread(target=_partial_loop, daemon=True).start()
    saw_done = False
    rc = None
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            if "HARVEST_DONE" in line:
                saw_done = True
                break          # the worker os._exit()s right after this line, but its vLLM engine-core child keeps the pipe open -> do not wait on EOF
        if saw_done:
            try:
                rc = p.wait(timeout=30)
            except Exception:  # noqa
                rc = 0
        else:
            rc = p.wait()
    finally:
        stop.set()
        # reap the whole process group (worker + engine core) whether we broke out or it died
        try:
            os.killpg(p.pid, signal.SIGTERM)
            time.sleep(3)
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:  # noqa
            pass
    if not (saw_done and os.path.exists(man_local)):   # (HARVEST_DONE is printed AFTER the manifest is written)
        if os.path.exists(out_local):
            shutil.copy(out_local, partial); vol.commit()
        raise RuntimeError(f"harvest worker exited rc={rc} without HARVEST_DONE (partial saved: {os.path.exists(partial)})")
    shutil.copy(out_local, final)
    man = json.load(open(man_local)); man.update({"shard": shard, "tag": tag, "out_name": out_name, "file": final})
    json.dump(man, open(done_p, "w"), indent=1)
    if os.path.exists(partial):
        os.remove(partial)
    vol.commit()
    print(f"[modal] shard {shard} of {tag} DONE: {man['n_done']} targets, {man['rollouts']} rollouts, {man['wall_s']} s", flush=True)
    return man


# ------------------------------------------------------------------------------------------------------------------ helpers
def _load_shards(out_name, tags, allow_partial=True):
    """{(bank_id, vec_idx): {tag: row}} over every shard file of the given tags; a shard's .partial.jsonl is used only when no final
    file exists (torn last line skipped)."""
    import glob
    rows = {}
    files = []
    for tag in tags:
        finals = sorted(glob.glob(f"{HROOT}/{out_name}/{tag}/shard_[0-9][0-9].jsonl"))
        partials = sorted(glob.glob(f"{HROOT}/{out_name}/{tag}/shard_[0-9][0-9].partial.jsonl")) if allow_partial else []
        have = {os.path.basename(f)[:8] for f in finals}
        use = finals + [f for f in partials if os.path.basename(f)[:8] not in have]
        for f in use:
            files.append(f)
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:  # noqa - torn tail of a partial file
                    continue
                rows.setdefault((r["bank_id"], r["vec_idx"]), {})[tag] = r
    return rows, files


# --------------------------------------------------------------------------------------------------------------- probe report
@app.function(image=cpu_image, cpu=4, memory=32 * 1024, volumes={"/data": vol}, timeout=3600)
def probe_report(out_name: str, tags_json: str):
    """Best-of-n curves (mean over targets of the prefix-max of the first n sample cosines, n = 1,2,4,...,N) per family x tag, plus the
    original text's cosine, the fraction of targets whose best rollout beats it, and the pooled-over-tags best. Written to
    /data/harvest/<out>/probe_report.json."""
    import numpy as np
    vol.reload()
    tags = json.loads(tags_json)
    rows, files = _load_shards(out_name, tags)
    rep = {"out_name": out_name, "tags": tags, "files": files, "n_targets": len(rows), "per_tag": {}, "pooled": {}}
    fams = sorted({r["family"] for d in rows.values() for r in d.values()})
    for tag in tags:
        rt = {k: d[tag] for k, d in rows.items() if tag in d}
        if not rt:
            continue
        N = max(r["n"] for r in rt.values())
        ns = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if n <= N]
        ent = {"n_targets": len(rt), "N": N, "temperature": next(iter(rt.values()))["temperature"], "ckpt": next(iter(rt.values()))["ckpt"], "families": {}}
        for fam in fams + ["all"]:
            sel = [r for r in rt.values() if fam == "all" or r["family"] == fam]
            if not sel:
                continue
            e = {"n": len(sel), "orig_cos": float(np.mean([r["orig_cos"] for r in sel])), "cos_mean": float(np.mean([r["cos_mean"] for r in sel])),
                 "cos_std_within": float(np.mean([r["cos_std"] for r in sel])), "best": float(np.mean([r["cos_max"] for r in sel])),
                 "beat_orig_frac": float(np.mean([r["cos_max"] > r["orig_cos"] for r in sel])), "len_mean": float(np.mean([r["len_mean"] for r in sel]))}
            if all("cos_all" in r for r in sel):
                e["best_of_n"] = {str(n): float(np.mean([max(r["cos_all"][:n]) for r in sel])) for n in ns}
                e["reward_best_n_tok"] = float(np.mean([r["best"]["n_tok"] for r in sel]))
                wk = [k for k in sel[0] if k.startswith("cos_all_w")]
                if wk and all(wk[0] in r for r in sel):
                    w = wk[0][len("cos_all_"):]
                    e[f"best_of_n_{w}"] = {str(n): float(np.mean([max(r[wk[0]][:n]) for r in sel])) for n in ns}
                    e[f"orig_cos_{w}"] = float(np.mean([r[f"orig_cos_{w}"] for r in sel]))
                    e[f"cos_mean_{w}"] = float(np.mean([r[f"cos_mean_{w}"] for r in sel]))
            ent["families"][fam] = e
        rep["per_tag"][tag] = ent
    for fam in fams + ["all"]:
        sel = [d for d in rows.values() if fam == "all" or next(iter(d.values()))["family"] == fam]
        if not sel:
            continue
        pooled_best = [max(r["cos_max"] for r in d.values()) for d in sel]
        orig = [next(iter(d.values()))["orig_cos"] for d in sel]
        rep["pooled"][fam] = {"n": len(sel), "best_over_tags": float(np.mean(pooled_best)), "orig_cos": float(np.mean(orig)),
                              "beat_orig_frac": float(np.mean([b > o for b, o in zip(pooled_best, orig)])),
                              "best_tag_hist": {t: int(sum(1 for d in sel if max(d, key=lambda k: d[k]["cos_max"]) == t)) for t in tags}}
    json.dump(rep, open(f"{HROOT}/{out_name}/probe_report.json", "w"), indent=1)
    vol.commit()
    return rep


# ------------------------------------------------------------------------------------------------------------------- finalize
@app.function(image=cpu_image, cpu=8, memory=64 * 1024, volumes={"/data": vol}, timeout=6 * 3600)
def finalize(out_name: str, tags_json: str, bank_out: str, k_keep: int = 1, select: str = "reward", min_cos: float = -1.0,
             require_beat_orig: bool = False, max_rep3: float = 0.5, overwrite: bool = False):
    """Distilled SFT bank /data/banks/<bank_out>: pool the top-k lists of every tag per target, drop degenerate texts (fraction of repeated
    3-grams > max_rep3, empty), optional min_cos / beat-original filters, sort by `select` (reward|cos), keep k_keep rows per target (the
    vector row is duplicated per kept text). records line i == vec_idx i; target_text = the rollout; provenance kept."""
    import numpy as np
    vol.reload()
    tags = json.loads(tags_json)
    out = f"/data/banks/{bank_out}"
    if os.path.exists(f"{out}/build_stats.json") and not overwrite:
        raise RuntimeError(f"{out} exists (overwrite=False)")
    os.makedirs(out, exist_ok=True)
    plan_ = json.load(open(f"{HROOT}/{out_name}/plan.json"))
    rows, files = _load_shards(out_name, tags)
    # vectors: from the plan shards (targets_XX.jsonl <-> dirs_XX.npy)
    vec_of = {}
    for sh in range(plan_["n_shards"]):
        tg = [json.loads(l) for l in open(f"{HROOT}/{out_name}/targets_{sh:02d}.jsonl")]
        dv = np.load(f"{HROOT}/{out_name}/dirs_{sh:02d}.npy")
        for r, v in zip(tg, dv):
            vec_of[(r["bank_id"], r["vec_idx"])] = (v, r)

    def rep3(text):
        w = text.split()
        if len(w) < 6:
            return 0.0
        g = [tuple(w[i: i + 3]) for i in range(len(w) - 2)]
        return 1.0 - len(set(g)) / len(g)

    recs, vecs, fam_counts, stats = [], [], {}, {"targets": 0, "kept": 0, "dropped_rep": 0, "dropped_cos": 0, "dropped_orig": 0, "no_candidate": 0,
                                                 "sum_best_cos": 0.0, "sum_orig_cos": 0.0, "beat_orig": 0, "by_tag": {}}
    for key, per_tag in rows.items():
        stats["targets"] += 1
        v, tgt = vec_of[key]
        orig_cos = next(iter(per_tag.values()))["orig_cos"]
        cands = []
        for tag, r in per_tag.items():
            for c in r["topk"]:
                t = c["text"].strip()
                if not t:
                    continue
                if rep3(t) > max_rep3:
                    stats["dropped_rep"] += 1; continue
                if c["cos"] < min_cos:
                    stats["dropped_cos"] += 1; continue
                if require_beat_orig and c["cos"] <= orig_cos:
                    stats["dropped_orig"] += 1; continue
                cands.append((c[select], tag, c))
        if not cands:
            stats["no_candidate"] += 1; continue
        cands.sort(key=lambda x: -x[0])
        seen = set(); kept = 0
        for _, tag, c in cands:
            if c["text"] in seen:
                continue
            seen.add(c["text"])
            recs.append({"vec_idx": len(recs), "target_text": c["text"], "family": tgt["family"], "src_bank": tgt["bank"], "src_vec_idx": tgt["vec_idx"],
                         "src_line": tgt["line"], "orig_text": tgt["target_text"][:300], "orig_cos": orig_cos, "cos": c["cos"], "reward": c["reward"],
                         "n_tok": c["n_tok"], "sampler": tag, "rank_in_target": kept, "src": tgt.get("src", {}), "kind": "harvest_bon"})
            vecs.append(v); fam_counts[tgt["family"]] = fam_counts.get(tgt["family"], 0) + 1
            stats["by_tag"][tag] = stats["by_tag"].get(tag, 0) + 1
            if kept == 0:
                stats["sum_best_cos"] += c["cos"]; stats["sum_orig_cos"] += orig_cos; stats["beat_orig"] += int(c["cos"] > orig_cos)
            kept += 1; stats["kept"] += 1
            if kept >= k_keep:
                break
    V = np.stack(vecs).astype(np.float32)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
    V.tofile(f"{out}/vecs.f32")
    with open(f"{out}/records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    n_t = max(stats["targets"] - stats["no_candidate"], 1)
    bs = {"kind": "harvest_bon", "n_examples": len(recs), "n_vecs": len(recs), "families": fam_counts, "layout": "records line i == vec_idx i; vecs.f32 unit rows",
          "d_model": D, "harvest": out_name, "tags": tags, "shard_files": files, "k_keep": k_keep, "select": select, "min_cos": min_cos,
          "require_beat_orig": require_beat_orig, "max_rep3": max_rep3, "plan": {k: plan_[k] for k in ("n_targets", "families", "spec", "seed")},
          "stats": {**stats, "mean_best_cos": stats["sum_best_cos"] / n_t, "mean_orig_cos": stats["sum_orig_cos"] / n_t, "beat_orig_frac": stats["beat_orig"] / n_t},
          "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(bs, open(f"{out}/build_stats.json", "w"), indent=1)
    json.dump({**bs, "trainer_args": {"note": "target_text = best-of-N rollout of the sampler; the prompt is the standard marker prompt"}},
              open(f"{out}/meta.json", "w"), indent=1)
    vol.commit()
    print(f"[finalize] {out}: {len(recs)} rows {fam_counts} | best cos {bs['stats']['mean_best_cos']:.3f} vs orig {bs['stats']['mean_orig_cos']:.3f} "
          f"(beat {bs['stats']['beat_orig_frac']:.2f}) | dropped rep {stats['dropped_rep']} cos {stats['dropped_cos']} orig {stats['dropped_orig']} | "
          f"no-candidate targets {stats['no_candidate']}", flush=True)
    return bs
