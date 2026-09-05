"""Modal app `maemm-uplift-banks`: the six 200k-row training banks of the CROSS-UPLIFT MATRIX experiment.

Question: does midtraining + RL on "50% real activations + 50% direction family X" lift the inverter on the OTHER
held-out eval families (cross-uplift), relative to a 100%-real-activations control? One bank per arm, all the same size
(200k rows), all drawn from the SAME two already-leak-checked source banks with ONE seed:

    arm                 rows                                              x rows from
    acts100             200k realact (control)                            /data/banks/mix_1m_v2  family realact (subset A + B)
    acts_sae            100k realact (subset A) + 100k sae                /data/banks/mix_1m_v2  family sae
    acts_bsf            100k realact (subset A) + 100k bsf                /data/banks/mix_1m_v2  family bsf
    acts_cluster        100k realact (subset A) + 100k cluster (probes)   /data/banks/mix_1m_v2  family cluster
    acts_realact_long   100k realact (subset A) + 100k realact_long       /data/banks/mix_1m_v2  family realact_long
    acts_mlp            100k realact (subset A) + 100k mlp/mlp_pair       /data/banks/mlp42      families mlp + mlp_pair (all rows shuffled, first 100k)

The realact subset A is the SAME 100k rows in every 50/50 arm (a seeded permutation of mix_1m_v2's 250k realact rows;
A = first 100k, B = next 100k; the control is A + B), so arms differ ONLY in the X half. Every arm bank is a seeded
shuffle of its rows: vecs.f32 [N, 5120] unit rows + records.jsonl (line i == vec_idx i; each record = the source record
plus src_bank / src_vec_idx / arm) + build_stats.json (per-family counts, provenance, leak table) + meta.json.

Leak check (asserted per arm, GPU): max cosine of EVERY bank row against EVERY eval-cache direction of the v2 cache
(/data/eval_universal_ho/eval_sets_heldout_v2.pt: the 11 cos families + sae_dirs + the new mlp / mlp_pair families) and
against every /data/pool_heldout row must be < 0.999 (same rule and threshold as data/modal_bank_everything.py and
data/mlp42_bank_worker.py, whose banks are the sources). The per-(bank family x reference family) max-cos table is
stored in build_stats.json for the report.

Deploy + spawn (profile safety-sahan; a spawned call on a deployed app survives the local client):
    MODAL_PROFILE=safety-sahan modal deploy data/modal_uplift_banks.py
    python -c "import modal; print(modal.Function.from_name('maemm-uplift-banks', 'build').spawn().object_id)"
    python -c "import modal; print(modal.Function.from_name('maemm-uplift-banks', 'peek').remote('acts_mlp'))"
"""
import modal

APP_NAME = "maemm-uplift-banks"
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy==2.4.6")
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
SMALL_GPUS = ["H100", "A100-80GB", "L40S", "A100-40GB"]   # leak check only (a few GB); the rest is I/O

D_MODEL = 5120
SRC_MIX = "/data/banks/mix_1m_v2"
SRC_MLP = "/data/banks/mlp42"
EVAL_CACHE_V2 = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
POOL_HELDOUT = "/data/pool_heldout"
OUT_ROOT = "/data/banks"
LEAK_COS = 0.999

# arm -> (source bank, family filter for the X half); None = the control (realact subset A + B)
ARMS = {
    "acts100": None,
    "acts_sae": (SRC_MIX, ("sae",)),
    "acts_bsf": (SRC_MIX, ("bsf",)),
    "acts_cluster": (SRC_MIX, ("cluster",)),
    "acts_realact_long": (SRC_MIX, ("realact_long",)),
    "acts_mlp": (SRC_MLP, ("mlp", "mlp_pair")),
}
ARM_ORDER = list(ARMS)   # the rng consumption order (fixed -> reproducible selections)


def _log(msg):
    import time
    print(f"[uplift-banks {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_records(bank):
    """(raw lines, family per line, n) of a bank's records.jsonl; asserts line i == vec_idx i."""
    import json
    import os
    import numpy as np
    lines, fams = [], []
    with open(f"{bank}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            assert int(r["vec_idx"]) == i, f"{bank}: line {i} has vec_idx {r['vec_idx']}"
            lines.append(line.rstrip("\n")); fams.append(r["family"])
    st = json.load(open(f"{bank}/build_stats.json"))
    assert st["n_examples"] == len(lines), f"{bank}: build_stats n_examples {st['n_examples']} != {len(lines)} lines"
    assert os.path.getsize(f"{bank}/vecs.f32") == len(lines) * D_MODEL * 4, f"{bank}: vecs.f32 size != n_rows x d"
    return lines, np.array(fams), len(lines)


@app.function(image=image, gpu=SMALL_GPUS, cpu=8, memory=65536, ephemeral_disk=512 * 1024, volumes={"/data": vol},
              timeout=4 * 3600)
def build(arms: str = "all", n_half: int = 100_000, seed: int = 2027, overwrite: bool = False):
    import json
    import os
    import shutil
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F

    T0 = time.time()
    vol.reload()
    todo = ARM_ORDER if arms == "all" else [a for a in arms.split(",") if a]
    assert all(a in ARMS for a in todo), f"unknown arm(s) in {todo}; known {ARM_ORDER}"
    for a in todo:
        if os.path.exists(f"{OUT_ROOT}/uplift_{a}/build_stats.json") and not overwrite:
            raise RuntimeError(f"{OUT_ROOT}/uplift_{a} already built (pass overwrite=True to rebuild)")
    for p in (f"{SRC_MIX}/vecs.f32", f"{SRC_MIX}/records.jsonl", f"{SRC_MIX}/build_stats.json", f"{SRC_MLP}/vecs.f32",
              f"{SRC_MLP}/records.jsonl", f"{SRC_MLP}/build_stats.json", EVAL_CACHE_V2, f"{POOL_HELDOUT}/vecs.f32"):
        assert os.path.exists(p), f"missing input {p}"
    dev = "cuda:0"

    # ---- source records ----
    t0 = time.time()
    lines_mix, fam_mix, n_mix = _load_records(SRC_MIX)
    lines_mlp, fam_mlp, n_mlp = _load_records(SRC_MLP)
    _log(f"source records: {SRC_MIX} {n_mix} rows {dict(zip(*np.unique(fam_mix, return_counts=True)))} | "
         f"{SRC_MLP} {n_mlp} rows {dict(zip(*np.unique(fam_mlp, return_counts=True)))} ({time.time() - t0:.0f}s)")
    lines = {SRC_MIX: lines_mix, SRC_MLP: lines_mlp}
    fam_of = {SRC_MIX: fam_mix, SRC_MLP: fam_mlp}
    n_of = {SRC_MIX: n_mix, SRC_MLP: n_mlp}

    # ---- selections (ONE rng, fixed consumption order over ARM_ORDER -> reproducible) ----
    rng = np.random.default_rng(seed)
    realact_rows = np.flatnonzero(fam_mix == "realact")
    assert len(realact_rows) >= 2 * n_half, f"only {len(realact_rows)} realact rows for 2 x {n_half}"
    realact_perm = rng.permutation(realact_rows)
    sub_a, sub_b = np.sort(realact_perm[:n_half]), np.sort(realact_perm[n_half:2 * n_half])
    x_sel = {}
    for a in ARM_ORDER:                      # consume the rng for EVERY arm so a partial rebuild matches a full build
        spec = ARMS[a]
        if spec is None:
            continue
        bank, fams = spec
        rows = np.flatnonzero(np.isin(fam_of[bank], list(fams)))
        take = min(n_half, len(rows))
        x_sel[a] = (bank, np.sort(rng.permutation(rows)[:take]), len(rows))
    _log(f"realact subset A {len(sub_a)} rows (min {sub_a[0]} max {sub_a[-1]}), B {len(sub_b)} | "
         + " ".join(f"{a}:{len(s[1])}/{s[2]}" for a, s in x_sel.items()))

    # ---- per-arm row lists: (bank, src_row) in a seeded shuffled order; dst i == records line i ----
    plans = {}
    for ai, a in enumerate(ARM_ORDER):
        if a not in todo:
            continue
        if ARMS[a] is None:
            src = [(SRC_MIX, int(r)) for r in np.concatenate([sub_a, sub_b])]
        else:
            bank, sel, _ = x_sel[a]
            src = [(SRC_MIX, int(r)) for r in sub_a] + [(bank, int(r)) for r in sel]
        N = len(src)
        perm = np.random.default_rng(seed * 100 + ai).permutation(N)      # staged g -> dst perm[g]
        dst_of_src = {}
        for g, (bank, r) in enumerate(src):
            dst_of_src.setdefault(bank, []).append((r, int(perm[g])))
        gather = {}
        for bank, pairs in dst_of_src.items():
            arr = np.array(sorted(pairs), np.int64)                       # sorted by src row for the sequential pass
            gather[bank] = (arr[:, 0], arr[:, 1])
        stage = f"/root/stage/{a}"
        os.makedirs(stage, exist_ok=True)
        vecs = np.memmap(f"{stage}/vecs.f32", np.float32, "w+", shape=(N, D_MODEL))
        plans[a] = {"N": N, "gather": gather, "vecs": vecs, "stage": stage, "src": src, "perm": perm}
        _log(f"plan {a}: {N} rows " + " ".join(f"{os.path.basename(b)}={len(g[0])}" for b, g in gather.items()))

    # ---- ONE sequential pass per source bank, scatter into every arm's staged memmap ----
    CH = 16384
    for bank in (SRC_MIX, SRC_MLP):
        need = [a for a in plans if bank in plans[a]["gather"]]
        if not need:
            continue
        N = n_of[bank]
        src_mm = np.memmap(f"{bank}/vecs.f32", np.float32, "r", shape=(N, D_MODEL))
        t0 = time.time(); n_copied = 0
        lo_needed = min(int(plans[a]["gather"][bank][0][0]) for a in need)
        hi_needed = max(int(plans[a]["gather"][bank][0][-1]) for a in need) + 1
        for c0 in range(lo_needed - lo_needed % CH, hi_needed, CH):
            c1 = min(c0 + CH, N)
            chunk = None
            for a in need:
                srows, drows = plans[a]["gather"][bank]
                i0, i1 = np.searchsorted(srows, c0), np.searchsorted(srows, c1)
                if i1 <= i0:
                    continue
                if chunk is None:
                    chunk = np.ascontiguousarray(src_mm[c0:c1])
                plans[a]["vecs"][drows[i0:i1]] = chunk[srows[i0:i1] - c0]
                n_copied += i1 - i0
            if (c0 // CH) % 8 == 0:
                _log(f"  {os.path.basename(bank)}: rows {c1}/{N} copied {n_copied} ({(c1 - lo_needed) * D_MODEL * 4 / 2**30 / max(time.time() - t0, 1e-6):.2f} GB/s)")
        for a in need:
            plans[a]["vecs"].flush()
        _log(f"gather from {bank}: {n_copied} rows in {time.time() - t0:.0f}s")

    # ---- leak-check references: every v2 eval-cache direction family + all pool_heldout rows ----
    es = torch.load(EVAL_CACHE_V2, map_location="cpu", weights_only=False)
    ref_names = sorted(k[:-5] for k in es if k.endswith("_dirs"))
    refs = [F.normalize(es[f"{n}_dirs"].float(), dim=-1) for n in ref_names]
    ho_n = os.path.getsize(f"{POOL_HELDOUT}/vecs.f32") // (4 * D_MODEL)
    ho = np.memmap(f"{POOL_HELDOUT}/vecs.f32", np.float32, "r", shape=(ho_n, D_MODEL))
    refs.append(F.normalize(torch.from_numpy(np.asarray(ho)).float(), dim=-1)); ref_names.append("pool_heldout(all)")
    ref = torch.cat(refs).to(dev)
    offs = np.cumsum([0] + [len(r) for r in refs])
    _log(f"leak refs: {len(ref)} directions = " + " ".join(f"{n}:{len(r)}" for n, r in zip(ref_names, refs))
         + f" | cache cos_families {es['meta'].get('cos_families')} extra {es['meta'].get('extra_families')}")

    results = {}
    for a in todo:
        P = plans[a]; N = P["N"]
        out = f"{OUT_ROOT}/uplift_{a}"
        t0 = time.time()
        # records in dst order
        recs = [None] * N
        fam_dst = [None] * N
        for g, (bank, r) in enumerate(P["src"]):
            d = int(P["perm"][g])
            rec = json.loads(lines[bank][r])
            src_idx = int(rec["vec_idx"])
            assert src_idx == r
            rec = {"vec_idx": d, **{k: v for k, v in rec.items() if k != "vec_idx"},
                   "src_bank": bank, "src_vec_idx": src_idx, "arm": a}
            recs[d] = rec; fam_dst[d] = rec["family"]
        fam_dst = np.array(fam_dst)
        fam_names = sorted(set(fam_dst.tolist()))
        counts = {f: int((fam_dst == f).sum()) for f in fam_names}
        # vectors: unit norms + leak table
        X = np.memmap(f"{P['stage']}/vecs.f32", np.float32, "r", shape=(N, D_MODEL))
        nrm_min, nrm_max = 1.0, 1.0
        leak = np.full((len(fam_names), len(ref_names)), -1.0)
        worst_rows = []
        fam_idx = np.array([fam_names.index(f) for f in fam_dst])
        with torch.no_grad():
            for c0 in range(0, N, 8192):
                xb = torch.from_numpy(np.ascontiguousarray(X[c0:c0 + 8192])).to(dev)
                nn = xb.norm(dim=-1)
                nrm_min, nrm_max = min(nrm_min, float(nn.min())), max(nrm_max, float(nn.max()))
                cos = xb @ ref.T
                for ri in range(len(ref_names)):
                    cm = cos[:, offs[ri]:offs[ri + 1]].max(1).values.cpu().numpy()
                    for fi in range(len(fam_names)):
                        m = fam_idx[c0:c0 + 8192] == fi
                        if m.any():
                            leak[fi, ri] = max(leak[fi, ri], float(cm[m].max()))
                mx, am = cos.max(1)
                top = torch.topk(mx, min(5, len(mx)))
                for v, j in zip(top.values.tolist(), top.indices.tolist()):
                    ri = int(np.searchsorted(offs, int(am[j]), side="right") - 1)
                    worst_rows.append({"row": c0 + j, "family": str(fam_dst[c0 + j]), "ref": ref_names[ri], "cos": round(v, 5)})
        worst_rows = sorted(worst_rows, key=lambda w: -w["cos"])[:10]
        max_cos = float(leak.max())
        assert abs(nrm_min - 1) < 1e-3 and abs(nrm_max - 1) < 1e-3, f"{a}: non-unit rows [{nrm_min}, {nrm_max}]"
        assert max_cos < LEAK_COS, f"{a}: leak check FAILED: max cos {max_cos} >= {LEAK_COS}: {worst_rows[:3]}"
        leak_tbl = {fam_names[fi]: {ref_names[ri]: round(float(leak[fi, ri]), 4) for ri in range(len(ref_names))}
                    for fi in range(len(fam_names))}
        _log(f"{a}: {N} rows {counts} | unit norms [{nrm_min:.5f}, {nrm_max:.5f}] | leak max cos {max_cos:.4f} < {LEAK_COS} OK "
             f"| worst {worst_rows[:2]} ({time.time() - t0:.0f}s)")
        for f in fam_names:
            _log(f"    {f:13s} " + " ".join(f"{n.split('(')[0][:12]}={leak_tbl[f][n]:.3f}" for n in ref_names))

        # ---- publish ----
        t0 = time.time()
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out, exist_ok=True)
        shutil.copyfile(f"{P['stage']}/vecs.f32", f"{out}/vecs.f32.tmp"); os.replace(f"{out}/vecs.f32.tmp", f"{out}/vecs.f32")
        with open(f"{out}/records.jsonl.tmp", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(f"{out}/records.jsonl.tmp", f"{out}/records.jsonl")
        assert os.path.getsize(f"{out}/vecs.f32") == N * D_MODEL * 4
        spec = ARMS[a]
        sources = {SRC_MIX: {"path": SRC_MIX, "n_rows_taken": int(sum(1 for b, _ in P["src"] if b == SRC_MIX)),
                             "realact_subset": "A+B" if spec is None else "A",
                             "realact_rows_A": [int(sub_a[0]), int(sub_a[-1]), len(sub_a)],
                             "realact_rows_B": [int(sub_b[0]), int(sub_b[-1]), len(sub_b)] if spec is None else None}}
        if spec is not None:
            bank, sel, avail = x_sel[a]
            sources.setdefault(bank, {"path": bank, "n_rows_taken": 0})
            sources[bank].update({"x_families": list(spec[1]), "x_rows_taken": int(len(sel)), "x_rows_available": int(avail),
                                  "x_rule": f"seeded permutation of every {'/'.join(spec[1])} row of the source, first {n_half}"})
            sources[bank]["n_rows_taken"] = int(sum(1 for b, _ in P["src"] if b == bank))
        stats = {"kind": f"cross-uplift arm {a}: " + ("200k real activations (control)" if spec is None else
                         f"100k real activations (subset A) + 100k {'/'.join(spec[1])} rows"),
                 "arm": a, "x_family": None if spec is None else list(spec[1]),
                 "n_examples": N, "n_vecs": N, "families": counts,
                 "layout": "seeded shuffle of all rows (records.jsonl line i == vec_idx i; src_bank/src_vec_idx = provenance)",
                 "seed": seed, "arm_shuffle_seed": seed * 100 + ARM_ORDER.index(a), "n_half": n_half,
                 "realact_subset_rule": f"np.random.default_rng({seed}).permutation(realact rows of {SRC_MIX}); A = first {n_half}, B = next {n_half}; "
                                        "every 50/50 arm uses A, the control uses A+B",
                 "sources": sources, "d_model": D_MODEL, "model": "Qwen/Qwen3.6-27B", "layer": 42,
                 "leak_check": {"threshold": LEAK_COS, "max_cos": round(max_cos, 5), "rows_at_or_above_threshold": 0,
                                "refs": ref_names, "n_ref_dirs": int(len(ref)), "eval_cache": EVAL_CACHE_V2,
                                "max_cos_table": leak_tbl, "worst_rows": worst_rows},
                 "unit_norm_range": [nrm_min, nrm_max], "created": time.time()}
        json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
        json.dump({"bank": out, "arm": a, "n_rows": N, "families": counts, "seed": seed,
                   "trainer_args": {"--data-dir": out, "--bank-file": "vecs.f32", "--direction-source": "cluster"},
                   "sources": sources, "eval_cache_cos_families": es["meta"].get("cos_families"),
                   "eval_cache_extra_families": es["meta"].get("extra_families")}, open(f"{out}/meta.json", "w"), indent=1)
        vol.commit()
        _log(f"{a}: published -> {out} ({N * D_MODEL * 4 / 2**30:.1f} GB, {time.time() - t0:.0f}s)")
        results[a] = {"out": out, "n": N, "families": counts, "leak_max_cos": round(max_cos, 5)}
        del X
        shutil.rmtree(P["stage"], ignore_errors=True)
    _log(f"DONE {len(results)} arms in {(time.time() - T0) / 60:.1f} min: {json.dumps(results)}")
    return results


@app.function(image=image, cpu=4, memory=16384, volumes={"/data": vol}, timeout=1800)
def peek(arm: str = "acts_mlp", n: int = 2):
    """build_stats summary + n sample rows per family (unit-norm check + the source row's vector must match)."""
    import json
    import os
    import numpy as np
    vol.reload()
    bank = f"{OUT_ROOT}/uplift_{arm}"
    st = json.load(open(f"{bank}/build_stats.json"))
    print(json.dumps({k: st[k] for k in ("kind", "n_examples", "families", "seed", "sources")}, indent=1), flush=True)
    print("leak max cos", st["leak_check"]["max_cos"], "| worst", st["leak_check"]["worst_rows"][:3], flush=True)
    N = st["n_examples"]
    assert os.path.getsize(f"{bank}/vecs.f32") == N * D_MODEL * 4
    vecs = np.memmap(f"{bank}/vecs.f32", np.float32, "r", shape=(N, D_MODEL))
    seen = {f: 0 for f in st["families"]}
    srcs = {}
    out = []
    with open(f"{bank}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            assert r["vec_idx"] == i
            f = r["family"]
            if seen[f] < n:
                seen[f] += 1
                sb = r["src_bank"]
                if sb not in srcs:
                    ns = json.load(open(f"{sb}/build_stats.json"))["n_examples"]
                    srcs[sb] = np.memmap(f"{sb}/vecs.f32", np.float32, "r", shape=(ns, D_MODEL))
                v = np.asarray(vecs[i], np.float32); s = np.asarray(srcs[sb][r["src_vec_idx"]], np.float32)
                same = bool(np.array_equal(v, s))
                print(f"[{f}] row {i} |v|={np.linalg.norm(v):.4f} src {os.path.basename(sb)}#{r['src_vec_idx']} identical={same} :: "
                      f"{r['target_text'][:160]!r}", flush=True)
                out.append({"row": i, "family": f, "identical_to_source": same, "text": r["target_text"][:160]})
                assert same, "vector differs from its source row"
            if all(c >= n for c in seen.values()):
                break
    return out


@app.local_entrypoint()
def main(stage: str = "peek", arm: str = "acts_mlp"):
    if stage == "build":
        print(build.remote())
    else:
        print(peek.remote(arm))
