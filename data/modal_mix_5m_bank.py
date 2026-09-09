"""Modal app `maemm-mix-5m-bank`: compose the ~5M-row, 8-family SFT MIDTRAIN bank /data/banks/mix_5m from finalized source banks,
with per-family row caps and multiple sources per family (bank-5m). Output is the exact bank format sft/pretrain.py + rl/rl.py read:
    vecs.f32          [N, 5120] float32 UNIT rows (f16 sources are up-cast and re-normalized)
    records.jsonl     line i == vec_idx i: the source record + {"vec_idx": i, "src_bank", "src_vec_idx"}; "family" preserved
    build_stats.json  n_examples, families, per-source provenance, windows-per-feature / rows-per-neuron summaries, leak table
    meta.json         same + trainer args
Rows are a seeded global shuffle. Default spec (== the 5M bank):
    realact       1,000,000  uniformly over the 11 UNUSED realact_short_20m_{n..x} parts (9M rows each; NOT in the 104M pretrain
                             corpus = realact_short_50m_all + parts h..m), proportional to part size; f16 -> f32
    realact_long  <= 1,000,000  /data/banks/everything_5m_fresh  (fresh 512-token store, deep positions)
    sae           <= 1,000,000  /data/banks/everything_5m_fresh  (unit encoder cols x fresh max-activating windows)
    bsf           all           /data/banks/everything_5m_fresh  (fresh store block projections)
    cluster       all           /data/banks/everything_5m_fresh  (probes; the fixed 241,741 cap)
    mlp / mlp_pair / mlp_triple  all  /data/banks/mlp42_5m_fresh (fresh-store scan)
Leak guard (asserted, GPU): max cosine of EVERY output row vs EVERY direction of the v2 eval cache (all `*_dirs` families, 14) and
every /data/pool_heldout row < 0.999 (the same rule as the source builders; rows above it are dropped and reported — the
realact parts were hash-excluded from eval documents but never direction-checked before). Per-(family x reference) max-cos
table stored for the report.
    MODAL_PROFILE=safety-sahan modal deploy data/modal_mix_5m_bank.py
    python -c "import modal; print(modal.Function.from_name('maemm-mix-5m-bank','build').spawn().object_id)"
"""
import json
import os

import modal

APP_NAME = os.environ.get("MAEMM_MIX5M_APP", "maemm-mix-5m-bank")
app = modal.App(APP_NAME)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy==2.4.6")
)
SMALL_GPUS = ["H100", "A100-80GB", "L40S", "A100-40GB"]   # leak check only (a few GB); the rest is I/O
D = 5120
EVAL_CACHE_V2 = "/data/eval_universal_ho/eval_sets_heldout_v2.pt"
POOL_HELDOUT = "/data/pool_heldout"
LEAK_COS = 0.999
UNUSED_PARTS = [f"/data/banks/realact_short_20m_{c}" for c in "nopqrstuvwx"]      # NOT in the 104M pretrain corpus
PRETRAIN_PARTS = ["realact_short_50m_all(= 20m,b,c,d,e,f,g)", "h", "i", "j", "k", "l", "m"]
DEFAULT_SPEC = [
    {"name": "realact_parts", "banks": UNUSED_PARTS, "dtype": "f16", "families": ["realact"], "n": 1_000_000, "all_one_family": "realact",
     "fresh": "unused realact_short_20m parts (never trained on)"},
    {"name": "everything_5m_fresh", "banks": ["/data/banks/everything_5m_fresh"], "dtype": "f32",
     "families": ["realact_long", "sae", "sae_dec", "bsf", "cluster"], "n": {"realact_long": 1_000_000, "sae": 1_000_000, "sae_dec": 1_000_000}, "fresh": "fresh store /data/acts27b_fresh (cluster: fixed probe set)"},
    {"name": "mlp42_5m_fresh", "banks": ["/data/banks/mlp42_5m_fresh"], "dtype": "f32", "families": ["mlp", "mlp_pair", "mlp_triple"], "n": None,
     "fresh": "fresh store /data/acts27b_fresh scan"},
]


def _log(msg):
    import time
    print(f"[mix5m {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pread_full(fd, n, off):
    buf = bytearray(n); mv = memoryview(buf); got = 0
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise IOError(f"short read at {off + got}")
        got += k
    return buf


def _scan_records(bank, all_one_family=None):
    """-> (n_rows, family per line (np.str_), line->vec_idx map or None if line i == vec_idx i, raw lines or None).
    Banks built by our builders keep line i == vec_idx i and are fully parsed (lines kept). The realact parts (9M lines, shuffled
    line order, single family) are scanned with a cheap vec_idx parse only; their selected lines are re-read later."""
    import numpy as np
    st = json.load(open(f"{bank}/build_stats.json"))
    n = int(st["n_examples"])
    if all_one_family:
        assert st["families"] == {all_one_family: n}, (bank, st["families"])
        line_of_vec = np.full(n, -1, np.int64)
        with open(f"{bank}/records.jsonl") as fh:
            for i, line in enumerate(fh):
                assert line.startswith('{"vec_idx": '), (bank, i, line[:40])
                line_of_vec[int(line[12:line.index(",", 12)])] = i
        assert (line_of_vec >= 0).all(), f"{bank}: records.jsonl does not cover every vec_idx"
        return n, None, line_of_vec, None
    fams, lines = [], []
    with open(f"{bank}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            assert int(r["vec_idx"]) == i, (bank, i, r["vec_idx"])
            fams.append(r["family"]); lines.append(line.rstrip("\n"))
    assert len(lines) == n, (bank, len(lines), n)
    return n, np.array(fams), None, lines


@app.function(image=image, gpu=SMALL_GPUS, cpu=16, memory=131072, ephemeral_disk=512 * 1024, volumes={"/data": vol}, timeout=8 * 3600)
def build(out_name: str = "mix_5m", spec_json: str = "", seed: int = 2030, overwrite: bool = False, threads: int = 48, exclude_json: str = ""):
    """exclude_json: JSON dict {source bank: path of a rows file} — rows listed under families[*].fail_vec_idx of that file are removed from
    the selection pool of that bank. Used for (a) the SAE end-anchor filter (end_anchor_rows.json: rows whose feature does not peak within
    the last 2 tokens when the target is re-tokenized standalone) and (b) the MLP eval-pair exclusion (eval_cos_rows.json: rows whose
    direction has cos > 0.99 to an eval mlp/mlp_pair direction or whose neuron set contains a train neuron dominating an eval pair)."""
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    import torch.nn.functional as F
    T0 = time.time()
    vol.reload()
    spec = json.loads(spec_json) if spec_json else DEFAULT_SPEC
    out = f"/data/banks/{out_name}"
    if os.path.exists(f"{out}/build_stats.json") and not overwrite:
        raise RuntimeError(f"{out} already finalized (overwrite=False)")
    dev = "cuda:0"
    rng = np.random.default_rng(seed)
    stage = "/root/mix"; os.makedirs(stage, exist_ok=True)

    # ---- optional per-bank row exclusions (end-anchor filter) ----
    excl_rows, excl_info = {}, {}
    for bank_p, fpath in (json.loads(exclude_json) if exclude_json else {}).items():
        ea = json.load(open(fpath))
        excl_rows[bank_p] = np.unique(np.concatenate([np.array(d["fail_vec_idx"], np.int64) for d in ea["families"].values()] or [np.zeros(0, np.int64)]))
        excl_info[bank_p] = {"file": fpath, "rule": ea.get("rule"), "n_excluded_rows": int(len(excl_rows[bank_p])),
                             "families": {f: {k: v for k, v in d.items() if k != "fail_vec_idx"} for f, d in ea["families"].items()}}
        _log(f"exclusions for {bank_p}: {len(excl_rows[bank_p])} rows ({ea.get('rule')})")
    # ---- sources: per-bank row lists per family, caps ----
    picks_bank, picks_row, picks_fam = [], [], []            # parallel lists -> arrays
    bank_ids, bank_info, src_lines, src_line_of_vec, per_source = [], {}, {}, {}, {}
    for grp in spec:
        fams = grp["families"]; cap = grp.get("n")
        rows_by_fam = {f: [] for f in fams}                    # fam -> list of (bank_id, row)
        for bank in grp["banks"]:
            t0 = time.time()
            assert os.path.exists(f"{bank}/build_stats.json") and os.path.exists(f"{bank}/vecs.{grp['dtype']}"), f"missing source {bank}"
            n, fam_arr, line_of_vec, lines = _scan_records(bank, grp.get("all_one_family"))
            bsz = os.path.getsize(f"{bank}/vecs.{grp['dtype']}")
            assert bsz == n * D * (2 if grp["dtype"] == "f16" else 4), f"{bank}: vecs size {bsz} != {n} rows"
            bid = len(bank_ids); bank_ids.append(bank)
            bank_info[bid] = {"bank": bank, "dtype": grp["dtype"], "n": n, "group": grp["name"]}
            if lines is not None:
                src_lines[bid] = lines
            if line_of_vec is not None:
                src_line_of_vec[bid] = line_of_vec
            for f in fams:
                rows = np.arange(n) if fam_arr is None else np.flatnonzero(fam_arr == f)
                if bank in excl_rows and len(excl_rows[bank]):
                    n_before = len(rows); rows = rows[~np.isin(rows, excl_rows[bank])]
                    if n_before != len(rows):
                        _log(f"  {bank} {f}: {n_before - len(rows)} rows excluded by the end-anchor filter -> {len(rows)}")
                rows_by_fam[f].append((bid, rows))
            _log(f"source {bank}: {n} rows, families {({f: sum(len(r) for b, r in rows_by_fam[f] if b == bid) for f in fams})} ({time.time() - t0:.0f}s)")
        for f in fams:
            avail = sum(len(r) for _, r in rows_by_fam[f])
            want = None
            if isinstance(cap, int):
                want = cap
            elif isinstance(cap, dict):
                want = cap.get(f)
            take_total = avail if want is None else min(int(want), avail)
            # proportional allocation over the group's banks (largest remainder), then a seeded sample inside each bank
            sizes = np.array([len(r) for _, r in rows_by_fam[f]], np.int64)
            if take_total == avail:
                alloc = sizes.copy()
            else:
                raw = take_total * sizes / sizes.sum(); alloc = np.floor(raw).astype(np.int64)
                rem = take_total - alloc.sum()
                for q in np.argsort(-(raw - np.floor(raw)))[:rem]:
                    alloc[q] += 1
            for (bid, rows), k in zip(rows_by_fam[f], alloc.tolist()):
                sel = np.sort(rng.permutation(rows)[:k]) if k < len(rows) else rows
                picks_bank.append(np.full(len(sel), bid, np.int64)); picks_row.append(sel); picks_fam.extend([f] * len(sel))
            per_source[f"{grp['name']}/{f}"] = {"available": int(avail), "taken": int(take_total), "requested": want, "banks": grp["banks"],
                                                "per_bank": {bank_ids[b]: int(k) for (b, _), k in zip(rows_by_fam[f], alloc.tolist())}, "fresh": grp.get("fresh")}
            _log(f"family {f} <- {grp['name']}: take {take_total} of {avail}")
    pb = np.concatenate(picks_bank); pr = np.concatenate(picks_row); pf = np.array(picks_fam)
    N = len(pb)
    perm = rng.permutation(N)                                   # output row i <- staged pick perm[i]
    pb, pr, pf = pb[perm], pr[perm], pf[perm]
    _log(f"{N} rows selected from {len(bank_ids)} source banks; shuffled ({time.time() - T0:.0f}s)")

    # ---- vectors: per bank, gather rows sorted by source row (sequential-ish reads), scatter into the local staging memmap ----
    vecs = np.memmap(f"{stage}/vecs.f32", np.float32, "w+", shape=(N, D))
    norm_stats = {"min": 9.0, "max": 0.0}
    for bid, bank in enumerate(bank_ids):
        t0 = time.time()
        info = bank_info[bid]
        idx = np.flatnonzero(pb == bid)                          # output rows fed by this bank
        order = idx[np.argsort(pr[idx], kind="stable")]
        src_rows = pr[order]
        row_b = D * (2 if info["dtype"] == "f16" else 4)
        dt = np.float16 if info["dtype"] == "f16" else np.float32
        fd = os.open(f"{bank}/vecs.{info['dtype']}", os.O_RDONLY)
        dense = len(src_rows) > 0.5 * info["n"]                  # dense selection: stream contiguous ranges; sparse: random preads
        CH = 32768
        for c0 in range(0, len(src_rows), CH):
            rs = src_rows[c0:c0 + CH]; outs = order[c0:c0 + CH]
            if dense:
                lo, hi = int(rs[0]), int(rs[-1]) + 1
                blk = np.frombuffer(_pread_full(fd, (hi - lo) * row_b, lo * row_b), dt).reshape(hi - lo, D)[rs - lo]
            else:
                blk = np.empty((len(rs), D), dt)
                with ThreadPoolExecutor(threads) as ex:
                    list(ex.map(lambda j: blk.__setitem__(j, np.frombuffer(_pread_full(fd, row_b, int(rs[j]) * row_b), dt)), range(len(rs)), chunksize=256))
            x = blk.astype(np.float32)
            nn = np.linalg.norm(x, axis=1)
            norm_stats["min"] = min(norm_stats["min"], float(nn.min())); norm_stats["max"] = max(norm_stats["max"], float(nn.max()))
            assert 0.98 < nn.min() and nn.max() < 1.02, (bank, nn.min(), nn.max())
            x /= nn[:, None]                                     # re-normalize (f16 sources carry ~1e-4 norm error)
            vecs[outs] = x
        os.close(fd)
        _log(f"vectors <- {bank}: {len(src_rows)} rows ({'dense' if dense else 'random'} reads, {time.time() - t0:.0f}s)")
    vecs.flush()

    # ---- leak guard: every row vs every v2 eval direction family + every pool_heldout row ----
    t0 = time.time()
    es = torch.load(EVAL_CACHE_V2, map_location="cpu", weights_only=False)
    ref_names = sorted(k[:-5] for k in es if k.endswith("_dirs") and torch.is_tensor(es[k]))
    refs = [F.normalize(es[f"{n}_dirs"].float().reshape(-1, D), dim=-1) for n in ref_names]
    ho_n = os.path.getsize(f"{POOL_HELDOUT}/vecs.f32") // (4 * D)
    ho = np.memmap(f"{POOL_HELDOUT}/vecs.f32", np.float32, "r", shape=(ho_n, D))
    ho_fam = np.array([json.loads(l)["family"] for l in open(f"{POOL_HELDOUT}/records.jsonl")])
    for hf in sorted(set(ho_fam.tolist())):
        refs.append(F.normalize(torch.from_numpy(np.asarray(ho[np.flatnonzero(ho_fam == hf)])).float(), dim=-1)); ref_names.append(f"pool_heldout/{hf}")
    ref = torch.cat(refs).to(dev)
    offs = np.cumsum([0] + [len(r) for r in refs])
    fam_names = sorted(set(pf.tolist()))
    fam_idx = np.array([fam_names.index(f) for f in pf], np.int8)
    leak = np.full((len(fam_names), len(ref_names)), -1.0)
    maxcos = np.zeros(N, np.float32)
    with torch.no_grad():
        LC = 16384                                              # [16384, ~98k] fp32 cos = 6.4 GB: fits every SMALL_GPUS type
        for c0 in range(0, N, LC):
            x = torch.from_numpy(np.ascontiguousarray(vecs[c0:c0 + LC])).to(dev)
            cos = x @ ref.T
            maxcos[c0:c0 + LC] = cos.max(1).values.cpu().numpy()
            fa = fam_idx[c0:c0 + LC]
            for ri in range(len(ref_names)):
                cm = cos[:, offs[ri]:offs[ri + 1]].max(1).values.cpu().numpy()
                for fi in range(len(fam_names)):
                    m = fa == fi
                    if m.any():
                        leak[fi, ri] = max(leak[fi, ri], float(cm[m].max()))
    bad = maxcos > LEAK_COS
    n_bad = int(bad.sum())
    leak_tbl = {f: {n: round(float(leak[fi, ri]), 4) for ri, n in enumerate(ref_names)} for fi, f in enumerate(fam_names)}
    dropped = {f: int((bad & (pf == f)).sum()) for f in fam_names}
    _log(f"leak guard vs {ref.shape[0]} reference dirs ({len(ref_names)} sets): {n_bad} rows > {LEAK_COS} -> dropped {dropped} ({time.time() - t0:.0f}s)")
    for f in fam_names:
        _log(f"  max-cos {f:13s} " + " ".join(f"{n.split('/')[-1][:11]}={leak_tbl[f][n]:.3f}" for n in ref_names))
    del ref; torch.cuda.empty_cache()

    # ---- publish: compact (skip leakers) sequentially into the volume; records line i == vec_idx i ----
    t0 = time.time()
    keep = np.flatnonzero(~bad)
    N_out = len(keep)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/vecs.f32.tmp", "wb") as fout:
        for c0 in range(0, N_out, 32768):
            fout.write(np.ascontiguousarray(vecs[keep[c0:c0 + 32768]]).tobytes())
    os.replace(f"{out}/vecs.f32.tmp", f"{out}/vecs.f32")
    assert os.path.getsize(f"{out}/vecs.f32") == N_out * D * 4
    _log(f"vecs.f32 written: {N_out} rows ({N_out * D * 4 / 2**30:.1f} GB, {time.time() - t0:.0f}s)")
    # records: parts' selected lines are re-read by line number (their line order is shuffled vs vec_idx)
    t0 = time.time()
    part_lines = {}
    for bid, bank in enumerate(bank_ids):
        if bid in src_line_of_vec:
            want_rows = pr[keep][pb[keep] == bid]
            want_lines = {int(src_line_of_vec[bid][r]): int(r) for r in want_rows.tolist()}
            got = {}
            with open(f"{bank}/records.jsonl") as fh:
                for i, line in enumerate(fh):
                    if i in want_lines:
                        got[want_lines[i]] = line.rstrip("\n")
            assert len(got) == len(want_lines), (bank, len(got), len(want_lines))
            part_lines[bid] = got
            _log(f"records <- {bank}: {len(got)} lines ({time.time() - t0:.0f}s)")
    fam_counts = {}
    with open(f"{out}/records.jsonl.tmp", "w") as fh:
        for i, g in enumerate(keep.tolist()):
            bid, row = int(pb[g]), int(pr[g])
            line = src_lines[bid][row] if bid in src_lines else part_lines[bid][row]
            r = json.loads(line)
            assert r["family"] == pf[g]
            r["src_bank"] = bank_ids[bid]; r["src_vec_idx"] = int(r.get("vec_idx", row)); r["vec_idx"] = i
            fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(f"{out}/records.jsonl.tmp", f"{out}/records.jsonl")
    _log(f"records.jsonl written ({time.time() - t0:.0f}s): {fam_counts}")

    # ---- summaries from the source banks' meta (windows per feature / neuron, fresh provenance) ----
    src_meta = {}
    for bank in bank_ids:
        mp = f"{bank}/meta.json"
        if os.path.exists(mp):
            m = json.load(open(mp))
            keep_keys = {"family_recipes", "neurons", "pairs", "triples", "distinct_windows", "leak_check", "leakage_check", "scan", "acts_store", "shortfalls"}
            src_meta[bank] = {k: v for k, v in m.items() if k in keep_keys}
    stats = {"kind": "5M-row 8-family SFT midtrain mix (bank-5m): fresh-store realact_long / sae / bsf / mlp / mlp_pair / mlp_triple + unused-part realact + fixed probes",
             "n_examples": N_out, "n_vecs": N_out, "families": fam_counts, "seed": seed, "spec": spec, "sources": per_source,
             "source_banks": bank_info, "layout": "seeded shuffle of all rows (records.jsonl line i == vec_idx i; src_bank/src_vec_idx point back)",
             "leak_check": {"threshold": LEAK_COS, "reference_sets": ref_names, "n_reference_dirs": int(offs[-1]), "rows_dropped": n_bad,
                            "rows_dropped_by_family": dropped, "max_cos_table": leak_tbl, "eval_cache": EVAL_CACHE_V2, "pool_heldout": POOL_HELDOUT},
             "source_norm_range": norm_stats, "pretrain_corpus_parts_excluded": PRETRAIN_PARTS, "unused_parts_used": UNUSED_PARTS,
             "row_exclusions": excl_info,
             "source_meta": src_meta, "d_model": D, "model": "Qwen/Qwen3.6-27B", "layer": 42, "created": time.time(), "wall_s": time.time() - T0}
    json.dump(stats, open(f"{out}/build_stats.json", "w"), indent=1)
    json.dump({**stats, "trainer_args": {"--data-dir": out, "--bank-file": "vecs.f32"}}, open(f"{out}/meta.json", "w"), indent=1)
    vol.commit()
    shutil.rmtree(stage, ignore_errors=True)
    _log(f"DONE {out}: {N_out} rows {fam_counts} in {(time.time() - T0) / 60:.1f} min")
    return {"out": out, "n_examples": N_out, "families": fam_counts, "leak_dropped": dropped, "minutes": (time.time() - T0) / 60}


@app.function(image=image, cpu=4, memory=16384, volumes={"/data": vol}, timeout=1800)
def peek(out_name: str = "mix_5m", n: int = 2):
    import numpy as np
    vol.reload()
    out = f"/data/banks/{out_name}"
    st = json.load(open(f"{out}/build_stats.json"))
    print(json.dumps({k: st[k] for k in ("n_examples", "families", "sources")}, indent=1))
    N = st["n_examples"]
    assert os.path.getsize(f"{out}/vecs.f32") == N * D * 4
    vecs = np.memmap(f"{out}/vecs.f32", np.float32, "r", shape=(N, D))
    seen = {}
    with open(f"{out}/records.jsonl") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line); f = r["family"]
            if seen.get(f, 0) >= n:
                continue
            seen[f] = seen.get(f, 0) + 1
            assert r["vec_idx"] == i
            print(f"[{f}] row {i} |v|={np.linalg.norm(vecs[i]):.4f} src={r['src_bank'].split('/')[-1]}:{r['src_vec_idx']} :: {r['target_text'][:100]!r}")
            if len(seen) == len(st["families"]) and all(c >= n for c in seen.values()):
                break
    return st["families"]
