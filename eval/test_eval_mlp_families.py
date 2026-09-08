"""Tests for the cache-v2 EXTRA eval families (layer-42 MLP neurons `mlp` / co-firing pairs `mlp_pair`) in
eval/eval_universal.py — CPU by default, no model download:

  1. score_mlp_fireback semantics on a TOY decoder stack (43 blocks with .mlp.down_proj, a fake whitespace tokenizer):
     the hook on the layer-42 down_proj input, the last-5-kept-token window, polarity, corpus-max normalization,
     right padding of mixed-length rows, and the min/max over the k members are checked against an explicit
     re-implementation that runs the toy layers by hand.
  2. mlp_metrics: fired10/25/50 and any_* fractions from known arrays.
  3. (if a v2 cache is reachable) structure of eval_sets_heldout_v2.pt: extra_families, per-family tensors, unit
     norms, k = 1 / 2, the 11 old cos families untouched; with --cols <down_proj_cols.f16> the mlp_dirs are re-derived
     from the down_proj columns (polarity * unit(col)) and the mlp_pair_dirs from meta acts_co, and must match.
  4. RANK metrics (2026-09-08): score_mlp_rank on the toy stack vs an explicit re-implementation (normalized rank at the
     member's best last-5 token, min rank over the window, raw rank, no-kept-token rows; its `na` == score_mlp_fireback's);
     mlp_corpus_percentile vs the empirical percentile of the samples a synthetic histogram was built from (and, with
     --stats <neuron_stats.npz>, sanity on the real stats: P(corpus max) ~ 1, P(-corpus max) ~ 0); mlp_rank_metrics on known
     arrays (k = 1 and pairs); mlp_chance_windows is per-row seeded (sharding-invariant) on a fake acts dir.

    PYTHONPATH=$PWD python eval/test_eval_mlp_families.py                       # toy tests only
    PYTHONPATH=$PWD python eval/test_eval_mlp_families.py --cache /tmp/eval_sets_heldout_v2.pt \
        --cols ~/shared/reports/maemm-mlp-neurons/raw/mlp42/down_proj_cols.f16   # + cache structure/derivation
The real-model path (HF actor + generation + fire-back) is exercised end-to-end by eval/eval_ckpt_daemon.py --once with
--eval-cache .../eval_sets_heldout_v2.pt (the GPU test); this file keeps the CPU-checkable pieces.
"""
import argparse
import contextlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_HERE), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import eval_universal as EU  # noqa: E402
from mxf.config import D_MODEL, READ_LAYER  # noqa: E402

OLD_COS_FAMILIES = ["bsf", "realact", "jlens", "cluster", "random", "realact_early", "realact_mid", "realact_long",
                    "indist_long", "indist_probe", "indist_realact"]


# ---------------------------------------------------------------------------------------------
# toy stand-ins
# ---------------------------------------------------------------------------------------------
class _Enc(dict):
    def to(self, device):
        return _Enc({k: v.to(device) for k, v in self.items()})


class FakeTok:
    """Whitespace tokenizer: ids = 1 + hash(word) % (vocab - 1); id 0 = BOS/sink. Right-pads to the longest row."""
    def __init__(self, vocab=97):
        self.vocab = vocab
        self.padding_side = "right"
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.pad_token_id = 0

    def _ids(self, text):
        return [1 + (sum(ord(c) * (7 ** k) for k, c in enumerate(w)) % (self.vocab - 1)) for w in text.split()]

    def __call__(self, batch, return_tensors="pt", padding=True, truncation=True, max_length=95, add_special_tokens=False):
        rows = [self._ids(t)[:max_length] for t in batch]
        T = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), T, dtype=torch.long); am = torch.zeros(len(rows), T, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r); am[i, :len(r)] = 1
        return _Enc({"input_ids": ids, "attention_mask": am})


class ToyBlock(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.mlp.up_proj = nn.Linear(d, d_ff, bias=False)
        self.mlp.down_proj = nn.Linear(d_ff, d, bias=False)

    def neuron_values(self, h):
        return F.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h)

    def forward(self, h):
        return (h + self.mlp.down_proj(self.neuron_values(h)),)


class ToyModel(nn.Module):
    """Looks enough like a PEFT-wrapped HF decoder for mxf.inject.get_layer / eval_universal._reencode_mlp."""
    def __init__(self, vocab=97, d=32, d_ff=48, n_layers=READ_LAYER + 1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList([ToyBlock(d, d_ff) for _ in range(n_layers)])

    def forward(self, input_ids, attention_mask=None):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        return h

    @contextlib.contextmanager
    def disable_adapter(self):
        yield


@torch.no_grad()
def expected_fireback(model, tok, texts, neuron, polarity, corpus_max):
    """Hand-rolled reference: run the toy stack to layer READ_LAYER, read the neuron values there, apply the eval's keep /
    last-5 rules, min/max over members."""
    enc = tok([t if t.strip() else " " for t in texts])
    B = enc["input_ids"].shape[0]
    ids = torch.cat([torch.zeros(B, 1, dtype=torch.long), enc["input_ids"]], 1)
    am = torch.cat([torch.ones(B, 1, dtype=torch.long), enc["attention_mask"]], 1)
    h = model.model.embed_tokens(ids)
    for layer in model.model.layers[:READ_LAYER]:
        h = layer(h)[0]
    blk = model.model.layers[READ_LAYER]
    a = blk.neuron_values(h)                                   # [B, T, d_ff]  == down_proj input at layer 42
    h42 = blk(h)[0]
    keep = am.bool().clone(); keep[:, 0] = False
    nrm = h42.norm(dim=-1)
    med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
    keep = keep & (nrm <= EU.NORM_FILTER_MULT * med)
    L = am.sum(1)
    pos = torch.arange(ids.shape[1])[None, :]
    last5 = keep & (pos >= (L - EU.MLP_LAST_K)[:, None])
    mins, maxs = [], []
    for b in range(B):
        vals = []
        for k in range(neuron.shape[1]):
            v = polarity[b, k] * a[b, :, neuron[b, k]]
            v = v[last5[b]]
            vals.append(float(v.max()) / float(corpus_max[b, k]) if len(v) else 0.0)
        mins.append(min(vals)); maxs.append(max(vals))
    return torch.tensor(mins), torch.tensor(maxs)


def test_fireback_toy():
    tok = FakeTok(); model = ToyModel().eval()
    texts = ["the quick brown fox jumps over the lazy dog again and again", "short one", "a b c d e f g h i j k l m n o p q",
             "   ", "one two three four five six seven eight nine ten eleven twelve"]
    n = len(texts)
    g = torch.Generator().manual_seed(1)
    neuron = torch.randint(0, 48, (n, 2), generator=g)
    polarity = torch.where(torch.rand(n, 2, generator=g) < 0.5, -1.0, 1.0)
    corpus_max = 0.5 + torch.rand(n, 2, generator=g)
    with torch.no_grad():
        na_min, na_max = EU.score_mlp_fireback(texts, neuron, polarity, corpus_max, model, tok, "cpu")
        e_min, e_max = expected_fireback(model, tok, texts, neuron, polarity, corpus_max)
    assert torch.allclose(na_min, e_min, atol=1e-5), (na_min, e_min)
    assert torch.allclose(na_max, e_max, atol=1e-5), (na_max, e_max)
    assert (na_max >= na_min).all()
    # k = 1 (singles): min == max, and passing 1-D neuron ids is accepted
    with torch.no_grad():
        s_min, s_max = EU.score_mlp_fireback(texts, neuron[:, 0], polarity[:, 0], corpus_max[:, 0], model, tok, "cpu")
    assert torch.allclose(s_min, s_max)
    # the k=1 value must equal the member-0 term of the k=2 computation
    e1_min, _ = expected_fireback(model, tok, texts, neuron[:, :1], polarity[:, :1], corpus_max[:, :1])
    assert torch.allclose(s_min, e1_min, atol=1e-5)
    # the window is the LAST 5 tokens only: prepending 5+ unrelated words must not change a row's value when the tail is fixed
    tail = "alpha beta gamma delta epsilon"
    with torch.no_grad():
        a1, _ = EU.score_mlp_fireback([tail], neuron[:1], polarity[:1], corpus_max[:1], model, tok, "cpu")
        a2, _ = EU.score_mlp_fireback(["zeta eta theta iota kappa lambda mu " + tail], neuron[:1], polarity[:1], corpus_max[:1], model, tok, "cpu")
    # (values differ because the toy has no attention, i.e. no context mixing -> the last-5 neuron values are identical)
    assert torch.allclose(a1, a2, atol=1e-5), (a1, a2)
    print(f"[test] fire-back toy: OK (n={n}, na_min {na_min.numpy().round(3).tolist()})")


@torch.no_grad()
def expected_rank(model, tok, texts, neuron, stats):
    """Hand-rolled reference for score_mlp_rank on the toy stack."""
    enc = tok([t if t.strip() else " " for t in texts])
    B = enc["input_ids"].shape[0]
    ids = torch.cat([torch.zeros(B, 1, dtype=torch.long), enc["input_ids"]], 1)
    am = torch.cat([torch.ones(B, 1, dtype=torch.long), enc["attention_mask"]], 1)
    h = model.model.embed_tokens(ids)
    for layer in model.model.layers[:READ_LAYER]:
        h = layer(h)[0]
    blk = model.model.layers[READ_LAYER]
    a = blk.neuron_values(h)                                   # [B, T, d_ff]
    h42 = blk(h)[0]
    keep = am.bool().clone(); keep[:, 0] = False
    nrm = h42.norm(dim=-1)
    med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
    keep = keep & (nrm <= EU.NORM_FILTER_MULT * med)
    L = am.sum(1)
    pos = torch.arange(ids.shape[1])[None, :]
    last5 = keep & (pos >= (L - EU.MLP_LAST_K)[:, None])
    pol = torch.as_tensor(stats["polarity"]); cm = torch.as_tensor(stats["corpus_max"])
    N = pol.numel()
    n, k = neuron.shape
    out = {key: torch.zeros(n, k) for key in ("av", "na")}
    out.update({key: torch.full((n, k), N, dtype=torch.long) for key in ("rank", "rank_best5", "raw_rank")})
    for b in range(B):
        ts = torch.nonzero(last5[b]).flatten().tolist()
        for j in range(k):
            i = int(neuron[b, j])
            if not ts:
                continue
            vals = [float(pol[i] * a[b, t, i] / cm[i]) for t in ts]
            t_best = ts[int(np.argmax(vals))]
            row_n = pol * a[b, t_best] / cm
            row_r = pol * a[b, t_best]
            out["av"][b, j] = pol[i] * a[b, t_best, i]
            out["na"][b, j] = max(vals)
            out["rank"][b, j] = int((row_n > row_n[i]).sum()) + 1
            out["raw_rank"][b, j] = int((row_r > row_r[i]).sum()) + 1
            out["rank_best5"][b, j] = min(int(((pol * a[b, t] / cm) > (pol[i] * a[b, t, i] / cm[i])).sum()) + 1 for t in ts)
    return out


def synthetic_stats(N, n_tok, seed=0, scale=None):
    """A stats dict in the load_mlp_neuron_stats layout from sampled signed values (returned too, for empirical checks)."""
    rng = np.random.default_rng(seed)
    scale = np.ones(N) if scale is None else np.asarray(scale, np.float64)
    vals = rng.standard_normal((n_tok, N)) * rng.lognormal(0, 1.0, (n_tok, N)) * scale[None, :]
    vals[rng.random((n_tok, N)) < 0.7] *= 0.01                                     # heavy tail: most tokens tiny
    pol = np.where(vals.max(0) >= -vals.min(0), 1.0, -1.0).astype(np.float32)
    cmax = np.abs(vals).max(0).astype(np.float32)
    lo, bpd, NB = -4.0, 16, 2 + 16 * 7
    mag = np.abs(vals)
    b = np.clip(np.floor((np.log10(np.maximum(mag, 1e-300)) - lo) * bpd).astype(np.int64) + 1, 0, NB - 1)
    b[mag < 10 ** lo] = 0
    hp = np.zeros((NB, N)); hn = np.zeros((NB, N))
    for j in range(N):
        hp[:, j] = np.bincount(b[vals[:, j] > 0, j], minlength=NB)
        hn[:, j] = np.bincount(b[vals[:, j] < 0, j], minlength=NB)
    pos_pol = pol[None, :] > 0
    st = {"path": "synthetic", "N": N, "n_tok": float(n_tok), "polarity": pol, "corpus_max": cmax,
          "hist_same": np.where(pos_pol, hp, hn), "hist_opp": np.where(pos_pol, hn, hp), "n_zero": (vals == 0).sum(0).astype(np.float64),
          "hist_lo": lo, "hist_bpd": bpd, "polarity_t": torch.from_numpy(pol.copy()), "corpus_max_t": torch.from_numpy(cmax.copy())}
    return st, vals


def test_rank_toy():
    tok = FakeTok(); model = ToyModel().eval()
    texts = ["the quick brown fox jumps over the lazy dog again and again", "short one", "a b c d e f g h i j k l m n o p q",
             "   ", "one two three four five six seven eight nine ten eleven twelve", "x"]
    n = len(texts)
    st, _ = synthetic_stats(48, 2000, seed=3, scale=np.linspace(0.2, 3.0, 48))
    g = torch.Generator().manual_seed(2)
    neuron = torch.randint(0, 48, (n, 2), generator=g)
    with torch.no_grad():
        got = EU.score_mlp_rank(texts, neuron, st, model, tok, "cpu", sbatch=4)
        exp = expected_rank(model, tok, texts, neuron, st)
        na_min, na_max = EU.score_mlp_fireback(texts, neuron, st["polarity_t"][neuron], st["corpus_max_t"][neuron], model, tok, "cpu")
    for key in ("rank", "rank_best5", "raw_rank"):
        assert torch.equal(got[key], exp[key]), (key, got[key], exp[key])
    assert torch.allclose(got["av"], exp["av"], atol=1e-5) and torch.allclose(got["na"], exp["na"], atol=1e-6)
    # the scorer's per-member `na` IS the fire-back value (min / max over members == score_mlp_fireback)
    assert torch.allclose(got["na"].min(1).values, na_min, atol=1e-6) and torch.allclose(got["na"].max(1).values, na_max, atol=1e-6)
    assert (got["rank_best5"] <= got["rank"]).all() and (got["rank"] >= 1).all() and (got["rank"] <= 48).all()
    assert got["pct"].shape == (n, 2) and (got["pct"] >= 0).all() and (got["pct"] <= 1).all()
    # 1-D neuron ids accepted == member 0 of the 2-D call
    with torch.no_grad():
        g1 = EU.score_mlp_rank(texts, neuron[:, 0], st, model, tok, "cpu")
    assert torch.equal(g1["rank"][:, 0], got["rank"][:, 0]) and torch.allclose(g1["na"][:, 0], got["na"][:, 0])
    # a row whose window has no kept token -> worst rank, zero value, percentile P(polarity*a <= 0)
    row_blank = [i for i, t in enumerate(texts) if not t.strip()]
    assert row_blank, "need a blank text in the toy set"
    # (the blank text becomes ' ' -> tokenizes to 0 words -> only the sink -> no kept token)
    r = row_blank[0]
    assert int(got["rank"][r, 0]) == 48 and float(got["na"][r, 0]) == 0.0, (got["rank"][r], got["na"][r])
    print(f"[test] rank toy: OK (ranks {got['rank'][:, 0].tolist()}, best5 {got['rank_best5'][:, 0].tolist()}, raw {got['raw_rank'][:, 0].tolist()})")


def test_percentile(stats_path=None):
    st, vals = synthetic_stats(5, 200_000, seed=7, scale=[0.1, 1.0, 5.0, 20.0, 0.01])
    signed = vals * st["polarity"][None, :]                                              # polarity-signed corpus values
    rng = np.random.default_rng(1)
    for j in range(5):
        qs = np.quantile(signed[:, j], [0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
        probe = np.concatenate([qs, [0.0, float(st["corpus_max"][j]), -float(st["corpus_max"][j]), 0.5 * float(st["corpus_max"][j])]])
        got = EU.mlp_corpus_percentile(probe, np.full(len(probe), j), st)
        emp = np.array([(signed[:, j] <= v).mean() for v in probe])
        # 16 log-bins/decade + interpolation: within 1% of the empirical percentile everywhere
        assert np.all(np.abs(got - emp) < 0.01), (j, np.round(got, 4), np.round(emp, 4))
    assert EU.mlp_corpus_percentile([float(st["corpus_max"][1])], [1], st)[0] > 1 - 1e-4   # the max sits inside its (interpolated) bin
    print("[test] corpus percentile (synthetic histogram vs empirical): OK")
    if stats_path and os.path.exists(stats_path):
        real = EU.load_mlp_neuron_stats(stats_path)
        N = real["N"]
        nid = np.arange(N)
        top = EU.mlp_corpus_percentile(real["corpus_max"], nid, real)
        bot = EU.mlp_corpus_percentile(-real["corpus_max"], nid, real)
        zero = EU.mlp_corpus_percentile(np.zeros(N), nid, real)
        assert np.all(top >= 0.999) and np.all(bot <= 0.001), (top.min(), bot.max())
        opp_share = real["hist_opp"].sum(0) / real["n_tok"]
        assert np.allclose(zero, opp_share, atol=1e-6)
        tenpct = EU.mlp_corpus_percentile(0.1 * real["corpus_max"], nid, real)
        print(f"[test] real stats {stats_path}: N={N} n_tok={real['n_tok']:.0f} | P(max) min {top.min():.5f} | P(-max) max {bot.max():.5f} | "
              f"P(<=0) median {np.median(zero):.3f} | P(<= 10% max) median {np.median(tenpct):.4f} (== 1 - fire frequency) OK")


def test_rank_metrics():
    r = np.array([[1], [1], [3], [10], [11], [17408], [2], [5]])
    m = EU.mlp_rank_metrics("mlp", r, r, r, np.array([[0.999], [0.5], [0.99], [0.2], [0.991], [0.1], [1.0], [0.98]]))
    assert m["eval/mlp/rank1_frac"] == 2 / 8 and m["eval/mlp/rank_le10"] == 6 / 8
    assert abs(m["eval/mlp/mean_rank"] - r.mean()) < 1e-9 and m["eval/mlp/median_rank"] == 4.0
    assert abs(m["eval/mlp/mrr"] - np.mean(1.0 / r)) < 1e-12
    assert m["eval/mlp/top1pct_frac"] == 4 / 8 and abs(m["eval/mlp/corpus_pct_mean"] - 0.72) < 1e-9
    assert m["eval/mlp/best5_rank1_frac"] == 2 / 8 and m["eval/mlp/raw_mrr"] == m["eval/mlp/mrr"]
    assert "eval/mlp/both_rank_le10" not in m
    rp = np.array([[1, 3], [2, 50], [12, 40], [1, 1]])
    mp = EU.mlp_rank_metrics("mlp_pair", rp, rp, rp, np.array([[0.999, 0.5], [0.99, 0.99], [0.1, 0.2], [1.0, 1.0]]))
    assert mp["eval/mlp_pair/rank1_frac"] == 3 / 8 and mp["eval/mlp_pair/rank_le10"] == 5 / 8
    assert mp["eval/mlp_pair/both_rank_le10"] == 2 / 4 and mp["eval/mlp_pair/any_rank_le10"] == 3 / 4
    assert abs(mp["eval/mlp_pair/mean_rank"] - np.mean(rp.mean(1))) < 1e-9 and mp["eval/mlp_pair/median_rank"] == np.median(rp.mean(1))
    assert mp["eval/mlp_pair/both_top1pct_frac"] == 2 / 4 and mp["eval/mlp_pair/top1pct_frac"] == 5 / 8
    ch = {3: {"na": 0.2, "na_any": 0.4, "av": [1.0, 2.0], "rank": [1, 5], "rank_best5": [1, 4], "raw_rank": [100, 200], "pct": [0.999, 0.9],
              "win": [[47800, 0, 16]] * 4},
          1: {"na": 0.05, "na_any": 0.05, "av": [0.1, 0.2], "rank": [30, 20], "rank_best5": [30, 20], "raw_rank": [3000, 2000], "pct": [0.5, 0.6],
              "win": [[47900, 10, 64]] * 4}}
    cm, cpd = EU.mlp_chance_metrics("mlp_pair", ch, 2)
    assert cm["eval/mlp_pair/chance_rank1_frac"] == 1 / 4 and cm["eval/mlp_pair/chance_norm_act"] == 0.125
    assert cm["eval/mlp_pair/chance_fired10"] == 0.5 and cm["eval/mlp_pair/chance_any_norm_act"] == 0.225
    assert cpd["row"] == [1, 3] and cpd["rank"] == [[30, 20], [1, 5]]
    print("[test] mlp_rank_metrics + chance aggregation: OK")


def test_chance_windows():
    import json
    import tempfile
    d = tempfile.mkdtemp()
    NS, T = 100, 64
    toks = (np.arange(NS)[:, None] * 1000 + np.arange(T)[None, :]).astype(np.int32)
    toks.tofile(os.path.join(d, "toks.i32"))
    json.dump({"n_seq": NS, "seq_len": T}, open(os.path.join(d, "meta.json"), "w"))
    spec_all, ids_all = EU.mlp_chance_windows(d, range(8), 4, 16, 64, fam_idx=0)
    spec_odd, ids_odd = EU.mlp_chance_windows(d, [1, 3, 5, 7], 4, 16, 64, fam_idx=0)
    assert spec_all.shape == (32, 3) and np.array_equal(spec_all.reshape(8, 4, 3)[1::2].reshape(-1, 3), spec_odd)
    assert ids_all[4 * 3:4 * 4] == ids_odd[4 * 1:4 * 2]
    lo = int(np.ceil(0.95 * NS))
    assert (spec_all[:, 0] >= lo).all() and (spec_all[:, 0] < NS).all() and (spec_all[:, 2] >= 16).all() and (spec_all[:, 2] <= 64).all()
    assert all(spec_all[q, 1] + spec_all[q, 2] <= T for q in range(32))
    assert all(ids_all[q] == toks[spec_all[q, 0], spec_all[q, 1]:spec_all[q, 1] + spec_all[q, 2]].tolist() for q in range(32))
    spec_f1, _ = EU.mlp_chance_windows(d, range(8), 4, 16, 64, fam_idx=1)
    assert not np.array_equal(spec_f1, spec_all)
    print("[test] chance windows: per-row seeded (sharding-invariant), held-out rows only, family-specific: OK")


def test_metrics():
    na = np.array([0.0, 0.05, 0.10, 0.2, 0.3, 0.6, 1.5, 0.25])
    m = EU.mlp_metrics("mlp", na)
    assert abs(m["eval/mlp/norm_act"] - na.mean()) < 1e-12
    assert m["eval/mlp/fired10"] == 6 / 8 and m["eval/mlp/fired25"] == 4 / 8 and m["eval/mlp/fired50"] == 2 / 8
    assert "eval/mlp/any_fired10" not in m
    m2 = EU.mlp_metrics("mlp_pair", na, np.array([1.0] * 8))
    assert m2["eval/mlp_pair/any_fired50"] == 1.0 and m2["eval/mlp_pair/fired50"] == 2 / 8
    print("[test] mlp_metrics: OK")


def test_cache(path, cols_path=None):
    es = torch.load(path, map_location="cpu", weights_only=False)
    xf = EU.extra_families(es)
    assert xf == ["mlp", "mlp_pair"], xf
    assert list(es["meta"]["cos_families"]) == OLD_COS_FAMILIES, es["meta"]["cos_families"]
    for fam in OLD_COS_FAMILIES + ["sae"]:
        d = es[f"{fam}_dirs"]
        assert tuple(d.shape) == (512, D_MODEL), (fam, d.shape)
    assert len(es["sae_feats"]) == 512 and es["corpus_peak"].shape == (512,)
    ks = {"mlp": 1, "mlp_pair": 2}
    for fam in xf:
        d = es[f"{fam}_dirs"]; nid = es[f"{fam}_neuron"]; pol = es[f"{fam}_polarity"]; cm = es[f"{fam}_corpus_max"]
        n = d.shape[0]
        assert d.shape[1] == D_MODEL and torch.allclose(d.norm(dim=-1), torch.ones(n), atol=1e-4)
        assert nid.shape == (n, ks[fam]) and pol.shape == (n, ks[fam]) and cm.shape == (n, ks[fam]), (nid.shape, pol.shape, cm.shape)
        assert nid.dtype == torch.long and int(nid.min()) >= 0 and int(nid.max()) < es["meta"]["mlp42"]["d_ff"]
        assert set(pol.unique().tolist()) <= {-1.0, 1.0} and (cm > 0).all()
        ho = set(es["meta"]["mlp42"]["heldout_neurons"])
        touching = [any(int(x) in ho for x in row) for row in nid.tolist()]
        assert all(touching), f"{fam}: every eval direction must touch a held-out neuron"
        print(f"[test] cache {fam}: n={n} k={ks[fam]} | dirs unit | neurons in [{int(nid.min())}, {int(nid.max())}] | "
              f"2-held-out members: {sum(len(set(r) & ho) == 2 for r in nid.tolist()) if ks[fam] == 2 else '-'}")
    m = es["meta"]["mlp42"]
    assert m["n_heldout_neurons"] == len(m["heldout_neurons"]) and not (set(m["heldout_neurons"]) & set(m["train_neurons"]))
    if cols_path and os.path.exists(cols_path):
        N = m["d_ff"]
        cols = torch.from_numpy(np.fromfile(cols_path, np.float16).reshape(N, D_MODEL).astype(np.float32))
        unit = F.normalize(cols, dim=-1)
        nid = es["mlp_neuron"][:, 0]; pol = es["mlp_polarity"][:, 0]
        rec = unit[nid] * pol[:, None]
        c = (rec * es["mlp_dirs"]).sum(-1)
        assert float(c.min()) > 0.9999, float(c.min())
        pn = es["mlp_pair_neuron"]; acts = torch.tensor(m["mlp_pair"]["acts_co"], dtype=torch.float32)
        wn = cols.norm(dim=-1)
        comp = acts[:, 0:1] * wn[pn[:, 0]][:, None] * unit[pn[:, 0]] * es["mlp_pair_polarity"][:, 0:1] \
            + acts[:, 1:2] * wn[pn[:, 1]][:, None] * unit[pn[:, 1]] * es["mlp_pair_polarity"][:, 1:2]
        c2 = (F.normalize(comp, dim=-1) * es["mlp_pair_dirs"]).sum(-1)
        assert float(c2.min()) > 0.9999, float(c2.min())
        print(f"[test] cache derivation from down_proj columns: mlp cos>={float(c.min()):.6f}, mlp_pair cos>={float(c2.min()):.6f} OK")
    print(f"[test] cache {path}: OK ({len(OLD_COS_FAMILIES)} old cos families unchanged + sae + extras {xf})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get(EU.ENV_EVAL_CACHE), help="eval_sets_heldout_v2.pt (optional)")
    ap.add_argument("--cols", default=None, help="down_proj_cols.f16 of /data/mlp42 (optional derivation check)")
    ap.add_argument("--stats", default=None, help="neuron_stats.npz of /data/mlp42 (optional real-stats percentile sanity)")
    a = ap.parse_args()
    test_fireback_toy()
    test_metrics()
    test_rank_toy()
    test_percentile(a.stats)
    test_rank_metrics()
    test_chance_windows()
    if a.cache and os.path.exists(a.cache):
        test_cache(a.cache, a.cols)
    else:
        print("[test] no v2 cache given/found -> cache structure test skipped")
    print("ALL_TESTS_PASSED")


if __name__ == "__main__":
    main()
