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
    a = ap.parse_args()
    test_fireback_toy()
    test_metrics()
    if a.cache and os.path.exists(a.cache):
        test_cache(a.cache, a.cols)
    else:
        print("[test] no v2 cache given/found -> cache structure test skipped")
    print("ALL_TESTS_PASSED")


if __name__ == "__main__":
    main()
