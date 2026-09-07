"""CPU unit tests for rl_disagg's trainer speed knobs (no GPU / HF model): --score-length-bucket's score reorder wrapper, the
prefix-cache gradient accumulator (one shared prefix backward == the per-micro-batch chain rule), the micro-batch candidate
list, and the flag defaults (both knobs off = legacy).

    python rl/test_rl_disagg_fast_cpu.py      (or pytest)
"""
import importlib.util
import os
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rl_disagg", os.path.join(_HERE, "rl_disagg.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

_BASE = ["--role", "launch", "--n-rollout", "1", "--n-trainer", "3"]


def test_flags_default_off():
    a = D.parse_args(_BASE)
    assert a.prefix_cache is False and a.score_length_bucket is False
    a = D.parse_args(_BASE + ["--prefix-cache", "--score-length-bucket"])
    assert a.prefix_cache and a.score_length_bucket


def test_mb_candidates():
    a = D.parse_args(_BASE)
    assert D._mb_candidates(a) == [64, 48, 40, 32, 24, 16, 12, 8, 6, 4]
    a = D.parse_args(_BASE + ["--prefix-cache"])
    assert D._mb_candidates(a) == [128, 96, 64, 48, 40, 32, 24, 16, 12, 8, 6, 4]
    a = D.parse_args(_BASE + ["--prefix-cache", "--mb-candidates", "32,16"])
    assert D._mb_candidates(a) == [32, 16]                      # an explicit list is never extended


class _FakeR:
    """Stands in for rl_hf: score() = a deterministic per-row function that ALSO records the batch composition."""
    def __init__(self, score_batch):
        self.batches = []
        self.score_batch = score_batch

    def score(self, texts, dirs_rep, actor, tok, device, a, with_fluency=False):
        r = torch.tensor([float(len(t)) * d[0].item() for t, d in zip(texts, dirs_rep)])
        for s in range(0, len(texts), self.score_batch):
            self.batches.append([len(t) for t in texts[s : s + self.score_batch]])
        if with_fluency:
            return r, -r, r * 2
        return r


def test_score_bucketed_is_a_pure_reorder():
    g = torch.Generator().manual_seed(0)
    n = 37
    lens = torch.randint(8, 97, (n,), generator=g).tolist()
    texts = ["x" * L for L in lens]
    dirs = torch.rand(n, 4, generator=g)
    R = _FakeR(score_batch=8)
    a = types.SimpleNamespace()
    plain = R.score(texts, dirs, None, None, "cpu", a)
    R.batches.clear()
    bucketed = D.score_bucketed(R, texts, dirs, None, None, "cpu", a, lens)
    assert torch.equal(plain, bucketed)                          # rewards land back on their own rows
    for b in R.batches:                                          # ... and every batch was length-homogeneous (sorted)
        assert b == sorted(b)
    assert [x for b in R.batches for x in b] == sorted(lens)
    r3 = D.score_bucketed(R, texts, dirs, None, None, "cpu", a, lens, with_fluency=True)
    ref = R.score(texts, dirs, None, None, "cpu", a, with_fluency=True)
    assert all(torch.equal(x, y) for x, y in zip(r3, ref))


class _Layer:
    def __init__(self, keys, values, states):
        self.keys, self.values = keys, values
        self.conv_states = {0: states}
        self.is_initialized = True


class _Cache:
    def __init__(self, layers):
        self.layers = layers


def test_prefix_grad_accumulator_matches_direct_backward():
    """A 'prefix' graph (params -> cache tensors), three 'suffix' losses on the cache (like three micro-batches). The
    accumulator route (suffix backwards into detached leaves, one prefix backward with the fp32 sums) must give the same
    parameter gradients as backpropagating each loss through the prefix directly."""
    torch.manual_seed(0)
    W = torch.randn(6, 6, dtype=torch.float64, requires_grad=True)
    x = torch.randn(1, 6, dtype=torch.float64)

    def prefix():
        h = torch.tanh(x @ W)
        return _Cache([_Layer(h * 2, h[:, :3], torch.sin(h))])

    def suffix_loss(cache, k):
        L = cache.layers[0]
        return (L.keys.pow(2).sum() * (k + 1) + L.values.sum() * 0.5 + L.conv_states[0].sum()) / 7.0
    # direct: every suffix loss through the prefix graph
    W.grad = None
    c = prefix()
    for k in range(3):
        suffix_loss(c, k).backward(retain_graph=k < 2)
    g_direct = W.grad.clone()
    # accumulator: suffix losses stop at detached leaves, one prefix backward
    W.grad = None
    acc = D._PrefixGradAccumulator(prefix())
    for k in range(3):
        suffix_loss(acc.cache, k).backward()
        acc.accumulate()
    acc.backward()
    assert torch.allclose(W.grad, g_direct, atol=1e-12, rtol=1e-12), (W.grad - g_direct).abs().max()
    try:
        acc.accumulate()
        assert False, "must refuse reuse after backward()"
    except RuntimeError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("ALL OK")
