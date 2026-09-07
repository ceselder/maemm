"""CPU unit tests for rl_disagg's --policy-base plumbing (no GPU, no vLLM, no HF model): flag defaults / 'none' unsetting, the
policy-base resolution + SAVE_DONE check, the BaseActor scoring shim (disable_adapter no-op, get_layer unwrapping), the KL
reference context (--ref adapter round trip vs the LoRA-disabled base), the scorer truncation (layers [0, READ_LAYER] kept,
layer-42 output unchanged, lm_head raises), load_scorer's identity path when the policy base IS MODEL, and run_meta.json.

    python -m pytest rl/test_rl_disagg_policy_base.py -q      (or: python rl/test_rl_disagg_policy_base.py)
"""
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # mxf.*
_spec = importlib.util.spec_from_file_location("rl_disagg", os.path.join(_HERE, "rl_disagg.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
from mxf.config import MODEL, READ_LAYER  # noqa: E402
from mxf.inject import get_layer, read_resid  # noqa: E402

_BASE = ["--role", "launch", "--n-rollout", "1", "--n-trainer", "3"]


# ------------------------------------------------------------------ flags
def test_defaults_are_the_original_base():
    a = D.parse_args(_BASE)
    assert a.policy_base is None and a.scorer_layers == 0
    assert D.policy_base_of(a) == MODEL and D.policy_base_is_model(a)


def test_none_unsets_launcher_defaults():
    # a launcher can only APPEND flags: `--init-adapter none` after the TRAIN_ARGS init must yield a fresh LoRA
    a = D.parse_args(_BASE + ["--init-adapter", "/data/sft/final", "--init-adapter", "none", "--ref-adapter", "None", "--policy-base", ""])
    assert a.init_adapter is None and a.ref_adapter is None and a.policy_base is None
    a = D.parse_args(_BASE + ["--policy-base", "/data/sft_mix/fullft2m/final"])
    assert D.policy_base_of(a) == "/data/sft_mix/fullft2m/final" and not D.policy_base_is_model(a)
    a = D.parse_args(_BASE + ["--policy-base", MODEL])
    assert D.policy_base_is_model(a)                     # explicit MODEL == default path


def test_check_policy_base_requires_save_done():
    d = tempfile.mkdtemp()
    a = D.parse_args(_BASE + ["--policy-base", d])
    try:
        D.check_policy_base(a); raise AssertionError("no config.json must fail")
    except AssertionError as e:
        assert "config.json" in str(e)
    open(f"{d}/config.json", "w").write("{}")
    try:
        D.check_policy_base(a); raise AssertionError("no SAVE_DONE must fail")
    except AssertionError as e:
        assert "SAVE_DONE" in str(e)
    open(f"{d}/SAVE_DONE", "w").write("{}")
    assert D.check_policy_base(a) == d
    assert D.check_policy_base(D.parse_args(_BASE)) == MODEL   # hub ids are not checked on disk


# ------------------------------------------------------------------ a tiny CausalLM-shaped model
class _Layer(torch.nn.Module):
    def __init__(self, d):
        super().__init__(); self.lin = torch.nn.Linear(d, d)

    def forward(self, h, **kw):
        return (torch.tanh(self.lin(h)),)


class _Inner(torch.nn.Module):
    def __init__(self, n_layers, d, V):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(V, d)
        self.layers = torch.nn.ModuleList([_Layer(d) for _ in range(n_layers)])
        self.config = types.SimpleNamespace(num_hidden_layers=n_layers)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        h = self.embed_tokens(input_ids)
        for layer in self.layers[: self.config.num_hidden_layers]:      # the HF loop shape
            h = layer(h)[0]
        return h


class _CausalLM(torch.nn.Module):
    def __init__(self, n_layers=48, d=8, V=32, tie=False):
        super().__init__()
        self.model = _Inner(n_layers, d, V)
        self.lm_head = torch.nn.Linear(d, V, bias=False)
        if tie:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.generation_config = types.SimpleNamespace(eos_token_id=1)
        self.config = types.SimpleNamespace(name="tiny")

    def forward(self, input_ids=None, attention_mask=None, **kw):
        return types.SimpleNamespace(logits=self.lm_head(self.model(input_ids, attention_mask)))


def test_base_actor_shim_is_the_clean_base():
    m = _CausalLM()
    sh = D.BaseActor(m)
    assert sh.get_base_model() is m and get_layer(sh, READ_LAYER) is m.model.layers[READ_LAYER]
    with sh.disable_adapter():
        pass                                              # a no-op context
    ids = torch.randint(0, 32, (2, 5)); am = torch.ones_like(ids)
    assert torch.equal(sh(input_ids=ids).logits, m(input_ids=ids).logits)
    assert sh.eval() is sh and not sh.training and sh.generation_config is m.generation_config
    h, mask = read_resid(sh, READ_LAYER, {"input_ids": ids, "attention_mask": am}, pool="all")
    assert h.shape == (2, 5, 8) and mask.all()


def test_truncated_scorer_keeps_layer_42_bit_identical_and_drops_the_head():
    torch.manual_seed(0)
    full = _CausalLM(n_layers=48)
    trunc = _CausalLM(n_layers=48); trunc.load_state_dict(full.state_dict())
    n_before, head_dropped = D._truncate_scorer(trunc, READ_LAYER + 1)
    assert n_before == 48 and head_dropped and len(trunc.model.layers) == READ_LAYER + 1
    ids = torch.randint(0, 32, (3, 7)); am = torch.ones_like(ids)
    h_full, _ = read_resid(full, READ_LAYER, {"input_ids": ids, "attention_mask": am}, pool="all")
    h_trunc, _ = read_resid(trunc, READ_LAYER, {"input_ids": ids, "attention_mask": am}, pool="all")
    assert torch.equal(h_full, h_trunc)                   # bit-identical layer-42 read
    try:
        trunc(input_ids=ids); raise AssertionError("logits must be unavailable on the truncated scorer")
    except RuntimeError as e:
        assert "lm_head" in str(e)
    # no-op cases: n_keep <= 0 or >= n_layers leave the model alone; tied head is kept
    m = _CausalLM(n_layers=48, tie=True)
    assert D._truncate_scorer(m, -1) == (48, False) and len(m.model.layers) == 48
    assert D._truncate_scorer(m, READ_LAYER + 1) == (48, False) and isinstance(m.lm_head, torch.nn.Linear)


def test_load_scorer_identity_when_policy_base_is_model():
    a = D.parse_args(_BASE)
    actor = object()
    assert D.load_scorer(a, actor, "cpu", "T0", use_gates=False) is actor       # no transformers import, no second model
    assert D.load_scorer(D.parse_args(_BASE + ["--policy-base", MODEL]), actor, "cpu", "T0", use_gates=True) is actor


# ------------------------------------------------------------------ KL reference
class _FakePeft:
    def __init__(self, with_ref):
        self.peft_config = {"default": 1, **({"ref": 2} if with_ref else {})}
        self.calls = []

    def set_adapter(self, name):
        self.calls.append(("set", name))

    @contextlib.contextmanager
    def disable_adapter(self):
        self.calls.append(("disable", True))
        yield
        self.calls.append(("disable", False))


def test_ref_policy_uses_ref_adapter_else_lora_off():
    m = _FakePeft(with_ref=True)
    with D._ref_policy(m):
        m.calls.append(("body",))
    assert m.calls == [("set", "ref"), ("body",), ("set", "default")]     # the pre-existing round trip, bit for bit
    m = _FakePeft(with_ref=False)
    with D._ref_policy(m):
        m.calls.append(("body",))
    assert m.calls == [("disable", True), ("body",), ("disable", False)]
    m = _FakePeft(with_ref=True)                                           # exception-safe restore
    try:
        with D._ref_policy(m):
            raise ValueError("x")
    except ValueError:
        pass
    assert m.calls[-1] == ("set", "default")


# ------------------------------------------------------------------ run_meta.json
def test_run_meta_records_policy_base_and_kl_ref():
    d = tempfile.mkdtemp()
    a = D.parse_args(_BASE + ["--policy-base", "/data/sft_mix/fullft2m/final", "--init-adapter", "none", "--kl-coef", "0.01",
                              "--save-dir", d, "--run-name", "smoke"])
    meta = D.write_run_meta(a, f"{d}/step_3", 16, step=3)
    on_disk = json.load(open(f"{d}/step_3/run_meta.json"))
    assert on_disk == json.loads(json.dumps(meta))
    assert on_disk["policy_base"] == "/data/sft_mix/fullft2m/final" and on_disk["policy_base_is_model"] is False
    assert on_disk["scorer_base"] == MODEL and on_disk["kl_ref"] == "policy_base_lora_off" and on_disk["init_adapter"] is None
    assert on_disk["lora"]["r"] == 64 and on_disk["lora"]["rslora"] is True and on_disk["step"] == 3
    a = D.parse_args(_BASE + ["--init-adapter", "/data/sft/final", "--kl-coef", "0.01", "--save-dir", d])
    meta = D.write_run_meta(a, d)
    assert meta["policy_base"] == MODEL and meta["policy_base_is_model"] and meta["kl_ref"] == "adapter" and meta["lora"] == "from init_adapter"
    a = D.parse_args(_BASE + ["--save-dir", d])
    assert D.write_run_meta(a, d)["kl_ref"] == "none"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
