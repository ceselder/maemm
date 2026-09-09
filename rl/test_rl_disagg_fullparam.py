"""CPU unit tests for rl_disagg --full-param (rl/rl_fullparam.py): flags, the publish manifest + FSDP2 shard geometry, the bf16
gather / fs shard round trip + checksums, the exception-free layer-42 read on a truncated sharded scorer, the FSDP2 loss
scaling (== the LoRA path's weighted grad mean), the trainable fp32 head hook, one update_disagg step on a tiny FSDP2
(world 1, gloo) Qwen3.5, the rollout-side fs sync against a fake engine, run_meta and the prefix accumulator's zero-grad
extra outputs. No GPU, no vLLM, no download.

    python -m pytest rl/test_rl_disagg_fullparam.py -q      (or: python rl/test_rl_disagg_fullparam.py)
"""
import importlib.util
import json
import os
import sys
import tempfile
import types

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # mxf.*, sft.*
sys.path.insert(0, _HERE)                           # rl_fullparam
_spec = importlib.util.spec_from_file_location("rl_disagg", os.path.join(_HERE, "rl_disagg.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
import rl_fullparam as FP  # noqa: E402
from mxf.inject import get_layer, hooked, make_inject_hook, read_resid  # noqa: E402

_BASE = ["--role", "launch", "--n-rollout", "1", "--n-trainer", "3"]
PROMPT = list(range(10, 22))
MARKER = 6


def _dist():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29581")
        dist.init_process_group("gloo", rank=0, world_size=1)


def _tiny_cfg(n_layers=4, vocab=512):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=n_layers,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            layer_types=(["linear_attention", "linear_attention", "full_attention", "linear_attention"] * 4)[:n_layers],
                            max_position_embeddings=256, pad_token_id=0, eos_token_id=1, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    return cfg


def _tiny_model(seed=0, n_layers=4):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(_tiny_cfg(n_layers)).to(torch.bfloat16)


def _fsdp_policy(seed=0):
    _dist()
    FT = FP.fullft_module()
    m = _tiny_model(seed)
    return FT.shard_full_model(m, 1, "cpu", log=lambda *a, **k: None, device_type="cpu")


# ------------------------------------------------------------------ flags
def test_full_param_flags_and_asserts():
    a = D.parse_args(_BASE + ["--full-param", "--init-adapter", "none", "--autocast-bf16"])
    assert a.full_param and a.init_adapter is None and a.publish_mode == "nccl" and a.wu_port == a.master_port + 111
    assert a.mb_target_frac == 0.85 and a.autocast_bf16 is False and a.scorer_shard is True and a.fs_keep_steps == 2
    assert a.chunked_head is True and a.suffix_ckpt is False and a.fsdp_prefetch == 2          # fast defaults; ckpt needs --prefix-cache
    f = D.parse_args(_BASE + ["--full-param", "--init-adapter", "none", "--prefix-cache"])
    assert f.suffix_ckpt is True and f.chunked_head is True and f.fsdp_prefetch == 2
    g = D.parse_args(_BASE + ["--full-param", "--init-adapter", "none", "--prefix-cache", "--no-suffix-ckpt", "--no-chunked-head", "--fsdp-prefetch", "0"])
    assert g.suffix_ckpt is False and g.chunked_head is False and g.fsdp_prefetch == 0       # the first validated (production) configuration
    b = D.parse_args(_BASE)
    assert not b.full_param and b.mb_target_frac == 0.85 and b.autocast_bf16 is False and b.suffix_ckpt is False and b.chunked_head is False and b.fsdp_prefetch == 0
    for bad in (["--full-param", "--init-adapter", "/x"], ["--full-param", "--inline-eval-every", "5"],
                ["--full-param", "--backend", "gloo"], ["--full-param", "--n-trainer", "1"], ["--full-param", "--publish-fp32"]):
        try:
            D.parse_args(_BASE + bad); assert False, bad
        except AssertionError as e:
            assert "--full-param" in str(e), (bad, e)
    c = D.parse_args(_BASE + ["--full-param", "--publish-mode", "fs", "--mb-target-frac", "0.7", "--role", "rollout"])
    assert c.publish_mode == "fs" and c.mb_target_frac == 0.7


# ------------------------------------------------------------------ shard geometry
def test_shard_rows_matches_torch_chunk():
    for n in (1, 2, 5, 7, 64, 65, 248320, 5120):
        for world in (1, 2, 3, 5, 6, 8):
            chunks = [c.shape[0] for c in torch.chunk(torch.zeros(n), world)]
            chunks += [0] * (world - len(chunks))
            assert FP.shard_rows(n, world) == chunks, (n, world)


def test_check_names_pick_unfused_tensors():
    names = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"] + \
            [f"model.language_model.layers.{i}.{s}" for i in range(4) for s in ("input_layernorm.weight", "mlp.down_proj.weight", "self_attn.o_proj.weight")]
    ck = FP.check_names(names)
    assert "model.language_model.norm.weight" in ck and "lm_head.weight" in ck
    assert "model.language_model.layers.0.mlp.down_proj.weight" in ck and "model.language_model.layers.3.mlp.down_proj.weight" in ck
    assert all(not n.endswith("input_layernorm.weight") for n in ck)


def test_find_vllm_param_mapping():
    params = {"language_model.model.layers.3.mlp.down_proj.weight": 1, "language_model.lm_head.weight": 2,
              "language_model.model.embed_tokens.weight": 3, "language_model.model.layers.3.linear_attn.in_proj_qkvz.weight": 4}
    assert FP.find_vllm_param(params, "model.language_model.layers.3.mlp.down_proj.weight")[0] == "language_model.model.layers.3.mlp.down_proj.weight"
    assert FP.find_vllm_param(params, "lm_head.weight")[0] == "language_model.lm_head.weight"
    assert FP.find_vllm_param(params, "model.language_model.layers.3.linear_attn.in_proj_qkv.weight") == (None, None)   # fused: not checkable
    hf = {"model.layers.3.mlp.down_proj.weight": 1}   # other naming: unique suffix
    assert FP.find_vllm_param(hf, "model.language_model.layers.3.mlp.down_proj.weight")[0] == "model.layers.3.mlp.down_proj.weight"


# ------------------------------------------------------------------ manifest + gather + fs round trip
def test_manifest_gather_and_fs_shards_roundtrip():
    ref = _tiny_model(0)
    ref_sd = {FP.map_name(k): v.clone() for k, v in ref.state_dict().items()}
    m = _fsdp_policy(0)
    man = FP.build_manifest(m, 1)
    assert man["names"] == [FP.map_name(n) for n, _ in m.named_parameters()] and man["n_trainer"] == 1
    assert man["names"][0] == "model.language_model.embed_tokens.weight" and "lm_head.weight" in man["names"]
    assert all(r == FP.shard_rows(s[0], 1) for s, r in zip(man["shapes"], man["rows"]))
    assert man["bytes_bf16"] == 2 * sum(v.numel() for v in ref_sd.values())
    n = 0
    for name, full in FP.iter_full_bf16(m, man):
        assert full.dtype == torch.bfloat16 and torch.equal(full, ref_sd[name]), name
        n += 1
    assert n == len(man["names"])
    w = tempfile.mkdtemp()
    tm = FP.write_shard_file(m, man, f"{w}/shard_0.safetensors", 0)
    assert tm["gb"] > 0 and os.path.exists(f"{w}/shard_0.safetensors")
    got = dict(FP.iter_fs_weights(w, man, "cpu"))
    assert set(got) == set(man["names"]) and all(torch.equal(got[k], ref_sd[k]) for k in got)
    loc = FP.all_reduce_dict_sum(FP.local_check_abs_sums(m, man), "cpu")
    fsck = FP.fs_checks(w, man)
    assert set(loc) == set(man["checks"]) == set(fsck) and len(loc) >= 4
    assert not FP.verify_checks(loc, fsck)
    bad = dict(fsck); k0 = sorted(bad)[0]; bad[k0] *= 1.01
    assert k0 in FP.verify_checks(loc, bad)
    # the engine side (vLLM naming: language_model.model.X / language_model.lm_head.weight) finds every check tensor and agrees on the |sum|
    eng = FP.engine_checks(_VllmNamed(ref), man["checks"])
    assert set(eng) == set(man["checks"]) and not FP.verify_checks(loc, eng)
    # a differently-named engine still resolves the unambiguous ones by whole-component suffix; the ambiguous final norm is skipped, never mis-matched
    eng_hf = FP.engine_checks(ref, man["checks"])
    assert "lm_head.weight" in eng_hf and "model.language_model.norm.weight" not in eng_hf and not FP.verify_checks(loc, eng_hf)


class _VllmNamed:
    """An HF model exposed under vLLM's Qwen3_5ForConditionalGeneration parameter names."""

    def __init__(self, hf):
        self.hf = hf

    def named_parameters(self):
        for k, v in self.hf.named_parameters():
            yield "language_model." + k, v


# ------------------------------------------------------------------ scorer read without _Stop
def test_read_resid_noraise_matches_read_resid_and_sharded_truncated_scorer():
    plain = _tiny_model(1).eval()
    ids = torch.randint(5, 500, (3, 9)); attn = torch.ones_like(ids); attn[2, 6:] = 0
    batch = {"input_ids": ids, "attention_mask": attn}
    h_ref, m_ref = read_resid(plain, 2, dict(batch), pool="all")
    h_new, m_new = FP.read_resid_noraise(plain, 2, dict(batch), pool="all")
    assert torch.equal(h_ref, h_new) and torch.equal(m_ref, m_new)
    assert torch.equal(read_resid(plain, 2, dict(batch), pool="mean"), FP.read_resid_noraise(plain, 2, dict(batch), pool="mean"))
    # truncated (layers [0, 3)) + TinyHead + FSDP2-sharded frozen copy: same layer-2 output, forward completes, no grads
    _dist()
    sc = _tiny_model(1)
    n_layers, dropped = FP.truncate_scorer_fsdp_safe(sc, 3)
    assert n_layers == 4 and dropped and len(sc.model.layers) == 3 and isinstance(sc.lm_head, FP.TinyHead)
    FP.shard_frozen_bf16(sc, 1, device_type="cpu")
    assert not any(p.requires_grad for p in sc.parameters())
    actor = D.BaseActor(sc)
    h_sc, _ = FP.read_resid_noraise(actor, 2, dict(batch), pool="all")
    assert torch.allclose(h_sc, h_ref, atol=2e-2, rtol=2e-2)     # bf16 compute either way; FSDP2 casts through the same dtype
    out = sc(**batch)
    assert out.logits.shape == (3, 9, 1) and float(out.logits.abs().sum()) == 0.0


# ------------------------------------------------------------------ loss scaling == weighted grad mean
def test_fsdp_mean_of_scaled_losses_equals_weighted_grad_mean():
    torch.manual_seed(0)
    world = 3
    grads = [torch.randn(50, dtype=torch.float64) for _ in range(world)]
    for weights in ([103.0, 102.0, 101.0], [8.0, 0.0, 5.0], [1.0, 1.0, 1.0]):
        tot = sum(weights)
        fsdp_mean = sum(FP.loss_scale(w, tot, world) * g for w, g in zip(weights, grads)) / world
        assert torch.allclose(fsdp_mean, FP.weighted_mean_reference(grads, weights), atol=1e-12)
    assert FP.loss_scale(3.0, 0.0, world) == 0.0
    assert FP.all_reduce_scalar(2.5, "cpu") == 2.5   # single process: identity


# ------------------------------------------------------------------ trainable fp32 head
def test_fp32_head_trainable_hook_on_fsdp_model():
    m = _fsdp_policy(2)
    ids = torch.randint(5, 500, (2, 7))
    before = m(input_ids=ids, use_cache=False).logits
    assert before.dtype == torch.bfloat16
    h = FP.install_fp32_head_trainable(m)
    out = m(input_ids=ids, use_cache=False).logits
    assert out.dtype == torch.float32 and out.shape == before.shape
    assert torch.allclose(out, before.float(), atol=0.1, rtol=0.05)   # same logits up to the bf16 rounding of the head output
    out.float().pow(2).mean().backward()
    assert m.lm_head.weight.grad is not None and float(m.lm_head.weight.grad.to_local().abs().sum()) > 0
    h.remove()
    m.zero_grad(set_to_none=True)


# ------------------------------------------------------------------ one update_disagg step on the tiny FSDP2 policy (no prefix cache)
def test_update_disagg_full_param_step_cpu():
    sp = importlib.util.spec_from_file_location("rl_hf", os.path.join(_HERE, "rl.py"))
    R = importlib.util.module_from_spec(sp); sys.modules["rl_hf"] = R; sp.loader.exec_module(R)
    m = _fsdp_policy(3)
    fp = D.FullParamCtx(FP, FP.fullft_module(), 1, 0)
    a = D.parse_args(_BASE + ["--full-param", "--init-adapter", "none", "--loss", "cispo", "--cispo-eps-max", "5", "--loss-agg", "prompt",
                              "--zero-var-filter", "--group-size", "2", "--groups-per-step", "2", "--kl-coef", "0", "--vocab-chunk", "4",
                              "--max-grad-norm", "1.0", "--lr", "1e-3"])
    opt = torch.optim.AdamW(list(m.parameters()), lr=a.lr, weight_decay=0.0)
    sub = get_layer(m, 1)
    p_len, T, n = len(PROMPT), 5, 4
    ids = torch.zeros((n, p_len + T), dtype=torch.long); attn = torch.zeros_like(ids)
    lens = [5, 3, 4, 2]
    for i, L in enumerate(lens):
        ids[i, :p_len] = torch.tensor(PROMPT); ids[i, p_len:p_len + L] = torch.randint(5, 500, (L,)); attn[i, :p_len + L] = 1
    old_lp = torch.full((n, T), -3.0); known = attn[:, p_len:].bool()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    dirs_rep = torch.nn.functional.normalize(torch.randn(n, 64), dim=-1)
    w0 = m.lm_head.weight.to_local().clone()
    st = D.update_disagg(m, opt, sub, ids, attn, p_len, MARKER, old_lp, known, adv, dirs_rep, a, "cpu", mb=2, keep=None, pfx=None, fp=fp)
    assert st["skipped"] == 0 and 0 < st["grad_norm"] < float("inf") and st["t_sync"] < 1.0
    assert not torch.equal(w0, m.lm_head.weight.to_local())          # AdamW moved the fp32 masters
    assert 0 <= st["clipfrac"] <= 1 and st["sampler_abs_dlogp"] > 0
    # the clone-mode injection hook fired (marker row changed) inside the FSDP2 forward
    cap = {}
    def grab(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
    with torch.no_grad():
        hd = sub.register_forward_hook(grab); m(input_ids=ids[:1], attention_mask=attn[:1], use_cache=False); hd.remove(); clean = cap["h"]
        hook = R.make_inject_hook([dirs_rep[:1]], [[MARKER]], 1.0, "cpu", torch.bfloat16, mode="add_clone")
        with hooked(sub, hook):
            hd = sub.register_forward_hook(grab); m(input_ids=ids[:1], attention_mask=attn[:1], use_cache=False); hd.remove(); inj = cap["h"]
    d = (inj[0, MARKER].float() - clean[0, MARKER].float())
    assert d.norm() > 0 and torch.nn.functional.cosine_similarity(d, dirs_rep[0], dim=0) > 0.9


# ------------------------------------------------------------------ rollout-side fs sync against a fake engine
def test_fullparam_sync_fs_with_fake_engine():
    m = _fsdp_policy(4)
    man = FP.build_manifest(m, 1)
    w = tempfile.mkdtemp()
    os.makedirs(f"{w}/lora/step_0"); os.makedirs(f"{w}/lora/step_1")
    FP.write_manifest(f"{w}/lora/manifest.json", man)
    D._atomic_write_text(f"{w}/lora/step_0/meta.json", json.dumps({"step": 0, "hnorm": 7.5, "mode": "initial", "checks": {}}))
    D._atomic_write_text(f"{w}/lora/latest", "0")
    FP.write_shard_file(m, man, f"{w}/lora/step_1/shard_0.safetensors", 0)
    checks = FP.all_reduce_dict_sum(FP.local_check_abs_sums(m, man), "cpu")
    D._atomic_write_text(f"{w}/lora/step_1/meta.json", json.dumps({"step": 1, "hnorm": 7.7, "mode": "fs", "checks": checks}))

    engine = _tiny_model(99)          # different weights: the load must overwrite them
    calls = []

    class FakeLLM:
        def collective_rpc(self, name, args=()):
            calls.append(name)
            assert name == "wu_load_fs"
            step_dir, mp = args
            sd = dict(_VllmNamed(engine).named_parameters())
            loaded = set()
            for ck, t in FP.iter_fs_weights(step_dir, json.load(open(mp)), "cpu"):
                vn, p = FP.find_vllm_param(sd, ck)
                assert p is not None, ck
                p.data.copy_(t); loaded.add(vn)
            return [{"n_vllm_params_loaded": len(loaded), "load_s": 0.01, "gb": 0.001, "gbps": 0.1, "checks": FP.engine_checks(_VllmNamed(engine), man["checks"])}]
    a = D.parse_args(_BASE + ["--full-param", "--init-adapter", "none", "--publish-mode", "fs", "--role", "rollout"])
    sync = D._FullParamSync(FakeLLM(), a, 0, w, "R0")
    assert sync.initial() == 0 and sync.cur_step == 0 and sync.hnorm == 7.5 and not sync.pending()
    D._atomic_write_text(f"{w}/lora/latest", "1")
    assert sync.poll() is True and sync.cur_step == 1 and sync.hnorm == 7.7 and calls == ["wu_load_fs"]
    ref = {FP.map_name(k): (v.full_tensor() if FP.is_dtensor(v) else v) for k, v in m.state_dict().items()}
    for k, v in engine.state_dict().items():
        assert torch.equal(v, ref[FP.map_name(k)].to(v.dtype)), k
    assert sync.poll() is False and calls == ["wu_load_fs"]
    # a corrupted engine load is caught by the checksums
    D._atomic_write_text(f"{w}/lora/step_1/meta.json", json.dumps({"step": 1, "hnorm": 7.7, "mode": "fs", "checks": {k: v * 1.5 for k, v in checks.items()}}))
    sync.cur_step = 0
    try:
        sync.poll(); assert False
    except RuntimeError as e:
        assert "checksums differ" in str(e)
    FP.prune_fs_steps(w, 1)
    assert os.path.exists(f"{w}/lora/step_1/shard_0.safetensors")


# ------------------------------------------------------------------ run_meta + prefix accumulator extras
def test_run_meta_records_full_param():
    a = D.parse_args(_BASE + ["--full-param", "--init-adapter", "none", "--policy-base", "/x/final", "--kl-coef", "0.01"])
    d = tempfile.mkdtemp()
    meta = D.write_run_meta(a, d, 8, step=3)
    assert meta["format"] == "rl_disagg_fullparam_v1" and meta["full_param"] and meta["lora"] is None
    assert meta["policy_base"] == "/x/final" and meta["kl_ref"] == "frozen_init_copy" and meta["publish_mode"] == "nccl"
    b = D.parse_args(_BASE)
    mb = D.write_run_meta(b, d, 8)
    assert mb["format"] == "rl_disagg_lora_v1" and not mb["full_param"] and mb["publish_mode"] == "lora_adapter"


def test_prefix_grad_accumulator_extra_outputs_get_zero_grad():
    x = torch.randn(4, dtype=torch.float64, requires_grad=True)
    c = x * 2.0                              # a "cache tensor" of the prefix graph
    extra = (x * 5.0).sum()                  # the prefix logits stand-in: must be traversed with a ZERO gradient
    cache = types.SimpleNamespace(layers=[types.SimpleNamespace(k=c, meta="keep")])
    acc = D._PrefixGradAccumulator(cache, extra_outputs=[extra])
    leaf = acc.cache.layers[0].k
    assert leaf.requires_grad and leaf.grad_fn is None and acc.cache.layers[0].meta == "keep"
    (leaf * 1.5).sum().backward(); acc.accumulate()
    (leaf * 0.5).sum().backward(); acc.backward()
    assert torch.allclose(x.grad, torch.full((4,), 4.0, dtype=torch.float64))   # 2 * (1.5 + 0.5); the extra path contributed 0
    assert acc._finished and acc.cache is None


# ------------------------------------------------------------------ chunked recomputed head == plain fp32 head path (same grads)
def test_chunked_head_matches_plain_fp32_head_gradients():
    sp = importlib.util.spec_from_file_location("rl_hf", os.path.join(_HERE, "rl.py"))
    R = importlib.util.module_from_spec(sp); sys.modules["rl_hf"] = R; sp.loader.exec_module(R)
    base_args = _BASE + ["--full-param", "--init-adapter", "none", "--loss", "cispo", "--cispo-eps-max", "5", "--loss-agg", "prompt",
                         "--group-size", "2", "--groups-per-step", "2", "--kl-coef", "0", "--vocab-chunk", "3", "--max-grad-norm", "1e9",
                         "--lr", "1e-3", "--fp32-head", "--entropy-coef", "0.01"]
    p_len, T, n = len(PROMPT), 5, 4
    torch.manual_seed(7)
    ids = torch.zeros((n, p_len + T), dtype=torch.long); attn = torch.zeros_like(ids)
    for i, L in enumerate([5, 3, 4, 2]):
        ids[i, :p_len] = torch.tensor(PROMPT); ids[i, p_len:p_len + L] = torch.randint(5, 500, (L,)); attn[i, :p_len + L] = 1
    old_lp = torch.full((n, T), -3.0); known = attn[:, p_len:].bool()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5]); dirs_rep = torch.nn.functional.normalize(torch.randn(n, 64), dim=-1)

    class _Rec:   # AdamW stand-in that snapshots the flat grad and never updates
        def __init__(self, params): self.params, self.grads, self.param_groups = list(params), None, [{"lr": 0.0}]
        def zero_grad(self, set_to_none=True):
            for p in self.params: p.grad = None
        def step(self): self.grads = torch.cat([(p.grad.to_local() if FP.is_dtensor(p.grad) else p.grad).detach().flatten().float() for p in self.params if p.grad is not None])

    def run(chunked):
        m = _fsdp_policy(11)
        fp = D.FullParamCtx(FP, FP.fullft_module(), 1, 0)
        a = D.parse_args(base_args + (["--chunked-head"] if chunked else []))
        if chunked:
            fp.head = FP.ChunkedHead(m)
        else:
            FP.install_fp32_head_trainable(m)
        opt = _Rec(m.parameters())
        st = D.update_disagg(m, opt, get_layer(m, 1), ids, attn, p_len, MARKER, old_lp, known, adv, dirs_rep, a, "cpu", mb=2, keep=None, pfx=None, fp=fp)
        return st, opt.grads
    st_a, g_a = run(False)
    st_b, g_b = run(True)
    rel = float((g_a - g_b).norm() / g_a.norm().clamp_min(1e-12))
    cos = float(torch.nn.functional.cosine_similarity(g_a, g_b, dim=0))
    assert g_a.shape == g_b.shape and rel < 1e-2 and cos > 0.9999, (rel, cos, float((g_a - g_b).abs().max()))   # bf16 kernel noise only
    for k in ("loss", "entropy", "grad_norm", "sampler_abs_dlogp"):
        assert abs(st_a[k] - st_b[k]) <= 1e-3 * max(1.0, abs(st_a[k])), (k, st_a[k], st_b[k])
    m = _fsdp_policy(11); fp = D.FullParamCtx(FP, FP.fullft_module(), 1, 0); fp.head = FP.ChunkedHead(m)
    out = m(input_ids=ids[:1], attention_mask=attn[:1], use_cache=False).logits
    assert out.shape[-1] == 512, "inactive chunked head must return the real logits"
    with fp.suffix_ctx():
        out = m(input_ids=ids[:1], attention_mask=attn[:1], use_cache=False).logits
        assert out.shape[-1] == 1 and fp.head.hidden.shape == (1, ids.shape[1], 64)
    assert not fp.head.active and fp.head.hidden is None


# ------------------------------------------------------------------ dummy micro-batch reaches every parameter (FSDP2 reduce-scatter sizes)
def test_dummy_micro_batch_gives_every_param_a_grad_with_chunked_head():
    """FSDP2 reduce-scatters only params WITH grads as one flat collective per group: a dummy micro-batch whose loss skips lm_head
    (chunked head) would make that rank's root reduce-scatter shorter than the others' -> hang. The dummy must go through the head."""
    sp = importlib.util.spec_from_file_location("rl_hf", os.path.join(_HERE, "rl.py"))
    R = importlib.util.module_from_spec(sp); sys.modules["rl_hf"] = R; sp.loader.exec_module(R)
    m = _fsdp_policy(21)
    fp = D.FullParamCtx(FP, FP.fullft_module(), 1, 0)
    fp.head = FP.ChunkedHead(m)
    a = D.parse_args(_BASE + ["--full-param", "--init-adapter", "none", "--chunked-head", "--fp32-head", "--vocab-chunk", "4", "--loss", "cispo",
                              "--loss-agg", "prompt", "--group-size", "2", "--groups-per-step", "2", "--kl-coef", "0", "--max-grad-norm", "1e9"])
    p_len, T, n = len(PROMPT), 4, 2
    ids = torch.zeros((n, p_len + T), dtype=torch.long); attn = torch.zeros_like(ids)
    for i in range(n):
        ids[i, :p_len] = torch.tensor(PROMPT); ids[i, p_len:] = torch.randint(5, 500, (T,)); attn[i] = 1
    # replicate the dummy block of update_disagg by hand: one short row through policy_logits + logp_from, zero-weight loss
    sub = get_layer(m, 1)
    dirs_rep = torch.nn.functional.normalize(torch.randn(n, 64), dim=-1)
    hook = R.make_inject_hook([dirs_rep[:1]], [[MARKER]], 1.0, "cpu", torch.bfloat16, mode="add_clone")
    with hooked(sub, hook), fp.suffix_ctx():
        lg = m(input_ids=ids[:1], attention_mask=attn[:1], use_cache=False, logits_to_keep=T + 1).logits[:, :-1]
        assert lg.shape[-1] == 1                                   # the chunked head returned its dummy
        lp1, _ = fp.logp_from(lg, ids[:1, p_len:], a.vocab_chunk, False, a.fp32_head)
        (lp1.float().sum() * 0.0).backward()
    missing = [nme for nme, p in m.named_parameters() if p.grad is None]
    assert not missing, f"params without a grad after the dummy pass (would desync FSDP2's reduce-scatter): {missing[:5]}"
    assert m.lm_head.weight.grad is not None and float(m.lm_head.weight.grad.to_local().abs().sum()) == 0.0   # zero, but PRESENT


# ------------------------------------------------------------------ full-param micro-batch probe planning
def test_plan_probe_step_walks_up_within_budget():
    GB = 2**30
    cands = [4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 96, 128]
    budget = 108 * GB
    m = {}
    assert D.plan_probe_step(m, cands, budget) == 4                       # first two attempts are free
    m[4] = 67 * GB
    assert D.plan_probe_step(m, cands, budget) == 6
    m[6] = 67 * GB                                                         # fixed-dominated: slope 0 -> next candidate, growth-capped at 2x
    assert D.plan_probe_step(m, cands, budget) == 8
    m[8] = 67 * GB
    assert D.plan_probe_step(m, cands, budget) == 12
    m[12] = 80 * GB                                                        # slope 3.25 GB/seq from (8, 12): 16 -> 93 fits, 24 -> 119 does not
    assert D.plan_probe_step(m, cands, budget) == 16
    m[16] = 105 * GB
    assert D.plan_probe_step(m, cands, budget) is None                     # 24 -> 105 + 6.25 * 8 = 155 > budget
    assert D.plan_probe_step({4: 1, 8: 2}, [4, 8], budget) is None         # nothing left to try
    assert D.plan_probe_step({4: 60 * GB, 8: 60 * GB}, cands, budget) == 12
    assert D.plan_probe_step({4: 60 * GB, 8: 60 * GB}, [4, 8, 32], budget) is None   # 32 > 2 x 8: never jump more than a doubling


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
