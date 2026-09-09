"""Fast vllm_lens worker extension for the disaggregated RL rollout ranks (train/rl_disagg.py).

Why this exists. vllm_lens 1.1.0's stock ``HiddenStatesExtension`` registers a forward hook on EVERY
decoder layer and, on EVERY forward step, for EVERY request, calls ``_find_steering_configs`` which
scans ALL registered steering keys with ``str.startswith`` -> O(layers x reqs x keys) Python per decode
step. With 256 concurrent requests that is ~64 x 256 x 256 = 4.2M string ops per step (~1.5 s), which
is exactly the "generation is latency-bound and batches sub-linearly" behaviour rl.py measured
(8 seqs: 9 s; 256 seqs: 65 s for ~40 decode steps). The stock plugin also FORCES ``enforce_eager``.

This subclass keeps the plugin's data protocol (``set_steering_data`` / ``_steering_id`` in
``SamplingParams.extra_args`` / ``output_residual_stream`` capture) but:
  * hooks ONLY the layers we steer or capture (``FAST_LENS_LAYERS``, default "1" = INJECT_LAYER);
  * resolves a request's steering config with ONE dict lookup on its ``_steering_id`` (no key scan);
  * skips pure-decode steps entirely (``max_query_len == 1``): the marker is a PROMPT position, so
    steering can only ever apply in the forward pass that prefills it. Capture at decode positions
    is therefore not supported in this extension (the trainer's verify probe uses max_tokens=1);
  * reads ``query_start_loc`` / ``seq_lens`` to the host ONCE per hooked step (not once per request);
  * adds ``set_steering_data_many`` / ``clear_steering_data_many`` so a 1024-request block costs one
    collective_rpc instead of 1024.

CUDA graphs: with ``cudagraph_mode=FULL_DECODE_ONLY`` (and compilation mode NONE so hooks still fire in
eager pieces) uniform-decode batches are graph replays where this hook does not run -- which is exactly
the pure-decode step we skip anyway -- and every batch containing prefill tokens runs eagerly with the
hook live. During graph CAPTURE (dummy run) there are no real requests so the hook is a no-op and nothing
steering-related is baked into the graph. rl_disagg.py verifies the result numerically (injected delta
cos > 0.99, magnitude ratio in [0.95, 1.05]) and, on the trainer, with the sampler-vs-policy |dlogp| at
step 0 (ratio must be ~1.0 -- a request whose injection was silently skipped shows up as a ~1.5-nat tail).

Selected via ``LLM(worker_extension_cls="fast_lens_ext.FastSteerExtension")`` -- this module must be
importable in the vLLM worker process (PYTHONPATH). The plugin only sets its own extension class when
none is given, so this simply takes precedence.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_lens._worker_ext import HiddenStatesExtension, _apply_steering, _get_layers

logger = logging.getLogger(__name__)

HOOK_LAYERS = [int(x) for x in os.environ.get("FAST_LENS_LAYERS", "1").split(",") if x.strip()]
SKIP_DECODE = os.environ.get("FAST_LENS_SKIP_DECODE", "1") == "1"

# diagnostics readable via collective_rpc(_fast_lens_stats)
_STATS = {"hook_calls": 0, "decode_skips": 0, "steered_reqs": 0, "steer_steps": 0, "errors": 0}


def _fast_lens_stats(worker=None):
    return dict(_STATS)


def _fast_hook_inner(ext: "FastSteerExtension", layer_idx: int, output):
    _STATS["hook_calls"] += 1
    if not is_forward_context_available():
        return None
    runner = ext.model_runner
    num_reqs = runner.input_batch.num_reqs
    if num_reqs == 0:
        return None
    ctx = get_forward_context()
    am = ctx.attn_metadata
    if am is None:
        return None
    if isinstance(am, list):
        am = am[0]
        if am is None:
            return None
    qsl = seq_lens = mql = None
    for meta in am.values():
        if hasattr(meta, "query_start_loc"):
            qsl = meta.query_start_loc
            seq_lens = getattr(meta, "seq_lens", None)
            mql = getattr(meta, "max_query_len", None)
            break
    if qsl is None:
        return None
    if SKIP_DECODE and mql is not None:
        try:
            mql_i = int(mql.item() if isinstance(mql, torch.Tensor) else mql)
        except Exception:  # noqa
            mql_i = 0
        if mql_i == 1:                       # uniform decode: no prompt position is being computed
            _STATS["decode_skips"] += 1
            return None

    req_ids = runner.input_batch.req_ids
    per_req: list = []
    want_cap: list = []
    need_steer = need_cap = False
    for i in range(num_reqs):
        rs = runner.requests.get(req_ids[i])
        extra = rs.sampling_params.extra_args if (rs is not None and rs.sampling_params is not None) else None
        cfgs = None
        cap = None
        if extra:
            sid = extra.get("_steering_id")
            if sid is not None:
                cfgs = ext._steering_data.get(sid)
            cap = extra.get("output_residual_stream")
            if isinstance(cap, list) and layer_idx not in cap:
                cap = None
        per_req.append(cfgs)
        want_cap.append(cap)
        need_steer |= bool(cfgs)
        need_cap |= cap is not None
    if not need_steer and not need_cap:
        return None

    qsl_h = qsl.tolist() if isinstance(qsl, torch.Tensor) else list(qsl)
    sl_h = None
    if seq_lens is not None:
        sl_h = seq_lens.tolist() if isinstance(seq_lens, torch.Tensor) else list(seq_lens)

    modified = None
    if need_steer:
        _STATS["steer_steps"] += 1
        if isinstance(output, tuple):
            modified = (output[0].clone(), *output[1:])
            target = modified[0]
        else:
            modified = output.clone()
            target = modified
        for i in range(num_reqs):
            cfgs = per_req[i]
            if not cfgs:
                continue
            start, end = int(qsl_h[i]), int(qsl_h[i + 1])
            n_query = end - start
            abs_start = int(sl_h[i]) - n_query if sl_h is not None else 0
            _apply_steering(cfgs, layer_idx, target, start, end, abs_start)
            _STATS["steered_reqs"] += 1

    if need_cap and getattr(ext, "_should_capture", True):
        src = modified if modified is not None else output
        if isinstance(src, tuple):
            hs = src[0] + src[1] if (len(src) > 1 and src[1] is not None) else src[0]
        else:
            hs = src
        for i in range(num_reqs):
            if want_cap[i] is None:
                continue
            start, end = int(qsl_h[i]), int(qsl_h[i + 1])
            act = hs[start:end].cpu()
            rid = req_ids[i]
            layer_states = ext._captured_states.setdefault(rid, {})
            layer_states.setdefault(layer_idx, []).append(act)
    return modified


def _make_fast_hook(ext, layer_idx):
    def hook(_module, _inp, output):
        try:
            return _fast_hook_inner(ext, layer_idx, output)
        except Exception:  # noqa
            _STATS["errors"] += 1
            logger.warning("fast_lens hook error on layer %d, skipping", layer_idx, exc_info=True)
            return None
    return hook


class FastSteerExtension(HiddenStatesExtension):
    """Drop-in for vllm_lens' HiddenStatesExtension: same RPC surface, O(1) per-request steering
    lookup, hooks only on FAST_LENS_LAYERS, pure-decode steps skipped. See module docstring."""

    def install_hooks(self) -> None:
        if self._hooks_installed:
            return
        self._hooks_installed = True
        self._captured_states = {}
        self._steering_data = {}
        tp = self.parallel_config.tensor_parallel_size
        self._should_capture = tp <= 1 or self.rank % tp == 0
        layers = _get_layers(self.model_runner.model)
        for li in HOOK_LAYERS:
            layers[li].register_forward_hook(_make_fast_hook(self, li))
        logger.info("fast_lens: hooks installed on layers %s (skip_decode=%s)", HOOK_LAYERS, SKIP_DECODE)

    def set_steering_data_many(self, pickled: bytes) -> int:
        """{key: [SteeringVector, ...]} in ONE rpc; tensors moved to the model device/dtype."""
        d: dict[str, list[Any]] = pickle.loads(pickled)
        p = next(self.model_runner.model.parameters())
        device, dtype = p.device, p.dtype
        n_layers = len(_get_layers(self.model_runner.model))
        for key, svs in d.items():
            out = []
            for sv in svs:
                for idx in sv.layer_indices:
                    if idx < 0 or idx >= n_layers:
                        raise ValueError(f"layer_index {idx} out of range [0, {n_layers})")
                out.append(sv.model_copy(update={"activations": sv.activations.to(device=device, dtype=dtype)}))
            self._steering_data[key] = out
        return len(d)

    def clear_steering_data_many(self, keys: list[str]) -> None:
        for k in keys:
            self._steering_data.pop(k, None)

    def fast_lens_stats(self) -> dict:
        return dict(_STATS)

    # ------------------------------------------------------------------------------------------
    # full-parameter weight sync (rl_disagg --full-param; protocol in rl/rl_fullparam.py). `self` is the vLLM Worker
    # (worker_extension_cls mixin): self.device, self.model_runner.model. Called through llm.collective_rpc between
    # generate() calls, i.e. at a block boundary -- never while the engine runs a forward.
    # ------------------------------------------------------------------------------------------
    def wu_init(self, host: str, port: int, rank: int, world_size: int, manifest_path: str, store_timeout: int = 1800) -> dict:
        """Join the weight-update NCCL group (rank 0 = trainer rank 0) and load the publish manifest."""
        import json
        import time
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
        t0 = time.time()
        self._wu_manifest = json.load(open(manifest_path))
        pg = StatelessProcessGroup.create(host, int(port), rank=int(rank), world_size=int(world_size), store_timeout=int(store_timeout))
        self._wu_comm = PyNcclCommunicator(pg, device=self.device)
        self._wu_pg = pg
        assert not self._wu_comm.disabled, "PyNcclCommunicator disabled in the vLLM worker (NCCL library missing?)"
        self._wu_n_loads = 0
        return {"rank": int(rank), "world": int(world_size), "n_tensors": len(self._wu_manifest["names"]),
                "gb": self._wu_manifest["bytes_bf16"] / 2**30, "init_s": time.time() - t0, "device": str(self.device)}

    def _wu_load(self, stream_of_weights):
        """Feed (ckpt name, tensor) lazily into vLLM's own model.load_weights (in place: parameters keep their storage, so
        CUDA graphs, the steering hook and everything else stay valid). Returns the set of vLLM params it touched."""
        model = self.model_runner.model
        loaded = model.load_weights(stream_of_weights)
        torch.cuda.synchronize()
        self._wu_n_loads += 1
        return loaded

    def wu_recv(self, step: int) -> dict:
        """publish-mode nccl: receive every tensor of the manifest by broadcast from group rank 0, in order, straight into
        load_weights. One staging buffer per tensor (fresh torch.empty; the caching allocator reuses the blocks)."""
        import time
        man = self._wu_manifest
        comm = self._wu_comm
        stream = torch.cuda.current_stream(self.device)
        t0 = time.time()
        n_bytes = [0]

        def gen():
            for name, shape in zip(man["names"], man["shapes"]):
                buf = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
                comm.broadcast(buf, src=0, stream=stream)
                n_bytes[0] += buf.numel() * 2
                yield name, buf
        loaded = self._wu_load(gen())
        t_load = time.time() - t0
        from rl_fullparam import engine_checks
        checks = engine_checks(self.model_runner.model, man["checks"])
        return {"step": int(step), "n_vllm_params_loaded": len(loaded), "gb": n_bytes[0] / 2**30, "load_s": t_load,
                "gbps": n_bytes[0] / 2**30 / max(t_load, 1e-6), "checks": checks, "n_loads": self._wu_n_loads}

    def wu_load_fs(self, step_dir: str, manifest_path: str) -> dict:
        """publish-mode fs: stream the trainer ranks' bf16 shard files (rl_fullparam.write_shard_file) into load_weights."""
        import json
        import time
        from rl_fullparam import engine_checks, iter_fs_weights
        man = json.load(open(manifest_path))
        if not hasattr(self, "_wu_n_loads"):
            self._wu_n_loads = 0
        t0 = time.time()
        loaded = self._wu_load(iter_fs_weights(step_dir, man, self.device))
        t_load = time.time() - t0
        checks = engine_checks(self.model_runner.model, man["checks"])
        return {"step_dir": step_dir, "n_vllm_params_loaded": len(loaded), "gb": man["bytes_bf16"] / 2**30, "load_s": t_load,
                "gbps": man["bytes_bf16"] / 2**30 / max(t_load, 1e-6), "checks": checks, "n_loads": self._wu_n_loads}

    def wu_checks(self, names: list) -> dict:
        from rl_fullparam import engine_checks
        return engine_checks(self.model_runner.model, list(names))
