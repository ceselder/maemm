"""--full-ft for sft/pretrain.py: every weight of Qwen3.6-27B trainable, sharded with torch FSDP2 (fully_shard).

Layout (per rank, 8xB200): fp32 sharded master params (13.5 GB) + fp32 sharded grads (13.5 GB) + fp32 AdamW moments
(27 GB) = ~54 GB of optimizer state, plus one decoder layer's bf16 all-gather buffer at a time. Compute happens in bf16
(MixedPrecisionPolicy param_dtype=bf16, reduce_dtype=fp32), exactly the precision the LoRA path ran the frozen base in.

Loading: each rank loads the bf16 HF model onto its own GPU (54 GB, the same path pretrain.py always used), then every
decoder layer is upcast to fp32 and sharded IMMEDIATELY (peak = the bf16 model + one fp32 layer), then the root
(embed_tokens / final norm / lm_head).

Checkpoints are FULL HF models in the base repo's on-disk layout (Qwen3_5ForConditionalGeneration config, weights named
model.language_model.*, lm_head.weight, bf16 safetensors shards + index, tokenizer files, and the base's untouched
vision-tower / MTP tensors copied in so the directory is byte-for-byte the same schema as Qwen/Qwen3.6-27B). Both
transformers' AutoModelForCausalLM and vLLM (language_model_only) load it exactly like the base. The directory is
written as <path>.tmp and renamed at the end; SAVE_DONE (json) is written last -- consumers must require it.
"""
import contextlib
import json
import os
import shutil
import time

import torch
import torch.distributed as dist

_LM_PREFIX = "model.language_model."


class FSDPNoSync:
    """`ddp.no_sync()` stand-in: FSDP2 reduce-scatters every micro-batch into the fp32 sharded grads (correct
    accumulation, no 54 GB unsharded-grad buffer), so gradient accumulation needs NO communication skipping."""

    def __call__(self):
        return contextlib.nullcontext()


def shard_full_model(model, world, device, log=print):
    """In-place: all params trainable, fp32 masters, FSDP2-sharded per decoder layer + root. Returns the model."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cuda", (world,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(True)
    t0 = time.time()
    layers = model.model.layers
    for layer in layers:
        layer.float()                                   # fp32 master for THIS layer only ...
        fully_shard(layer, mesh=mesh, mp_policy=mp)     # ... then shard it before touching the next one
    for m in (model.model.embed_tokens, model.model.norm, model.lm_head):
        m.float()
    fully_shard(model, mesh=mesh, mp_policy=mp)         # root group: embed_tokens + norm + lm_head
    torch.cuda.synchronize()
    n = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[fullft] FSDP2 sharded {len(layers)} layers + root over {world} ranks in {time.time() - t0:.0f}s | "
        f"{n / 1e9:.2f}B params, {n_tr / 1e9:.2f}B trainable | resident {torch.cuda.memory_allocated() / 2**30:.1f} GB "
        f"(peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)")
    model.no_sync = FSDPNoSync()   # pretrain.py calls ddp.no_sync() for grad accumulation
    return model


def clip_grad_norm(params, max_norm):
    """clip_grad_norm_ over FSDP2 DTensor grads; returns the total norm as a python float."""
    gn = torch.nn.utils.clip_grad_norm_(params, max_norm)
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(gn, DTensor):
            gn = gn.full_tensor()
    except ImportError:
        pass
    return float(gn)


@contextlib.contextmanager
def injection_probe(model, inject_layer, log=print, tag="fullft"):
    """One-batch proof that the layer-`inject_layer` injection hook fires under FSDP2: compare the marker row
    (suffix index 0 in the prefix-cache path) of the injection layer's output BEFORE the injection hook (a forward
    hook registered before it) with the next layer's INPUT (after every hook). Norm-matched addition of a unit
    vector gives ratio ~ sqrt(2 + 2 cos) ~ 1.41; ratio 1.00 means the hook did NOT fire."""
    cap = {}
    L = model.model.layers

    def pre_inj(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            cap["pre"] = h[:, 0].detach().float().norm(dim=-1)

    def next_in(_m, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is not None and h.shape[1] > 1 and "pre" in cap and "post" not in cap:
            cap["post"] = h[:, 0].detach().float().norm(dim=-1)

    h1 = L[inject_layer].register_forward_hook(pre_inj)
    h2 = L[inject_layer + 1].register_forward_pre_hook(next_in, with_kwargs=True)
    try:
        yield
    finally:
        h1.remove(); h2.remove()
        if "pre" in cap and "post" in cap:
            r = (cap["post"] / cap["pre"].clamp_min(1e-6))
            log(f"[{tag}] injection check @layer {inject_layer}: marker-row norm ratio post/pre = {r.mean():.3f} "
                f"(min {r.min():.3f} max {r.max():.3f}; ~1.41 expected, 1.00 = hook NOT firing) -> "
                f"{'OK' if r.mean() > 1.2 else 'FAIL'}")
        else:
            log(f"[{tag}] injection check: probe captured nothing ({sorted(cap)})")


# ---------------------------------------------------------------------------------------------------------------
# full-model checkpoints in the base repo layout
# ---------------------------------------------------------------------------------------------------------------
def _base_snapshot_dir(model_id):
    """Local snapshot dir of the base repo (offline: resolves from HF_HOME)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id)


def prepare_nontext_shard(model_id, out_path, log=print):
    """Copy every base tensor that is NOT part of the text model (vision tower, MTP head) into one safetensors file,
    so a checkpoint directory carries the exact tensor set of the base repo. Returns the list of tensor names."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    snap = _base_snapshot_dir(model_id)
    idx = json.load(open(f"{snap}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    extra = sorted(k for k in wm if not k.startswith(_LM_PREFIX) and k != "lm_head.weight")
    if os.path.exists(out_path):
        return extra
    tensors, by_file = {}, {}
    for k in extra:
        by_file.setdefault(wm[k], []).append(k)
    t0 = time.time()
    for fn, keys in by_file.items():
        with safe_open(f"{snap}/{fn}", framework="pt", device="cpu") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k).contiguous()
    tmp = out_path + ".tmp"
    save_file(tensors, tmp, metadata={"format": "pt"})
    os.replace(tmp, out_path)
    log(f"[fullft] non-text base tensors ({len(extra)}, {sum(t.numel() * t.element_size() for t in tensors.values()) / 2**30:.2f} GB) "
        f"staged at {out_path} in {time.time() - t0:.0f}s")
    return extra


def map_name(name):
    """Qwen3_5ForCausalLM param name -> base repo (Qwen3_5ForConditionalGeneration) tensor name."""
    if name.startswith("model."):
        return _LM_PREFIX + name[len("model."):]
    return name   # lm_head.weight


def save_full_ckpt(model, path, tok, model_id, is_main, world, nontext_shard=None, shard_bytes=4 * 2**30, log=print, extra_meta=None):
    """COLLECTIVE (every rank must call it): all-gather each FSDP2-sharded param (one at a time), rank 0 writes bf16
    safetensors shards in the base layout + config/tokenizer files + the non-text shard + SAVE_DONE. Atomic via
    <path>.tmp -> rename."""
    from safetensors.torch import save_file
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover
        DTensor = ()

    t0 = time.time()
    tmp = path + ".tmp"
    if is_main:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
    snap = _base_snapshot_dir(model_id) if is_main else None
    base_keys = set(json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]) if is_main else None

    weight_map, cur, cur_bytes, parts, total_bytes, n_tensors = {}, {}, 0, [], 0, 0

    def flush():
        nonlocal cur, cur_bytes
        if not cur:
            return
        fn = f"part-{len(parts):05d}.safetensors"
        save_file(cur, f"{tmp}/{fn}", metadata={"format": "pt"})
        parts.append(fn)
        for k in cur:
            weight_map[k] = fn
        cur, cur_bytes = {}, 0

    for name, p in model.named_parameters():   # identical order on every rank -> the all-gathers line up
        with torch.no_grad():
            full = p.full_tensor() if isinstance(p, DTensor) else p.detach()
        if is_main:
            t = full.detach().to(torch.bfloat16).cpu().contiguous()
            k = map_name(name)
            assert k in base_keys, f"{name} -> {k} is not a tensor of {model_id}"
            cur[k] = t
            cur_bytes += t.numel() * t.element_size()
            total_bytes += t.numel() * t.element_size()
            n_tensors += 1
            if cur_bytes >= shard_bytes:
                flush()
        del full
    if is_main:
        flush()
        n_parts = len(parts)
        final_names = {fn: f"model-{i + 1:05d}-of-{n_parts:05d}.safetensors" for i, fn in enumerate(parts)}
        for fn, new in final_names.items():
            os.replace(f"{tmp}/{fn}", f"{tmp}/{new}")
        weight_map = {k: final_names[fn] for k, fn in weight_map.items()}
        if nontext_shard and os.path.exists(nontext_shard):
            from safetensors import safe_open
            shutil.copy(nontext_shard, f"{tmp}/model-nontext.safetensors")
            with safe_open(nontext_shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weight_map[k] = "model-nontext.safetensors"
        missing = base_keys - set(weight_map)
        assert not missing, f"checkpoint misses {len(missing)} base tensors, e.g. {sorted(missing)[:5]}"
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                  open(f"{tmp}/model.safetensors.index.json", "w"), indent=1)
        for fn in os.listdir(snap):   # config.json (ConditionalGeneration), tokenizer files, chat template, generation config
            if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or fn.startswith("."):
                continue
            src = os.path.join(snap, fn)
            if os.path.isfile(src):
                shutil.copy(os.path.realpath(src), f"{tmp}/{fn}")
        try:
            tok.save_pretrained(tmp)
        except Exception as e:  # noqa — base tokenizer files were copied above already
            log(f"[fullft] tok.save_pretrained failed ({e}); base tokenizer files kept")
        meta = {"format": "full_model_bf16_base_layout", "n_tensors_text": n_tensors, "n_parts": n_parts,
                "bytes_text": total_bytes, "saved_at": time.time(), "save_s": time.time() - t0, **(extra_meta or {})}
        json.dump(meta, open(f"{tmp}/SAVE_DONE", "w"), indent=1)   # written LAST, before the rename
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.replace(tmp, path)
        log(f"[fullft] saved full model -> {path} ({total_bytes / 2**30:.1f} GB text weights in {n_parts} shards"
            f"{' + non-text shard' if nontext_shard else ''}) in {time.time() - t0:.0f}s")
    if world > 1:
        dist.barrier()
