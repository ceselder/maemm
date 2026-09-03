"""--fp8-base: run the FROZEN base nn.Linear layers of a PEFT-wrapped model in torchao float8 (experimental, default off).

Blackwell tensor cores run fp8 GEMMs at ~2x bf16 throughput. Only the base weights are frozen here (LoRA A/B train), so
the base linears need just two GEMMs per step -- output = x @ W^T and grad_input = grad_out @ W -- and this module runs
exactly those two in fp8 via torchao's float8 training stack (casts, Float8TrainingTensor, torch._scaled_mm dispatch).

What gets converted: every nn.Linear that (i) has a frozen weight (requires_grad=False), (ii) is not a LoRA A/B module,
lm_head or an embedding, and (iii) passes the shape filter -- both dims multiples of 16 (hardware requirement) and both
>= MIN_DIM (fp8 buys nothing on tiny GEMMs and the 5120->48 GatedDeltaNet gate projections in_proj_b/in_proj_a feed fp32
recurrence params, so they stay bf16). On Qwen3.6-27B this is q/k/v/o_proj, in_proj_qkv/z, out_proj, gate/up/down_proj:
~24.3B of the ~27B params; lm_head (1.27B) and the embeddings keep bf16.

Recipe (env MAEMM_FP8_RECIPE, torchao Float8LinearRecipeName):
  rowwise (default): e4m3 for activations, weights and grad_output, per-row (per-token) scales on the activation operands,
                     per-output-channel (fwd) / per-input-channel (grad_input) scales on the weight, scales rounded down to
                     powers of two. The accurate option.
  tensorwise:        torchao's default -- e4m3 act/weight, e5m2 grad_output, one scale per tensor, cuBLAS tensorwise
                     kernel. Fastest, least accurate.
  rowwise_with_gw_hp: rowwise fwd, tensorwise weight for grad_input (grad_weight is never computed here anyway).

Why not just `torchao.float8.convert_to_float8_training` on the base layers -- two frozen-weight specialisations:
  1. stock `matmul_with_hp_or_float8_args.backward` computes grad_weight UNCONDITIONALLY (no needs_input_grad check): a
     third fp8 GEMM + two casts per layer that the bf16 path never runs (aten::mm skips the weight grad when
     requires_grad=False). Here backward computes grad_input only.
  2. stock Float8Linear re-casts the weight dynamically in every forward AND every backward, and torchao's
     `preprocess_addmm` needs the weight operand column-major in both GEMMs (two different layouts -> a transposed copy per
     step). The weight is frozen, so both fp8 layouts are cast ONCE at conversion (W as [N,K] for the forward, W^T as [K,N]
     for grad_input) and the bf16 master copy is dropped: 2 x 1 byte/param = the same memory as bf16, zero per-step weight
     casts/transposes, numerics identical to casting dynamically (the weight never changes).
Activations and grad_output are still cast per step with torchao's own `hp_tensor_to_float8_dynamic`, and the GEMMs go
through torchao's Float8TrainingTensor -> torch._scaled_mm dispatch, so the numerics ARE torchao's recipe numerics.

Composes with torch.compile (the autograd.Function is `allow_in_graph`, like torchao's; AOTAutograd traces the casts and
Inductor fuses them into the GEMM prologue) and with torch.autocast(bf16) (input is cast to the autocast dtype first, as
Float8Linear does). Not touched: LoRA A/B (bf16/fp32 as before), the loss (fp32 logits upcast in HF), attention kernels.

Usage (pretrain.py): `--fp8-base` -> convert_frozen_base_to_fp8(model) right after PEFT wrapping, before compile/DDP.
"""
import os

import torch
import torch.nn as nn
from torchao.float8 import Float8LinearConfig
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_linear_utils import swap_linear_layers
from torchao.float8.float8_scaling_utils import get_maybe_axiswise_dim, hp_tensor_to_float8_dynamic
from torchao.float8.float8_training_tensor import Float8TrainingTensor, GemmInputRole
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated

MIN_DIM = 1024                                  # both in/out features must be >= this (and multiples of 16)
EXCLUDE_FQN = ("lora_", "lm_head", "embed")     # substrings of the module FQN that are never converted
DEFAULT_RECIPE = "rowwise"
RECOMPILE_LIMIT = 256                           # torch._dynamo recompile_limit floor once fp8 layers exist (see convert)


@torch._dynamo.allow_in_graph
class _FrozenBaseFp8Matmul(torch.autograd.Function):
    """output = fp8(x) @ W^T ; grad_input = fp8(grad_output) @ W ; NO grad_weight (the weight is frozen).

    w_fwd: fp8 W  as [N, K] contiguous  -> used transposed as the column-major [K, N] operand of the forward GEMM
    w_bwd: fp8 W^T as [K, N] contiguous -> used transposed as the column-major [N, K] operand of the grad_input GEMM
    s_fwd/s_bwd: fp32 scales, [1, N] / [1, K] for rowwise (per output / per input channel) or 0-d for tensorwise.
    Mirrors torchao's matmul_with_hp_or_float8_args minus the weight casts and the grad_weight branch."""

    @staticmethod
    def forward(ctx, input_hp, w_fwd, s_fwd, w_bwd, s_bwd, orig_dtype, linear_mm_config, config):
        ctx.save_for_backward(w_bwd, s_bwd)
        ctx.orig_dtype, ctx.linear_mm_config, ctx.config = orig_dtype, linear_mm_config, config
        c = config
        x_fp8 = hp_tensor_to_float8_dynamic(
            input_hp, c.cast_config_input.target_dtype, linear_mm_config,
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=c.cast_config_input.scaling_granularity,
            axiswise_dim=get_maybe_axiswise_dim(-1, c.cast_config_input.scaling_granularity),
            round_scales_to_power_of_2=c.round_scales_to_power_of_2,
        )
        w_t_fp8 = Float8TrainingTensor(  # [K, N], column-major, scaled along dim 0 (= per output channel n)
            w_fwd.t(), s_fwd, orig_dtype, linear_mm_config, GemmInputRole.WEIGHT,
            get_maybe_axiswise_dim(0, c.cast_config_weight.scaling_granularity),
        )
        shp = x_fp8.shape
        out = torch.mm(x_fp8.reshape(-1, shp[-1]), w_t_fp8)
        return out.reshape(*shp[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        w_bwd, s_bwd = ctx.saved_tensors
        c = ctx.config
        go = grad_output.reshape(-1, grad_output.shape[-1])
        go_fp8 = hp_tensor_to_float8_dynamic(
            go, c.cast_config_grad_output.target_dtype, ctx.linear_mm_config,
            gemm_input_role=GemmInputRole.GRAD_OUTPUT,
            scaling_granularity=c.cast_config_grad_output.scaling_granularity,
            axiswise_dim=get_maybe_axiswise_dim(-1, c.cast_config_grad_output.scaling_granularity),
            round_scales_to_power_of_2=c.round_scales_to_power_of_2,
        )
        w_fp8 = Float8TrainingTensor(  # [N, K], column-major, scaled along dim 0 (= per input channel k)
            w_bwd.t(), s_bwd, ctx.orig_dtype, ctx.linear_mm_config, GemmInputRole.WEIGHT,
            get_maybe_axiswise_dim(0, c.cast_config_weight_for_grad_input.scaling_granularity),
        )
        grad_input = torch.mm(go_fp8, w_fp8)
        return grad_input.reshape(*grad_output.shape[:-1], grad_input.shape[-1]), None, None, None, None, None, None, None


@torch.no_grad()
def _precast_frozen_weight(weight, config):
    """One-time fp8 cast of a frozen [N, K] weight into both GEMM layouts (see _FrozenBaseFp8Matmul)."""
    W = weight.detach()
    cw, cwg = config.cast_config_weight, config.cast_config_weight_for_grad_input
    p2 = config.round_scales_to_power_of_2
    # forward operand is W^T [K, N] scaled along dim 0 -> scale [1, N] (or 0-d); stored as W [N, K] contiguous
    Wt = W.t()
    s_fwd = tensor_to_scale(Wt, cw.target_dtype, scaling_granularity=cw.scaling_granularity,
                            axiswise_dim=get_maybe_axiswise_dim(0, cw.scaling_granularity), round_scales_to_power_of_2=p2)
    w_fwd = to_fp8_saturated(Wt.to(torch.float32) * s_fwd, cw.target_dtype).t().contiguous()
    # grad_input operand is W [N, K] scaled along dim 0 -> scale [1, K] (or 0-d); stored as W^T [K, N] contiguous
    s_bwd = tensor_to_scale(W, cwg.target_dtype, scaling_granularity=cwg.scaling_granularity,
                            axiswise_dim=get_maybe_axiswise_dim(0, cwg.scaling_granularity), round_scales_to_power_of_2=p2)
    w_bwd = to_fp8_saturated(W.to(torch.float32) * s_bwd, cwg.target_dtype).t().contiguous()
    return w_fwd, s_fwd, w_bwd, s_bwd


class FrozenBaseFloat8Linear(Float8Linear):
    """torchao Float8Linear for a frozen weight: fp8 weight pre-cast in both GEMM layouts, backward = grad_input only."""

    def forward(self, input):
        if torch.is_autocast_enabled():  # same as Float8Linear: F.linear's autocast semantics
            input = input.to(torch.get_autocast_gpu_dtype())
        out = _FrozenBaseFp8Matmul.apply(input, self.weight, self.w_scale_fwd, self.weight_t, self.w_scale_bwd,
                                         self.orig_dtype, self.linear_mm_config, self.config)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    @classmethod
    def from_frozen(cls, mod, config):
        assert not mod.weight.requires_grad, "FrozenBaseFloat8Linear is for frozen weights only"
        with torch.device("meta"):
            new = cls(mod.in_features, mod.out_features, bias=False, config=config)
        w_fwd, s_fwd, w_bwd, s_bwd = _precast_frozen_weight(mod.weight, config)
        # `weight` stays an nn.Parameter (fp8, frozen) so parameter counts / DDP state sync / PEFT's state_dict walk see
        # the same names and numel as before; the second layout + scales are non-persistent buffers (never saved).
        new.weight = nn.Parameter(w_fwd, requires_grad=False)
        new.register_buffer("weight_t", w_bwd, persistent=False)
        new.register_buffer("w_scale_fwd", s_fwd, persistent=False)
        new.register_buffer("w_scale_bwd", s_bwd, persistent=False)
        new.bias = mod.bias
        new.orig_dtype = mod.weight.dtype
        return new


def eligible(mod, fqn):
    """(convert?, reason) for one module. Follows torchao's auto-filter rules (multiples of 16, size thresholds) plus
    the frozen / name exclusions specific to a LoRA setup."""
    if not isinstance(mod, nn.Linear) or isinstance(mod, Float8Linear):
        return False, "not_linear"
    if any(s in fqn for s in EXCLUDE_FQN):
        return False, "excluded_name"
    if mod.weight.requires_grad:
        return False, "trainable"
    N, K = mod.weight.shape
    if N % 16 or K % 16:
        return False, "not_mult_of_16"
    if min(N, K) < MIN_DIM:
        return False, f"small_dim(<{MIN_DIM})"
    return True, "ok"


def convert_frozen_base_to_fp8(model, recipe=None, emulate=False, verbose=True):
    """In-place: swap every eligible frozen nn.Linear under `model` for FrozenBaseFloat8Linear. Call AFTER PEFT wrapping
    and BEFORE torch.compile / DDP. Returns a summary dict (also printed when verbose)."""
    recipe = recipe or os.environ.get("MAEMM_FP8_RECIPE", DEFAULT_RECIPE)
    config = Float8LinearConfig.from_recipe_name(recipe)
    if emulate:  # CPU / unit-test path: fp32 emulation of the scaled GEMMs, same casts
        import dataclasses
        config = dataclasses.replace(config, emulate=True)
    summary = {"recipe": recipe, "converted": 0, "params_fp8": 0, "skipped": {}, "shapes": {}}

    def filt(mod, fqn):
        ok, why = eligible(mod, fqn)
        if ok:
            summary["converted"] += 1
            summary["params_fp8"] += mod.weight.numel()
            key = f"{mod.in_features}->{mod.out_features}"
            summary["shapes"][key] = summary["shapes"].get(key, 0) + 1
        elif isinstance(mod, nn.Linear):
            summary["skipped"][why] = summary["skipped"].get(why, 0) + 1
        return ok

    swap_linear_layers(model, lambda m: FrozenBaseFloat8Linear.from_frozen(m, config), module_filter_fn=filt)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # the bf16 masters of converted layers are gone; return their blocks to the allocator
    # fp8 must never run in eager: the unfused cast (to(fp32) * scale -> clamp -> to(fp8)) + post-scale is ~5 ATen passes
    # over every activation and grad_output, ~2x SLOWER than a bf16 layer (B200, 5120->17408, M=3072 fwd+bwd: eager fp8
    # 1.19 ms vs compiled fp8 0.51 ms vs bf16 0.70 ms). Dynamo's default recompile_limit (8) is exceeded by transformers'
    # per-layer cache guards (`cache_params.layers[i].device is None` -- Qwen3.5 builds a DynamicCache in every training
    # forward unless use_cache=False is passed) and by the decoder-layer mask-rank guard, after which those frames silently
    # fall back to eager. With the cache present the GatedDeltaNet forward needs one entry per (layer, padded length):
    # 48 x 3 = 144 for pretrain.py's 64/128/192 buckets (measured: limit 64 still tripped in the smoke run), hence 256.
    # Passing use_cache=False in the training forward removes the per-layer guards entirely (3 entries); this is the belt.
    for name in ("recompile_limit", "cache_size_limit",  # 2.10 name / pre-2.10 alias (both still honoured)
                 "accumulated_recompile_limit", "accumulated_cache_size_limit"):  # total across all `self` of one code object
        if getattr(torch._dynamo.config, name, RECOMPILE_LIMIT) < RECOMPILE_LIMIT:
            setattr(torch._dynamo.config, name, RECOMPILE_LIMIT)
    if verbose:
        print(f"[fp8] recipe={recipe} converted {summary['converted']} frozen linears "
              f"({summary['params_fp8'] / 1e9:.2f}B params) to fp8; skipped {summary['skipped']}; "
              f"shapes {summary['shapes']}", flush=True)
    return summary
