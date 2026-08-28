"""Smoke test: vLLM 0.19 + vllm_lens per-request steering for Qwen/Qwen3.6-27B, single GPU.

Verifies, in order:
  1. the model loads in vLLM (vendored Qwen3_5Config; hybrid GatedDeltaNet + full attention),
  2. plain greedy generation is coherent (vanilla chat prompt),
  3. vllm_lens SteeringVector injection at INJECT_LAYER on the marker token actually FIRES:
     (a) the worker-side marker-write counter increments (direct evidence, immune to the
         "identical outputs" ambiguity -- greedy divergence alone is NOT a valid check),
     (b) NUMERIC: capture the layer-INJECT_LAYER residual stream for a clean and a steered
         request; the delta at the marker must equal scale * ||h_clean|| * unit(v)
         (cos ~ 1, norm ratio ~ scale) -- proves the norm-match convention matches the
         HF trainer's `h <- h + ||h||*v`,
     (c) outputs differ across directions at a strong scale (behavioral evidence).

Matches the RL trainer's intended vLLM rollout path (train/rl.py greedy_eval):
  SteeringVector(activations=v.view(1,1,-1).cpu().float(), layer_indices=[INJECT_LAYER],
                 scale=STEER_COEFF, norm_match=True, position_indices=[marker])

Env: vllm 0.19.0 + vllm_lens 1.1.0 (with its dist-info intact -- a wheel installed without
dist-info silently skips the vllm plugin entry point, so steering no-ops) + transformers 5.15.0.
CUDA_VISIBLE_DEVICES defaults to 2 below (dev-box convention: GPUs 0/1 were training) --
override via env.

Run (from the repo root):
  PYTHONPATH=$PWD python scripts/vllm_smoke.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
# In-process engine core so the vllm_lens steer counter is readable from this process
# (collective_rpc with a callable also works either way; this just keeps debugging simple).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector

from mxf.config import D_MODEL, INJECT_LAYER, MODEL, STEER_COEFF
from mxf.prompts import build_prompt_ids

MAX_TOKENS = 48
MIN_TOKENS = 16  # trainer uses min_new_tokens=16; without it the bare prompt emits EOS immediately


def read_steer_count(llm):
    """Marker position-writes in the worker since the last call (then reset)."""
    def _read(worker):
        from vllm_lens._worker_ext import get_and_reset_steer_count
        return get_and_reset_steer_count()
    try:
        return sum(llm.collective_rpc(_read))
    except Exception:
        from vllm_lens._worker_ext import get_and_reset_steer_count
        return get_and_reset_steer_count()


def _get_layer_act(out):
    """[seq, d] float tensor of the single captured layer.

    out.activations == {"residual_stream": tensor[n_captured_layers, seq, d]}; we request
    exactly one layer (output_residual_stream=[INJECT_LAYER]) so index 0 is it.
    """
    act = getattr(out, "activations", None)
    assert act is not None, "no activations captured -- vllm_lens plugin not active?"
    rs = act["residual_stream"]
    assert rs.shape[0] == 1, f"expected 1 captured layer, got {tuple(rs.shape)}"
    return rs[0].float()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids, mpos = build_prompt_ids(tok)
    marker = mpos[0]
    print(f"[smoke] prompt len={len(prompt_ids)} marker@{marker} "
          f"(token {prompt_ids[marker]!r} = {tok.decode([prompt_ids[marker]])!r})")

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.60,
        max_model_len=1024,
        # FLASHINFER lacks the metadata vllm_lens needs -> injection silently no-ops.
        # TRITON_ATTN is the verified backend (same pin as RL/rl_hf.py --attn-backend).
        attention_backend="TRITON_ATTN",
        enforce_eager=True,  # vllm_lens plugin forces this anyway; explicit for clarity
        # Text-only rollouts: skip the vision tower. Also REQUIRED on this box -- the vision
        # encoder's profile run imports vllm_flash_attn's CuTe path -> quack-kernels 0.5.0,
        # which crashes against nvidia-cutlass-dsl 4.7.0 (cutlass.cute.core.ThrMma missing).
        language_model_only=True,
    )

    # ---- 1. vanilla coherence check (no marker, no steering) ----------------
    chat = tok.apply_chat_template([{"role": "user", "content": "In one sentence, what is the ocean?"}],
                                   tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(chat, "keys"):  # transformers 5.x returns a BatchEncoding
        chat = chat["input_ids"]
    while isinstance(chat[0], list):
        chat = chat[0]
    v_out = llm.generate([{"prompt_token_ids": list(chat)}],
                         [SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)])
    vanilla = v_out[0].outputs[0].text
    print(f"\n[vanilla | plain chat prompt]\n{vanilla!r}")
    assert len(vanilla.strip()) > 10, "vanilla generation empty/garbage -- model not loading right"

    # ---- 2. steered generations at the marker -------------------------------
    def gen(vec=None, scale=STEER_COEFF, capture=False):
        extra = {}
        if vec is not None:
            extra["apply_steering_vectors"] = [SteeringVector(
                activations=vec.view(1, 1, -1).cpu().float(),
                layer_indices=[INJECT_LAYER],
                scale=scale,
                norm_match=True,
                position_indices=[marker],
            )]
        if capture:
            extra["output_residual_stream"] = [INJECT_LAYER]
        kw = dict(temperature=0.0, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS)
        if extra:
            kw["extra_args"] = extra
        return llm.generate([{"prompt_token_ids": list(prompt_ids)}],
                            [SamplingParams(**kw)])[0]

    g = torch.Generator().manual_seed(0)
    vA = torch.randn(D_MODEL, generator=g)
    vA = vA / vA.norm()
    vB = torch.randn(D_MODEL, generator=g)
    vB = vB / vB.norm()

    read_steer_count(llm)  # zero the counter

    out_clean = gen(None, capture=True)
    base = out_clean.outputs[0].text
    c_base = read_steer_count(llm)
    print(f"\n[base | no steering | writes={c_base}]\n{base!r}")

    outA_o = gen(vA, capture=True)
    outA = outA_o.outputs[0].text
    cA = read_steer_count(llm)
    print(f"\n[dirA | scale={STEER_COEFF} norm_match | writes={cA}]\n{outA!r}")

    outB = gen(vB).outputs[0].text
    cB = read_steer_count(llm)
    print(f"\n[dirB | scale={STEER_COEFF} norm_match | writes={cB}]\n{outB!r}")

    outA8 = gen(vA, scale=8.0).outputs[0].text
    cA8 = read_steer_count(llm)
    print(f"\n[dirA | scale=8.0 norm_match | writes={cA8}]\n{outA8!r}")

    # ---- 3. numeric injection check at INJECT_LAYER --------------------------
    # capture is post-injection; prefill is deterministic, so
    # steered - clean at the marker == STEER_COEFF * ||h_clean_full|| * unit(vA)  (bf16 noise aside)
    h_clean = _get_layer_act(out_clean)[marker]
    h_steer = _get_layer_act(outA_o)[marker]
    delta = h_steer - h_clean
    cos = torch.nn.functional.cosine_similarity(delta, vA, dim=0).item()
    ratio = (delta.norm() / (STEER_COEFF * h_clean.norm())).item()
    print(f"\n[numeric] ||h_clean||={h_clean.norm():.3f} ||delta||={delta.norm():.3f} "
          f"cos(delta, vA)={cos:.5f} norm-ratio (want ~1.0)={ratio:.4f}")

    # ---- verdict -------------------------------------------------------------
    assert c_base == 0, f"counter should be 0 with no steering, got {c_base}"
    assert cA >= 1 and cB >= 1 and cA8 >= 1, (
        f"steer counter zero (A={cA} B={cB} A8={cA8}) -- injection NOT firing "
        "(wrong attention backend? plugin not registered? vllm_lens drift?)")
    assert cos > 0.99, f"delta not aligned with injected dir (cos={cos:.4f}) -- wrong write"
    assert 0.9 < ratio < 1.1, f"injection magnitude off (ratio={ratio:.4f}) -- norm-match drift vs HF"
    assert not (base == outA == outB == outA8), (
        "all outputs identical across distinct directions -- steering silently no-ops")
    print(f"\n[smoke] PASS: counter fires (A={cA} B={cB} A8={cA8}), numeric inject exact "
          f"(cos={cos:.4f}, ratio={ratio:.4f}), outputs differ: A!=base={outA != base} "
          f"B!=base={outB != base} A!=B={outA != outB} A8!=base={outA8 != base}")


if __name__ == "__main__":
    main()
