#!/usr/bin/env python3
"""Weekly digest infrastructure figures (Sep 1-8 2026) for the Qwen3.6-27B activation->text inverter.

Produces three figures (PNG + PDF, same stem) plus replot-ready JSON:

  1. tp8_vs_fsdp2          tensor-parallel 8-way vs sharded data parallel (FSDP2) full fine-tune
  2. corpus_growth         real-activation pretraining corpus growth + midtrain bank composition
  3. sft_throughput_ladder full fine-tune throughput ladder on 8xB200 for the 104M-example run

Sources (all numbers are copied verbatim from these; nothing is estimated here except the
104M-example ETA = 104e6 / ex_s / 3600 and the wall-time split in fig 1b = GPU-busy share x 0.77):
  * /home/celeste/maemm-pub-tp/docs/fullft_tp.md  section 9 (measured table, profile, exactness table)
  * /home/celeste/maemm-pub-tp/sft/results/tp_bench_eff4096.json, tp_bench_eff512.json,
    tp_bench_mb256_fp32reduce.json, tp_exactness_v2.json / v3 / v4_fsdp_split_control.json
  * Modal volume maemm-data: banks/<name>/build_stats.json (n_examples, families, created) and
    everything_eq240k/build_stats.json, fetched 2026-09-08 with
      modal volume get maemm-data banks/<name>/build_stats.json /tmp/bs/bs_<name>.json --force
  * project memory: maemm-overnight-last5-run.md (Sep 3 - Sep 8 entries) and maemm-sft-length-bucket.md

Usage:  python3 scripts/plot_week_infra.py [--out DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --------------------------------------------------------------------------------------------
# style (dataviz reference palette; categorical hues assigned in fixed slot order, never cycled)
# --------------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
BLUE_DARK = "#104281"   # sequential step 650 (used to emphasise one bar in a one-hue ranking)
BLUE_MID = "#5598e7"    # sequential step 350
DPI = 150
WIDTH_IN = 16.0         # 16 in x 150 dpi = 2400 px wide

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False,
})


def style_ax(ax, ygrid=True, xgrid=False):
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(length=3, color=AXIS)


def save(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def dump(obj, out_dir, stem):
    with open(os.path.join(out_dir, "data", f"{stem}.json"), "w") as f:
        json.dump(obj, f, indent=2)


def eta_h(ex_s: float, n: float = 104e6) -> float:
    return n / ex_s / 3600.0


# ============================================================================================
# FIGURE 1 - tensor parallel 8-way vs FSDP2
# ============================================================================================
# fullft_tp.md section 9, measured table (8xB200, Qwen3.6-27B, eff batch 4,096, median of steps >= 2)
TP_THROUGHPUT = [
    {"config": "FSDP2\n(sharded data\nparallel)\nmicro-batch\n64 x 8",
     "key": "fsdp2_mb64x8", "ex_s": 629, "ex_s_per_rank": 78.6, "s_step": 6.46, "peak_gb": 138,
     "tflops_per_gpu": "267", "kind": "fsdp2"},
    {"config": "tensor\nparallel 8-way\nmicro-batch\n512 x 8",
     "key": "tp8_mb512x8", "ex_s": 696, "ex_s_warm_range": [711, 814], "s_step": 5.77,
     "s_step_warm_range": [5.0, 5.9], "peak_gb": 166.7, "tflops_per_gpu": "281-362", "kind": "tp8"},
    {"config": "tensor\nparallel 8-way\nmicro-batch\n256 x 16",
     "key": "tp8_mb256x16", "ex_s": 607, "s_step": 6.49, "peak_gb": 112,
     "tflops_per_gpu": "251-291", "kind": "tp8"},
]
# fullft_tp.md section 9, torch.profiler of one TP-8 mb-512 micro-batch step (--profile-step 3):
# shares of GPU time; GPU busy ~4.5 s of a ~5.8 s step = 77 % of wall, remainder CPU launch overhead.
GPU_BUSY_FRAC = 0.77
TP_PROFILE_GPU_SHARE = [  # (label, % of GPU time, detail)
    ("GEMMs (matrix multiplies)", 29, "aten::mm 1.31 s for 1.83 PFLOP per GPU = ~1.4 PFLOP/s while running (~60 % of B200 peak)"),
    ("NCCL all-reduce\n(not overlapped)", 17, "2,586 calls x 293 us, ~116 MB, ~400 GB/s algbw"),
    ("gated-delta-net chunk\nkernels (Triton)", 15, "fla chunk_gated_delta_rule fwd + bwd"),
    ("depthwise conv1d\n(PyTorch fallback)", 10, "causal_conv1d CUDA package not installed; 432 x (0.3 ms fwd + 0.5 ms bwd)"),
    ("elementwise glue\n(copy / add / mul / sum)", 22, "~70k kernel launches"),
    ("attention backward\n(SDPA mem-efficient)", 3, ""),
    ("fused RMSNorm", 3, ""),
]
GPU_IDLE_LABEL = "GPU idle: CPU kernel-\nlaunch bound"
# fullft_tp.md section 9, exactness table: gradient cosine vs ONE FSDP2 reference run (global / median / worst)
TP_EXACTNESS = [
    {"leg": "FSDP2 re-run\n(noise floor)", "key": "fsdp2_rerun", "global_cos": 0.99972, "median_cos": 0.99972,
     "worst_cos": 0.992, "n_params_cos_below_0999": 23, "step0_loss_delta": -9e-4, "kind": "fsdp2"},
    {"leg": "FSDP2, micro-batch\nsplit in 2 (control)", "key": "fsdp2_split2", "global_cos": 0.99997, "median_cos": 0.99998,
     "worst_cos": 0.998, "n_params_cos_below_0999": 1, "step0_loss_delta": 0.0, "kind": "fsdp2"},
    {"leg": "TP-8, fused norm (default)", "key": "tp8_fused_norm", "global_cos": 0.9926, "median_cos": 0.9939,
     "worst_cos": 0.86, "n_params_cos_below_0999": 819, "step0_loss_delta": 5.1e-3, "kind": "tp8"},
    {"leg": "TP-8, eager norm", "key": "tp8_eager_norm", "global_cos": 0.9907, "median_cos": 0.9938,
     "worst_cos": 0.69, "n_params_cos_below_0999": 816, "step0_loss_delta": 3.5e-4, "kind": "tp8"},
    {"leg": "TP-8, fused norm,\nfp32 all-reduce", "key": "tp8_fused_fp32reduce", "global_cos": 0.9943, "median_cos": 0.9952,
     "worst_cos": 0.85, "n_params_cos_below_0999": 791, "step0_loss_delta": 4.1e-3, "kind": "tp8"},
    {"leg": "TP-8, eager norm,\nfp32 all-reduce", "key": "tp8_eager_fp32reduce", "global_cos": 0.9950, "median_cos": 0.9956,
     "worst_cos": 0.78, "n_params_cos_below_0999": 796, "step0_loss_delta": 7.9e-4, "kind": "tp8"},
]
COS_CRITERION = 0.999
KIND_COLOR = {"fsdp2": BLUE, "tp8": ORANGE, "lora": AQUA}
KIND_LABEL = {"fsdp2": "sharded data parallel (FSDP2)", "tp8": "tensor parallel 8-way", "lora": "LoRA r64 adapter"}


def inner_text_color(kind):
    """Ink colour for text drawn on top of a bar of the given kind (white only on the dark blue)."""
    return "white" if kind == "fsdp2" else INK


def fig_tp8_vs_fsdp2(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 6.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.2]})
    fig.get_layout_engine().set(wspace=0.06, w_pad=0.15)
    fig.suptitle(
        "Tensor parallelism gives at most 1.1x over sharded data parallel because the step is not GEMM-bound\n"
        "(matmuls are 29 % of GPU time, the GPU idles 23 % of wall) - 8xB200, Qwen3.6-27B full fine-tune, effective batch 4,096",
        fontsize=14, fontweight="bold", x=0.01, ha="left")

    # (a) throughput -------------------------------------------------------------------------
    ax = axes[0]
    style_ax(ax)
    xs = list(range(len(TP_THROUGHPUT)))
    base = TP_THROUGHPUT[0]["ex_s"]
    for i, r in enumerate(TP_THROUGHPUT):
        ax.bar(i, r["ex_s"], width=0.6, color=KIND_COLOR[r["kind"]], edgecolor=SURFACE, linewidth=1.5)
        if "ex_s_warm_range" in r:
            lo, hi = r["ex_s_warm_range"]
            ax.plot([i, i], [lo, hi], color=INK2, linewidth=1.6)
            ax.plot([i - 0.08, i + 0.08], [hi, hi], color=INK2, linewidth=1.6)
            ax.text(i + 0.12, hi, f"warm steps\n{lo}-{hi}", fontsize=8.5, color=INK2, va="center")
        ratio = r["ex_s"] / base
        ax.text(i, r["ex_s"] + 12, f"{r['ex_s']} ex/s" + ("" if i == 0 else f"  ({ratio:.2f}x)"),
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(i, r["ex_s"] * 0.5, f"peak\n{r['peak_gb']} GB\nper GPU", ha="center", va="center", fontsize=9.5,
                color=inner_text_color(r["kind"]))
    ax.axhline(base, color=INK2, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(len(xs) - 0.55, base - 8, "FSDP2 baseline", ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["config"] for r in TP_THROUGHPUT], fontsize=8.4)
    ax.set_ylabel("training examples / s (whole 8-GPU node)")
    ax.set_ylim(0, 900)
    ax.set_title("(a) Throughput, effective batch 4,096:\nTP-8 = 1.11x (micro-batch 512) / 0.97x (256)", fontsize=10.5, loc="left")

    # (b) GPU-time breakdown ----------------------------------------------------------------
    ax = axes[1]
    style_ax(ax, ygrid=False, xgrid=True)
    rows = [(n, s, s * GPU_BUSY_FRAC) for n, s, _ in TP_PROFILE_GPU_SHARE]
    rows.append((GPU_IDLE_LABEL, None, (1 - GPU_BUSY_FRAC) * 100))
    rows.sort(key=lambda r: r[2])  # ascending so the largest is on top
    labels = [r[0] for r in rows]
    wall = [r[2] for r in rows]
    colors, hatches = [], []
    for n, s, w in rows:
        if s is None:
            colors.append(MUTED); hatches.append("////")
        elif n.startswith("GEMM"):
            colors.append(BLUE_DARK); hatches.append("")
        else:
            colors.append(BLUE_MID); hatches.append("")
    bars = ax.barh(range(len(rows)), wall, height=0.62, color=colors, edgecolor=SURFACE, linewidth=1.5)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h); b.set_edgecolor(SURFACE)
    for i, (n, s, w) in enumerate(rows):
        txt = f"{w:.0f} % of wall" if s is None else f"{w:.0f} % of wall  ({s} % of GPU time)"
        ax.text(w + 0.4, i, txt, va="center", ha="left", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9.2)
    ax.set_xlim(0, 37)
    ax.set_xlabel("share of one micro-batch step's wall-clock time (%)")
    ax.set_title("(b) Where a TP-8 micro-batch step's time goes\n(profiler, micro-batch 512): GEMMs 22 % of wall,\nlaunch-bound idle 23 %, other kernels 55 %", fontsize=10.5, loc="left")

    # (c) exactness (horizontal dot plot, zoomed x-axis; dots not bars because the axis does not start at 0)
    ax = axes[2]
    style_ax(ax, ygrid=False, xgrid=True)
    n = len(TP_EXACTNESS)
    ys = [n - 1 - i for i in range(n)]  # first leg on top
    for y, r in zip(ys, TP_EXACTNESS):
        c = KIND_COLOR[r["kind"]]
        ax.plot([0.99, r["global_cos"]], [y, y], color=c, linewidth=2, alpha=0.3)
        ax.plot(r["global_cos"], y, marker="o", markersize=10, color=c, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(r["global_cos"], y + 0.22, f"{r['global_cos']:.5f}" if r["global_cos"] > 0.999 else f"{r['global_cos']:.4f}",
                ha="center", va="bottom", fontsize=9.2, color=INK)
    ax.axvline(COS_CRITERION, color=INK2, linestyle=(0, (4, 3)), linewidth=1.1)
    ax.text(COS_CRITERION - 0.00015, -0.55, f"acceptance criterion\ncos >= {COS_CRITERION}", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(0.99, 1.0003)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["leg"] for r in TP_EXACTNESS], fontsize=9.2)
    ax.set_xlabel("gradient cosine vs FSDP2 reference (step 0)")
    ax.set_xticks([0.990, 0.992, 0.994, 0.996, 0.998, 1.000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    ax.set_title("(c) Gradient cosine vs the FSDP2 reference:\nevery TP-8 leg is ~20x further than a re-run", fontsize=10.5, loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=9, label=KIND_LABEL["fsdp2"]),
                       Line2D([], [], marker="o", color=ORANGE, linestyle="", markersize=9, label=KIND_LABEL["tp8"])],
              loc="center right", fontsize=9)

    fig.text(0.01, -0.04,
             "Bench: 8xB200 (Modal), torch 2.10 / flash-linear-attention 0.5.2 / patched transformers; tiny bank replicated to 132k rows, targets 8-57 tokens; "
             "median of logged steps >= 2 after JIT warm-up. Exactness: same data window, micro-batch 32 per data-parallel rank = 256 under TP, compared before the optimizer step.",
             fontsize=8.5, color=INK2, ha="left", va="top", wrap=True)
    save(fig, out_dir, "tp8_vs_fsdp2")

    dump({
        "figure": "tp8_vs_fsdp2",
        "experiment": "Qwen3.6-27B full fine-tune on 8xB200 (Modal), effective batch 4,096, prefix cache on, pad-multiple 1",
        "source_doc": "/home/celeste/maemm-pub-tp/docs/fullft_tp.md section 9",
        "source_json": ["/home/celeste/maemm-pub-tp/sft/results/tp_bench_eff4096.json",
                        "/home/celeste/maemm-pub-tp/sft/results/tp_bench_eff512.json",
                        "/home/celeste/maemm-pub-tp/sft/results/tp_bench_mb256_fp32reduce.json",
                        "/home/celeste/maemm-pub-tp/sft/results/tp_exactness_v2.json",
                        "/home/celeste/maemm-pub-tp/sft/results/tp_exactness_v3.json",
                        "/home/celeste/maemm-pub-tp/sft/results/tp_exactness_v4_fsdp_split_control.json"],
        "panel_a_throughput": [{k: v for k, v in r.items() if k != "config"} | {"label": r["config"].replace("\n", " "),
                                "ratio_vs_fsdp2": round(r["ex_s"] / TP_THROUGHPUT[0]["ex_s"], 3),
                                "eta_104M_examples_h": round(eta_h(r["ex_s"]), 1)} for r in TP_THROUGHPUT],
        "panel_b_profile": {
            "gpu_busy_fraction_of_wall": GPU_BUSY_FRAC,
            "gpu_idle_fraction_of_wall": round(1 - GPU_BUSY_FRAC, 2),
            "note": "GPU busy ~4.5 s of a ~5.8 s unprofiled step; the rest is CPU launch overhead (Triton launchers 1-6 ms CPU per call; 432 GDN backward calls + 1,159 norm calls per step). wall share = GPU-time share x 0.77.",
            "categories": [{"name": n, "pct_of_gpu_time": s, "pct_of_wall": round(s * GPU_BUSY_FRAC, 1), "detail": d}
                           for n, s, d in TP_PROFILE_GPU_SHARE],
        },
        "panel_c_exactness": {"criterion_global_cos": COS_CRITERION,
                              "reference": "one FSDP2 run: step-0 loss 5.6048, grad norm 76.7",
                              "legs": [{k: v for k, v in r.items() if k != "leg"} | {"label": r["leg"].replace("\n", " ")} for r in TP_EXACTNESS]},
    }, out_dir, "tp8_vs_fsdp2")


# ============================================================================================
# FIGURE 2 - corpus growth + midtrain bank composition
# ============================================================================================
# build_stats.json per bank on Modal volume maemm-data (fetched 2026-09-08). 'created' = epoch seconds (UTC).
REALACT_PARTS = [  # name, n_examples, created (epoch s)
    ("realact_short_20m_b", 4_500_000, 1788472352.645931),
    ("realact_short_20m_c", 4_500_000, 1788472359.7185261),
    ("realact_short_20m_d", 3_000_000, 1788472579.100832),
    ("realact_short_20m",  11_000_000, 1788472589.3193626),
    ("realact_short_20m_g", 9_000_000, 1788760078.3011696),
    ("realact_short_20m_e", 9_000_000, 1788760116.871445),
    ("realact_short_20m_f", 9_000_000, 1788760133.1710954),
    ("realact_short_20m_h", 9_000_000, 1788764725.718798),
    ("realact_short_20m_i", 9_000_000, 1788764797.993346),
    ("realact_short_20m_j", 9_000_000, 1788764893.666661),
    ("realact_short_20m_k", 9_000_000, 1788764903.891253),
    ("realact_short_20m_l", 9_000_000, 1788768368.679430),
    ("realact_short_20m_m", 9_000_000, 1788768430.700399),
    ("realact_short_20m_o", 9_000_000, 1788768558.591266),
    ("realact_short_20m_n", 9_000_000, 1788769283.294238),
    ("realact_short_20m_p", 9_000_000, 1788773294.394353),
    ("realact_short_20m_r", 9_000_000, 1788773487.890164),
    ("realact_short_20m_s", 9_000_000, 1788773543.268630),
    ("realact_short_20m_q", 9_000_000, 1788774006.856775),
    ("realact_short_20m_t", 9_000_000, 1788776664.241769),
    ("realact_short_20m_u", 9_000_000, 1788776901.708848),
    ("realact_short_20m_v", 9_000_000, 1788777270.019974),
    ("realact_short_20m_w", 9_000_000, 1788777523.811680),
    ("realact_short_20m_x", 9_000_000, 1788780156.113402),
]
MERGED_BANKS = [  # merged (concatenated) banks - NOT added to the cumulative count
    ("realact_short_20m_all", 23_000_000, 1788473553.6635113, ["realact_short_20m", "realact_short_20m_b", "realact_short_20m_c", "realact_short_20m_d"]),
    ("realact_short_50m_all", 50_000_000, 1788765041.1476195, ["realact_short_20m", "realact_short_20m_b", "realact_short_20m_c", "realact_short_20m_d",
                                                              "realact_short_20m_e", "realact_short_20m_f", "realact_short_20m_g"]),
]
TRAINING_CORPORA = [  # (examples, label)
    (23_000_000, "23M: LoRA pretraining corpus\n(merged bank, Sep 3)"),
    (104_000_000, "104M: full fine-tune corpus\n(50M merged bank + 6 parts of 9M)"),
    (203_000_000, "203M: total available\n(24 parts)"),
]
FAMILY_ORDER = ["realact", "realact_long", "sae", "bsf", "cluster", "mlp", "mlp_pair"]
FAMILY_LABEL = {
    "realact": "real activations (short context)",
    "realact_long": "real activations (long context)",
    "sae": "SAE feature directions",
    "bsf": "learned subspace-featurizer directions",
    "cluster": "cluster-probe directions",
    "mlp": "MLP-neuron directions",
    "mlp_pair": "MLP-neuron pair directions",
}
FAMILY_COLOR = dict(zip(FAMILY_ORDER, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET]))
MIX_BANKS = [  # bottom-to-top order in the plot
    {"name": "mix_1m_mlp", "label": "7-family mix (Sep 5)\n1,209,088 rows", "n_examples": 1_209_088, "created": 1788571006.3011339,
     "families": {"realact": 250000, "realact_long": 250000, "sae": 235851, "cluster": 241741, "mlp_pair": 12337, "bsf": 117682, "mlp": 101477}},
    {"name": "everything_eq240k", "label": "5-family equal mix (Sep 7)\n1,320,647 rows", "n_examples": 1_320_647, "created": 1788760520.725914,
     "families": {"realact": 241741, "realact_long": 241741, "sae": 353683, "bsf": 241741, "cluster": 241741}},
    {"name": "mix_eq_1p45m", "label": "6-family equal mix (Sep 7)\n1,450,446 rows = 6 x 241,741", "n_examples": 1_450_446, "created": 1788762820.336234,
     "families": {"realact": 241741, "cluster": 241741, "mlp": 241741, "realact_long": 241741, "sae": 241741, "bsf": 241741}},
]
MLP42_BIG = {"name": "mlp42_big", "n_examples": 416_574, "created": 1788762180.130396, "families": {"mlp": 405998, "mlp_pair": 10576},
             "role": "source bank for the mlp family of mix_eq_1p45m (MLP-neuron directions, train neurons only)"}


def _utc(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc)


def fig_corpus_growth(out_dir):
    parts = sorted(REALACT_PARTS, key=lambda r: r[2])
    times = [_utc(r[2]) for r in parts]
    cum = []
    tot = 0
    for r in parts:
        tot += r[1]
        cum.append(tot)
    t0 = _utc(parts[0][2]) - dt.timedelta(hours=6)
    t_end = _utc(parts[-1][2]) + dt.timedelta(hours=14)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 6.6), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.get_layout_engine().set(wspace=0.08, w_pad=0.15)
    d_first = times[0].strftime("%b %-d"); d_last = times[-1].strftime("%b %-d")
    first_wave_total = cum[3]                       # the four Sep-3 parts
    i_second = 4                                    # first part of the Sep-7 wave
    second_wave_hours = (times[-1] - times[i_second]).total_seconds() / 3600
    second_wave_added = cum[-1] - cum[i_second - 1]
    growth = cum[-1] / first_wave_total
    fig.suptitle(
        f"Real-activation pretraining corpus grew {growth:.1f}x in four days ({first_wave_total/1e6:.0f}M on {d_first} -> {cum[-1]/1e6:.0f}M on {d_last}) "
        "and the midtrain bank is now family-balanced\n"
        "(Qwen3.6-27B layer-42 residual activations from FineFineWeb; banks on the Modal volume, counts from each bank's build_stats.json)",
        fontsize=13.5, fontweight="bold", x=0.01, ha="left")

    # (a) cumulative step plot ----------------------------------------------------------------
    ax = axes[0]
    style_ax(ax)
    xs = [t0] + times + [t_end]
    ys = [0] + cum + [cum[-1]]
    ax.step(xs, [y / 1e6 for y in ys], where="post", color=BLUE, linewidth=2.2)
    ax.plot(times, [c / 1e6 for c in cum], linestyle="", marker="o", markersize=5, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1)
    for n, lab in TRAINING_CORPORA:
        ax.axhline(n / 1e6, color=INK2, linestyle=(0, (4, 3)), linewidth=0.9, alpha=0.8)
        ax.text(t0 + dt.timedelta(hours=1.5), n / 1e6 + 3, lab, fontsize=9, color=INK2, va="bottom", ha="left")
    # merged-bank markers
    for name, n, created, _ in MERGED_BANKS:
        t = _utc(created)
        ax.plot(t, n / 1e6, marker="D", markersize=8, color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="")
    ax.text(_utc(MERGED_BANKS[1][2]) + dt.timedelta(hours=1.5), 50 - 4, "50M merged bank\n(7 parts, Sep 7 07:10 UTC)", fontsize=8.8, color=ORANGE, va="top")
    ax.text(times[-1] + dt.timedelta(hours=1.0), cum[-1] / 1e6 - 2, f"{cum[-1]/1e6:.0f}M\n{len(parts)} parts", fontsize=9.5, color=BLUE, va="top", ha="left", fontweight="bold")
    ax.text(times[3] + dt.timedelta(hours=2), 23 - 2, f"23M\n4 parts, {d_first}", fontsize=9.5, color=BLUE, va="top", ha="left", fontweight="bold")
    ax.set_ylabel("cumulative real-activation training examples collected (millions)")
    ax.set_ylim(0, 225)
    ax.set_xlim(t0, t_end)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.set_xlabel("bank completion time (UTC, 2026)")
    ax.legend(handles=[Line2D([], [], color=BLUE, linewidth=2.2, marker="o", markersize=5, label=f"cumulative examples ({len(parts)} collection parts)"),
                       Line2D([], [], color=ORANGE, marker="D", linestyle="", markersize=8, label="merged training bank written (23M, 50M)"),
                       Line2D([], [], color=INK2, linestyle=(0, (4, 3)), label="corpus used / available for a training run")],
              loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=8.8)
    ax.set_title(f"(a) Collection: {first_wave_total/1e6:.0f}M on {d_first}, then {second_wave_added/1e6:.0f}M more in {second_wave_hours:.1f} hours on {d_last} "
                 f"({len(parts) - i_second} parts x 9M)", fontsize=11, loc="left")

    # (b) composition stacked horizontal bars ------------------------------------------------
    ax = axes[1]
    style_ax(ax, ygrid=False, xgrid=True)
    for yi, b in enumerate(MIX_BANKS):
        left = 0
        for fam in FAMILY_ORDER:
            n = b["families"].get(fam, 0)
            if n == 0:
                continue
            ax.barh(yi, n, left=left, height=0.58, color=FAMILY_COLOR[fam], edgecolor=SURFACE, linewidth=1.5)
            if n >= 60_000:
                ax.text(left + n / 2, yi, f"{n/1e3:.0f}k", ha="center", va="center", fontsize=8.8,
                        color="white" if fam in ("realact", "mlp", "mlp_pair") else INK)
            left += n
        ax.text(left + 15_000, yi, f"{b['n_examples']:,}", va="center", ha="left", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yticks(range(len(MIX_BANKS)))
    ax.set_yticklabels([b["label"] for b in MIX_BANKS], fontsize=9.5)
    ax.set_xlim(0, 1.75e6)
    ax.set_xticks([0, 0.5e6, 1.0e6, 1.5e6])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v else "0"))
    ax.set_xlabel("rows in the bank (activation vector + target text)")
    ax.legend(handles=[Patch(color=FAMILY_COLOR[f], label=FAMILY_LABEL[f]) for f in FAMILY_ORDER],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.8, title="direction family", title_fontsize=9)
    ax.set_title("(b) Midtrain bank composition: from a lopsided 7-family mix\nto exactly 241,741 rows per family x 6", fontsize=11, loc="left")
    fig.text(0.01, -0.03, f"MLP-neuron directions for the 6-family mix come from a {MLP42_BIG['n_examples']:,}-row source bank "
                          f"({MLP42_BIG['families']['mlp']:,} single neurons + {MLP42_BIG['families']['mlp_pair']:,} pairs); singles were used first. "
                          "Collection parts: four on Sep 3 (11M / 4.5M / 4.5M / 3M), twenty of 9M on Sep 7. Dates are bank completion times from build_stats.json.",
             fontsize=8.5, color=INK2, ha="left", va="top", wrap=True)
    save(fig, out_dir, "corpus_growth")

    dump({
        "figure": "corpus_growth",
        "source": "Modal volume maemm-data: banks/<name>/build_stats.json and everything_eq240k/build_stats.json (fields n_examples, families, created), fetched 2026-09-08",
        "panel_a_parts": [{"bank": n, "n_examples": k, "created_epoch_s": c, "created_utc": _utc(c).isoformat(), "cumulative_examples": cu}
                          for (n, k, c), cu in zip(parts, cum)],
        "panel_a_merged_banks": [{"bank": n, "n_examples": k, "created_epoch_s": c, "created_utc": _utc(c).isoformat(), "parts": p}
                                 for n, k, c, p in MERGED_BANKS],
        "panel_a_training_corpora": [{"examples": n, "label": lab.replace("\n", " ")} for n, lab in TRAINING_CORPORA],
        "panel_a_summary": {"total_examples": cum[-1], "n_parts": len(parts), "first_wave_examples": first_wave_total,
                            "first_wave_completed_utc": times[3].isoformat(),
                            "second_wave_examples_added": second_wave_added, "second_wave_n_parts": len(parts) - i_second,
                            "second_wave_first_part_utc": times[i_second].isoformat(), "second_wave_last_part_utc": times[-1].isoformat(),
                            "second_wave_hours": round(second_wave_hours, 2),
                            "growth_factor_first_wave_to_total": round(growth, 2)},
        "panel_b_mix_banks": [{k: v for k, v in b.items() if k != "label"} | {"created_utc": _utc(b["created"]).isoformat()} for b in MIX_BANKS],
        "panel_b_mlp_source_bank": MLP42_BIG | {"created_utc": _utc(MLP42_BIG["created"]).isoformat()},
        "family_labels": FAMILY_LABEL,
        "skipped": "SAE-eval hold-out set sizes not included (not in build_stats).",
    }, out_dir, "corpus_growth")


# ============================================================================================
# FIGURE 3 - full fine-tune throughput ladder
# ============================================================================================
# Each row: sourced from project memory (maemm-overnight-last5-run.md, dated entries) unless noted.
LADDER = [
    {"key": "lora_eff512", "kind": "lora", "when": "Sep 3-4",
     "label": "LoRA r64 adapter\neff. batch 512\nmicro-batch 64\n(23M pretraining)",
     "ex_s": 528, "s_step": 0.97, "eff_batch": 512, "peak_gb": 143.5,
     "peak_note": "peak from the 1xB200 smoke with the production flags (prefix cache, micro-batch 64)",
     "source": "memory Sep 7 05:32Z: 'baseline 528' ex/s (eff-512 LoRA run realact20m_prefix_lr1e-4, 0.97-1.0 s/step); Sep 3 20:46Z smoke peak 143.5 GB"},
    {"key": "fullft_eff512_first", "kind": "fsdp2", "when": "Sep 5",
     "label": "full fine-tune\nfirst try\neff. batch 512\nmicro-batch 32 x 2",
     "ex_s": 273, "ex_s_range": [256, 293], "s_step": 1.875, "s_step_range": [1.75, 2.0], "eff_batch": 512, "peak_gb": 114,
     "mfu_pct": 10,
     "source": "memory Sep 5 04:20Z / Sep 7 09:40Z: 1.75-2.0 s/step = 256-290 ex/s, MFU 10 %, peak 114 GB/rank (2M-example lr sweep arms)"},
    {"key": "fullft_eff16k_mb32", "kind": "fsdp2", "when": "Sep 7 bench",
     "label": "full fine-tune\nbench, eff. 16,384\nmicro-batch 32 x 64",
     "ex_s": 260, "eff_batch": 16384, "peak_gb": 114,
     "source": "memory Sep 7 10:55Z FFT bench: mb32 x ga64 = 260 ex/s (114 GB)"},
    {"key": "fullft_eff16k_mb64", "kind": "fsdp2", "when": "Sep 7 bench",
     "label": "full fine-tune\nbench, eff. 16,384\nmicro-batch 64 x 32",
     "ex_s": 468, "eff_batch": 16384, "peak_gb": 140,
     "source": "memory Sep 7 10:55Z FFT bench: mb64 x ga32 = 468 ex/s (140 GB) <- picked; mb128 OOM"},
    {"key": "fullft_eff4096_leg1", "kind": "fsdp2", "when": "Sep 7 19:00Z",
     "label": "104M run, leg 1\neff. batch 4,096\nmicro-batch 64 x 8\nprefix cache on",
     "ex_s": 465, "ex_s_per_rank": 58, "s_step": 8.8, "eff_batch": 4096, "peak_gb": 140, "tflops_per_gpu": 192,
     "source": "memory Sep 7 19:14Z-20:14Z: 8.6-9.4 then 8.8 s/step, 59 ex/s/rank ~ 470 ex/s, peak 140 GB; Sep 7 20:40Z bench baseline 58 ex/s/rank (192 TFLOP/s, 140 GB)"},
    {"key": "fullft_prefix_share", "kind": "fsdp2", "when": "Sep 7 bench",
     "label": "+ one shared\nprefix forward\nper step (exact)",
     "ex_s": 616, "ex_s_per_rank": 77, "eff_batch": 4096, "peak_gb": 140,
     "source": "memory Sep 7 20:40Z MFU bench at the production shape: --prefix-share-step 77 ex/s/rank (exact)"},
    {"key": "fullft_prefetch", "kind": "fsdp2", "when": "Sep 8 00:44Z",
     "label": "+ FSDP parameter\nprefetch, depth 2\n(exact); production\n6.3 s/step",
     "ex_s": 641, "ex_s_per_rank": 80, "s_step": 6.3, "eff_batch": 4096, "peak_gb": 140.5,
     "tp_doc_remeasure": {"ex_s": 629, "s_step": 6.46, "peak_gb": 138, "tflops_per_gpu": 267},
     "source": "memory Sep 7 20:40Z bench 80 ex/s/rank (exact, 140 GB); Sep 8 00:44Z production leg median 6.3 s/step, peak 140.5 GB; Sep 8 01:2xZ '641 ex/s baseline'; fullft_tp.md re-measure 629 ex/s / 6.46 s / 138 GB / 267 TFLOP/s"},
    {"key": "fullft_keep_unsharded", "kind": "rejected", "when": "Sep 8 00:18Z",
     "label": "+ 40 layers kept\nunsharded + bf16\nAdam moment\n(rejected: thrash\nat 162 GB)",
     "ex_s": 482, "s_step": 8.5, "s_step_range": [5, 16], "eff_batch": 4096, "peak_gb": 162.1,
     "bench": {"ex_s": 720, "ex_s_per_rank": 90, "peak_gb": 156, "tflops_per_gpu": 300},
     "source": "memory Sep 7 20:40Z bench ~90 ex/s/rank (300 TFLOP/s, ~156 GB); Sep 8 00:18Z real run peaked 162.1 GB, 5-16 s/step median 8.5 -> reverted"},
    {"key": "tp8_mb512", "kind": "tp8", "when": "Sep 8 bench",
     "label": "tensor parallel\n8-way, micro-\nbatch 512 x 8\n(unsafe for long\ntargets)",
     "ex_s": 696, "s_step": 5.77, "eff_batch": 4096, "peak_gb": 166.7,
     "source": "fullft_tp.md section 9 measured table"},
    {"key": "tp8_mb256", "kind": "tp8", "when": "Sep 8 bench",
     "label": "tensor parallel\n8-way, micro-\nbatch 256 x 16",
     "ex_s": 607, "s_step": 6.49, "eff_batch": 4096, "peak_gb": 112,
     "source": "fullft_tp.md section 9 measured table"},
]
LADDER_COLOR = {"lora": AQUA, "fsdp2": BLUE, "tp8": ORANGE, "rejected": MUTED}
LADDER_LABEL = {"lora": "LoRA r64 adapter (rest of model frozen)", "fsdp2": "full fine-tune, sharded data parallel (FSDP2)",
                "tp8": "full fine-tune, tensor parallel 8-way", "rejected": "rejected variant (memory thrash on the real run)"}


def fig_sft_ladder(out_dir):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 7.4), layout="constrained")
    fig.suptitle(
        "Full fine-tune throughput ladder on 8xB200 (Qwen3.6-27B): one shared prefix forward per step + FSDP prefetch bought 1.4x (465 -> 641 ex/s),\n"
        "tensor parallelism at most 1.1x more; the 104M-example run needs about 45 h at the production setting",
        fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    style_ax(ax)
    xs = list(range(len(LADDER)))
    for i, r in enumerate(LADDER):
        c = LADDER_COLOR[r["kind"]]
        b = ax.bar(i, r["ex_s"], width=0.64, color=c, edgecolor=SURFACE, linewidth=1.5)
        if r["kind"] == "rejected":
            b[0].set_hatch("////")
        if "ex_s_range" in r:
            lo, hi = r["ex_s_range"]
            ax.plot([i, i], [lo, hi], color=INK2, linewidth=1.6)
            ax.plot([i - 0.08, i + 0.08], [lo, lo], color=INK2, linewidth=1.6)
            ax.plot([i - 0.08, i + 0.08], [hi, hi], color=INK2, linewidth=1.6)
        if "bench" in r:
            ax.plot(i, r["bench"]["ex_s"], marker="o", markersize=10, markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.8, linestyle="")
            ax.text(i + 0.14, r["bench"]["ex_s"], f"bench {r['bench']['ex_s']} ex/s\nat {r['bench']['peak_gb']} GB", fontsize=8.5, color=INK2, va="center", ha="left")
        # value + ETA above the bar
        ax.text(i, r["ex_s"] + 14, f"{r['ex_s']} ex/s", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(i, r["ex_s"] + 52, f"104M ex: {eta_h(r['ex_s']):.0f} h", ha="center", va="bottom", fontsize=9, color=INK2)
        # peak memory inside the bar
        peak_txt = f"peak\n{r['peak_gb']:g} GB" + ("†" if r["key"] == "lora_eff512" else "")
        ax.text(i, min(r["ex_s"] * 0.5, 130), peak_txt, ha="center", va="center", fontsize=9.2, color=inner_text_color(r["kind"]))
    # reference lines (labelled in the legend, not inline, so they never collide with bar labels)
    prod = next(r for r in LADDER if r["key"] == "fullft_prefetch")
    leg1 = next(r for r in LADDER if r["key"] == "fullft_eff4096_leg1")
    ax.axhline(prod["ex_s"], color=BLUE, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.8)
    ax.axhline(leg1["ex_s"], color=INK2, linestyle=(0, (2, 3)), linewidth=0.9, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in LADDER], fontsize=8.6)
    ax.set_ylabel("training examples / s (whole 8-GPU node)")
    ax.set_ylim(0, 1060)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.legend(handles=[Patch(color=LADDER_COLOR[k], label=LADDER_LABEL[k], hatch="////" if k == "rejected" else None) for k in ("lora", "fsdp2", "rejected", "tp8")]
              + [Line2D([], [], marker="o", markersize=9, markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.8, linestyle="", label="isolated benchmark value (before the real run)"),
                 Line2D([], [], color=INK2, linestyle=(0, (2, 3)), label=f"104M run, leg 1: {leg1['ex_s']} ex/s"),
                 Line2D([], [], color=BLUE, linestyle=(0, (4, 3)), label=f"production setting: {prod['ex_s']} ex/s")],
              loc="upper left", fontsize=9, ncol=2)
    fig.text(0.01, -0.03,
             "Configurations in the order they were tried (Sep 3 -> Sep 8, 2026). ETA = 104,000,000 examples / (ex/s) / 3600, excluding checkpoint saves, kernel JIT and leg restarts. "
             "Effective batch differs across the first four bars (512 / 16,384) and is 4,096 for the rest; whiskers = reported range. "
             "† LoRA peak memory is from a 1xB200 smoke with the production flags. Prefix cache and exact no-padding suffixes were on for every full fine-tune bar.",
             fontsize=8.5, color=INK2, ha="left", va="top", wrap=True)
    save(fig, out_dir, "sft_throughput_ladder")

    dump({
        "figure": "sft_throughput_ladder",
        "experiment": "Qwen3.6-27B, 8xB200 (Modal); full fine-tune = FSDP2 fp32 masters + bf16 compute unless tensor-parallel; LoRA row = r64 adapter",
        "eta_formula": "eta_h = 104e6 / ex_s / 3600",
        "sources": ["memory maemm-overnight-last5-run.md (Sep 3-8 entries)", "memory maemm-sft-length-bucket.md",
                    "/home/celeste/maemm-pub-tp/docs/fullft_tp.md section 9"],
        "rows": [{k: v for k, v in r.items() if k != "label"} | {"label": r["label"].replace("\n", " "), "eta_104M_examples_h": round(eta_h(r["ex_s"]), 1)} for r in LADDER],
        "derived": {"speedup_prefix_share_plus_prefetch_vs_leg1": round(641 / 465, 3),
                    "speedup_prefix_share_only_vs_leg1": round(616 / 465, 3),
                    "speedup_tp8_mb512_vs_production": round(696 / 641, 3),
                    "speedup_tp8_mb256_vs_production": round(607 / 641, 3)},
        "related_but_not_plotted": {
            "pad_multiple_1": "exact no-padding suffixes: +7-10 % ex/s at micro-batch 64 (90.1 -> 96.1 ex/s, 1xB200 LoRA smoke), peak 145 -> 136 GB; already on for every full fine-tune bar (memory maemm-sft-length-bucket.md)",
            "prefix_cache_speedup_lora_1xB200_sep3": "naive 24.2 ex/s -> compiled 29.3 -> prefix cache mb64 62.4 (2.13x) -> shared prefix 2x64 67.3 (2.3x) on 1xB200 with LoRA (memory Sep 3 20:45Z); full fine-tune never ran without the prefix cache, so no bar",
            "not_sourced": "283 TFLOP/s per GPU and 12.5 % of B200 peak for the production setting were NOT found in memory or docs; closest measured value is 267 TFLOP/s per GPU (fullft_tp.md FSDP2 re-measure at 629 ex/s) = 11.9 % of 2.25 PFLOP/s dense bf16",
        },
    }, out_dir, "sft_throughput_ladder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/celeste/shared/reports/maemm-week-digest")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    fig_tp8_vs_fsdp2(a.out)
    fig_corpus_growth(a.out)
    fig_sft_ladder(a.out)
    for stem in ("tp8_vs_fsdp2", "corpus_growth", "sft_throughput_ladder"):
        for ext in ("png", "pdf"):
            print(os.path.join(a.out, f"{stem}.{ext}"))
        print(os.path.join(a.out, "data", f"{stem}.json"))


if __name__ == "__main__":
    main()
