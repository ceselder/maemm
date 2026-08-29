"""Two plots for the Qwen3.6-27B universal-inverter data ablations (held-out, mean over 5 families:
SAE norm_act, real-act cos, cluster cos, BSF cos, J-lens cos).

A) mix_ablation: mean held-out vs RL step for 4 token-matched training mixes {BSF, probe, realact, all3}.
B) sft_vs_rl: SFT-only line (mean held-out vs SFT step) + the post-50-RL line, with RL-lift connectors —
   shows SFT contributes little (flat-ish line) while RL adds a big ~constant lift at every SFT level.

Reads scripts/out/plot_data.json. Writes scripts/out/{mix_ablation,sft_vs_rl}.{png,pdf}.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = json.load(open("/home/celeste/max-activating-examples/scripts/out/plot_data.json"))

plt.rcParams.update({"figure.facecolor": "#fbf7f0", "axes.facecolor": "#fbf7f0", "font.size": 11,
    "axes.grid": True, "grid.color": "#ddd4c7", "grid.linewidth": 0.7, "axes.edgecolor": "#8a8178"})

# ---------- A) mix ablation RL curves ----------
STY = {"mix_rp": ("50/50 real-acts + probes (no BSF)", "#c0392b", "D", 2.8),
       "mix_all3": ("33/33/33 balanced mix", "#b0682f", "o", 2.4),
       "mix_realact": ("real-acts only", "#7d4b6b", "s", 2.0),
       "mix_probe": ("probe-dict only", "#5f7a5a", "^", 2.0),
       "mix_bsf": ("BSF only", "#4a6d8c", "v", 2.0)}
order = sorted(D["mix_rl"], key=lambda k: -D["mix_rl"][k][-1][1])
fig, ax = plt.subplots(figsize=(9.2, 6.0))
for k in order:
    pts = D["mix_rl"][k]; xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    lbl, c, mk, lw = STY[k]
    ax.plot(xs, ys, marker=mk, color=c, lw=lw, ms=7, label=f"{lbl}  (final {ys[-1]:.3f})")
    ax.text(xs[-1] + 1.5, ys[-1], f"{ys[-1]:.3f}", fontsize=8.5, color=c, va="center")
ax.set_xlabel("RL step (from each mix's converged SFT)")
ax.set_ylabel("mean held-out eval  (avg of SAE, real-act, cluster, BSF, J-lens)")
ax.set_xlim(-3, 112); ax.legend(frameon=False, fontsize=10, loc="lower right", title="training-data mix")
ax.set_title("Mixing beats any single source — and the lead grows with RL",
             fontsize=15, color="#2b2b2b", fontweight="bold", pad=10)
fig.text(0.5, 0.005, "Qwen3.6-27B universal inverter · mean held-out over 5 families (n=512 each) · "
         "example-matched mixes (179,919 spans)", ha="center", fontsize=8, color="#6b6259")
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"/home/celeste/max-activating-examples/scripts/out/mix_ablation.{ext}", dpi=150, bbox_inches="tight")

# ---------- B) SFT line + RL emergence ----------
sl = np.array(D["sft_line"]); rl = np.array(D["rl50_by_sft"])
# align on common SFT steps
sd = {int(s): m for s, m in sl}; rd = {int(s): m for s, m in rl}
steps = sorted(set(sd) & set(rd))
sft_y = np.array([sd[s] for s in steps]); rl_y = np.array([rd[s] for s in steps])
fig2, ax2 = plt.subplots(figsize=(9.6, 6.0))
ax2.fill_between(steps, sft_y, rl_y, color="#b0682f", alpha=0.13, label="RL contribution (the lift)")
for s, a, b in zip(steps, sft_y, rl_y):           # RL "emerges" upward from each SFT point
    ax2.plot([s, s], [a, b], color="#b0682f", lw=1.0, alpha=0.5)
ax2.plot(steps, sft_y, "-o", color="#5f7a5a", lw=2.4, ms=6, label="SFT only (no RL)")
ax2.plot(steps, rl_y, "-o", color="#b0682f", lw=2.6, ms=6, label="after 50 RL steps")
ax2.set_xscale("log")
ax2.set_xlabel("SFT amount (SFT step, log)")
ax2.set_ylabel("mean held-out eval  (avg of 5 families)")
ax2.set_ylim(0, max(rl_y) * 1.15)
ax2.legend(frameon=False, fontsize=10, loc="center right")
lift0 = rl_y[0] - sft_y[0]; liftN = rl_y[-1] - sft_y[-1]
ax2.set_title("Final checkpoint (after RL) improves with more SFT data",
              fontsize=15, color="#2b2b2b", fontweight="bold", pad=10)
fig2.text(0.5, 0.005, "Qwen3.6-27B · mean held-out over 5 families · orange = after 50 RL steps, "
          "green = SFT only · RL @ lr 1e-5 from each SFT checkpoint", ha="center", fontsize=8, color="#6b6259")
fig2.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"):
    fig2.savefig(f"/home/celeste/max-activating-examples/scripts/out/sft_vs_rl.{ext}", dpi=150, bbox_inches="tight")
print("mix finals:", {k: round(D["mix_rl"][k][-1][1], 3) for k in order})
print(f"SFT-only mean {sft_y.min():.3f}->{sft_y.max():.3f} | RL50 mean {rl_y.min():.3f}->{rl_y.max():.3f} | lift {lift0:.3f}->{liftN:.3f}")
