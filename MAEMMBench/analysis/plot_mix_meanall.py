"""Mean held-out (avg over 5 families) for every training-data mix — the headline comparison.
Bars = final (post-RL) mean; faded bars behind = SFT-only (pre-RL) mean, so the RL lift per mix is visible.
Token counts annotated (mixes are EXAMPLE-matched at 179,919 spans; cluster spans are ~24% shorter, so tokens
differ — but the ranking does not track tokens, so it's not a token-budget artifact).
Reads scripts/out/plot_data.json. Writes scripts/out/mix_meanall.{png,pdf}.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = json.load(open("/home/celeste/max-activating-examples/scripts/out/plot_data.json"))["mix_rl"]
TOK_M = {"mix_rp": 6.36, "mix_realact": 7.23, "mix_bsf": 7.03, "mix_all3": 6.59, "mix_probe": 5.49}   # mean-span × 179,919
LBL = {"mix_rp": "50/50\nrealact+probe", "mix_all3": "33/33/33\nBSF+probe+realact", "mix_realact": "realact\nonly",
       "mix_probe": "probe\nonly", "mix_bsf": "BSF\nonly"}
COL = {"mix_rp": "#c0392b", "mix_all3": "#b0682f", "mix_realact": "#7d4b6b", "mix_probe": "#5f7a5a", "mix_bsf": "#4a6d8c"}
mixes = sorted(D, key=lambda k: -D[k][-1][1])          # by final mean
final = {k: D[k][-1][1] for k in mixes}
sftonly = {k: D[k][0][1] for k in mixes}               # RL step 0 = converged SFT

plt.rcParams.update({"figure.facecolor": "#fbf7f0", "axes.facecolor": "#fbf7f0", "font.size": 11,
    "axes.grid": True, "grid.color": "#ddd4c7", "grid.linewidth": 0.7, "axes.edgecolor": "#8a8178"})
x = np.arange(len(mixes)); fig, ax = plt.subplots(figsize=(11.5, 6.0))
for i, k in enumerate(mixes):
    ax.bar(i, sftonly[k], 0.6, color=COL[k], alpha=0.28)                       # SFT-only (faded)
    ax.bar(i, final[k], 0.6, color=COL[k], alpha=0.95, label=None)             # final (solid, overlaps)
    ax.text(i, final[k] + 0.006, f"{final[k]:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=COL[k])
    ax.text(i, sftonly[k] - 0.018, f"SFT {sftonly[k]:.2f}", ha="center", va="top", fontsize=7.5, color="#6b6259")
    ax.text(i, 0.012, f"{TOK_M[k]:.1f}M tok", ha="center", va="bottom", fontsize=7.5, color="white")
ax.set_xticks(x); ax.set_xticklabels([LBL[k] for k in mixes], fontsize=9.5)
ax.set_ylabel("mean held-out eval  (avg of SAE, real-act, cluster, BSF, J-lens; n=512/family)")
ax.set_ylim(0, 0.44)
ax.set_title("Mixing BSF + real-acts + probes beats any single source",
             fontsize=15, color="#2b2b2b", fontweight="bold", pad=10)
fig.text(0.5, 0.005, "Qwen3.6-27B · mean held-out (5 families) · solid = final (post-RL), faded = SFT-only · "
         "example-matched (179,919 spans); token counts annotated", ha="center", fontsize=8, color="#6b6259")
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"/home/celeste/max-activating-examples/scripts/out/mix_meanall.{ext}", dpi=150, bbox_inches="tight")
print("final:", {k: round(final[k], 3) for k in mixes}, "| SFT-only:", {k: round(sftonly[k], 3) for k in mixes})
