"""Per-family final held-out (RL step 100) for the 4 training mixes (Qwen3.6-27B universal inverter).
Shows WHERE the balanced 33/33/33 mix wins: it matches/leads everywhere and dominates the held-out SAE
family. Reads scripts/out/mix_finals_perfamily.json. Writes scripts/out/mix_perfamily.{png,pdf}.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = json.load(open("/home/celeste/max-activating-examples/scripts/out/mix_finals_perfamily.json"))
FAM = ["sae", "realact", "cluster", "bsf", "jlens"]
FAM_LBL = ["SAE\nnorm_act", "real-act\ncos", "cluster\ncos", "BSF\ncos", "J-lens\ncos"]
MIX = ["mix_all3", "mix_realact", "mix_probe", "mix_bsf"]
LBL = {"mix_all3": "33/33/33 balanced", "mix_realact": "realact only", "mix_probe": "probe only", "mix_bsf": "BSF only"}
COL = {"mix_all3": "#b0682f", "mix_realact": "#7d4b6b", "mix_probe": "#5f7a5a", "mix_bsf": "#4a6d8c"}

plt.rcParams.update({"figure.facecolor": "#fbf7f0", "axes.facecolor": "#fbf7f0", "font.size": 11,
    "axes.grid": True, "grid.color": "#ddd4c7", "grid.linewidth": 0.7, "axes.edgecolor": "#8a8178"})
x = np.arange(len(FAM)); w = 0.2
fig, ax = plt.subplots(figsize=(10.2, 6.0))
for i, mix in enumerate(MIX):
    ys = [D[mix][f] for f in FAM]
    mean = np.mean(ys)
    b = ax.bar(x + (i - 1.5) * w, ys, w, color=COL[mix], label=f"{LBL[mix]}  (mean {mean:.3f})")
    if mix == "mix_all3":
        for r in b:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.006, f"{r.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=COL[mix], fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(FAM_LBL, fontsize=9.5)
ax.set_ylabel("held-out eval at RL step 100  (cosine / SAE norm_act)")
ax.set_ylim(0, 0.85)
ax.legend(frameon=False, fontsize=9.5, loc="upper right", title="training-data mix")
ax.set_title("Mixing beats any individual pick — especially on SAEs",
             fontsize=15, color="#2b2b2b", fontweight="bold", pad=10)
fig.text(0.5, 0.005, "Qwen3.6-27B universal inverter · per-family held-out at final RL checkpoint (n=512 each) · "
         "example-matched mixes", ha="center", fontsize=8, color="#6b6259")
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(f"/home/celeste/max-activating-examples/scripts/out/mix_perfamily.{ext}", dpi=150, bbox_inches="tight")
print("means:", {m: round(np.mean([D[m][f] for f in FAM]), 3) for m in MIX})
print("SAE:", {m: round(D[m]["sae"], 3) for m in MIX})
