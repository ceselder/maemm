"""Evidence: the probe-only training mix generalizes WORSE than the BSF/universal-mixed inverter.

Both models compared at their PLATEAUED RL checkpoint on the identical held-out suite (Qwen3.6-27B,
eval_universal.py, n=512/family, old sae27b SAE). Not training-budget-matched — the probe mix actually
saw MORE SFT — so we compare converged endpoints and annotate how much SFT/RL each saw.

Panel A (the evidence): held-out eval per family, probe-mix vs BSF/universal, both at plateau.
Panel B (context, NOT apples-to-apples): training reward vs RL step — both plateau at a similar reward,
which is exactly why reward alone is misleading and the held-out bars are the real signal.

Data:
  probe-mix   = old ProbeMaxxer RL (rl_1M_yolo2/step_135), SFT 62,496 steps on 1M cluster/probe spans.
  BSF/universal = universal RL (ref run, step 195),        SFT 25,679 steps on 822k mixed bank (incl BSF).
Writes scripts/out/probemix_vs_bsf.{png,pdf} + data/data_probemix_vs_bsf.json.
"""
import json, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

FAM = ["sae", "realact", "cluster", "bsf", "jlens", "random"]
FAM_LBL = ["SAE\nnorm_act", "real-act\ncos", "cluster/probe\ncos", "BSF\ncos", "J-lens\ncos", "random\n(control)"]
probe = {"sae": 0.658, "realact": 0.413, "cluster": 0.267, "bsf": 0.318, "jlens": 0.104, "random": 0.029}
bsf   = {"sae": 0.856, "realact": 0.496, "cluster": 0.265, "bsf": 0.343, "jlens": 0.119, "random": 0.034}
BUD = {"probe": {"sft": 62496, "rl": 135, "data": "1M cluster/probe spans"},
       "bsf":   {"sft": 25679, "rl": 195, "data": "822k mixed bank (BSF+probe+realact+SAE+Jlens)"}}

def load_reward(path):
    s, r = [], []
    for ln in open(path):
        m = re.search(r"(\d+)\s+([\d.]+)", ln) or re.search(r"step (\d+) \| r ([\d.]+)", ln)
        if m: s.append(int(m[1])); r.append(float(m[2]))
    return np.array(s), np.array(r)
os_, or_ = load_reward("/home/celeste/max-activating-examples/scripts/out/oldrl_yolo2_reward.txt")
us_, ur_ = load_reward("/home/celeste/max-activating-examples/scripts/out/uni_rl_reward_raw.txt")
def roll(a, w=15): return a if len(a) < w else np.convolve(a, np.ones(w)/w, mode="valid")

json.dump({"families": FAM, "probe_mix": probe, "bsf_universal": bsf, "budgets": BUD,
           "reward_probe": {"step": os_.tolist(), "r": or_.tolist()},
           "reward_bsf": {"step": us_.tolist(), "r": ur_.tolist()}},
          open("/home/celeste/max-activating-examples/scripts/out/data_probemix_vs_bsf.json", "w"), indent=1)

plt.rcParams.update({"figure.facecolor": "#fbf7f0", "axes.facecolor": "#fbf7f0", "font.size": 11,
    "axes.grid": True, "grid.color": "#ddd4c7", "grid.linewidth": 0.7, "axes.edgecolor": "#8a8178"})
CLAY, PLUM, INK = "#b0682f", "#7d4b6b", "#2b2b2b"
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})

# --- Panel A: grouped bars ---
x = np.arange(len(FAM)); w = 0.38
pv = [probe[f] for f in FAM]; bv = [bsf[f] for f in FAM]
b1 = axA.bar(x - w/2, pv, w, color=PLUM, label="probe-only mix (old ProbeMaxxer)")
b2 = axA.bar(x + w/2, bv, w, color=CLAY, label="BSF / universal mix")
for bars in (b1, b2):
    for r in bars:
        axA.text(r.get_x()+r.get_width()/2, r.get_height()+0.008, f"{r.get_height():.2f}",
                 ha="center", va="bottom", fontsize=7.5, color=INK)
axA.set_xticks(x); axA.set_xticklabels(FAM_LBL, fontsize=8.5)
axA.set_ylabel("held-out eval  (cosine / SAE norm_act)"); axA.set_ylim(0, 0.95)
axA.legend(frameon=False, fontsize=9.5, loc="upper right")
axA.set_title("A. Held-out generalization at plateau — BSF/universal mix wins on every family "
              "(ties cluster,\nthe probe mix's home turf)", fontsize=9.2, color=INK, fontweight="bold")

# --- Panel B: reward curves ---
axB.plot(os_, or_, ".", color=PLUM, ms=3, alpha=0.3)
axB.plot(roll(os_), roll(or_), "-", color=PLUM, lw=2.2, label="probe-only mix RL")
axB.plot(us_, ur_, ".", color=CLAY, ms=3, alpha=0.3)
axB.plot(roll(us_), roll(ur_), "-", color=CLAY, lw=2.2, label="BSF / universal mix RL")
axB.set_xlabel("RL step"); axB.set_ylabel("training reward  (max-token cos × 1000)")
axB.legend(frameon=False, fontsize=9.5, loc="lower right")
axB.set_title("B. Training reward (⚠ NOT matched: lr 1e-6 vs 1e-5, different data/steps) — both plateau at a\n"
              "similar reward, so reward alone hides the panel-A gap", fontsize=9.2, color=INK)

sub = ("⚠ CONFOUNDED, treat as SUGGESTIVE not clean: probe RL used lr≈1e-6, universal 1e-5 (10×) — and lower LR "
       "alone lowers the plateau (see LR sweep).\n"
       f"Probe mix: SFT {BUD['probe']['sft']:,} steps ({BUD['probe']['data']}) + RL≈{BUD['probe']['rl']} steps, lr≈1e-6"
       f"   |   BSF/universal: SFT {BUD['bsf']['sft']:,} steps (mixed incl. BSF) + RL {BUD['bsf']['rl']} steps, lr 1e-5"
       "   →  clean probe-vs-BSF test = the running mix ablation (identical lr/kl/steps/tokens)")
fig.suptitle("Old probe-only inverter scores lower on held-out than the universal/BSF mix — but training was NOT matched "
             "(Qwen3.6-27B, at plateau; suggestive)",
             fontsize=10.8, fontweight="bold", color=INK, y=1.005)
fig.text(0.5, -0.035, sub, ha="center", fontsize=7.8, color="#6b6259")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"/home/celeste/max-activating-examples/scripts/out/probemix_vs_bsf.{ext}", dpi=150, bbox_inches="tight")
print("wrote probemix_vs_bsf.png |", {f: (probe[f], bsf[f]) for f in FAM})
