"""Sample rollouts BEFORE / DURING / AFTER the constant-lr RL collapse, from the transcripts.jsonl each RL run writes every 5 steps
(rows: step, group, vec_idx, family, target, text, cos, reward, adv, n_tok). Produces a report folder with a per-step summary figure
(mean reward, mean generated length, share of degenerate texts vs RL step, onset marked) and a page of side-by-side rollouts.

    for d in ckpts_rl_I_8x4096_nowarm ckpts_rl_F_16x512 ckpts_rl_H_16x512_lr5e-6; do
        modal volume get maemm-data $d/transcripts.jsonl /tmp/collapse_tx/$d.jsonl --force; done
    python3 scripts/dump_collapse_rollouts.py
"""
import html
import json
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser("~/shared/reports/maemm-collapse-rollouts")
TX = "/tmp/collapse_tx"
# run -> (label, transcripts file, runaway onset step (grad-norm > clip for consecutive steps), lr, rollouts/step)
RUNS = {
    "RL-I": ("RL-I: 8 samples × 4096 directions/step (32k rollouts), lr 1e-5, no warmup", "ckpts_rl_I_8x4096_nowarm.jsonl", 178, 1e-5, 32768),
    "RL-F": ("RL-F: 16 samples × 512 directions/step (8k rollouts), lr 1e-5", "ckpts_rl_F_16x512.jsonl", 246, 1e-5, 8192),
    "RL-H": ("RL-H: 16 samples × 512 directions/step (8k rollouts), lr 5e-6", "ckpts_rl_H_16x512_lr5e-6.jsonl", 452, 5e-6, 8192),
}
PHASES = [("before", -70, -25), ("during", -5, 15), ("after", 50, 10_000)]
N_SHOW = 6
rng = random.Random(0)


def degenerate(t):
    """Heuristic 'slop' flags: non-ASCII share, repeated token runs, very short."""
    toks = t.split()
    non_ascii = sum(ord(c) > 127 for c in t) / max(len(t), 1)
    rep = 0.0
    if len(toks) >= 6:
        rep = 1 - len(set(toks)) / len(toks)
    return non_ascii > 0.3 or rep > 0.5 or len(toks) < 3


def load(fn):
    rows = [json.loads(l) for l in open(f"{TX}/{fn}")]
    by_step = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r)
    return by_step


summary, picks = {}, {}
for run, (label, fn, onset, lr, rps) in RUNS.items():
    by_step = load(fn)
    steps = sorted(by_step)
    per_step = []
    for s in steps:
        rs = by_step[s]
        per_step.append({"step": s, "n": len(rs), "reward_mean": float(np.mean([r["reward"] for r in rs])), "cos_mean": float(np.mean([r["cos"] for r in rs])),
                         "len_mean": float(np.mean([r["n_tok"] for r in rs])), "degenerate_frac": float(np.mean([degenerate(r["text"]) for r in rs])),
                         "reward_max": float(np.max([r["reward"] for r in rs]))})
    summary[run] = {"label": label, "onset": onset, "lr": lr, "rollouts_per_step": rps, "steps_logged": steps, "per_step": per_step}
    picks[run] = {}
    for name, lo, hi in PHASES:
        cand = [s for s in steps if onset + lo <= s <= onset + hi]
        if not cand:
            continue
        s = cand[len(cand) // 2] if name != "after" else cand[min(len(cand) - 1, 2)]
        rs = by_step[s]
        # spread the shown samples over families; prefer distinct groups
        fams = sorted({r["family"] for r in rs})
        chosen, used = [], set()
        for f in fams:
            pool = [r for r in rs if r["family"] == f and r["group"] not in used]
            if pool:
                r = rng.choice(pool); chosen.append(r); used.add(r["group"])
        for g in sorted({r["group"] for r in rs} - used):          # fill up with other groups (transcripts log 8-16 samples from a few groups)
            if len(chosen) >= N_SHOW:
                break
            r = rng.choice([r for r in rs if r["group"] == g]); chosen.append(r); used.add(g)
        picks[run][name] = {"step": s, "window": [onset + lo, onset + hi], "stats": next(p for p in per_step if p["step"] == s),
                            "samples": [{k: r[k] for k in ("group", "vec_idx", "family", "target", "text", "cos", "reward", "n_tok")} for r in chosen[:N_SHOW]]}

os.makedirs(f"{OUT}/data", exist_ok=True)
json.dump({"summary": summary, "picks": picks, "phases": {n: [lo, hi] for n, lo, hi in PHASES},
           "degenerate_rule": "non-ASCII share > 0.3 or repeated-token share > 0.5 or < 3 words"}, open(f"{OUT}/data/collapse_rollouts.json", "w"), indent=1)

# ---- figure: per-step summary from the transcripts, onset marked
fig, axes = plt.subplots(3, len(RUNS), figsize=(5.2 * len(RUNS), 9.5), sharex="col")
for j, (run, S) in enumerate(summary.items()):
    x = [p["step"] for p in S["per_step"]]
    for i, (k, lab) in enumerate([("reward_mean", "mean reward of the logged rollouts"), ("len_mean", "mean generated length (tokens)"), ("degenerate_frac", "share of degenerate texts")]):
        ax = axes[i, j]
        ax.plot(x, [p[k] for p in S["per_step"]], "o-", ms=3, lw=1.4, color=["#2a7f62", "#4a6fa5", "#b5542b"][i])
        ax.axvline(S["onset"], color="#111", ls="--", lw=1); ax.grid(alpha=0.25)
        for name, lo, hi in PHASES:
            if name in picks[run]:
                ax.axvspan(max(S["onset"] + lo, min(x)), min(S["onset"] + hi, max(x)), color={"before": "#2a7f62", "during": "#c99a2e", "after": "#b5542b"}[name], alpha=0.10)
        if i == 0:
            ax.set_title(S["label"], fontsize=9.5)
        if j == 0:
            ax.set_ylabel(lab, fontsize=9)
        if i == 2:
            ax.set_xlabel("RL step (dashed = runaway onset; shaded = before / during / after windows sampled below)", fontsize=8.5)
fig.suptitle("Constant-lr RL collapse in the transcripts: reward holds or climbs through the onset, generated text gets shorter and degenerate afterwards\n"
             "(sampled rollouts logged every 5 steps by three runs of the Qwen3.6-27B activation→text inverter)", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/collapse_transcript_stats.{ext}", dpi=150)
plt.close(fig)


# ---- report
def esc(t):
    return html.escape(t).replace("\n", "⏎ ")


def card(r):
    return (f'<div class="rollout"><div class="meta"><b>{esc(r["family"])}</b> · direction {r["vec_idx"]} · reward {r["reward"]:.3f} · cos {r["cos"]:.3f} · {r["n_tok"]} tok</div>'
            f'<div class="tgt"><span class="lab">target text</span> {esc(r["target"][:300])}</div>'
            f'<div class="gen"><span class="lab">inverter output</span> {esc(r["text"][:400])}</div></div>')


sections = []
for run, S in summary.items():
    cols = []
    for name, _, _ in PHASES:
        if name not in picks[run]:
            continue
        p = picks[run][name]; st = p["stats"]
        cols.append(f'<div class="phase"><h3>{name} — step {p["step"]}</h3><p class="subtitle">window steps {p["window"][0]}–{min(p["window"][1], max(S["steps_logged"]))} · '
                    f'this step: mean reward {st["reward_mean"]:.3f}, mean length {st["len_mean"]:.0f} tok, degenerate {st["degenerate_frac"] * 100:.0f}%</p>'
                    + "".join(card(r) for r in p["samples"]) + "</div>")
    sections.append(f'<h2 id="{run}">{S["label"]} — runaway onset step {S["onset"]} (lr × steps = {S["onset"] * S["lr"] * 1e3:.2f}e-3)</h2><div class="phases">{"".join(cols)}</div>')

css = """<style>.phases{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.phase h3{margin:.2em 0}.rollout{border:1px solid #e3ddd2;border-radius:8px;padding:8px 10px;margin:8px 0;background:#fffdf8;font-size:.86em}
.meta{color:#6b6257;margin-bottom:4px}.lab{display:inline-block;min-width:7.5em;color:#8a7f70;font-variant:small-caps}.tgt{margin:2px 0}.gen{margin:2px 0;color:#1d2a44}@media(max-width:1100px){.phases{grid-template-columns:1fr}}</style>"""
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Rollouts before, during and after the RL collapse</title>
<link rel="stylesheet" href="/reports/static/claude.css">{css}</head><body><main>
<h1>What the inverter says before, during and after the constant-lr RL collapse: reward keeps climbing through the onset while the text turns short and degenerate</h1>
<p class="subtitle">Sampled rollouts (target text of the direction → inverter output, with its reward = clean-base cosine to the target activation, max over the last 5 tokens)
logged every 5 RL steps by three runs. Onset = first step with gradient norm above the clip for consecutive steps. Windows: before = onset −70…−25, during = onset −5…+15, after = onset +50 onward.</p>
<div class="tldr"><b>TL;DR</b> The collapse is not a reward collapse first: in every run the mean reward of the logged rollouts holds or rises past the onset, while the outputs shorten and the share of degenerate texts
(non-ASCII, repetition, &lt; 3 words) rises. The samples below show the same three stages in each run — coherent paraphrases before, terse high-reward fragments during, and repetitive or multilingual slop after.
Exact numbers per step are in <code>data/collapse_rollouts.json</code>.</div>
<figure><img src="collapse_transcript_stats.png" alt="per-step transcript stats"><figcaption>Per-step statistics of the logged rollouts (16–64 per step). <a href="collapse_transcript_stats.pdf">PDF</a></figcaption></figure>
<nav class="toc">{" · ".join(f'<a href="#{r}">{r}</a>' for r in summary)}</nav>
{"".join(sections)}
<details><summary>Appendix</summary><ul>
<li>Source: <code>/data/ckpts_rl_*/transcripts.jsonl</code> on the Modal volume maemm-data (written with <code>--transcript-every 5</code>). Onsets from the RL-lr-level / batch-scaling reports.</li>
<li>Script: <code>~/maemm-pub/scripts/dump_collapse_rollouts.py</code> (seeded sampling; one sample per family per phase, distinct groups).</li>
<li>Reward = ScaleRL/CISPO last-5 cosine reward incl. the length penalty; cos = raw cosine.</li></ul></details>
</main></body></html>"""
open(f"{OUT}/report.html", "w").write(page)
print("wrote", OUT, "| runs:", {r: {"steps": (min(S["steps_logged"]), max(S["steps_logged"])), "phases": {n: p["step"] for n, p in picks[r].items()}} for r, S in summary.items()})
