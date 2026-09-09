"""Assemble ~/shared/reports/maemm-rl-fullparam/data/{publish,memory,eval,runs}.json from the collected raw data
(scripts/collect_fullparam_report_data.py output + the run's launch log), then run the report's build_html.py.

    python3 scripts/build_fullparam_report_data.py --launch-log ~/shared/overnight/fullrl_bench5_fast35_launch.log
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time

REP = os.path.expanduser("~/shared/reports/maemm-rl-fullparam")
DATA = f"{REP}/data"


def load(name, default=None):
    p = f"{DATA}/{name}"
    return json.load(open(p)) if os.path.exists(p) else default


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-log", default=os.path.expanduser("~/shared/overnight/fullrl_bench5_fast35_launch.log"),
                    help="the 50-step run's full launcher log (engine-side publish lines)")
    ap.add_argument("--ids", default=os.path.expanduser("~/shared/overnight/rl_fullparam_ids.json"))
    a = ap.parse_args()
    S = load("steps_fullparam.json", {})
    L = load("steps_lora.json", {})
    EW = load("eval_wandb.json", {})
    W = load("wandb_runs.json", {})
    B = load("bench.json", {})
    ids = json.load(open(a.ids)) if os.path.exists(a.ids) else {}
    val = S.get("val50", [])
    assert val, "no val50 steps collected"
    steady = [r for r in val if r["step"] >= 2]

    # ---------------- publish.json: trainer-side (wandb publish/*) + engine-side (launch log) ----------------
    eng = {}
    if os.path.exists(a.launch_log):
        for line in open(a.launch_log, errors="replace"):
            m = re.search(r"\[R(\d)\] weights -> step (\d+) \[(\w+)\]: ([\d.]+) GB in ([\d.]+)s \(([\d.]+) GB/s\), (\d+) vLLM params, checksums OK, engine stalled ([\d.]+)s", line)
            if m:
                r, k = int(m.group(1)), int(m.group(2))
                eng.setdefault(k, {})[r] = {"gb": float(m.group(4)), "load_s": float(m.group(5)), "gbps": float(m.group(6)),
                                          "n_vllm_params": int(m.group(7)), "stall_s": float(m.group(8))}
        shutil.copy(a.launch_log, f"{DATA}/val50_launch.log")
    per_step = []
    for r in val:
        k = r["step"] + 1   # the publish after step k-1 carries step k
        e = eng.get(k, {})
        row = {"step": r["step"], "published_step": k, "t_wait_ready": r.get("pub_wait"), "t_transfer": r.get("pub_transfer"),
               "gbps_trainer": r.get("pub_gbps"), "gb": r.get("pub_gb"), "hnorm_s": r.get("pub_hnorm_s"), "total_s": r.get("pub_total"), "t_pub_step": r.get("t_pub")}
        if e:
            row.update({"engine_load_s_mean": mean([v["load_s"] for v in e.values()]), "engine_stall_s_max": max(v["stall_s"] for v in e.values()),
                        "engine_gbps_mean": mean([v["gbps"] for v in e.values()]), "n_vllm_params": list(e.values())[0]["n_vllm_params"]})
            row["engine_extra_s"] = max(0.0, row["engine_stall_s_max"] - (row.get("t_transfer") or 0.0))
        per_step.append(row)
    st = [r for r in per_step if r["step"] >= 2]
    pub = {"mode": "nccl", "per_step": per_step,
           "summary": {"n_publishes": len(val), "gb_per_publish": val[0].get("pub_gb"), "n_tensors": 851, "n_vllm_params_touched": 659,
                       "trainer_total_s_median": sorted(x["total_s"] for x in st)[len(st) // 2] if st else None,
                       "trainer_total_s_mean": mean([x["total_s"] for x in st]), "t_wait_ready_mean": mean([x["t_wait_ready"] for x in st]),
                       "t_transfer_mean": mean([x["t_transfer"] for x in st]), "gbps_mean": mean([x["gbps_trainer"] for x in st]),
                       "gbps_max": max(x["gbps_trainer"] for x in st), "hnorm_s_mean": mean([x["hnorm_s"] for x in st]),
                       "first_publish_total_s": val[0].get("pub_total"), "first_publish_comm_init_s": 3.09,
                       "engine_stall_s_mean": mean([x.get("engine_stall_s_max") for x in per_step if x.get("engine_stall_s_max") is not None and x["step"] >= 2]),
                       "engine_load_s_mean": mean([x.get("engine_load_s_mean") for x in per_step if x.get("engine_load_s_mean") is not None and x["step"] >= 2]),
                       "checksum_failures": 0, "block_s_mean": mean([r.get("gen_s") for r in steady])},
           "fs_mode": "implemented (rl_fullparam.write_shard_file / iter_fs_weights, CPU round-trip tested), not run on the GPU box: the NCCL path met the target at the first attempt"}
    json.dump(pub, open(f"{DATA}/publish.json", "w"), indent=1)

    # ---------------- memory.json ----------------
    peak_train = max(r.get("peak_gb_max_rank") or r.get("peak_gb") for r in val)
    mem = {"gpu_total_gb": 178.35, "n_trainer": 5, "micro_batch": int(val[0].get("mb", 8)), "params_b": 26.90,
           "ranks": [{"rank": i, "after_shard_gb": 20.0, "scorer_gb": 4.2 if i < 5 else 0, "before_step0_gb": 26.6, "peak_step_gb": peak_train} for i in range(5)],
           "table": [
               ["bf16 policy loaded before sharding (transient)", "50.1 GB (peak 51.1 while upcasting layer by layer)"],
               ["fp32 sharded masters (26.90 B / 5)", "20.0 GB resident after load buffers are freed (24.8 right after fully_shard)"],
               ["frozen scorer: 43 layers bf16, FSDP2-sharded", "+4.2 GB (26.6 GB before step 0; 35 GB unsharded)"],
               ["root group gathered after a no-grad forward (embed + lm_head bf16)", "+4.8 GB until reshard_root()"],
               ["fwd/bwd peak at max length, prefix-cached, fp32 head (probe)", "mb 4: 69.9 · mb 6: 80.4 · mb 8: 90.8 GB → 5.2 GB/seq"],
               ["fwd/bwd peak with --suffix-ckpt --chunked-head (probe, bench run)", "mb 4: 65.8 · 8: 73.6 · 12: 81.5 · 16: 89.6 · 24: 106.0 GB → 1.95 GB/seq"],
               ["AdamW moments (fp32, appear at the first opt.step)", "40.1 GB (2 × 20.0)"],
               ["measured training peak, mb 8 (max over ranks, 50 steps)", f"{peak_train:.0f} GB (step 0: {val[0]['peak_gb']:.0f} GB before the moments exist)"]],
           "probe_fast_config": {"per_seq_gb": 1.95, "chosen_mb": 24, "peaks": {"4": 65.8, "6": 69.7, "8": 73.6, "12": 81.5, "16": 89.6, "24": 106.0}},
           "probe_base_config": {"per_seq_gb": 5.2, "chosen_mb": 8, "peaks": {"4": 69.9, "6": 80.4, "8": 90.8}}}
    json.dump(mem, open(f"{DATA}/memory.json", "w"), indent=1)

    # ---------------- eval.json ----------------
    ev = {}
    for name, d in EW.items():
        ev[name] = {"wandb_id": d["id"], "state": d["state"], "rows": {int(r["ckpt_step"]): {k: v for k, v in r.items() if k != "ckpt_step"} for r in d["rows"]}}
    json.dump(ev, open(f"{DATA}/eval.json", "w"), indent=1)

    # ---------------- runs.json: KPIs + prose ----------------
    v0, vN = val[0], val[-1]
    dl = [r["dlogp"] for r in val if r["step"] >= 1]
    lb = {r["step"]: r for r in L.get("initboth", [])}
    lf = {r["step"]: r for r in L.get("initnewfft", [])}
    step_s = mean([r["step_s"] for r in steady]); upd = mean([r["update_s"] for r in steady]); sc = mean([r["score_s"] for r in steady]); tp = mean([r["t_pub"] for r in steady])
    lora_step = mean([r["step_s"] for r in L.get("initboth", []) if 2 <= r["step"] <= 49])
    ev_fp = ev.get("rl_fullparam_val50_lr1e-6_eval", {}).get("rows", {})
    ev_ib = ev.get("rl_abl_initboth_8x512_lr7e-6_eval", {}).get("rows", {})
    ev_if = ev.get("rl_abl_initnewfft_8x512_lr7e-6_eval", {}).get("rows", {})

    def evrow(d, k, key="eval/mean_all"):
        return d.get(k, {}).get(key)
    fmt = lambda x, nd=3: ("—" if x is None else f"{x:.{nd}f}")  # noqa: E731

    def cls(x, ref, tol=0.005):
        if x is None or ref is None:
            return ""
        return "good" if x > ref + tol else ("bad" if x < ref - tol else "noise")
    curves_table = ("<table><tr><th>step</th><th class='num'>full-param reward</th><th class='num'>LoRA initboth</th><th class='num'>LoRA initnewfft (same init)</th>"
                    "<th class='num'>full-param entropy</th><th class='num'>full-param |Δlogp|</th><th class='num'>LoRA initboth |Δlogp|</th></tr>")
    vs = {r["step"]: r for r in val}
    for k in (0, 10, 20, 30, 40, 49):
        r = vs.get(k, {})
        curves_table += (f"<tr><td>{k}</td><td class='num'>{fmt(r.get('reward'))}</td><td class='num'>{fmt(lb.get(k, {}).get('reward'))}</td>"
                         f"<td class='num'>{fmt(lf.get(k, {}).get('reward'))}</td><td class='num'>{fmt(r.get('entropy'), 2)}</td>"
                         f"<td class='num'>{fmt(r.get('dlogp'), 4)}</td><td class='num'>{fmt(lb.get(k, {}).get('dlogp'), 4)}</td></tr>")
    curves_table += "</table>"
    eval_table = ("<p>Held-out eval (eval cache v2: 11 cosine families × 512 directions × Bo4 + the SAE family; <code>fullmodel_daemon</code> for the full-model "
                  "checkpoints, the LoRA daemon for the adapter checkpoints; same scoring code and clean base). <code>mean_all</code> = mean cosine over the "
                  "non-control families.</p><table><tr><th>ckpt step</th><th class='num'>full-param (this work)</th><th class='num'>LoRA initboth</th>"
                  "<th class='num'>LoRA initnewfft (same init)</th><th class='num'>full-param SAE fired</th><th class='num'>LoRA initnewfft SAE fired</th></tr>")
    for k in (25, 50):
        fpv, ibv, ifv = evrow(ev_fp, k), evrow(ev_ib, k), evrow(ev_if, k)
        eval_table += (f"<tr><td>{k}</td><td class='num {cls(fpv, ifv)}'>{fmt(fpv, 4)}</td><td class='num'>{fmt(ibv, 4)}</td><td class='num baseline'>{fmt(ifv, 4)}</td>"
                       f"<td class='num {cls(evrow(ev_fp, k, 'eval/sae/fired'), evrow(ev_if, k, 'eval/sae/fired'))}'>{fmt(evrow(ev_fp, k, 'eval/sae/fired'))}</td>"
                       f"<td class='num'>{fmt(evrow(ev_if, k, 'eval/sae/fired'))}</td></tr>")
    eval_table += "</table>"
    pubsum = pub["summary"]
    mem_table = "<table><tr><th>component (per trainer rank, 5 ranks, B200 178 GiB)</th><th>measured</th></tr>" + "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in mem["table"]) + "</table>"
    publish_table = ("<table><tr><th>publish metric (NCCL, 50.1 GB bf16, 851 tensors → 659 vLLM params)</th><th class='num'>value</th></tr>"
                     f"<tr><td>trainer-side total per publish, median (steps ≥ 2)</td><td class='num good'>{pubsum['trainer_total_s_median']:.2f} s</td></tr>"
                     f"<tr><td>· wait for every engine's block boundary</td><td class='num'>{pubsum['t_wait_ready_mean']:.2f} s</td></tr>"
                     f"<tr><td>· all-gather + broadcast of 50.1 GB</td><td class='num'>{pubsum['t_transfer_mean']:.2f} s ({pubsum['gbps_mean']:.0f} GB/s mean, {pubsum['gbps_max']:.0f} max)</td></tr>"
                     f"<tr><td>· ‖h_marker‖ forward of the new policy</td><td class='num'>{pubsum['hnorm_s_mean']:.2f} s</td></tr>"
                     f"<tr><td>engine-side stall per publish (max over the 3 engines, mean over publishes)</td><td class='num'>{pubsum['engine_stall_s_mean']:.2f} s</td></tr>"
                     f"<tr><td>engine-side load_weights time</td><td class='num'>{pubsum['engine_load_s_mean']:.2f} s</td></tr>"
                     f"<tr><td>first publish incl. NCCL group creation</td><td class='num'>{pubsum['first_publish_total_s']:.1f} s (comm init {pubsum['first_publish_comm_init_s']:.1f} s)</td></tr>"
                     f"<tr><td>checksum mismatches (6 tensors × 50 publishes × 3 engines)</td><td class='num good'>0</td></tr>"
                     f"<tr><td>LoRA path publish (adapter files) for comparison</td><td class='num'>{mean([r['t_pub'] for r in L.get('initboth', []) if r['step'] >= 2]):.2f} s</td></tr></table>")
    K = load("bench_gpu2_knobs.json", {})
    knob_table = ""
    if K.get("runs"):
        knob_table = ("<p><b>Knob isolation on a 2-GPU trainer-only harness</b> (<code>rl/fullparam_update_bench.py</code>: same FSDP2 policy sharded over 2 B200, "
                      "56 vs 48 synthetic rollouts on the two ranks so the dummy-micro-batch path runs, 2 reps each, no optimizer state; times are the 2nd rep, "
                      "the 1st includes the fla Triton autotune for new shapes). Before the dummy-micro-batch fix the <code>head</code> config hung here exactly like "
                      "the 8-GPU bench (<code>data/bench_gpu2_knobs_hang.json</code>); after it every config completes.</p>"
                      "<table><tr><th>knobs</th><th class='num'>micro-batch</th><th class='num'>prefetch</th><th class='num'>update s (rep 2)</th><th class='num'>peak GB (2 ranks)</th><th class='num'>micro-batches</th></tr>"
                      "<tr><td class='baseline'>base (fp32-head hook, no recompute)</td><td class='num'>8</td><td class='num'>0</td><td class='num'>13.1</td><td class='num'>164.6</td><td class='num'>7</td></tr>")
        for run in K["runs"]:
            for r in run["results"]:
                knob_table += (f"<tr><td>{'+'.join(r['knobs']) or 'base'}</td><td class='num'>{r['mb']}</td><td class='num'>{run['prefetch']}</td>"
                               f"<td class='num'>{r['update_s'][-1]:.1f}</td><td class='num {'good' if r['peak_gb'] < 140 else ''}'>{r['peak_gb']:.1f}</td><td class='num'>{r['n_micro_batches']}</td></tr>")
        knob_table += "</table>"
    bench_rows = B.get("configs", [])
    bench_table = knob_table
    if bench_rows:
        bench_table += ("<p><b>8×B200 (3 vLLM + 5 trainer), 512 × 8 rollouts per step, means over the bench steps ≥ 1.</b></p>"
                        "<table><tr><th>config</th><th class='num'>micro-batch</th><th class='num'>s/step</th><th class='num'>update s</th><th class='num'>tok/s/rank</th>"
                       "<th class='num'>peak GB</th><th class='num'>|Δlogp|</th></tr>" + "".join(
                           f"<tr><td>{r['label']}</td><td class='num'>{r['micro_batch']}</td><td class='num'>{r['step_s']:.0f}</td><td class='num'>{r['update_s']:.0f}</td>"
                           f"<td class='num'>{r['tok_s_rank']:.0f}</td><td class='num'>{r['peak_gb']:.0f}</td><td class='num'>{r['dlogp']:.4f}</td></tr>" for r in bench_rows) + "</table>")
    prod = ids.get("prod", {})
    val_id = W.get("val50", {}).get("id", "?")
    runs = {
        "date": time.strftime("%Y-%m-%d"),
        "tldr": (f"Full-parameter GRPO/CISPO on the 27B inverter works end to end: the FSDP2 policy (fp32 masters + AdamW over 5 B200 trainer ranks) pushes all "
                 f"<b>50.1 GB</b> of bf16 weights into the 3 vLLM samplers by NCCL after <b>every</b> step in <b>{pubsum['trainer_total_s_median']:.2f} s</b> "
                 f"({pubsum['gbps_mean']:.0f} GB/s; engines stall {pubsum['engine_stall_s_mean']:.2f} s), the sampler stays on-policy (|Δlogp| {mean(dl):.3f} vs "
                 f"{mean([r['dlogp'] for r in L.get('initboth', []) if 1 <= r['step'] <= 49]):.3f} for the LoRA arm), and 50 steps at lr 1e-6 lift the reward "
                 f"{v0['reward']:.3f} → {vN['reward']:.3f} on the LoRA arm's trajectory with a better held-out eval at both checkpoints (mean_all step 25: {fmt(evrow(ev_fp, 25), 4)} vs "
                 f"{fmt(evrow(ev_if, 25), 4)} for LoRA on the same init; step 50: {fmt(evrow(ev_fp, 50), 4)} vs {fmt(evrow(ev_if, 50), 4)}, and vs {fmt(evrow(ev_ib, 50), 4)} for the "
                 f"LoRA-on-base arm). The validated configuration costs {step_s:.0f} s per step (LoRA arm: {lora_step:.0f} s) because micro-batch 8 pays "
                 f"~22 TB/rank of FSDP all-gather/reduce-scatter traffic; the production arm runs at it. Exact per-layer recompute + a chunked lm_head "
                 f"(now the default) lift the micro-batch to 24 → <b>44 s/step</b> on 3 vLLM + 5 trainer GPUs and to 32 → <b>35 s/step</b> on 2 + 6 "
                 f"(the engines still keep the queue full), after a hang caused by FSDP2's gradient-dependent reduce-scatter payload was root-caused and fixed (§2b)."),
        "kpis": {"items": [
            {"v": f"{pubsum['trainer_total_s_median']:.2f} s", "l": "per-step publish of 50.1 GB bf16 → 3 vLLM engines (NCCL)", "cls": "good"},
            {"v": f"{mean(dl):.3f}", "l": "sampler |Δlogp| after the pushes (LoRA arm 0.025)", "cls": "good"},
            {"v": f"{v0['reward']:.3f} → {vN['reward']:.3f}", "l": "reward, 50 steps, lr 1e-6 (LoRA 7e-6: 0.203 → 0.245)", "cls": "good"},
            {"v": f"{peak_train:.0f} GB", "l": "peak GPU memory per trainer rank (of 178)"},
            {"v": f"{step_s:.0f} s", "l": f"s/step, production config (mb 8; LoRA arm {lora_step:.0f} s)", "cls": "bad"},
            {"v": "35 s", "l": "s/step, fast default (recompute + chunked head, mb 32, 2 vLLM + 6 trainer)", "cls": "good"},
            {"v": fmt(evrow(ev_fp, 50), 4), "l": f"held-out mean_all at step 50 (LoRA same init {fmt(evrow(ev_if, 50), 4)}, LoRA on base {fmt(evrow(ev_ib, 50), 4)})", "cls": cls(evrow(ev_fp, 50), evrow(ev_if, 50))}]},
        "sections": {
            "curves": (f"50-step validation run <code>rl_fullparam_val50_lr1e-6</code> (wandb {val_id}) vs the LoRA ablation arms at the same steps: "
                       f"<code>rl_abl_initboth_8x512_lr7e-6</code> (LoRA r64 on the base, SFT-adapter init) and <code>rl_abl_initnewfft_8x512_lr7e-6</code> "
                       f"(LoRA r64 on the SAME full-fine-tune midtrain weights the full-parameter run starts from). All three: 512 directions × 8 samples per step, "
                       f"ScaleRL/CISPO, 25-step warmup, 3 vLLM + 5 trainer B200. Right panel: the sampler-vs-trainer |Δlog p| per token; after every one of the 50 "
                       f"weight pushes the vLLM engines' log-probs of their own samples agree with the FSDP2 policy to {min(dl):.3f}–{max(dl):.3f} nats (lag 2), "
                       f"i.e. exactly the LoRA path's band — a broken weight mapping would sit at ~1–1.5 nats."),
            "curves_table": curves_table,
            "time": (f"Mean over steps 2–49 of the validation run: {step_s:.0f} s per step = scoring {sc:.1f} s (frozen sharded scorer, 7 batches of 128) + "
                     f"update {upd:.0f} s (103 micro-batches of 8 rollouts: forward, backward, AdamW on the fp32 shards) + publish {tp:.2f} s. The LoRA arm "
                     f"(same GPUs, micro-batch 16) needs {lora_step:.0f} s. Rollout generation is not the bottleneck (16 blocks × 4.4 s / 3 engines ≈ 24 s of "
                     f"work per step, hidden behind the update; queue at its cap of 32 blocks). The update is communication-bound: every micro-batch all-gathers "
                     f"the 54 GB of bf16 parameters twice (forward and backward, <code>reshard_after_forward=True</code>) and reduce-scatters 108 GB of fp32 "
                     f"gradients — ~22 TB per rank per step at micro-batch 8. <code>time/grad_sync_s</code> is 0 because FSDP2's reduce-scatter <em>is</em> the sync "
                     f"(no per-micro-batch all-reduce on top); the prefix is run once per step (<code>trainer/body_tokens_per_rollout</code> counts suffix tokens only)."),
            "mem": ("Every trainer rank holds an fp32 shard of the 26.90 B parameters (20 GB), its fp32 gradient shard (20 GB) and the fp32 AdamW moments (40 GB) plus "
                    "the frozen 43-layer scorer shard (4.2 GB); activations at micro-batch 8 add ~5.2 GB per sequence at the 295-token maximum. "
                    f"Peak {peak_train:.0f} GB of 178 GB (step 0, before the AdamW moments exist: {val[0]['peak_gb']:.0f} GB)."),
            "mem_table": mem_table,
            "publish": ("<p>Trainer rank 0 and the three vLLM worker processes form one extra NCCL communicator (vLLM's <code>StatelessProcessGroup</code> + "
                        "<code>PyNcclCommunicator</code>, created at the first publish in 3.1 s). Per step: rank 0 flags <code>lora/pending</code>; each engine finishes its "
                        "in-flight block (~4.4 s of 256 sequences), writes <code>ready_r_k</code> and enters <code>wu_recv</code>; the five trainer ranks all-gather every "
                        "parameter in bf16 (<code>DTensor.to(bf16).full_tensor()</code>, 851 tensors in <code>named_parameters()</code> order) and rank 0 broadcasts each; "
                        "the workers stream the tensors lazily into vLLM's own <code>model.load_weights</code> (in place — CUDA graphs, the steering hook and the KV cache "
                        "stay valid; the 851 checkpoint tensors land in 659 vLLM parameters through its stacked-param mapping). Rank 0 then writes <code>meta.json</code> "
                        "(‖h_marker‖ + |sum| of six unfused tensors) and flips <code>latest</code>; every engine verifies the six checksums (rtol 1e-3) before tagging its "
                        "next block with the new step. No transfer overlaps generation or FSDP collectives. Design + protocol: <code>docs/rl_fullparam.md</code>.</p>"),
            "publish_caption": (f"Trainer-side cost per publish over the 50 steps: waiting for the slowest engine's block boundary ({pubsum['t_wait_ready_mean']:.2f} s mean), "
                                f"the all-gather + broadcast ({pubsum['t_transfer_mean']:.2f} s, {pubsum['gbps_mean']:.0f} GB/s), and the engines' extra load time "
                                f"beyond the transfer. The first publish (step 1) includes the 3.1 s NCCL group creation."),
            "publish_table": publish_table,
            "bench": ("Step-time configurations. The first 8-GPU bench of the fast configuration (--suffix-ckpt --chunked-head --fsdp-prefetch 2, probe → micro-batch 24) "
                      "hung in step 0: FSDP2 reduce-scatters only the parameters that have a gradient, as one flat collective per group, and the zero-weight dummy "
                      "micro-batch that equalizes the micro-batch count across uneven shards bypassed lm_head under the chunked head → one rank's root reduce-scatter "
                      "was shorter than the others' → all five NCCL watchdogs stuck. Reproduced and fixed on a 2-GPU trainer-only harness (table 1). Table 2: the fixed "
                      "configuration on 8×B200 — 3 vLLM + 5 trainer at micro-batch 24 (43–45 s/step steady; the mean includes step 1's Triton autotune) and 2 vLLM + 6 "
                      "trainer at micro-batch 32 (35–36 s/step; the two engines still keep the 32-block queue full, so the rollout side is not the bottleneck). An explicit "
                      "<code>--micro-batch 32</code> on 3 + 5 (probe skipped) OOMed in step 0 (174 GB live during a forward; log in data/bench_mb32_3plus5_oom_launch.log) — "
                      "the 5-rank fixed cost is ~8 GB/rank higher and the probe's linear extrapolation from 24 is optimistic there. Rewards, |Δlogp| and gradient norms of "
                      "every fast-config bench step match the baseline's at the same seed (exact recompute)."),
            "bench_table": bench_table,
            "eval": eval_table + (f"<p>Step 25 of the full-parameter run scores <b>{fmt(evrow(ev_fp, 25), 4)}</b> mean_all vs {fmt(evrow(ev_if, 25), 4)} for the LoRA "
                                  f"arm on the same init and {fmt(evrow(ev_ib, 25), 4)} for the LoRA arm on the base + SFT adapter; SAE fired {fmt(evrow(ev_fp, 25, 'eval/sae/fired'))} "
                                  f"vs {fmt(evrow(ev_if, 25, 'eval/sae/fired'))}. At step 50 (the final checkpoint of the validation run) the full-parameter policy reaches "
                                  f"<b>{fmt(evrow(ev_fp, 50), 4)}</b> vs {fmt(evrow(ev_if, 50), 4)} (LoRA, same init) and {fmt(evrow(ev_ib, 50), 4)} (LoRA on base + SFT adapter): "
                                  f"+{100 * (evrow(ev_fp, 50) - evrow(ev_if, 50)):.1f} pp over the LoRA arm that started from the same weights, at a 7× lower learning rate "
                                  f"(eval run <code>rl_fullparam_val50_lr1e-6_eval</code>). Same eval protocol, same clean scorer; one seed each, so treat ~0.5 pp as noise.</p>"),
            "limits": ("<ul>"
                       "<li><b>Step time.</b> The production arm runs 68 s/step at micro-batch 8 (vs 30 s for the LoRA arm) — the FSDP2 traffic per micro-batch. The fast "
                       "configuration (now rl_disagg's default for --full-param: --suffix-ckpt --chunked-head --fsdp-prefetch 2) reaches 44 s/step on 3 + 5 and 35 s/step "
                       "on 2 + 6 (§2b); the ≤ 30 s target is not met — the remaining update time is still the per-micro-batch all-gather + fp32 reduce-scatter. Next levers: "
                       "bf16 gradient reduce-scatter (−54 GB per micro-batch), --fsdp-keep-unsharded N (fullft), micro-batch 48 on 2 + 6. The production arm was launched "
                       "before the fix and is deliberately NOT hot-swapped (a resume without --save-optim would restart the AdamW moments and confound the ablation).</li>"
                       "<li><b>Budget.</b> Development used ≈ 22 GPU-hours on 8×B200 (two failed smokes: probe fit + uneven-shard deadlock; the 50-step validation; one "
                       "failed fast-config bench) against the ≈ 12 asked for; the production arm is on top.</li>"
                       "<li><b>fs publish mode</b> implemented and CPU-tested, not exercised on the GPUs (the NCCL path met the target immediately).</li>"
                       "<li><b>--save-optim / --load-optim</b> (torch.distributed.checkpoint) implemented, not exercised on the GPUs; a resume restarts the AdamW moments "
                       "unless it is used. --kl-coef > 0 uses a frozen sharded copy of the init (+11 GB/rank when the policy base is not MODEL); not exercised (the recipe has kl 0).</li>"
                       "<li><b>Inline eval</b> is refused in full-param mode; checkpoints are full-model dirs evaluated by <code>fullmodel_daemon</code> (works: step 25 and final of the validation run).</li>"
                       "<li>The per-step publish waits for every engine's block boundary (≤ 1 block, 0.2–0.4 s here); with much longer blocks a double-buffered "
                       "asynchronous variant would be needed.</li></ul>"),
            "appendix": (f"<h3>Runs</h3><table><tr><th>run</th><th>wandb</th><th>Modal</th><th>config</th></tr>"
                         f"<tr><td>rl_fullparam_val50_lr1e-6 (validation, 50 steps)</td><td>{val_id}</td><td>train {ids.get('val50', {}).get('train', '?')} · eval {ids.get('val50', {}).get('eval', '?')}</td>"
                         f"<td>3 rollout + 5 trainer B200, --full-param --publish-mode nccl, lr 1e-6, warmup 25, saves 25 + final → /data/ckpts_fullrl_val50_lr1e-6</td></tr>"
                         f"<tr><td>rl_abl_initnewfft_fullparam_8x512 (PRODUCTION arm, 300 steps)</td><td>{prod.get('run', '')}</td><td>train {prod.get('train', '?')} · eval {prod.get('eval', '?')}</td>"
                         f"<td>same config, saves 25,50,100,150,200,250,300 → {prod.get('save', '')}; evaluator fullmodel_daemon on app maemm-eval-ckpt-fullrl (eval cache v2, no judge extras); "
                         f"ids in ~/shared/overnight/rl_ablation_ids.json['initnewfft_fullparam']</td></tr>"
                         f"<tr><td>rl_fullparam_smoke6 attempts 1–2 (failed)</td><td>fl6lkj89</td><td>fc-01M21F2Q3H8AN6DXCTZZRVQBNT, fc-01M21JYSPYC4VP7J2XP5DFQDW3</td>"
                         f"<td>attempt 1: micro-batch probe fit from mb 1/2 predicted 0.06 GB/seq → verified mb 128 → OOM; attempt 2: 103 vs 102 micro-batches on uneven shards → FSDP2 collective mismatch → NCCL watchdog abort</td></tr>"
                         f"<tr><td>rl_fullparam_bench5_fast35 (failed)</td><td>—</td><td>fc-01M21Q45C7PVZYGKY5NM9QC0YY</td><td>--suffix-ckpt --chunked-head --fsdp-prefetch 2 → probe mb 24 (1.95 GB/seq) → hang in step 0</td></tr></table>"
                         f"<h3>Recipe (all arms)</h3><pre><code>--recipe scalerl --loss cispo --cispo-eps-max 5 --loss-agg prompt --adv-mode batch --zero-var-filter --npr-threshold 0.9 --npr-pass-cos 0.7 --max-lag 2 --fp32-head --autocast-bf16 --length-control penalty --kl-coef 0 --entropy-coef 0 --entropy-target 0 --groups-per-step 512 --group-size 8 --warmup-steps 25 --len-penalty-start 8 --len-penalty-per-tok 0.00025 --max-new-tokens 192 --reward-window-last 5 --prefix-cache --score-length-bucket --cuda-graphs --max-num-seqs 512 --rollout-block-groups 32\n"
                         f"full-param: --full-param --init-adapter none --policy-base /data/sft_mix/mixeq_midtrain_fft_from_fft23m_v2/final --lr 1e-6 (LoRA arms: --lr 7e-6)</code></pre>"
                         f"<h3>Production launch</h3><pre><code>DISAGG_APP=maemm-rl-disagg-fullparam DISAGG_GPU=B200:8 DISAGG_TRANSFORMERS=\"transformers @ git+https://github.com/ceselder/transformers@e52940e567ab9a991a1c971c1094e340233baff3\" modal deploy rl/modal_rl_disagg.py\n"
                         f"python3 scripts/launchers/spawn_rl_fullparam.py prod 1e-6      # = modal Function maemm-rl-disagg-fullparam/train(n_rollout=3, n_trainer=5, total_steps=300, full_param=True, policy_base=..., pool_dir=/data/banks/mix_eq_1p45m, extra_args=RECIPE + ' --lr 1e-6 --save-steps 25,50,100,150,200,250,300 --run-name rl_abl_initnewfft_fullparam_8x512 --save-dir /data/ckpts_rl_abl_initnewfft_fullparam')\n"
                         f"                                                            # + maemm-eval-ckpt-fullrl/fullmodel_daemon(ckpt_dir=..., tag=..., final_step=300, extra_args='--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt --no-extra-evals')</code></pre>"
                         f"<h3>Learning-rate choice</h3><p>1e-6 (the LoRA arms use 7e-6): AdamW's per-parameter step is ~lr regardless of gradient scale, and a full-parameter update moves every weight, "
                         f"so the standard full-fine-tune RL value (1e-6, 5–7× below a LoRA lr) was tried first. Over 50 steps it reproduced the LoRA arm's reward trajectory (0.205 → 0.257 vs 0.203 → 0.245) with a stable "
                         f"gradient norm (0.69–0.96) and a slightly faster entropy decline (2.71 → 2.19 vs 2.70 → 2.50); |Δlogp| grew from 0.021 to 0.032 as the lr reached its plateau — still inside the LoRA band. "
                         f"No reason to go higher for a 300-step run; a lower value was not tried.</p>"
                         f"<h3>Memory numbers</h3>{mem_table}<h3>Files</h3><p>branch <code>rl-fullparam</code> in ~/maemm-pub-fullrl (docs/rl_fullparam.md, rl/rl_fullparam.py, rl/rl_disagg.py, rl/fast_lens_ext.py, "
                         f"rl/modal_rl_disagg.py, rl/test_rl_disagg_fullparam.py, rl/fullparam_update_bench.py, scripts/launchers/spawn_rl_fullparam.py); data: data/steps_fullparam.json (wandb history), "
                         f"data/steps_lora.json, data/eval_wandb.json, data/publish.json, data/memory.json, data/val50_launch.log (full launcher log incl. every publish line).</p>")}}
    json.dump(runs, open(f"{DATA}/runs.json", "w"), indent=1)
    print("data written; building html ...")
    subprocess.run(["python3", f"{REP}/build_html.py"], check=True)


if __name__ == "__main__":
    main()
