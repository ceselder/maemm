"""ONE evaluator per RL run, on H200s (app maemm-eval-ckpt-h200)."""
import modal
f = modal.Function.from_name("maemm-eval-ckpt-h200", "daemon")
a = f.spawn(ckpt_dir="/data/ckpts_rl_A_randctx", tag="rl_A_randctx", rl_run_id="", wandb_name="rl_A_randctx_from_realact23m_eval", final_step=400)
b = f.spawn(ckpt_dir="/data/ckpts_rl_B_randctx_probes", tag="rl_B_randctx_probes", rl_run_id="", wandb_name="rl_B_randctx_probes_from_rp500k_eval", final_step=400)
print("evalA", a.object_id); print("evalB", b.object_id)
