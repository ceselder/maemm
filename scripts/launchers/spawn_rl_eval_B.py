import modal
f = modal.Function.from_name("maemm-eval-ckpt-h200", "daemon")
print(f.spawn(ckpt_dir="/data/ckpts_rl_B_randctx_probes", tag="rl_B_randctx_probes", rl_run_id="", wandb_name="rl_B_randctx_probes_from_rp500k_eval", final_step=400).object_id)
