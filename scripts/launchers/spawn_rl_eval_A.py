import modal
f = modal.Function.from_name("maemm-eval-ckpt", "daemon")
print(f.spawn(ckpt_dir="/data/ckpts_rl_A_randctx", tag="rl_A_randctx", rl_run_id="", wandb_name="rl_A_randctx_from_realact23m_eval", final_step=400).object_id)
