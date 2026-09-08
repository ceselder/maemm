import modal
print(modal.Function.from_name("maemm-eval-ckpt", "daemon").spawn(ckpt_dir="/data/ckpts_rl_C_mix1m", tag="rl_C_mix1m", rl_run_id="", wandb_name="rl_C_mix1m_from_realact23m_mixsft_eval", final_step=400).object_id)
