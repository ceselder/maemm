import modal
print(modal.Function.from_name("maemm-eval-ckpt", "daemon").spawn(ckpt_dir="/data/ckpts_rl_D_mix1m_lr1e-5", tag="rl_D_mix1m_lr1e-5", rl_run_id="", wandb_name="rl_D_mix1m_lr1e-5_from_realact23m_mixsft_eval", final_step=400).object_id)
