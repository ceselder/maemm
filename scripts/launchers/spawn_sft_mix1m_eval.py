import modal
print(modal.Function.from_name("maemm-eval-ckpt", "daemon").spawn(ckpt_dir="/data/sft_mix/mix1m_from_realact23m", tag="sft_mix1m_from_realact23m", rl_run_id="", wandb_name="mix1m_from_realact23m_eval", final_step=5000).object_id)
