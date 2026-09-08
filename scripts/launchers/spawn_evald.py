import modal
c=modal.Function.from_name("maemm-eval-ckpt","daemon").spawn(ckpt_dir="/data/ckpts_last5_disagg_2x6", tag="last5_disagg_2x6", rl_run_id="ouf0jv3t", wandb_name="rl_everything_8x256_disagg_entropy2.0_last5win_eval", final_step=1000)
print(c.object_id)
