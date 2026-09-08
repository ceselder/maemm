import modal
e=modal.Function.from_name("maemm-eval-ckpt","daemon").spawn(ckpt_dir="/data/ckpts_disagg_scalerl", tag="disagg_scalerl", rl_run_id="x2s43el1", wandb_name="rl_everything_8x256_disagg_scalerl_lr7e-6_eval", final_step=1000)
print(e.object_id)
