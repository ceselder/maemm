import modal
print(modal.Function.from_name("maemm-eval-ckpt-mlp", "daemon").spawn(ckpt_dir="/data/ckpts_rl_E_mixmlp", tag="rl_E_mixmlp", rl_run_id="", wandb_name="rl_E_mixmlp_from_mixsft_eval", final_step=400, extra_args="--eval-cache /data/eval_universal_ho/eval_sets_heldout_v2.pt").object_id)
