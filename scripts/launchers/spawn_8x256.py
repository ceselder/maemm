import modal
modal.FunctionCall.from_id("fc-01M1JSWATD1B9A5BYCQ6DD6F3S").cancel(terminate_containers=True); print("cancelled 8x128 run (step ~20, no ckpt yet)")
f=modal.Function.from_name("maemm-rl-disagg","train")
c=f.spawn(n_rollout=2, n_trainer=6, total_steps=1000, extra_args="--cuda-graphs --max-num-seqs 512 --rollout-block-groups 64 --groups-per-step 256 --save-dir /data/ckpts_last5_disagg_2x6 --run-name rl_everything_8x256_disagg_2x6_last5win --transcript-every 5")
print(c.object_id)
