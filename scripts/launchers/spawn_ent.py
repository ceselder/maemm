import modal
modal.FunctionCall.from_id("fc-01M1JT3Y2FBGHVQ83ZQW2WFKRM").cancel(terminate_containers=True); print("cancelled pending w25/kl0.01 call")
f=modal.Function.from_name("maemm-rl-disagg","train")
c=f.spawn(n_rollout=2, n_trainer=6, total_steps=1000, extra_args="--cuda-graphs --max-num-seqs 512 --rollout-block-groups 64 --groups-per-step 256 --warmup-steps 25 --kl-coef 0 --entropy-coef 0.005 --entropy-target 2.0 --save-dir /data/ckpts_last5_disagg_2x6 --run-name rl_everything_8x256_disagg_entropy2.0_last5win --transcript-every 5")
print(c.object_id)
