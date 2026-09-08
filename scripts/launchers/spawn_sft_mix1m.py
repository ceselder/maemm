import modal
extra = ("--prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 20 "
         "--init-adapter /data/sft_mix/realact20m_prefix_lr1e-4/final")
c = modal.Function.from_name("maemm-sft-8xb200", "train").spawn(
    run_name="mix1m_from_realact23m", data_dir="/data/banks/mix_1m_v2", n_ckpts=8, epochs=1,
    batch_size=32, lr=1e-4, max_seq=160, backend="nccl", extra_args=extra)
print(c.object_id)
