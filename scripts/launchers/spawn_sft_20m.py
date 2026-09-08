"""Spawn the scaled SFT on the DEPLOYED maemm-sft-8xb200 app (fork image). Run after /data/banks/realact_short_20m/build_stats.json exists."""
import sys
import modal
run_name = sys.argv[1] if len(sys.argv) > 1 else "realact20m_prefix_lr1e-4"
extra = ("--prefix-cache --grad-ckpt 0 --autocast-bf16 --log-steps 50 "
         "--save-examples 250000,500000,1000000,2000000,5000000,10000000")
c = modal.Function.from_name("maemm-sft-8xb200", "train").spawn(
    run_name=run_name, data_dir="/data/banks/realact_short_20m_all", n_ckpts=40, epochs=1,
    batch_size=64, lr=1e-4, max_seq=160, backend="nccl", extra_args=extra)
print(c.object_id)
