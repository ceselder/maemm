"""ONE eval daemon for the SFT run's step_* checkpoints (eval/modal_eval_ckpt.py daemon)."""
import sys
import modal
run_name = sys.argv[1] if len(sys.argv) > 1 else "realact20m_prefix_lr1e-4"
c = modal.Function.from_name("maemm-eval-ckpt", "daemon").spawn(
    ckpt_dir=f"/data/sft_mix/{run_name}", tag=f"sft_{run_name}", rl_run_id="",
    wandb_name=f"{run_name}_eval", final_step=40000)   # ~20M / 512 eff batch ≈ 39k optimizer steps
print(c.object_id)
