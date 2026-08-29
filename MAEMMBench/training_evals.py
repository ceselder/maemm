"""In-training evals (greedy held-out + SAE greedy/best-of-N) — thin re-export.

The canonical implementations live in train/rl.py, deliberately: they run INSIDE the RL loop
(`--eval-every` / `--sae-eval-every`), share the trainer's vLLM engine + scoring code, and moving
them out would fork that logic. This module makes them importable from the benchmark package:

    from MAEMMBench.training_evals import greedy_eval, sae_eval, sae_score

- greedy_eval(llm, prompt_ids, marker, eval_dirs, eval_base, actor, tok, device, a)
    -> ({eval/greedy_act_mean, eval/greedy_act_max[, eval/greedy_beat_frac]}, texts)
  One greedy (T=0) rollout per reserved held-out bank direction, scored by the training scorer
  (clean base, max-over-kept-positions act . unit(dir)). Includes the "all rollouts identical"
  assert that catches steering silently not firing (the FLASHINFER failure mode).

- sae_eval(llm, prompt_ids, marker, sae_dirs, sae_feats, sae, dataset_max, actor, tok, device, a)
    -> ({eval/sae_act_mean, eval/sae_norm_act, eval/sae_beat_frac
         [, eval/sae_bo{N}_act_mean, eval/sae_bo{N}_norm_act, eval/sae_bo{N}_beat_frac]}, texts)
  Zero-shot held-out SAE features: inject each unit encoder column, greedy-decode, score the TRUE
  feature activation vs the feature's corpus peak; optional best-of-N sampled arm (--sae-eval-bo).

- sae_score(texts, feats, sae, actor, tok, device, a) -> [N] max-position post-ReLU encoder acts
  (clean base @READ_LAYER, standalone re-tokenization, 10x-median norm filter, pos-0 sink drop).

NOTE: importing this module imports train.rl (torch / transformers / peft / wandb at module
level) — do it on a box with the training deps installed.
"""
from train.rl import greedy_eval, sae_eval, sae_score  # noqa: F401
