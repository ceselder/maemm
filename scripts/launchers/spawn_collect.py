import modal
c = modal.Function.from_name("maemm-collect-bank", "collect").spawn(n_examples=20_000_000, out_name="realact_short_20m", batch=64, per_window=8, p_lo=8, p_hi=256, w_lo=8, w_hi=32, max_wins=4, seed=7)
print(c.object_id)
