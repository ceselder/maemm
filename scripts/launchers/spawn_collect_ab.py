import modal
A = modal.Function.from_name("maemm-collect-bank", "collect").spawn(n_examples=11_000_000, out_name="realact_short_20m", batch=64, per_window=8, p_lo=8, p_hi=256, w_lo=8, w_hi=32, max_wins=4, seed=7)
print("A", A.object_id)
B = modal.Function.from_name("maemm-collect-bank", "collect_b200").spawn(n_examples=9_000_000, out_name="realact_short_20m_b", batch=64, per_window=8, p_lo=8, p_hi=256, w_lo=8, w_hi=32, max_wins=4, seed=8, exclude_from="realact_short_20m")
print("B", B.object_id)
