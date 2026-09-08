import modal
C = modal.Function.from_name("maemm-collect-bank", "collect").spawn(n_examples=4_500_000, out_name="realact_short_20m_c", batch=64, per_window=8, p_lo=8, p_hi=256, w_lo=8, w_hi=32, max_wins=4, seed=9, exclude_from="realact_short_20m,realact_short_20m_b")
print(C.object_id)
