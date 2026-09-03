"""Refine the diversity mean-pool: pool only tokens UP TO & INCLUDING the peak (max-activation) token,
not all tokens. Prevents gaming the bonus with varied filler AFTER the peak (which the max-over-token
reward ignores). Idempotent."""
f = "/root/pmx/RL/rl_hf.py"
s = open(f).read()
if "_pstar" in s:
    print("already refined"); raise SystemExit
old = ("            _msum = (h * keep.unsqueeze(-1)).sum(1); _mcnt = keep.sum(1, keepdim=True).clamp(min=1)\n"
       "            meanact[idxs] = (_msum / _mcnt).float().cpu()\n")
new = ("            _filled = proj.masked_fill(~keep, torch.finfo(proj.dtype).min)\n"
       "            _pstar = _filled.argmax(1)                       # peak (max-activation) token per rollout\n"
       "            _tt = torch.arange(h.shape[1], device=h.device)\n"
       "            _pmask = keep & (_tt.unsqueeze(0) <= _pstar.unsqueeze(1))   # kept tokens up to & incl peak (excl after)\n"
       "            _msum = (h * _pmask.unsqueeze(-1)).sum(1); _mcnt = _pmask.sum(1, keepdim=True).clamp(min=1)\n"
       "            meanact[idxs] = (_msum / _mcnt).float().cpu()\n")
assert s.count(old) == 1, "anchor count %d" % s.count(old)
open(f, "w").write(s.replace(old, new))
print("refined diversity pooling -> prefix-up-to-peak")
