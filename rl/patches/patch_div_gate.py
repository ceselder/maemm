f="/root/pmx/RL/rl_hf.py"; s=open(f).read()
if "_gmask" in s:
    print("already gate-masked")
else:
    old="            r = r + a.div_coef * div\n"
    new=("            _gmask = gate.float() if use_fluency else torch.ones_like(div)\n"
         "            r = r + a.div_coef * div * _gmask\n")
    assert s.count(old)==1, "anchor %d"%s.count(old)
    open(f,"w").write(s.replace(old,new)); print("gate-masked div bonus")
