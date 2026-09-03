"""Add within-group activation-orthogonal diversity bonus to rl_hf.py. Opt-in via --div-coef (0=off,
default -> byte-identical behavior to before). Idempotent."""
f = "/root/pmx/RL/rl_hf.py"
s = open(f).read()
if "--div-coef" in s:
    print("already patched"); raise SystemExit

edits = [
    # 1a: score signature
    ("def score(texts, dirs_rep, actor, tok, device, a, with_fluency=False):",
     "def score(texts, dirs_rep, actor, tok, device, a, with_fluency=False, return_act=False):"),
    # 1b: meanact accumulator (anchor on the following logp line to disambiguate)
    ("    r = torch.zeros(len(texts))\n    logp = torch.full((len(texts),), -20.0) if with_fluency else None\n",
     "    r = torch.zeros(len(texts))\n    meanact = torch.zeros(len(texts), D_MODEL)\n"
     "    logp = torch.full((len(texts),), -20.0) if with_fluency else None\n"),
    # 1c: accumulate mean-pooled clean act over kept tokens
    ("            r[idxs] = torch.where(has, best, 0).cpu()\n",
     "            r[idxs] = torch.where(has, best, 0).cpu()\n"
     "            _msum = (h * keep.unsqueeze(-1)).sum(1); _mcnt = keep.sum(1, keepdim=True).clamp(min=1)\n"
     "            meanact[idxs] = (_msum / _mcnt).float().cpu()\n"),
    # 1d: return meanact when asked
    ("    return (r, logp, dis) if with_fluency else r\n",
     "    if return_act:\n"
     "        return (r, logp, dis, meanact) if with_fluency else (r, meanact)\n"
     "    return (r, logp, dis) if with_fluency else r\n"),
    # 2: arg
    ("    ap.add_argument(\"--len-penalty-per-tok\", type=float, default=cfg.len_penalty_per_tok)\n",
     "    ap.add_argument(\"--len-penalty-per-tok\", type=float, default=cfg.len_penalty_per_tok)\n"
     "    ap.add_argument(\"--div-coef\", type=float, default=0.0,\n"
     "                    help=\"within-group activation-orthogonal diversity bonus (0=off)\")\n"),
    # 3: score call + unpack meanact
    ("        scored = score(texts, dirs_rep, actor, tok, device, a, with_fluency=use_fluency)\n"
     "        if use_fluency:\n"
     "            r, flu, dis = scored\n"
     "        else:\n"
     "            r = scored\n",
     "        scored = score(texts, dirs_rep, actor, tok, device, a, with_fluency=use_fluency, return_act=(a.div_coef > 0))\n"
     "        meanact = None\n"
     "        if use_fluency and a.div_coef > 0:\n"
     "            r, flu, dis, meanact = scored\n"
     "        elif use_fluency:\n"
     "            r, flu, dis = scored\n"
     "        elif a.div_coef > 0:\n"
     "            r, meanact = scored\n"
     "        else:\n"
     "            r = scored\n"),
    # 4: diversity bonus after len penalty
    ("        if a.len_penalty_start is not None:\n"
     "            over = torch.tensor([max(0, len(g) - a.len_penalty_start) for g in gen_ids],\n"
     "                                dtype=torch.float32)\n"
     "            r = r - a.len_penalty_per_tok * over\n",
     "        if a.len_penalty_start is not None:\n"
     "            over = torch.tensor([max(0, len(g) - a.len_penalty_start) for g in gen_ids],\n"
     "                                dtype=torch.float32)\n"
     "            r = r - a.len_penalty_per_tok * over\n"
     "        div_mean = 0.0\n"
     "        if a.div_coef > 0 and meanact is not None:\n"
     "            ma = meanact.to(device).view(B, G, D_MODEL)\n"
     "            vhat = dirs.to(device)\n"
     "            dots = torch.einsum(\"bgd,bd->bg\", ma, vhat)\n"
     "            perp = F.normalize(ma - dots.unsqueeze(-1) * vhat.unsqueeze(1), dim=-1)\n"
     "            sim = torch.einsum(\"bgd,bhd->bgh\", perp, perp)\n"
     "            div = (1.0 - (sim.sum(2) - 1.0) / max(G - 1, 1)).flatten().cpu()\n"
     "            r = r + a.div_coef * div\n"
     "            div_mean = div.mean().item()\n"),
    # 5: log
    ("               \"reward/max\": raw_r.max().item(), \"reward/shaped_mean\": r.mean().item(),\n",
     "               \"reward/max\": raw_r.max().item(), \"reward/shaped_mean\": r.mean().item(),\n"
     "               \"reward/div_mean\": div_mean,\n"),
]
for old, new in edits:
    c = s.count(old)
    assert c == 1, f"anchor count {c} for: {old[:70]!r}"
    s = s.replace(old, new)
open(f, "w").write(s)
print("patched rl_hf.py with --div-coef diversity bonus")
