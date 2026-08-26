"""Train a SASA block-sparse subspace featurizer on collected L42 activations (Qwen3.6-27B).

Group-top-k autoencoder: G groups of b dims; sparsity at the GROUP level (k groups active/token);
each group is a b-dim subspace. Inputs ZCA-whitened + unit-norm (the decisive 8B fix). Decoupled
weight decay on the factored dict = nuclear-norm rank adaptation (variational 1/2(||E||^2+||D||^2));
NO decoder renorm. Logs EV / mean eff-rank / dead-block count to wandb.

Saves: sasa.pt = {E,D,bias,G,b,d}, blocks_Q.pt = per-block orthonormal basis Q_j [d,r_j] (for projecting
activations through a subspace), + copies whitening. Q_j = orthonormal basis of the block's decoder columns.

Usage: CUDA_VISIBLE_DEVICES=1 python bsf/train_sasa.py --acts-dir bsf27b/acts --G 32768 --b 8 --k 32
"""
import argparse, os, json, numpy as np, torch, torch.nn as nn


class SASA(nn.Module):
    def __init__(self, d, G, b):
        super().__init__()
        self.d, self.G, self.b = d, G, b
        self.E = nn.Parameter(torch.randn(d, G * b) / d ** 0.5)
        self.D = nn.Parameter(torch.randn(G * b, d) / (G * b) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x, k):
        z = (x @ self.E).view(-1, self.G, self.b)          # [B,G,b]
        gn = z.norm(dim=-1)                                # [B,G] group activation
        idx = gn.topk(k, dim=-1).indices                   # [B,k]
        mask = torch.zeros_like(gn).scatter_(-1, idx, 1.0)
        zs = z * mask.unsqueeze(-1)
        xh = zs.reshape(-1, self.G * self.b) @ self.D + self.bias
        return xh, z, gn, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="/root/pmx/bsf27b/acts")
    ap.add_argument("--out-dir", default="/root/pmx/bsf27b/sasa")
    ap.add_argument("--G", type=int, default=32768)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--wandb", default="bsf-sasa-27b")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)

    meta = json.load(open(f"{a.acts_dir}/meta.json"))
    N, T, d = meta["n_seq"], meta["seq_len"], meta["d_model"]
    acts = np.memmap(f"{a.acts_dir}/acts.f16", dtype=np.float16, mode="r", shape=(N, T, d)).reshape(N * T, d)
    ntok = N * T
    mu = torch.from_numpy(np.load(f"{a.acts_dir}/whiten_mu.npy")).float().to(dev)
    zca = torch.from_numpy(np.load(f"{a.acts_dir}/whiten_zca.npy")).float().to(dev)

    def load_batch(rng):
        idx = rng.integers(0, ntok, size=a.batch)
        x = torch.from_numpy(np.asarray(acts[idx]).astype(np.float32)).to(dev)
        x = (x - mu) @ zca                                  # whiten
        return torch.nn.functional.normalize(x, dim=-1)     # unit-norm

    model = SASA(d, a.G, a.b).to(dev)
    # DECOUPLED weight decay (AdamW) on the factored dict E,D = nuclear-norm rank adaptation.
    # (Must NOT be added to the loss — that dwarfs recon and just crushes the weights → EV~0.)
    opt = torch.optim.AdamW([
        {"params": [model.E, model.D], "weight_decay": a.wd},
        {"params": [model.bias], "weight_decay": 0.0},
    ], lr=a.lr)
    rng = np.random.default_rng(a.seed)

    import wandb
    wandb.init(project=a.wandb, config=vars(a) | {"ntok": ntok})
    fired_ever = torch.zeros(a.G, device=dev)

    for step in range(a.steps + 1):
        x = load_batch(rng)
        xh, z, gn, mask = model(x, a.k)
        recon = ((x - xh) ** 2).sum(-1).mean()
        loss = recon                                        # WD is decoupled (in the optimizer), not here
        opt.zero_grad(); loss.backward(); opt.step()
        fired_ever += mask.sum(0).detach()

        if step % a.log_every == 0:
            with torch.no_grad():
                var = ((x - x.mean(0)) ** 2).sum(-1).mean()
                ev = 1 - recon / var
                # per-active-block participation-ratio eff-rank (tr(C)^2/||C||_F^2), no eig
                zsel = z * mask.unsqueeze(-1)                # [B,G,b]
                C = torch.einsum("bgi,bgj->gij", zsel, zsel)  # [G,b,b] block code 2nd moments
                tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)      # [G]
                fro2 = (C ** 2).sum((-2, -1))                          # [G]
                active = mask.sum(0) > 0
                effr = (tr[active] ** 2 / fro2[active].clamp(min=1e-12)).mean()
                dead = int((fired_ever == 0).sum())
                l0 = a.k
            wandb.log({"loss": loss.item(), "recon": recon.item(), "ev": ev.item(),
                       "eff_rank": effr.item(), "dead_blocks": dead, "l0": l0, "step": step})
            print(f"[sasa] step {step} loss {loss.item():.4f} EV {ev.item():.3f} effr {effr.item():.2f} dead {dead}", flush=True)

        if step and step % a.save_every == 0:
            torch.save({"E": model.E.detach().cpu(), "D": model.D.detach().cpu(),
                        "bias": model.bias.detach().cpu(), "G": a.G, "b": a.b, "d": d}, f"{a.out_dir}/sasa.pt")

    # ---- finalize: per-block orthonormal Q from decoder columns (for subspace projection) ----
    D = model.D.detach().view(a.G, a.b, d)                  # [G,b,d]
    Q = torch.zeros(a.G, a.b, d)
    for g in range(a.G):
        q, _ = torch.linalg.qr(D[g].T)                      # [d,b] orthonormal cols
        Q[g] = q.T
    torch.save({"E": model.E.detach().cpu(), "D": model.D.detach().cpu(), "bias": model.bias.detach().cpu(),
                "G": a.G, "b": a.b, "d": d}, f"{a.out_dir}/sasa.pt")
    torch.save({"Q": Q.cpu(), "fired_ever": fired_ever.cpu(), "G": a.G, "b": a.b, "d": d}, f"{a.out_dir}/blocks_Q.pt")
    for f in ["whiten_mu.npy", "whiten_zca.npy"]:
        os.system(f"cp {a.acts_dir}/{f} {a.out_dir}/{f}")
    json.dump({"G": a.G, "b": a.b, "k": a.k, "d": d, "alive": int((fired_ever > 0).sum())},
              open(f"{a.out_dir}/meta.json", "w"), indent=1)
    print(f"[sasa] DONE -> {a.out_dir} | alive blocks {int((fired_ever>0).sum())}/{a.G}", flush=True)


if __name__ == "__main__":
    main()
