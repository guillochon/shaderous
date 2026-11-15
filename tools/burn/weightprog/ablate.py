"""Direction ablation in the last hidden layer (weight-space programming, technique 5).
Usage: ablate.py base.pt out_prefix alpha [alpha ...]
Region states: the model's own rollouts of the held-out Si/S trajectories between 3 ms and
0.1 s (where it turns Si into S ~40x too fast). General states: a sample of training states.
v = mean(h_region) - mean(h_general), with its projection onto the top-k principal components
of the general activations removed (a direction general states hardly use), normalized.
Edit: W' = W - alpha (W v) v^T, b' = b + alpha (W v)(v . mean(h_general))."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train

base, prefix, alphas = sys.argv[1], sys.argv[2], [float(a) for a in sys.argv[3:]]
m = train.load_model(base)
net = m.net
def hidden(r, T, dt, X):
    x = train.features(r, T, dt, X, m.rates)
    return net[:-1](x)                    # after the last SiLU

tt = np.load(train.HERE / "targets_test.npz"); t = tt["t"]
s = tt["kind"] == 1
rho, X, T = tt["rho"][s], tt["X"][s], tt["T"][s]
H = []
with torch.no_grad():
    Xc, Tc, done = X[:, 0], T[:, 0], 0.0
    for tc in (0.003, 0.01, 0.03, 0.1):
        k = int(np.argmin(np.abs(t - tc)))
        steps = max(1, int(round((t[k] - done) / 1e-3)))
        Xm, Tm, _ = train.roll(m, rho, Xc, Tc, np.full(len(rho), (t[k] - done) / steps), steps)
        Xc, Tc, done = Xm.cpu().numpy(), Tm.cpu().numpy(), t[k]
        si = Xc[:, 5] > 0.2                           # still Si-rich (before the runaway ends)
        r_ = torch.tensor(np.log10(rho[si]), dtype=torch.float32, device=train.dev)
        H.append(hidden(r_, torch.log10(Tm[si]), torch.full_like(r_, -3.0), Xm[si]))
    Hr = torch.cat(H)
    gen = train.Pairs(*train.load({"fuel.npz", "mix.npz", "tracehe.npz", "targets.npz"} | {f"dagger_{i}.npz" for i in range(20, 31)}))
    (r, Tl, dt, X0), _, _ = gen.sample(60000, 0.2)
    Hg = hidden(r, Tl, dt, X0)
print(f"{len(Hr)} region states, {len(Hg)} general")
mu_g = Hg.mean(0)
v = Hr.mean(0) - mu_g
U, S_, Vt = torch.linalg.svd(Hg - mu_g, full_matrices=False)
for kpc in (16, 32, 64):
    P = Vt[:kpc]
    w = v - P.T @ (P @ v)
    w = w / w.norm()
    sep = ((Hr - mu_g) @ w).mean() / ((Hg - mu_g) @ w).std()
    print(f"k {kpc}: region mean along v {((Hr - mu_g) @ w).mean():.3f}, general std {((Hg - mu_g) @ w).std():.4f}, separation {sep:.1f} sd")
P = Vt[:32]
w = v - P.T @ (P @ v); w = w / w.norm()
W, b = net[-1].weight.data, net[-1].bias.data
for a in alphas:
    ck = torch.load(train.HERE / base)
    st = ck["state"]
    key = [k for k in st if k.endswith(".weight")][-1]
    bk = key.replace("weight", "bias")
    Wv = W @ w
    st[key] = W - a * torch.outer(Wv, w)
    st[bk] = b + a * Wv * (w @ mu_g)
    torch.save(ck, train.HERE / f"{prefix}{a:g}.pt")
    print("wrote", f"{prefix}{a:g}.pt")
