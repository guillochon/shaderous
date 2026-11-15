import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train
NAMES = ["he4","c12","o16","ne20","mg24","si28","s32","ar36","ca40","ti44","cr48","fe52","ni56"]
m = train.load_model("model_dagger_30.pt")
tt = np.load(train.HERE / "targets_test.npz"); t = tt["t"]
s = np.where(tt["kind"] == 1)[0]
rho, X, T = tt["rho"][s], tt["X"][s], tt["T"][s]
# the worst trajectory from regime.py: rho 6.4e7 T0 3.69e9, si 0.76
i = int(np.argmin(np.abs(np.log10(rho / 6.4e7)) + np.abs(np.log10(T[:, 0] / 3.69e9)) * 10))
print(f"rho {rho[i]:.2e} T0 {T[i,0]:.3e}")
for tc in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1):
    k = int(np.argmin(np.abs(t - tc)))
    dt = t[k + 1] - t[k]
    Dr = X[i, k + 1] - X[i, k]
    with torch.no_grad():
        Dm = m(torch.tensor([np.log10(rho[i])], dtype=torch.float32, device=train.dev),
               torch.tensor([np.log10(T[i, k])], dtype=torch.float32, device=train.dev),
               torch.tensor([np.log10(dt)], dtype=torch.float32, device=train.dev),
               torch.tensor(X[i, k][None], dtype=torch.float32, device=train.dev))[0].cpu().numpy()
    top = np.argsort(-np.abs(Dr) - np.abs(Dm))[:5]
    print(f"t {t[k]:.2e} dt {dt:.1e} T {T[i,k]:.3e} ni56 {X[i,k,12]:.4f}: " + ", ".join(f"{NAMES[j]} ref {Dr[j]:+.1e} model {Dm[j]:+.1e}" for j in top))
