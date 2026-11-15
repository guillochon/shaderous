import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train
tt = np.load(train.HERE / "targets_test.npz"); t = tt["t"]
s = tt["kind"] == 1
rho, X, T = tt["rho"][s], tt["X"][s], tt["T"][s]
def feats(k):
    return train.features(torch.tensor(np.log10(rho), dtype=torch.float32, device=train.dev),
                          torch.tensor(np.log10(T[:, k]), dtype=torch.float32, device=train.dev),
                          torch.full((len(rho),), -3.0, device=train.dev),
                          torch.tensor(X[:, k], dtype=torch.float32, device=train.dev), "full").cpu().numpy()
for tc in (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0):
    k = int(np.argmin(np.abs(t - tc)))
    f = feats(k)
    d = (f[:, 47] - f[:, 48]) * 20          # log10 fwd - log10 rev, Si28 + a <-> S32
    d2 = (f[:, 49] - f[:, 50]) * 20         # S32 + a <-> Ar36
    print(f"t {t[k]:.1e}: log10(fwd/rev) Si->S p10/50/90 {np.quantile(d,0.1):+.2f} {np.median(d):+.2f} {np.quantile(d,0.9):+.2f};  S->Ar {np.quantile(d2,0.1):+.2f} {np.median(d2):+.2f} {np.quantile(d2,0.9):+.2f}")
