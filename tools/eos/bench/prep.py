"""Assets for the EOS benchmark (bench.js), written to tools/eos/bench/assets/:
  table.f32   the Hermite table: (h, h_x, h_y, h_xy) per node of the 0.02 dex grid, x along
              log rho Ye and y along log T, derivatives per grid step (h_xy by differences)
  states.f32  held-out (rho, e_th, Abar, 0) from the inverse data (the fit's test split)
  truth.f32   their exact (T, p_th, c_s^2, c0^2)
  meta.json   grid, constants, the pair series and the inverse network's weights

    tools/burn/.venv/Scripts/python tools/eos/bench/prep.py [inverse_net.pt]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial import chebyshev as C

HERE = Path(__file__).parent
EOS = HERE.parent
sys.path.insert(0, str(EOS))
import electron_gas as eg  # noqa: E402
from fit_inverse import prepare  # noqa: E402

PAIR_LO, PAIR_HI, PAIR_DEG = 8.5, 11.0, 16


def main():
    net = sys.argv[1] if len(sys.argv) > 1 else "inv_w64d2.pt"
    out = HERE / "assets"
    out.mkdir(exist_ok=True)
    g = np.load(EOS / "data" / "eos_grid.npz")["data"]
    lr = np.unique(np.round(g[:, 0], 6))
    lt = np.unique(np.round(g[:, 1], 6))
    d = lr[1] - lr[0]
    nr, nt = len(lr), len(lt)
    H = g[:, 2].reshape(nr, nt)
    Hx = g[:, 3].reshape(nr, nt) * d * math.log(10)
    Hy = g[:, 4].reshape(nr, nt) * d * math.log(10)
    Hxy = np.gradient(Hx, axis=1)
    np.stack([H, Hx, Hy, Hxy], -1).astype("<f4").tofile(out / "table.f32")

    # Pairs: ln P_p = ln P_0 + q, P_0 the nonrelativistic Boltzmann pairs, q a Chebyshev series in log T.
    T = 10 ** np.linspace(PAIR_LO, PAIR_HI, 801)
    beta = eg.K * T / eg.MEC2
    P = np.array([eg.pair(t)[0] for t in T])
    P0 = 4 * eg.K * T * (2 * np.pi * eg.ME * eg.K * T / eg.H ** 2) ** 1.5 * np.exp(-1 / beta)
    mid, half = (PAIR_LO + PAIR_HI) / 2, (PAIR_HI - PAIR_LO) / 2
    q = C.chebfit((np.log10(T) - mid) / half, np.log(P / P0), PAIR_DEG)
    dq = C.chebder(q)
    d2q = C.chebder(dq)
    pad = lambda c: list(np.pad(c, (0, len(q) - len(c))))

    a = np.load(EOS / "data" / "inverse.npz")["data"]
    test = np.random.default_rng(0).random(len(a)) < 0.1
    t = a[test]
    states = np.stack([10 ** t[:, 0], 10 ** t[:, 3], 10 ** t[:, 2], 0 * t[:, 0]], 1)
    truth = np.stack([10 ** t[:, 1], t[:, 4], t[:, 5], t[:, 6]], 1)
    states.astype("<f4").tofile(out / "states.f32")
    truth.astype("<f4").tofile(out / "truth.f32")

    ck = torch.load(EOS / net, weights_only=False)
    st = ck["state"]
    lin = sorted({k.split(".")[1] for k in st if k.startswith("net.")}, key=int)
    weights, layers = [], []
    for i in lin:
        W, b = st[f"net.{i}.weight"].numpy(), st[f"net.{i}.bias"].numpy()
        layers.append([W.shape[1], W.shape[0]])
        weights += list(W.ravel()) + list(b)
    meta = {
        "table": {"r0": float(lr[0]), "t0": float(lt[0]), "d": float(d), "nr": nr, "nt": nt},
        "pair": {"mid": mid, "half": half, "q": list(q), "dq": pad(dq), "d2q": pad(d2q)},
        "consts": {"k": eg.K, "mu": eg.MU, "mec2": eg.MEC2, "arad": eg.ARAD, "me": eg.ME, "h": eg.H, "pref": eg.PREF},
        "net": {"layers": layers, "weights": [float(x) for x in weights], "lo": st["lo"].tolist(), "hi": st["hi"].tolist(),
                "ylo": [float(x) for x in ck["ylo"]], "yhi": [float(x) for x in ck["yhi"]]},
    }
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"table {nr} x {nt}; {len(states)} held-out states; network {layers}")


if __name__ == "__main__":
    main()
