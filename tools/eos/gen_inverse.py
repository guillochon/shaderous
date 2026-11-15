"""Training data for the inverse network: what the hydro has, (rho, e_th, Abar), to what it
needs, (T, p_th, c_s). Samples (log rho, log T, log Abar) and computes the rest exactly: the
electron-positron gas (electron_gas.state), ideal ions (Abar nucleons each), radiation. Ye = 1/2,
as for every nucleus of the alpha chain. Below-normal priority, all cores but two.

    tools/burn/.venv/Scripts/python tools/eos/gen_inverse.py [points]

Columns of data/inverse.npz: log10 rho, log10 T, log10 Abar, log10 e_th (erg/g, the specific
energy above the zero-temperature degenerate electrons), p_th (erg/cm^3), c_s^2 (cm^2/s^2), and
c0^2 = dP_cold/drho, the zero-temperature electrons' share of c_s^2.
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import psutil

import electron_gas as eg

HERE = Path(__file__).parent
YE = 0.5
DOMAIN = {"log_rho": (0.3, 11.3), "log_T": (4.0, 11.0), "log_abar": (np.log10(4.0), np.log10(56.0))}


def thermal(rho, T, abar):
    """(e_th per gram, p_th) at rho, T, Abar: everything above the zero-temperature electrons."""
    n = rho * YE / eg.MU
    g, gn, gT, eta, pf = eg.state(n, T)
    kT = eg.K * T
    ni = rho / (abar * eg.MU)
    e = (-n * kT * gT + 1.5 * ni * kT + eg.ARAD * T ** 4) / rho
    p = n * kT * gn + ni * kT + eg.ARAD * T ** 4 / 3.0
    return e, p


def cold_p(rho):
    return eg.cold(rho * YE / eg.MU)[1]


def one(q):
    lr, lt, la = q
    rho, T, abar = 10.0 ** lr, 10.0 ** lt, 10.0 ** la
    try:
        e, p = thermal(rho, T, abar)
        d = 1e-4
        # c_s^2 = (dP/drho)_T + T (dP/dT)_rho^2 / (rho^2 c_v), all from the total pressure.
        pp = cold_p(rho * (1 + d)) + thermal(rho * (1 + d), T, abar)[1]
        pm = cold_p(rho * (1 - d)) + thermal(rho * (1 - d), T, abar)[1]
        dPdr = (pp - pm) / (2 * d * rho)
        ep, Pp = thermal(rho, T * (1 + d), abar)
        em, Pm = thermal(rho, T * (1 - d), abar)
        dPdT = (Pp - Pm) / (2 * d * T)
        cv = (ep - em) / (2 * d * T)
        cs2 = dPdr + T * dPdT ** 2 / (rho ** 2 * cv)
        c0 = (cold_p(rho * (1 + d)) - cold_p(rho * (1 - d))) / (2 * d * rho)
        return (lr, lt, la, np.log10(e), p, cs2, c0)
    except Exception:   # noqa: BLE001 - dropped, and counted
        return (lr, lt, la) + (np.nan,) * 4


def init():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


if __name__ == "__main__":
    init()
    npts = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
    rng = np.random.default_rng(2)
    pts = list(zip(*(rng.uniform(*DOMAIN[k], npts) for k in ("log_rho", "log_T", "log_abar"))))
    t0 = time.time()
    with Pool(max(1, os.cpu_count() - 2), initializer=init) as pool:
        rows = pool.map(one, pts, chunksize=32)
    a = np.array(rows)
    bad = ~np.isfinite(a).all(axis=1)
    (HERE / "data").mkdir(exist_ok=True)
    np.savez(HERE / "data" / "inverse.npz", data=a[~bad])
    print(f"{len(a) - bad.sum()} points ({bad.sum()} failed) in {time.time() - t0:.0f} s")
