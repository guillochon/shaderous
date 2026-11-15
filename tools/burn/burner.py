"""Burn the 13-isotope alpha chain at constant density, heating the gas as it burns: the
reference that the neural network learns to reproduce.

The state is the molar abundances Y_i (= X_i / A_i) and the temperature. The energy each
reaction releases goes into the gas's thermal energy, by the equation of state the simulator
uses for burning white dwarf matter (src/eos/helmholtz.ts: the electron-positron gas at any
degeneracy, ideal ions, radiation; read from the same table through tools/eos/helm.py). At
constant density the thermal energy e(T, n_i) follows the energy released, eps, and burning
also changes the number of ions, n_i = sum Y_i per atomic mass unit, each holding 3/2 k T:
    de/dt = eps  =>  dT/dt = (eps - (3/2) k T / m_u * sum dY_i/dt) / c_v,   c_v = de/dT
as the simulator has it: the energy added to the gas's, and T read back with the new ions.
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from pynucastro.screening import screen5

import sys as _sys
_sys.path.append(str(Path(__file__).resolve().parents[1] / "eos"))
import helm  # noqa: E402

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("aprox13", HERE / "aprox13.py")
net = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(net)

N = net.nnuc
A = net.A.astype(np.float64)
Z = net.Z.astype(np.float64)
NAMES = list(net.names)
KB = 1.380649e-16
MU = 1.66053906660e-24
ARAD = 7.5657e-15
T_MAX = 1.0e10  # where the rate fits end


def cv(Y, rho, T):
    """Specific heat at constant volume, erg / g / K, by the simulator's equation of state."""
    _, de = helm.thermo(rho, np.log(T), float(np.sum(Y)))
    return float(de) / T


def _rhs(t, s, rho):
    Y = np.maximum(s[:N], 0.0)
    T = min(max(s[N], 1e7), T_MAX)
    dY = net.rhs(t, Y, rho, T, screen5)
    eps = net.energy_release(dY)
    # The ions' share of the energy changes with their number at fixed T.
    return np.append(dY, (eps - 1.5 * KB * T / MU * float(np.sum(dY))) / cv(Y, rho, T))


def burn(X, rho, T, dt, rtol=1e-7, atol=1e-12, t_eval=None):
    """Burn mass fractions X for dt seconds at density rho, starting at temperature T.
    Returns (X, T, e) at dt, e the energy released per gram (or arrays over t_eval)."""
    Y0 = np.asarray(X, dtype=np.float64) / A
    s0 = np.append(Y0, T)
    sol = solve_ivp(_rhs, (0.0, dt), s0, method="BDF", args=(rho,), rtol=rtol,
                    atol=np.append(np.full(N, atol), 1.0), t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = np.maximum(sol.y[:N], 0.0)
    X1 = Y * A[:, None]
    X1 /= X1.sum(axis=0, keepdims=True)
    e = np.array([net.energy_release(Y[:, k] - Y0) for k in range(Y.shape[1])])
    if t_eval is None:
        return X1[:, -1], sol.y[N, -1], e[-1]
    return X1, sol.y[N], e


def composition(**fractions):
    """Mass fractions by name, e.g. composition(c12=0.5, o16=0.5)."""
    X = np.zeros(N)
    for k, v in fractions.items():
        X[NAMES.index(k.capitalize() if k[1].isdigit() else k[:2].capitalize() + k[2:])] = v
    return X / X.sum()


if __name__ == "__main__":
    import time
    print("nuclei:", NAMES)
    X0 = composition(c12=0.5, o16=0.5)
    for rho, T in [(1e7, 3e9), (1e8, 2e9), (2e6, 4e9)]:
        t0 = time.time()
        X, T1, e = burn(X0, rho, T, 1e-2)
        top = sorted(zip(X, NAMES), reverse=True)[:4]
        print(f"rho={rho:.0e} T={T:.0e}: T1={T1:.3e} e={e:.3e} erg/g  {[(n, round(x, 3)) for x, n in top]}  {time.time() - t0:.2f}s")
