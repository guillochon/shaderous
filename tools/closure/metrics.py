"""Statistics a coarse run is judged by against the restricted reference (numpy, on one
conserved state U[6, n, n] of a periodic box)."""
import numpy as np


def ke_spectrum(U):
    """Kinetic energy per shell of whole wavenumbers, of sqrt(rho) v (so that the shells sum to
    the mean kinetic energy density). Returns E[0 .. n/2]."""
    rho = np.maximum(U[0], 1e-30)
    n = rho.shape[-1]
    w = 1.0 / np.sqrt(rho)
    e = 0.5 * (abs(np.fft.fft2(U[1] * w)) ** 2 + abs(np.fft.fft2(U[2] * w)) ** 2) / n**4
    k = np.fft.fftfreq(n, 1 / n)
    shell = np.rint(np.hypot(k[:, None], k[None, :])).astype(int)
    return np.bincount(shell.ravel(), e.ravel())[: n // 2 + 1]


def scalar_spectrum(U):
    """Variance of the passive scalar per shell."""
    s = U[5] / np.maximum(U[0], 1e-30)
    n = s.shape[-1]
    e = abs(np.fft.fft2(s - s.mean())) ** 2 / n**4
    k = np.fft.fftfreq(n, 1 / n)
    shell = np.rint(np.hypot(k[:, None], k[None, :])).astype(int)
    return np.bincount(shell.ravel(), e.ravel())[: n // 2 + 1]


def mixing(U):
    """Of the passive scalar s (0 to 1 at the start):
    thickness  4 <s_bar (1 - s_bar)> with s_bar the mean along x: the mixing layers' width,
               as a fraction of the box (1 when the mean profile is flat at 1/2);
    mixedness  4 <s (1 - s)>: how much of the gas is mixed cell by cell;
    variance   of s."""
    s = U[5] / np.maximum(U[0], 1e-30)
    sbar = s.mean(axis=-1)
    return {"thickness": float(4 * np.mean(sbar * (1 - sbar))), "mixedness": float(np.mean(4 * s * (1 - s))), "variance": float(s.var())}


def band_energy(E, lo, hi):
    return float(E[lo:hi].sum())


def spectrum_error(E, Eref, kmin=2):
    """Mean absolute log10 ratio of a spectrum to the reference's over the shells from kmin up."""
    a, b = np.maximum(E[kmin:], 1e-300), np.maximum(Eref[kmin:], 1e-300)
    return float(np.mean(abs(np.log10(a / b))))


def enstrophy(U):
    rho = np.maximum(U[0], 1e-30)
    vx, vy = U[1] / rho, U[2] / rho
    n = rho.shape[-1]
    w = 0.5 * n * ((np.roll(vy, -1, -1) - np.roll(vy, 1, -1)) - (np.roll(vx, -1, -2) - np.roll(vx, 1, -2)))
    return float(0.5 * np.mean(w * w))


def totals(U):
    """Mean mass, momenta, total energy, kinetic energy and scalar mass per unit volume."""
    rho = np.maximum(U[0], 1e-30)
    return {"mass": float(U[0].mean()), "px": float(U[1].mean()), "py": float(U[2].mean()), "energy": float(U[3].mean()),
            "kinetic": float((0.5 * (U[1] ** 2 + U[2] ** 2) / rho).mean()), "scalar": float(U[5].mean())}
