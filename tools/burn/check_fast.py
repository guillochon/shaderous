"""Check that the vectorized rate terms (train.full_rate_terms) reproduce the term-by-term
reference (train.full_rate_terms_reference), and time both."""
import time

import numpy as np
import torch

import train

dev = train.dev
rng = np.random.default_rng(1)
B = 4000
X = torch.tensor(rng.dirichlet(np.ones(13) * 0.2, B), dtype=torch.float32, device=dev)
X[:200] = 0
X[:200, 1] = 0.5
X[:200, 2] = 0.5                                       # pure C/O: most species zero
lr = torch.tensor(rng.uniform(4, 10, B), dtype=torch.float32, device=dev)
lT = torch.tensor(rng.uniform(8, 10.5, B), dtype=torch.float32, device=dev)


def tm(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


with torch.no_grad():
    a = train.full_rate_terms_reference(lr, lT, X)
    b = train.full_rate_terms(lr, lT, X)
    fa = (a / np.log(10)).clamp(-30, 10)
    fb = (b / np.log(10)).clamp(-30, 10)
    print(f"max |diff| of log10 terms as features clip them: {(fa - fb).abs().max().item():.2e}; "
          f"all finite: {torch.isfinite(b).all().item()}")
    for n in (600, 8192):
        idx = torch.arange(n, device=dev) % B
        x, r_, T_ = X[idx], lr[idx], lT[idx]
        print(f"batch {n}: reference {tm(lambda: train.full_rate_terms_reference(r_, T_, x)):.1f} ms, "
              f"vectorized {tm(lambda: train.full_rate_terms(r_, T_, x)):.1f} ms")
