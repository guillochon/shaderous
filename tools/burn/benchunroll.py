"""Time train.py's unrolled term (forward and backward) for n = 2, 8 and 32 steps, with
torch.compile (BURN_COMPILE=1) or without, and a plain training step for comparison: how much a
GPU gains on the term, whose cost is overhead per chained step more than arithmetic.

    tools/burn/.venv/Scripts/python tools/burn/benchunroll.py [batch]
"""
import os
import sys
import time

import torch

import train  # puts tools/eos on the path
import helm  # noqa: E402


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    model = train.load_model("model_dagger_36a.pt")
    model.train()
    rho, X, T, t = train.load({"fuel.npz", "mix.npz"})
    P = train.Pairs(rho, X, T, t)
    fwd = model
    if os.environ.get("BURN_COMPILE"):
        fwd = torch.compile(model)
        helm.thermo_t = torch.compile(helm.thermo_t)
    print(torch.cuda.get_device_name(), "compile" if fwd is not model else "eager", flush=True)
    (r, Tl, dt, X0), Dt, et = P.sample(8192, 0.2)
    for k in range(6):
        if k == 1:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        loss, _, _ = train.loss_fn(model(r, Tl, dt, X0), Dt, et, 4, 0.3, model.parts, 1)
        model.zero_grad(set_to_none=True)
        loss.backward()
    torch.cuda.synchronize()
    print(f"plain step, batch 8192: {(time.perf_counter() - t0) / 5 * 1e3:.0f} ms", flush=True)
    for n in (2, 8, 32):
        lr_, lT, span, U0, UD, ue = P.sample_unrolled(batch, 0.2)
        t_first = time.perf_counter()
        for k in range(4):
            if k == 1:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                first = t0 - t_first
            loss = train.log_loss(train.unrolled(fwd, lr_, lT, span, U0, n), UD, ue)
            model.zero_grad(set_to_none=True)
            loss.backward()
        torch.cuda.synchronize()
        print(f"unrolled n={n:2d}, batch {batch}: {(time.perf_counter() - t0) / 3 * 1e3:.0f} ms (first call {first:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
