"""Tune a gated rate unit (rateedit.py with_unit) by rollouts: its strength alpha and an
offset of its threshold (a higher threshold keeps it shut on states that want only a little
slowing, such as hot C/O, while it still fires on the Si/S runaway), judged on lpedit3's dev
set (400 trajectories per regime, 1000 general) by score.py's metric.

    tools/burn/.venv/Scripts/python tools/burn/weightprog/tunegate.py edited.pt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lpedit3 as L  # noqa: E402
import rateedit as R  # noqa: E402
import train  # noqa: E402


def gate_of(edited):
    """(w, b) of the extra unit of an edited model."""
    m = train.load_model(edited)
    Ln = [x for x in m.net if isinstance(x, torch.nn.Linear)]
    k, u = len(Ln) - 2, Ln[0].out_features - 1
    return Ln[k].weight[u, :u] / R.RELU_A, Ln[k].bias[u] / R.RELU_A


def main():
    edited = sys.argv[1]
    base = L.load_base("model_dagger_35b.pt")
    w, b = gate_of(edited)
    _, devset = L.split()
    for off in (0.0, -0.05, -0.1, -0.2):
        for alpha in ((1.5, 2.0, 3.0) if off else (1.5,)):
            m = R.with_unit(base, w, b + off, alpha)
            r = L.dev_score(m, devset)
            print(f"offset {off:+.2f} alpha {alpha}: " + L.summary(r), flush=True)
            torch.save({"state": m.state_dict(), "width": 129, "depth": 4, "arch": m.arch},
                       train.HERE / f"tune_o{-off:g}_a{alpha}.pt")


if __name__ == "__main__":
    main()
