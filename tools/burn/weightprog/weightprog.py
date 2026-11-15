"""Weight-space programming for the burner: add hand-written gated paths to a trained Burner.

Each gate is a list of input boxes (feature index, lo, hi) ANDed together, and a dict of output
deltas {output index: value}. The paths widen every hidden layer; the original units neither
read from nor write to the new ones, so where every gate is 0 the network is the original.

  layer 1: per box, 4 units: relu(K(x-lo)), relu(K(x-lo)-1), relu(K(x-hi)), relu(K(x-hi)-1)
           box = [r(y_lo) - r(y_lo-1)] - [r(y_hi) - r(y_hi-1)], y = K (x - edge)
  layer 2: per gate, 1 unit: relu(sum(boxes) - (n - 1))      (AND, in [0, 1])
  layers 3..depth: pass-through, silu(g + B) - B ~ g
  output:  + delta * g
relu(y) is computed as silu(A y) / A (A large), the network's own activation.
"""
import torch
from torch import nn
import train

A, B = 20.0, 20.0


def program(model, gates, K=25.0):
    old = model.net
    lin = [m for m in old if isinstance(m, nn.Linear)]
    depth = len(lin) - 1                      # hidden layers
    w = lin[0].out_features
    n1 = sum(4 * len(g["boxes"]) for g in gates)
    ng = len(gates)
    widths = [w + n1] + [w + ng] * (depth - 1)
    new = train.Burner(1, depth, model.arch).to(train.dev)
    layers, nin = [], lin[0].in_features
    for k in range(depth):
        layers += [nn.Linear(nin, widths[k]), nn.SiLU()]
        nin = widths[k]
    layers.append(nn.Linear(nin, lin[-1].out_features))
    net = nn.Sequential(*layers).to(train.dev)
    L = [m for m in net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for m in L:
            m.weight.zero_(); m.bias.zero_()
        for k in range(depth + 1):            # the original block
            o, i = lin[k].weight.shape
            L[k].weight[:o, :i] = lin[k].weight
            L[k].bias[:o] = lin[k].bias
        # layer 1: ramps
        u = w
        ramp_of = []                          # per gate: list of (unit index, sign) feeding the AND
        for g in gates:
            feeds = []
            for (idx, lo, hi) in g["boxes"]:
                for edge, sgn in ((lo, 1.0), (hi, -1.0)):
                    for off, s2 in ((0.0, 1.0), (-1.0, -1.0)):
                        # unit = silu(A * (K (x - edge) + off))
                        for fi, c in (idx.items() if isinstance(idx, dict) else ((idx, 1.0),)):
                            L[0].weight[u, fi] = A * K * c
                        L[0].bias[u] = A * (-K * edge + off)
                        feeds.append((u, sgn * s2 / A))
                        u += 1
            ramp_of.append(feeds)
        # layer 2: AND
        for gi, g in enumerate(gates):
            row = w + gi
            for (ui, c) in ramp_of[gi]:
                L[1].weight[row, ui] = A * c
            L[1].bias[row] = -A * (len(g["boxes"]) - 1)
        # pass-through layers: layer 2 holds silu(A s)  ~ A relu(s) = A g  ->  carry g + B
        for k in range(2, depth):
            for gi in range(ng):
                row = w + gi
                L[k].weight[row, row] = (1.0 / A) if k == 2 else 1.0
                L[k].bias[row] = B if k == 2 else 0.0
        last = depth                          # output layer index in L
        for gi, g in enumerate(gates):
            col = w + gi
            for oi, d in g["deltas"].items():
                if depth > 2:                 # reads g + B
                    L[last].weight[oi, col] += d
                    L[last].bias[oi] -= d * B
                else:                         # reads silu(A s) ~ A g
                    L[last].weight[oi, col] += d / A
    new.net = net
    new.width = model.width
    new.eval()
    return new


def gate_values(model, gates, x):
    """The gates' values on feature rows x (for checking where they fire)."""
    net = model.net
    h = x
    for k, m in enumerate(net):
        h = m(h)
        if k == 3:   # after layer 2's SiLU
            w = h.shape[1] - len(gates)
            return h[:, w:] / A
