"""What the tabulated equations of state share, in numpy and torch: the bicubic Hermite lookup
of a free energy from (h, h_x, h_y, h_xy) per node, as src/eos/table.ts and eos/table.wgsl have
it. helm.py (the white dwarf table, for the burner) and make_table.py (which builds it) use it.
The functions take arrays of either library: they are plain arithmetic.
"""


def basis(t):
    return [2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2]


def basis1(t):
    return [6 * t**2 - 6 * t, 3 * t**2 - 4 * t + 1, -6 * t**2 + 6 * t, 3 * t**2 - 2 * t]


def basis2(t):
    return [12 * t - 6, 6 * t - 4, -12 * t + 6, 6 * t - 2]


def comb(nodes, A, B):
    """sum over a cell's four nodes, (0, 0), (0, 1), (1, 0), (1, 1), of A_a(u) B_b(v) (h, h_x,
    h_y, h_xy): basis entries 0 and 2 weigh the value at nodes 0 and 1, entries 1 and 3 the
    slope there. `nodes` are the four nodes' texels, the channels on the last axis."""
    s = 0.0
    for a in (0, 1):
        for b in (0, 1):
            q = nodes[a * 2 + b]
            s = s + A[2*a] * B[2*b] * q[..., 0] + A[2*a+1] * B[2*b] * q[..., 1] + A[2*a] * B[2*b+1] * q[..., 2] + A[2*a+1] * B[2*b+1] * q[..., 3]
    return s
