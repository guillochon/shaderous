"""Figures of the blade sections on the GPU (plans/wind-tunnel.md 8.1.6).

Reads runs/sections/<section>_<cells>/ (section_fields.mjs: the runner's slices; section_cells.mjs:
the gas shares cut.npy) and, for the convergence figure, the log of the GPU test cut-sections.

  python tools/cutcell/section_plots.py flow CELLS          the flow around the five sections
  python tools/cutcell/section_plots.py wall SECTION        the wall cells' speed along the lower side, by grid
  python tools/cutcell/section_plots.py polars LOG          C_l and C_d against cells on the chord

Figures go to runs/sections/figures/. Vorticity is taken from gas cells only: a difference across
the wall (a solid cell holds no velocity of the flow's) becomes one-sided.
"""
import glob, json, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.environ.get('SECTIONS_ROOT', 'runs/sections')
OUT = f'{ROOT}/figures'
SECTIONS = [('section-tip', 'Tip as built (floor 0.20)', 0.2, 0.2, False),
            ('section-075', '0.75 R as built (floor 0.15)', 0.2, 0.15, False),
            ('section-round', 'Round nose, floor 0.15 aft', 0.2, 0.15, True),
            ('naca0020', 'NACA 0020, sharp', 0.2, 0.0, False),
            ('naca0012', 'NACA 0012, sharp', 0.12, 0.0, False)]
ALPHA, CENTER = 9.0, (-1.4963, 0.0051)
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
DIV = LinearSegmentedColormap.from_list('div', ['#104281', '#3987e5', '#cde2fb', '#f0efec', '#fbd5c4', '#eb6834', '#8f2f0c'])
SEQ = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': INK2, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def outline(t, minT, round_nose, n=200):
    """The section's outline in the box (src/geometry/builtin.ts sectionOutline, turned and placed as sectionSpec)."""
    xs = [0.5 * (1 - math.cos(math.pi * k / n)) for k in range(n + 1)]
    def yt(x):
        v = 5 * t * (0.2969 * math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        return v if (minT <= 0 or (round_nose and x < 0.3)) else max(v, minT / 2)
    pts = [(x - 0.25, -yt(x)) for x in xs] + [(x - 0.25, yt(x)) for x in reversed(xs)]
    th = -math.radians(ALPHA)
    c, s = math.cos(th), math.sin(th)
    return np.array([(CENTER[0] + x * c - y * s, CENTER[1] + x * s + y * c) for x, y in pts])


def load(section, cells):
    """The gas shares and the slices of one run: phi[j, i], and per output time {var: array[j, i]}."""
    d = f'{ROOT}/{section}_{cells}'
    meta = json.load(open(f'{d}/cut.json'))
    phi = np.load(f'{d}/cut.npy')[:, :, 0]
    frames = {}
    for out in json.load(open(f'{d}/manifest.json'))['outputs']:
        for s in out.get('slices', []):
            frames.setdefault(round(s['t'], 6), {})[s['variable']] = np.load(f"{d}/{s['file']}")
    return meta, phi, frames


def gas_derivative(f, gas, dx, axis):
    """df/dx along an axis from gas cells only: central where both neighbors hold gas, one-sided where one does."""
    fp, fm = np.roll(f, -1, axis), np.roll(f, 1, axis)
    gp, gm = np.roll(gas, -1, axis), np.roll(gas, 1, axis)
    out = np.where(gp & gm, (fp - fm) / (2 * dx), np.where(gp, (fp - f) / dx, np.where(gm, (f - fm) / dx, np.nan)))
    return np.where(gas, out, np.nan)


def window(meta, x0=-2.3, x1=1.0, y0=-0.8, y1=0.8):
    lo, dx = meta['lo'], meta['dx']
    i0, i1 = int((x0 - lo[0]) / dx), int(math.ceil((x1 - lo[0]) / dx))
    j0, j1 = int((y0 - lo[1]) / dx), int(math.ceil((y1 - lo[1]) / dx))
    return (slice(j0, j1), slice(i0, i1)), [lo[0] + i0 * dx, lo[0] + i1 * dx, lo[1] + j0 * dx, lo[1] + j1 * dx]


def flow(cells):
    fig, axs = plt.subplots(len(SECTIONS), 2, figsize=(12, 12.5))
    for row, (key, name, t, mt, rn) in zip(axs, SECTIONS):
        meta, phi, frames = load(key, cells)
        dx = meta['dx']
        gas = phi > 0
        times = sorted(frames)
        last = frames[times[-1]]
        w = gas_derivative(last['vel2'], gas, dx, 1) - gas_derivative(last['vel1'], gas, dx, 0)
        mean = [frames[k] for k in times if k >= 8 - 1e-9]
        um = np.mean([f['vel1'] for f in mean], axis=0)
        vm = np.mean([f['vel2'] for f in mean], axis=0)
        sl, ext = window(meta)
        sp = np.where(gas, np.hypot(um, vm), np.nan)[sl]
        body = outline(t, mt, rn)
        a, b = row
        a.imshow(np.ma.masked_invalid(w[sl]), origin='lower', extent=ext, cmap=DIV, vmin=-8, vmax=8, interpolation='nearest')
        b.imshow(np.ma.masked_invalid(sp), origin='lower', extent=ext, cmap=SEQ, vmin=0, vmax=1.5, interpolation='nearest')
        xs = ext[0] + (np.arange(sp.shape[1]) + 0.5) * dx
        ys = ext[2] + (np.arange(sp.shape[0]) + 0.5) * dx
        b.streamplot(xs, ys, np.where(gas, um, np.nan)[sl], np.where(gas, vm, np.nan)[sl], color=INK2, linewidth=0.6, density=1.4, arrowstyle='-', broken_streamlines=False)
        for ax in (a, b):
            ax.fill(body[:, 0], body[:, 1], color='#383835', lw=0, zorder=3)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(-0.8, 0.8); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        a.set_title(name, loc='left', fontsize=10.5, fontweight='bold')
        b.set_title(f'{len(mean)} slices, t = 8 to {times[-1]:.0f}', loc='left', fontsize=9.5, color=INK2)
    axs[0, 0].text(0, 1.42, f'Vorticity at t = {times[-1]:.0f}', transform=axs[0, 0].transAxes, fontsize=12)
    axs[0, 1].text(0, 1.42, 'Mean speed and streamlines, t = 8 to 24', transform=axs[0, 1].transAxes, fontsize=12)
    axs[0, 0].text(0, 1.18, f'GPU, cut cells, {cells} cells on the chord; 9°, Re 5000; the drawn body is the exact outline, cells with no gas left blank.', transform=axs[0, 0].transAxes, fontsize=8.5, color=INK2)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/flow_{cells}.png', dpi=150, bbox_inches='tight')
    print(f'{OUT}/flow_{cells}.png')


def wall(section):
    """The speed along the wall in the cut cells of the lower side, against the distance along the chord, at each grid."""
    key, name, t, mt, rn = next(s for s in SECTIONS if s[0] == section)
    th = -math.radians(ALPHA)
    c, s = math.cos(th), math.sin(th)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = [BLUE, ORANGE, AQUA, MAGENTA]
    runs = sorted(int(m.group(1)) for p in glob.glob(f'{ROOT}/{section}_*') if (m := re.search(r'_(\d+)$', p)) and os.path.exists(f'{p}/manifest.json'))
    for col, cells in zip(colors, runs):
        meta, phi, frames = load(section, cells)
        times = [k for k in sorted(frames) if k >= 8 - 1e-9]
        um = np.mean([frames[k]['vel1'] for k in times], axis=0)
        vm = np.mean([frames[k]['vel2'] for k in times], axis=0)
        dx, lo = meta['dx'], meta['lo']
        J, I = np.nonzero((phi > 0) & (phi < 1))
        X = lo[0] + (I + 0.5) * dx - CENTER[0]
        Y = lo[1] + (J + 0.5) * dx - CENTER[1]
        xi, eta = X * c + Y * s, -X * s + Y * c
        sel = (eta < 0) & (xi > -0.2) & (xi < 0.7)
        ut = (um[J, I] * c + vm[J, I] * s)[sel]
        order = np.argsort(xi[sel])
        small = phi[J, I][sel][order] < 0.5
        ax.plot(xi[sel][order], ut[order], color=col, lw=1.5, label=f'{cells} cells on the chord')
        ax.scatter(xi[sel][order][small], ut[order][small], s=10, color=col, zorder=3)
    ax.set_xlabel('distance along the chord from the quarter chord')
    ax.set_ylabel('mean speed along the wall, cut cells')
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f'{name}, lower side: the wall cells by grid (dots: cells under half full)', loc='left', fontsize=11)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/wall_{section}.png', dpi=150, bbox_inches='tight')
    print(f'{OUT}/wall_{section}.png')


def polars(log):
    """C_l and C_d against cells on the chord, from the metrics the GPU test cut-sections printed."""
    text = open(log, encoding='utf-8', errors='replace').read()
    m = json.loads(text[text.index('{', text.index('cut-sections:')):text.index('}', text.index('cut-sections:')) + 1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = [ORANGE, YELLOW, AQUA, BLUE, '#6250d6']
    for col, (key, name, *_) in zip(colors, SECTIONS):
        cells = sorted(int(k.split('_')[1]) for k in m if k.startswith(key + '_') and k.endswith('_cd') and 'staircase' not in k)
        for ax, q in ((a1, 'cl'), (a2, 'cd')):
            ax.plot(cells, [m[f'{key}_{c}_{q}'] for c in cells], color=col, lw=2, marker='o', ms=6, label=name)
            st = [c for c in (32, 128) if f'{key}_{c}_staircase_{q}' in m]
            ax.plot(st, [m[f'{key}_{c}_staircase_{q}'] for c in st], color=col, lw=0, marker='x', ms=7)
    for ax, lab in ((a1, 'C_l'), (a2, 'C_d')):
        ax.set_xscale('log', base=2)
        ax.set_xlabel('cells on the chord')
        ax.set_ylabel(lab)
        ax.grid(True, color=GRID)
    a1.legend(frameon=False, fontsize=8.5)
    fig.suptitle('Blade sections at 9°, Re 5000, on the GPU: lines cut cells, crosses the staircase', x=0.01, ha='left', fontsize=11.5)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/polars.png', dpi=150, bbox_inches='tight')
    print(f'{OUT}/polars.png')


if __name__ == '__main__':
    what, arg = sys.argv[1], sys.argv[2]
    {'flow': lambda: flow(int(arg)), 'wall': lambda: wall(arg), 'polars': lambda: polars(arg)}[what]()
