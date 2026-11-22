"""Blade element momentum estimate for builtin:turbine-fast (src/geometry/builtin.ts TURBINE_FAST); plans/wind-tunnel.md 8.1.5. python tools/cutcell/turbine_bem.py

Units: R = 1 (the mesh), U = 1. The preset scales R to 0.8 box units; Re uses that.
"""
import math, sys

TSR_D, ALPHA_D, HUB, C_ROOT, C_TIP, THICK, MINT, B = 4.0, 6.0, 0.2, 0.3, 0.15, 0.2, 0.03, 3
R_BOX, NU, BOX = 0.8, 1e-4, 3.0


def chord(r):
    frac = max(0.0, (r - HUB) / (1 - HUB))
    return C_ROOT + (C_TIP - C_ROOT) * frac


def phi_design(r):
    return math.atan((2 / 3) / (TSR_D * max(r, HUB)))


def twist(r):  # section's chord to the rotor plane (radians)
    return phi_design(r) - math.radians(ALPHA_D)


def polar(alpha, re, kind):
    """(Cl, Cd). 'ideal': thin airfoil 2 pi alpha, Cd 0.01, stall at 12 deg.
    'lowre': slope and drag from this code's own NACA 0012 at Re 1000 (8.1.4: Cl 0.25, Cd 0.13 at 5 deg)
    scaled in Re: slope 2.9/rad, Cd = 0.13 sqrt(1000/Re) + 0.6 alpha^2 (laminar friction ~ Re^-1/2)."""
    if kind == 'ideal':
        a = max(-math.radians(12), min(math.radians(12), alpha))
        return 2 * math.pi * a, 0.01 + 0.5 * alpha * alpha
    if kind == 'lowre':
        a = max(-math.radians(15), min(math.radians(15), alpha))
        return 2.9 * a, 0.13 * math.sqrt(1000 / re) + 0.6 * alpha * alpha
    if kind.startswith('cd'):  # the simulation's own: Cl 4.5/rad (8.1.5's inversion), a constant Cd
        a = max(-math.radians(15), min(math.radians(15), alpha))
        return 4.5 * a, float(kind[2:])
    raise ValueError(kind)


def station(r, tsr, kind, induction):
    lam = tsr * r
    c = chord(r)
    sigma = B * c / (2 * math.pi * r)
    th = twist(r)
    a = ap = 0.0
    for _ in range(500):
        phi = math.atan2(1 - a, lam * (1 + ap))
        al = phi - th
        w = math.hypot(1 - a, lam * (1 + ap))
        re = w * c * R_BOX / NU
        cl, cd = polar(al, re, kind)
        cn = cl * math.cos(phi) + cd * math.sin(phi)
        ct = cl * math.sin(phi) - cd * math.cos(phi)
        if not induction:
            break
        # Prandtl tip and hub loss
        s = abs(math.sin(phi)) + 1e-9
        ftip = 2 / math.pi * math.acos(min(1, math.exp(-B / 2 * (1 - r) / (r * s))))
        fhub = 2 / math.pi * math.acos(min(1, math.exp(-B / 2 * (r - HUB) / (HUB * s))))
        F = max(1e-3, ftip * fhub)
        k = sigma * cn / (4 * F * math.sin(phi) ** 2)
        if k <= 2 / 3:
            an = k / (1 + k)
        else:  # Glauert/Buhl high-load branch (Spera)
            ac = 0.2
            K = 1 / k
            an = 0.5 * (2 + K * (1 - 2 * ac) - math.sqrt((K * (1 - 2 * ac) + 2) ** 2 + 4 * (K * ac * ac - 1)))
        kp = sigma * ct / (4 * F * math.sin(phi) * math.cos(phi))
        apn = kp / (1 - kp) if abs(1 - kp) > 1e-6 else ap
        an = min(max(an, -0.5), 0.95)
        apn = min(max(apn, -0.5), 1.0)
        if abs(an - a) < 1e-9 and abs(apn - ap) < 1e-9:
            break
        a = 0.7 * a + 0.3 * an
        ap = 0.7 * ap + 0.3 * apn
    # torque per unit span (rho = U = R = 1): dQ/dr = 1/2 W^2 B c Ct r
    dq = 0.5 * w * w * B * c * ct * r
    dt = 0.5 * w * w * B * c * cn
    return dict(r=r, phi=math.degrees(phi), alpha=math.degrees(al), a=a, ap=ap, re=re, cl=cl, cd=cd, dq=dq, dt=dt)


def rotor(tsr, kind, induction, n=400):
    rs = [HUB + (1 - HUB) * (i + 0.5) / n for i in range(n)]
    dr = (1 - HUB) / n
    st = [station(r, tsr, kind, induction) for r in rs]
    Q = sum(s['dq'] for s in st) * dr
    T = sum(s['dt'] for s in st) * dr
    return st, tsr * Q / (0.5 * math.pi), T / (0.5 * math.pi)


if __name__ == '__main__':
    dx = {128: BOX / 128, 256: BOX / 256}
    print('geometry: r/R chord/R twist(deg) thickness/R cells-on-chord@128 @256 thickness-cells@128 @256')
    for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        c = chord(r)
        t = max(THICK * c, MINT)
        print(f'{r:.1f} {c:.3f} {math.degrees(twist(r)):6.2f} {t:.3f} {c*R_BOX/dx[128]:5.1f} {c*R_BOX/dx[256]:5.1f} {t*R_BOX/dx[128]:4.1f} {t*R_BOX/dx[256]:4.1f}')
    for kind in ['cd0.08', 'cd0.15', 'cd0.28']:
        print(f'polar {kind} with induction: ' + ' '.join(f'TSR {t}: {rotor(t, kind, True)[1]:+.3f}' for t in [2, 3, 4, 5, 6]))
    for kind in ['ideal', 'lowre']:
        for ind in [False, True]:
            print(f'\n== polar {kind}, induction {ind}')
            for tsr in [2, 3, 4, 5, 6]:
                st, cp, ct = rotor(tsr, kind, ind)
                print(f'TSR {tsr}: C_p {cp:+.3f}  C_T {ct:.3f}')
                if tsr == 4 or '-v' in sys.argv:
                    # torque by tenth of span, matching the simulation's bins (r in R)
                    for lo in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        sel = [s for s in st if lo <= s['r'] < lo + 0.1]
                        q = sum(s['dq'] for s in sel) * (0.8 / 400)
                        m = sel[len(sel) // 2]
                        print(f'   r {lo:.1f}-{lo+0.1:.1f}: alpha {m["alpha"]:+6.2f} phi {m["phi"]:6.2f} a {m["a"]:.2f} Re {m["re"]:7.0f} Cl {m["cl"]:+.3f} Cd {m["cd"]:.3f} dQ {q:+.5f}')
