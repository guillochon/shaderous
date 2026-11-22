"""From the simulation's thrust and torque by span (256^3, cut cells, TSR 4, t = 1.2; 8.1.4), the
sections' lift and drag coefficients, with the axial induction from momentum on each annulus."""
import math, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from turbine_bem import chord, twist

R, U, OM, B = 0.8, 1.0, 5.0, 3
# r/R bin low edge: (thrust, torque), pressure + shear
sim = {
    0.4: (0.07940 + 0.00039, -0.000657 - 0.000345),
    0.5: (0.09093 + 0.00033, -0.004167 - 0.000409),
    0.6: (0.09120 + 0.00031, -0.008543 - 0.000571),
    0.7: (0.09963 + 0.00028, -0.012318 - 0.000707),
    0.8: (0.08139 + 0.00027, -0.018075 - 0.000842),
    0.9: (0.06508 + 0.00032, -0.028380 - 0.001151),
    0.2: (0.03686 + 0.00047, 0.001862 - 0.000125),
    0.3: (0.06106 + 0.00044, 0.003062 - 0.000238),
}
print('r/R   a     phi   alpha  W     Cl     Cd    L/D   (pressure+shear; F = 1)')
for lo in sorted(sim):
    T, Q = sim[lo]
    rm = (lo + 0.05) * R
    dr = 0.1 * R
    # momentum: dT = 4 pi r rho U^2 a (1 - a) dr
    k = T / (4 * math.pi * rm * U * U * dr)
    a = 0.5 * (1 - math.sqrt(max(0.0, 1 - 4 * k)))
    # tangential: dQ = 4 pi r^3 rho U Omega a' (1 - a) dr
    ap = Q / (4 * math.pi * rm ** 3 * U * OM * (1 - a) * dr)
    phi = math.atan2(U * (1 - a), OM * rm * (1 + ap))
    W = math.hypot(U * (1 - a), OM * rm * (1 + ap))
    c = chord(lo + 0.05) * R
    q = 0.5 * W * W * c * dr * B
    fn, ft = T / q, (Q / rm) / q  # normal (thrust) and tangential (driving) coefficients
    cl = fn * math.cos(phi) + ft * math.sin(phi)
    cd = fn * math.sin(phi) - ft * math.cos(phi)
    al = phi - twist(lo + 0.05)
    print(f'{lo+0.05:.2f} {a:.3f} {math.degrees(phi):6.2f} {math.degrees(al):+6.2f} {W:.2f} {cl:+.3f} {cd:.3f} {cl/cd:5.2f}')
