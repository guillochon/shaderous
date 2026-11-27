"""Front position (km) at each output of each run under a folder, and the speed between the
last two outputs: the farthest point where the pressure exceeds twice the fuel's."""
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
L = 1000.0  # km per box
for run in sorted(p.parent for p in root.rglob("manifest.json")):
    files = sorted(run.glob("*-profile-x*.csv"))
    if not files:
        continue
    pts = []
    for f in files:
        t = float(f.name.split("-t")[1].split("-")[0])
        rows = [(float(r["s"]), float(r["value"])) for r in csv.DictReader(open(f))]
        p0 = rows[-1][1]
        x = max((s for s, v in rows if v > 2 * p0), default=0.0)
        pmax = max(v for _, v in rows)
        pts.append((t, x * L, pmax / p0))
    ok = [q for q in pts[1:] if 0 < q[1] < 950]
    r2 = ok[-1][2] if ok else pts[-1][2]
    v = (ok[-1][1] - ok[-2][1]) / (ok[-1][0] - ok[-2][0]) if len(ok) >= 2 else float("nan")
    q = str(json.load(open(run / "manifest.json")).get("setup", ""))
    print(f"{run.name:40s} x(t)={[round(x) for _, x, _ in pts]} km  v={v:7.0f} km/s  pmax/p0={r2:5.1f}  {q[-60:] if isinstance(q, str) else ''}")
