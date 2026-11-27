#!/bin/sh
# The C/O tube (detonation-co, 2 km cells) with the burning limiter: f, ignition temperature and
# layer width. Serve a static build on port 5190 first (vite build, vite preview). Front speeds:
#   python tools/burn/limiter/front.py runs/kush/tube
node tools/headless/cli.mjs sweep --problem detonation-co --vary p.limiter=0,0.05,0.1,0.2,0.4 --vary p.tign=4,8 \
  --vary p.width=15,30,60,200 --until 0.06 --every 0.02 --profile x:0,0.5:1,0.5:press --profile-format csv \
  --url http://localhost:5190 --low-priority --quiet --out runs/kush/tube
