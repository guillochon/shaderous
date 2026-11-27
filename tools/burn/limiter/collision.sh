#!/bin/sh
# Collision of two 0.64 Msun dwarfs: CFL, resolution and limiter, to t = 10 with .athdf every 0.5
# (gas leaves the box soon after t = 8, so the yield is read at its plateau before that).
cd /c/Users/guill/shaderous
run() {
  name=$1; shift
  node tools/headless/cli.mjs run --problem wd-collision-burning --until 10 --every 0.5 --athdf \
    --url http://localhost:5190 --low-priority --quiet --out runs/kush/coll2/$name "$@" > runs/kush/coll-$name.log 2>&1 < /dev/null
  echo "$name exit $?"
}
run base
run cfl0.6 --set cfl=0.6
run cfl0.4 --set cfl=0.4
run n64 --n 64
run f0.1 --set p.limiter=0.1
run f0.4 --set p.limiter=0.4
run n256 --n 256
echo all done
