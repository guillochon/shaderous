# Order of blocks in the dispatch lists

Whether the order in which a pass takes its blocks matters for speed, as RAMSES-GPU sorts its octs along a Hilbert curve for memory locality (Teyssier et al. 2026, arXiv:2607.20704). Measured 2026-11-26 on an RTX 3080 in Chrome, with a static build of commit 6fea0f8.

## Method

The block lists are in slot order within each level (`BlockTree.all()`). We captured one step's dispatches of the assembly, predictor and update passes, timed each by replaying it back to back on the page's own resources (the slope of time against the number of repeats), then rewrote every list in place in Morton order within each level and timed the same dispatches again, then did the same with a random order, and restored the lists. The mesh and the data were the same for all three orders.

## Results

Time per pass, microseconds.

Sedov blast, 3D, $256^3$ equivalent, three levels: 512 leaves, about $2.1 \times 10^6$ cells.

| Pass | Slot order | Morton | Random |
|---|---|---|---|
| assemble | 757.5 | 750 | 812.5 |
| predict | 637.5 | 620 | 640 |
| update, no refined faces | 800 | 777.5 | 812.5 |
| update, matching | 335 | 330 | 315 |

Kelvin-Helmholtz, 3D, $512^3$ equivalent, two levels, after 200 steps: 6063 leaves (6859 blocks with parents), about $2.5 \times 10^7$ cells, block budget full.

| Pass | Slot order (two timings) | Morton | Random |
|---|---|---|---|
| assemble | 7420 / 7487.5 | 7437.5 | 8215 |
| predict | 7825 / 7975 | 7882.5 | 8157.5 |
| update, no refined faces | 9162.5 / 9282.5 | 9212.5 | 9227.5 |
| update, matching | 5375 / 5390 | 5345 | 5415 |

Shock-bubble, 3D, $512^3$ equivalent, two levels, after 900 steps of re-meshing: 2220 leaves.

| Pass | Slot order | Morton | Random |
|---|---|---|---|
| assemble | 4550 | 4550 | 4740 |
| predict | 3620 | 3670 | 3810 |
| update, no refined faces | 5450 | 5520 | 5610 |
| update, matching | 1220 | 1250 | 1190 |

Mean distance between consecutive leaves of the list, in blocks of the finest level, for the shock-bubble: 3.64 in slot order, 2.85 in Morton order, 12.72 in random order.

## Conclusion

Morton order changed no pass by more than 3 percent (the Sedov predictor and update, 2.7 and 2.8 percent faster), and on the two large meshes by at most 1.3 percent either way. Slot order already keeps neighboring blocks close in the list: blocks take free slots in the order in which the tree creates them, and freed slots are reused nearby. A random order costs up to 10 percent (the assembly on the largest mesh), so locality matters, but the existing order already provides it.
