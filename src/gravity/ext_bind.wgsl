// The point masses for gravity/ext.wgsl, at the binding the hydro passes reserve for them.
// Requires P.
@group(0) @binding(20) var<storage, read> BH: Masses;
#include "gravity/ext.wgsl"
