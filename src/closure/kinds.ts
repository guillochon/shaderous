// The turbulence closure's kinds (closure.ts), apart from its passes: the setup format and the
// option table read these without the GPU code.
/**
 * How the closure sets a cell's a and b where its gates are open: at their limits (`constant`),
 * or by the trained network (`network`), which in every flow it was tried on returns 99% of
 * those limits and costs ten times as much (docs/closure.md).
 */
export const CLOSURE_KINDS = ['off', 'constant', 'network'] as const;
export type ClosureKind = (typeof CLOSURE_KINDS)[number];
export const CLOSURE_LABELS: Record<ClosureKind, string> = { off: 'Off', constant: 'Constants', network: 'Trained network' };

