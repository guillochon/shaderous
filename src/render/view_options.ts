// What the view can be set to: one list, read by the renderer's types, links, the scripting
// API and the generated schema. This file imports nothing, so anything may read it.

/** How the gas is drawn: a fog seen through the box, one plane, or a sum along an axis. */
export const VIEW_MODES = ['volume', 'slice', 'projection'] as const;
export type ViewMode = (typeof VIEW_MODES)[number];
