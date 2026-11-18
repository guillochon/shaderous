// What the view can be set to: one table, read by the renderer's types, links, the scripting
// API, the generated schema and the docs. This file imports nothing, so anything may read it.

/** How the gas is drawn: a fog seen through the box, one plane, or a sum along an axis. */
export const VIEW_MODES = ['volume', 'slice', 'projection'] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

/** Largest magnification of the slice view. */
export const MAX_ZOOM = 256;
/** Volume view: the nearest and farthest the camera comes to the point it orbits, in box lengths. */
export const MIN_DIST = 0.225;
export const MAX_DIST = 6;
/** Most contour lines the slice and projection views draw. */
export const MAX_CONTOURS = 40;
/** Most field lines drawn. */
export const MAX_FIELD_LINES = 1024;

export interface ViewOption {
  /** its name in the view's state, and in `shaderous.view()` */
  key: string;
  /** its key in a link */
  link: string;
  /** 'int' is a whole number; 'mode' one of VIEW_MODES */
  type: 'number' | 'int' | 'boolean' | 'mode';
  /** a link's value is held to these (the fields and species of the run set the largest index themselves) */
  min?: number;
  max?: number;
  /** one line for the docs */
  doc: string;
}

/**
 * Every setting of the view, in the order links write them. Adding one here adds it to links,
 * to `shaderous.view()`, to the MCP tools' descriptions and to the docs; the renderer's
 * ViewState must then hold it too, or the build stops (render/renderer.ts).
 */
export const VIEW_OPTIONS = [
  { key: 'mode', link: 'mode', type: 'mode', doc: '`volume`, `slice` or `projection`' },
  { key: 'field', link: 'field', type: 'int', min: 0, doc: 'Which field is drawn: an index into `describe().fields`' },
  { key: 'species', link: 'species', type: 'int', min: 0, doc: 'Which species, for the mass-fraction field' },
  { key: 'log', link: 'log', type: 'boolean', doc: 'Logarithmic color scale' },
  { key: 'lo', link: 'lo', type: 'number', doc: 'Lower end of the color range; setting either end turns the automatic range off' },
  { key: 'hi', link: 'hi', type: 'number', doc: 'Upper end of the color range' },
  { key: 'sliceAxis', link: 'axis', type: 'int', min: 0, max: 2, doc: 'Slice and projection axis (0 x, 1 y, 2 z)' },
  { key: 'slicePos', link: 'slice', type: 'number', min: 0, max: 0.999, doc: 'Slice position along its axis (0 to 1)' },
  { key: 'minRho', link: 'rhocut', type: 'number', min: 0, max: 1, doc: 'Volume view: least density drawn, as a fraction of the way from the smallest to the largest on a log scale' },
  { key: 'opacity', link: 'opacity', type: 'number', min: 1, max: 300, doc: 'Volume view: opacity of the fog' },
  { key: 'yaw', link: 'yaw', type: 'number', doc: 'Volume view: camera angle about the vertical, in radians' },
  { key: 'pitch', link: 'pitch', type: 'number', min: -1.5, max: 1.5, doc: 'Volume view: camera angle above the horizontal, in radians' },
  { key: 'dist', link: 'dist', type: 'number', min: MIN_DIST, max: MAX_DIST, doc: 'Volume view: camera distance, in box lengths' },
  { key: 'tx', link: 'tx', type: 'number', min: 0, max: 1, doc: 'Volume view: the point the camera orbits, in fractions of the box (0.5: the center)' },
  { key: 'ty', link: 'ty', type: 'number', min: 0, max: 1, doc: 'The same, in y' },
  { key: 'tz', link: 'tz', type: 'number', min: 0, max: 1, doc: 'The same, in z' },
  { key: 'zoom', link: 'zoom', type: 'number', min: 1, max: MAX_ZOOM, doc: 'Slice view: magnification (1 fits the box)' },
  { key: 'cx', link: 'cx', type: 'number', min: 0, max: 1, doc: 'Slice view: the point at the center of the view, in fractions of the box' },
  { key: 'cy', link: 'cy', type: 'number', min: 0, max: 1, doc: 'The same, along the other axis of the slice' },
  { key: 'showBlocks', link: 'blocks', type: 'boolean', doc: 'Outline the blocks of an adaptive mesh' },
  { key: 'contours', link: 'contours', type: 'boolean', doc: 'Contour lines (isosurfaces in the volume view)' },
  { key: 'nContours', link: 'ncont', type: 'int', min: 1, max: MAX_CONTOURS, doc: 'How many contours' },
  { key: 'fog', link: 'fog', type: 'boolean', doc: 'Volume view: draw the gas as fog (off leaves the isosurfaces alone)' },
  { key: 'surfaceOpacity', link: 'isoa', type: 'number', min: 0.01, max: 1, doc: 'Volume view: how opaque the isosurfaces are' },
  { key: 'light', link: 'light', type: 'boolean', doc: 'A white background instead of the dark one' },
  { key: 'box', link: 'box', type: 'boolean', doc: 'Volume view: outline the box (on by default)' },
  { key: 'fieldLines', link: 'flines', type: 'boolean', doc: 'A run with a magnetic field: draw field lines (on by default with a field). In 2D they are contours of the flux function' },
  { key: 'fieldLineCount', link: 'nflines', type: 'int', min: 1, max: MAX_FIELD_LINES, doc: 'How many field lines' },
  { key: 'mirrorImages', link: 'mirror', type: 'boolean', doc: 'An octant (or 2D quadrant) of a mirror-symmetric run: show the whole configuration, the computed part with its mirror images (on by default)' },
] as const satisfies readonly ViewOption[];

/** What `shaderous.view()` takes besides: settings that are not part of the view's own state. */
export const VIEW_EXTRAS = [
  { key: 'colormap', link: 'cmap', doc: 'A name from `describe().colormaps`' },
  { key: 'autoRange', link: '', doc: 'Fit the color range to the data' },
] as const;

/** The view's state as the table gives it: the renderer's ViewState must be exactly this. */
export type ViewFromOptions = {
  [O in (typeof VIEW_OPTIONS)[number] as O['key']]: O['type'] extends 'boolean' ? boolean : O['type'] extends 'mode' ? ViewMode : number;
};
