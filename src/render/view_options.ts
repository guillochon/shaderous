// What the view can be set to: one table, read by the renderer's types, links, the scripting
// API, the generated schema and the docs. This file imports nothing, so anything may read it.

/** How the gas is drawn: a fog seen through the box, one plane, or a sum along an axis. */
export const VIEW_MODES = ['volume', 'slice', 'projection'] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

/** Largest magnification of the slice view. */
export const MAX_ZOOM = 256;
/**
 * Largest magnification of a 1D run's view, whose cells may be 262144 across: down to some 16
 * cells across the view (app.ts maxZoom), its map over what is in view (renderer.ts mapWindow).
 */
export const MAX_ZOOM_1D = 16384;
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
  { key: 'zoom', link: 'zoom', type: 'number', min: 1, max: MAX_ZOOM_1D, doc: `Slice view: magnification (1 fits the box), up to ${MAX_ZOOM} (in 1D, up to cells / 16, at most ${MAX_ZOOM_1D})` },
  { key: 'cx', link: 'cx', type: 'number', min: 0, max: 1, doc: 'Slice view: the point at the center of the view, in fractions of the box' },
  { key: 'cy', link: 'cy', type: 'number', min: 0, max: 1, doc: 'The same, along the other axis of the slice' },
  { key: 'showBlocks', link: 'blocks', type: 'boolean', doc: 'Outline the blocks of an adaptive mesh' },
  { key: 'contours', link: 'contours', type: 'boolean', doc: 'Contour lines (isosurfaces in the volume view)' },
  { key: 'nContours', link: 'ncont', type: 'int', min: 1, max: MAX_CONTOURS, doc: 'How many contours' },
  { key: 'fog', link: 'fog', type: 'boolean', doc: 'Volume view: draw the gas as fog (off leaves the isosurfaces alone)' },
  { key: 'surfaceOpacity', link: 'isoa', type: 'number', min: 0.01, max: 1, doc: 'Volume view: how opaque the isosurfaces are' },
  { key: 'exact', link: 'exact', type: 'boolean', doc: 'Volume view: isosurfaces of the field’s log interpolated between cell centers (off, the default: of the value smoothed by a cubic B-spline, which places a surface within the cells it crosses and takes away their terraces)' },
  { key: 'light', link: 'light', type: 'boolean', doc: 'A white background instead of the dark one' },
  { key: 'box', link: 'box', type: 'boolean', doc: 'Volume view: outline the box (on by default)' },
  { key: 'fieldLines', link: 'flines', type: 'boolean', doc: 'A run with a magnetic field: draw field lines (on by default with a field). In 2D they are contours of the flux function' },
  { key: 'fieldLineCount', link: 'nflines', type: 'int', min: 1, max: MAX_FIELD_LINES, doc: 'How many field lines' },
  { key: 'fieldLineFloor', link: 'flfloor', type: 'number', min: 0.00001, max: 0.5, doc: 'Field lines end where |B| falls below this fraction of the largest |B|, and start only where it is above twice that (default 0.01): lower it to follow a weak field farther out, such as to the surface of a star' },
  { key: 'mirrorImages', link: 'mirror', type: 'boolean', doc: 'An octant (or 2D quadrant) of a mirror-symmetric run: show the whole configuration, the computed part with its mirror images (on by default)' },
  { key: 'turnWithFrame', link: 'turn', type: 'boolean', doc: 'Volume view of a run in a rotating frame: turn the box at the frame’s rate, so the picture is the one a camera at rest sees (a rotor turning, its wake left behind)' },
] as const satisfies readonly ViewOption[];

/** What `shaderous.view()` takes besides: settings that are not part of the view's own state. */
export const VIEW_EXTRAS = [
  { key: 'colormap', link: 'cmap', doc: 'A name from `describe().colormaps`' },
  { key: 'tf', link: 'tf', doc: 'Volume view: the opacity curve, as `x:a` points separated by commas (2 to 16 of them; x a fraction of the color range, a the opacity, each 0 to 1), as in a link. The default is `0:0,0.25:0.02,0.6:0.25,1:0.9`' },
  { key: 'autoRange', link: '', doc: 'Fit the color range to the data' },
] as const;

/** The view's state as the table gives it: the renderer's ViewState must be exactly this. */
export type ViewFromOptions = {
  [O in (typeof VIEW_OPTIONS)[number] as O['key']]: O['type'] extends 'boolean' ? boolean : O['type'] extends 'mode' ? ViewMode : number;
};
