import { describe, expect, it } from 'vitest';
import { cameraFrame, defaultView, setFrameTurn } from '../../src/render/renderer';

// The volume view turning with a rotating frame (ViewState.turnWithFrame): a camera at rest,
// seen from the frame, turned back about the frame's axis by Omega t.
describe('view turning with a rotating frame', () => {
  const close = (a: number[], b: number[]) => a.forEach((x, i) => expect(x).toBeCloseTo(b[i], 9));

  it('turns the camera back by Omega t about the axis, through the box center', () => {
    const v = { ...defaultView(), turnWithFrame: true };
    setFrameTurn(null, 0);
    const still = cameraFrame(v, 1);
    // A quarter turn about +x: (x, y, z) -> (x, z, -y) for the angle -pi/2.
    setFrameTurn([2, 0, 0], Math.PI / 4);
    const t = cameraFrame(v, 1);
    close(t.eye, [still.eye[0], still.eye[2], -still.eye[1]]);
    close(t.fwd, [still.fwd[0], still.fwd[2], -still.fwd[1]]);
    close(t.up, [still.up[0], still.up[2], -still.up[1]]);
    setFrameTurn(null, 0);
  });

  it('leaves the camera alone when off, in a slice, or without a frame', () => {
    const still = cameraFrame(defaultView(), 1);
    setFrameTurn([0, 0, 1], 1);
    close(cameraFrame(defaultView(), 1).eye, still.eye);
    close(cameraFrame({ ...defaultView(), turnWithFrame: true, mode: 'slice' }, 1).eye, still.eye);
    setFrameTurn([0, 0, 0], 1);
    close(cameraFrame({ ...defaultView(), turnWithFrame: true }, 1).eye, still.eye);
    setFrameTurn(null, 0);
  });
});
