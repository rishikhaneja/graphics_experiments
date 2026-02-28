import { mat4, vec3 } from "gl-matrix";

export class Light {
  position: vec3 = vec3.fromValues(3, 3, 0);
  /** Optional per-frame animation. Receives DOMHighResTimeStamp. */
  onUpdate: ((time: number) => void) | null = null;

  private _orthoSize = 4.0;
  private _near = 0.1;
  private _far = 12.0;

  update(time: number): void {
    if (this.onUpdate) this.onUpdate(time);
  }

  /** Builds the light-space view-projection for shadow mapping. */
  viewProjection(out: mat4): mat4 {
    const view = mat4.create();
    const proj = mat4.create();
    mat4.lookAt(view, this.position, [0, 0, 0], [0, 1, 0]);
    mat4.ortho(proj, -this._orthoSize, this._orthoSize, -this._orthoSize, this._orthoSize, this._near, this._far);
    mat4.multiply(out, proj, view);
    return out;
  }
}
