import { mat4, vec3, quat } from "gl-matrix";

export class Transform {
  position: vec3 = vec3.create();
  rotation: quat = quat.create();
  scale: vec3 = vec3.fromValues(1, 1, 1);
  parent: Transform | null = null;

  /** Builds the local model matrix: T × R × S */
  localMatrix(out: mat4): mat4 {
    mat4.fromRotationTranslationScale(out, this.rotation, this.position, this.scale);
    return out;
  }

  /** Builds the world matrix (walks up parent chain). */
  worldMatrix(out: mat4): mat4 {
    this.localMatrix(out);
    if (this.parent) {
      const parentWorld = mat4.create();
      this.parent.worldMatrix(parentWorld);
      mat4.multiply(out, parentWorld, out);
    }
    return out;
  }
}
