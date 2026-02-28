import { mat4 } from "gl-matrix";
import type { MeshHandle, DrawCall } from "./Renderer";
import type { Material } from "./Material";
import { Transform } from "./Transform";

export class Entity {
  transform = new Transform();
  mesh: MeshHandle;
  material: Material;
  /** Optional per-frame behavior. Called with DOMHighResTimeStamp. */
  onUpdate: ((time: number) => void) | null = null;

  private _modelMatrix = mat4.create();

  constructor(mesh: MeshHandle, material: Material) {
    this.mesh = mesh;
    this.material = material;
  }

  update(time: number): void {
    if (this.onUpdate) this.onUpdate(time);
  }

  /** Produces the DrawCall the renderer needs. */
  drawCall(): DrawCall {
    this.transform.worldMatrix(this._modelMatrix);
    return {
      mesh: this.mesh,
      texture: this.material.texture,
      normalMap: this.material.normalMap,
      model: this._modelMatrix,
    };
  }
}
