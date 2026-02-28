// Scene — holds the entity list, camera, and light; builds DrawCall[] and FrameUniforms each frame.

import { mat4 } from "gl-matrix";
import type { DrawCall, FrameUniforms } from "./Renderer";
import type { OrbitCamera } from "./OrbitCamera";
import type { Light } from "./Light";
import type { Entity } from "./Entity";

export class Scene {
  entities: Entity[] = [];
  camera: OrbitCamera;
  light: Light;

  constructor(camera: OrbitCamera, light: Light) {
    this.camera = camera;
    this.light = light;
  }

  add(entity: Entity): void {
    this.entities.push(entity);
  }

  /** Update all entities and the light. */
  update(time: number): void {
    for (const entity of this.entities) {
      entity.update(time);
    }
    this.light.update(time);
  }

  /** Build the DrawCall[] and FrameUniforms the renderer needs. */
  buildFrame(aspect: number): { drawCalls: DrawCall[]; uniforms: FrameUniforms } {
    const viewProj = mat4.create();
    this.camera.viewProjection(viewProj, aspect);

    const lightViewProj = mat4.create();
    this.light.viewProjection(lightViewProj);

    return {
      drawCalls: this.entities.map((e) => e.drawCall()),
      uniforms: {
        viewProj,
        lightPos: this.light.position,
        cameraPos: this.camera.position(),
        lightViewProj,
      },
    };
  }
}
