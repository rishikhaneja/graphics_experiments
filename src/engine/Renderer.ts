// Renderer — the backend-agnostic interface that main.ts programs against.
// Each backend (WebGL2, WebGPU) implements this interface.
//
// The interface is intentionally minimal: just enough to draw a lit, textured
// mesh with an orbit camera. We don't try to abstract every GPU concept —
// that would be a full engine. Instead we expose the operations our demo needs.

import { mat4, vec3 } from "gl-matrix";

export interface VertexLayout {
  location: number;
  size: number; // number of float components
}

export interface TextureDesc {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface Renderer {
  /** Upload mesh data and describe its vertex layout. Returns an opaque handle. */
  createMesh(data: Float32Array, layout: VertexLayout[]): MeshHandle;

  /** Upload a texture from raw RGBA pixels. Returns an opaque handle. */
  createTexture(desc: TextureDesc): TextureHandle;

  /** Begin a frame: clear the screen. */
  beginFrame(): void;

  /** Set per-frame uniforms and draw a mesh. */
  draw(mesh: MeshHandle, texture: TextureHandle, uniforms: DrawUniforms): void;

  /** Resize the rendering surface to match the canvas. */
  resize(): void;
}

export interface MeshHandle {
  readonly vertexCount: number;
}

export interface TextureHandle {
  // Opaque — backends store their own data.
}

export interface DrawUniforms {
  model: mat4;
  viewProj: mat4;
  lightPos: vec3;
  cameraPos: vec3;
  normalMap?: TextureHandle;
  lightViewProj?: mat4; // light-space VP matrix for shadow mapping
}
