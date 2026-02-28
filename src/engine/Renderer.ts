// Renderer — the backend-agnostic interface that main.ts programs against.

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

export interface DrawCall {
  mesh: MeshHandle;
  texture: TextureHandle;
  normalMap?: TextureHandle;
  model: mat4;
}

export interface FrameUniforms {
  viewProj: mat4;
  lightPos: vec3;
  cameraPos: vec3;
  lightViewProj?: mat4;
}

export interface RenderOptions {
  shadows: boolean;
  normalMaps: boolean;
  postProcessing: boolean;
}

export interface Renderer {
  readonly aspect: number;
  createMesh(data: Float32Array, layout: VertexLayout[]): MeshHandle;
  createTexture(desc: TextureDesc): TextureHandle;
  resize(): void;

  /** Render a full frame: shadow pass (if lightViewProj) + main pass for all objects. */
  renderFrame(drawCalls: DrawCall[], uniforms: FrameUniforms, options: RenderOptions): void;
}

export interface MeshHandle {
  readonly vertexCount: number;
}

export interface TextureHandle {
  // Opaque — backends store their own data.
}
