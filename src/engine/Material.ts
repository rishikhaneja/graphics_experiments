import type { TextureHandle } from "./Renderer";

export interface Material {
  texture: TextureHandle;
  normalMap?: TextureHandle;
}
