import type { TextureHandle } from "./Renderer";

export interface Material {
  texture: TextureHandle;
  normalMap?: TextureHandle;
  /** If true, this surface skips lighting and outputs its albedo directly at HDR brightness. */
  emissive?: boolean;
}
