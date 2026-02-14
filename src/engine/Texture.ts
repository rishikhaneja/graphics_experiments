// Texture — wraps a WebGL texture object. Supports creating from raw pixel
// data (Uint8Array) with configurable size and filtering.

export interface TextureOptions {
  width: number;
  height: number;
  data: Uint8Array;
  filter?: GLenum; // default: NEAREST
}

export class Texture {
  readonly handle: WebGLTexture;

  constructor(private gl: WebGL2RenderingContext, options: TextureOptions) {
    const texture = gl.createTexture();
    if (!texture) throw new Error("Failed to create texture");

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      options.width, options.height, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, options.data
    );

    const filter = options.filter ?? gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);

    this.handle = texture;
  }

  bind(unit: number = 0): void {
    this.gl.activeTexture(this.gl.TEXTURE0 + unit);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.handle);
  }
}
