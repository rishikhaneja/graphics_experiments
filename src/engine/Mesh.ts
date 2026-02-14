// Mesh — bundles a VBO + VAO with a typed vertex layout description.
// You describe your attributes once and the Mesh sets up the VAO pointers.
//
// This removes the tedious, error-prone stride/offset calculations from the
// main code. The pattern is:
//   1. Define attributes: [{ name, size }, ...]
//   2. Provide vertex data as a Float32Array
//   3. Mesh computes stride, offsets, and creates the VAO

export interface VertexAttribute {
  location: number; // shader layout(location = N)
  size: number;     // number of components (e.g. 3 for vec3)
}

export class Mesh {
  readonly vao: WebGLVertexArrayObject;
  readonly vertexCount: number;

  constructor(
    private gl: WebGL2RenderingContext,
    data: Float32Array,
    attributes: VertexAttribute[]
  ) {
    // Total floats per vertex = sum of all attribute sizes.
    const floatsPerVertex = attributes.reduce((sum, a) => sum + a.size, 0);
    this.vertexCount = data.length / floatsPerVertex;

    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

    const vao = gl.createVertexArray();
    if (!vao) throw new Error("Failed to create VAO");
    gl.bindVertexArray(vao);

    const stride = floatsPerVertex * Float32Array.BYTES_PER_ELEMENT;
    let offset = 0;

    for (const attr of attributes) {
      gl.enableVertexAttribArray(attr.location);
      gl.vertexAttribPointer(attr.location, attr.size, gl.FLOAT, false, stride, offset);
      offset += attr.size * Float32Array.BYTES_PER_ELEMENT;
    }

    gl.bindVertexArray(null);
    this.vao = vao;
  }

  draw(): void {
    this.gl.bindVertexArray(this.vao);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, this.vertexCount);
  }
}
