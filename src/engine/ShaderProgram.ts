// ShaderProgram — compiles vertex + fragment GLSL, links them, and provides
// typed uniform setters. This is the first abstraction we extract because
// every draw call needs a program, and the boilerplate for compiling/linking
// is identical every time.

import { mat4, vec3 } from "gl-matrix";

export class ShaderProgram {
  readonly handle: WebGLProgram;
  private uniformLocations = new Map<string, WebGLUniformLocation | null>();

  constructor(
    private gl: WebGL2RenderingContext,
    vertexSource: string,
    fragmentSource: string
  ) {
    const vs = this.compile(gl.VERTEX_SHADER, vertexSource);
    const fs = this.compile(gl.FRAGMENT_SHADER, fragmentSource);

    const program = gl.createProgram();
    if (!program) throw new Error("Failed to create program");

    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Program link error:\n${log}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);

    this.handle = program;
  }

  use(): void {
    this.gl.useProgram(this.handle);
  }

  // Lazy-cache uniform locations — avoids repeated lookups per frame.
  getUniform(name: string): WebGLUniformLocation | null {
    if (!this.uniformLocations.has(name)) {
      this.uniformLocations.set(name, this.gl.getUniformLocation(this.handle, name));
    }
    return this.uniformLocations.get(name)!;
  }

  setMat4(name: string, value: mat4): void {
    this.gl.uniformMatrix4fv(this.getUniform(name), false, value);
  }

  setVec3(name: string, x: number, y: number, z: number): void {
    this.gl.uniform3f(this.getUniform(name), x, y, z);
  }

  setVec3v(name: string, value: vec3 | Float32Array | number[]): void {
    this.gl.uniform3fv(this.getUniform(name), value);
  }

  setInt(name: string, value: number): void {
    this.gl.uniform1i(this.getUniform(name), value);
  }

  private compile(type: GLenum, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error("Failed to create shader");

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compile error:\n${log}`);
    }
    return shader;
  }
}
