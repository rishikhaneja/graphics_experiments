// WebGL2Renderer — implements the Renderer interface using WebGL2.
//
// Step 9 adds normal mapping:
//   - Vertex shader now accepts a tangent attribute (location 3) and passes
//     the TBN matrix to the fragment shader.
//   - Fragment shader samples a normal map (texture unit 1) and transforms
//     the sample from tangent space to world space using the TBN matrix.
//   - When no normal map is bound, the shader falls back to the geometric normal.

import type { Renderer, VertexLayout, TextureDesc, MeshHandle, TextureHandle, DrawUniforms } from "./Renderer";

interface GL2MeshHandle extends MeshHandle {
  vao: WebGLVertexArrayObject;
}

interface GL2TextureHandle extends TextureHandle {
  texture: WebGLTexture;
}

const VERT_SRC = `#version 300 es

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;
layout(location = 3) in vec3 aTangent;

uniform mat4 uModel;
uniform mat4 uViewProj;

out vec2 vTexCoord;
out vec3 vWorldPos;
out mat3 vTBN;

void main() {
  vec4 worldPos = uModel * vec4(aPosition, 1.0);
  vWorldPos = worldPos.xyz;

  mat3 modelMat3 = mat3(uModel);
  vec3 N = normalize(modelMat3 * aNormal);
  vec3 T = normalize(modelMat3 * aTangent);
  // Re-orthogonalize T w.r.t. N (Gram-Schmidt)
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T);
  vTBN = mat3(T, B, N);

  vTexCoord = aTexCoord;
  gl_Position = uViewProj * worldPos;
}
`;

const FRAG_SRC = `#version 300 es
precision mediump float;

in vec2 vTexCoord;
in vec3 vWorldPos;
in mat3 vTBN;

uniform sampler2D uTexture;
uniform sampler2D uNormalMap;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform bool uHasNormalMap;

out vec4 fragColor;

void main() {
  vec3 texColor = texture(uTexture, vTexCoord).rgb;

  vec3 N;
  if (uHasNormalMap) {
    // Sample normal map (stored as RGB [0,1], remap to [-1,1])
    vec3 mapNormal = texture(uNormalMap, vTexCoord).rgb * 2.0 - 1.0;
    N = normalize(vTBN * mapNormal);
  } else {
    N = normalize(vTBN[2]); // Just the geometric normal (column 2)
  }

  vec3 L = normalize(uLightPos - vWorldPos);
  vec3 V = normalize(uCameraPos - vWorldPos);
  vec3 R = reflect(-L, N);

  float ambient = 0.15;
  float diffuse = max(dot(N, L), 0.0);
  float specular = pow(max(dot(R, V), 0.0), 32.0);

  vec3 color = texColor * (ambient + diffuse) + vec3(1.0) * specular * 0.5;
  fragColor = vec4(color, 1.0);
}
`;

export class WebGL2Renderer implements Renderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private canvas: HTMLCanvasElement;

  private uModel: WebGLUniformLocation | null;
  private uViewProj: WebGLUniformLocation | null;
  private uTexture: WebGLUniformLocation | null;
  private uNormalMap: WebGLUniformLocation | null;
  private uLightPos: WebGLUniformLocation | null;
  private uCameraPos: WebGLUniformLocation | null;
  private uHasNormalMap: WebGLUniformLocation | null;

  // 1x1 flat normal texture — used when no normal map is provided.
  private flatNormalTex: WebGLTexture;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2");
    if (!gl) throw new Error("WebGL2 not supported");
    this.gl = gl;

    this.program = this.buildProgram(VERT_SRC, FRAG_SRC);
    this.uModel = gl.getUniformLocation(this.program, "uModel");
    this.uViewProj = gl.getUniformLocation(this.program, "uViewProj");
    this.uTexture = gl.getUniformLocation(this.program, "uTexture");
    this.uNormalMap = gl.getUniformLocation(this.program, "uNormalMap");
    this.uLightPos = gl.getUniformLocation(this.program, "uLightPos");
    this.uCameraPos = gl.getUniformLocation(this.program, "uCameraPos");
    this.uHasNormalMap = gl.getUniformLocation(this.program, "uHasNormalMap");

    gl.enable(gl.DEPTH_TEST);

    // Create a 1x1 flat normal texture (pointing straight up in tangent space: 0,0,1 → 128,128,255)
    this.flatNormalTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.flatNormalTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([128, 128, 255, 255]));
  }

  createMesh(data: Float32Array, layout: VertexLayout[]): GL2MeshHandle {
    const gl = this.gl;
    const floatsPerVertex = layout.reduce((s, a) => s + a.size, 0);

    const vbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    const stride = floatsPerVertex * 4;
    let offset = 0;
    for (const attr of layout) {
      gl.enableVertexAttribArray(attr.location);
      gl.vertexAttribPointer(attr.location, attr.size, gl.FLOAT, false, stride, offset);
      offset += attr.size * 4;
    }
    gl.bindVertexArray(null);

    return { vao, vertexCount: data.length / floatsPerVertex };
  }

  createTexture(desc: TextureDesc): GL2TextureHandle {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, desc.width, desc.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, desc.data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return { texture };
  }

  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.gl.viewport(0, 0, w, h);
    }
  }

  beginFrame(): void {
    const gl = this.gl;
    gl.clearColor(0.08, 0.08, 0.12, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }

  draw(mesh: GL2MeshHandle, texture: GL2TextureHandle, u: DrawUniforms): void {
    const gl = this.gl;
    gl.useProgram(this.program);

    gl.uniformMatrix4fv(this.uModel, false, u.model);
    gl.uniformMatrix4fv(this.uViewProj, false, u.viewProj);
    gl.uniform3fv(this.uLightPos, u.lightPos as unknown as Float32Array);
    gl.uniform3fv(this.uCameraPos, u.cameraPos as unknown as Float32Array);

    // Albedo texture on unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture.texture);
    gl.uniform1i(this.uTexture, 0);

    // Normal map on unit 1 (or flat fallback)
    gl.activeTexture(gl.TEXTURE1);
    const hasNormal = !!u.normalMap;
    if (hasNormal) {
      gl.bindTexture(gl.TEXTURE_2D, (u.normalMap as GL2TextureHandle).texture);
    } else {
      gl.bindTexture(gl.TEXTURE_2D, this.flatNormalTex);
    }
    gl.uniform1i(this.uNormalMap, 1);
    gl.uniform1i(this.uHasNormalMap, hasNormal ? 1 : 0);

    gl.bindVertexArray(mesh.vao);
    gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
  }

  get aspect(): number {
    return this.canvas.width / this.canvas.height;
  }

  private buildProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const gl = this.gl;
    const vs = this.compileShader(gl.VERTEX_SHADER, vertSrc);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`Link error: ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  private compileShader(type: GLenum, src: string): WebGLShader {
    const gl = this.gl;
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error(`Shader error: ${gl.getShaderInfoLog(s)}`);
    }
    return s;
  }
}
