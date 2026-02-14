// WebGL2Renderer — implements the Renderer interface using WebGL2.
//
// Step 11 restructures rendering to support multiple objects per frame.
// renderFrame() does:
//   1. Shadow pass — render ALL objects into the shadow map
//   2. Main pass — render ALL objects with lighting + shadow sampling

import type { Renderer, VertexLayout, TextureDesc, MeshHandle, TextureHandle, DrawCall, FrameUniforms } from "./Renderer";

interface GL2MeshHandle extends MeshHandle {
  vao: WebGLVertexArrayObject;
}

interface GL2TextureHandle extends TextureHandle {
  texture: WebGLTexture;
}

// ---- Shadow pass shader (depth only) ----
const SHADOW_VERT = `#version 300 es
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;
layout(location = 3) in vec3 aTangent;

uniform mat4 uModel;
uniform mat4 uLightViewProj;

void main() {
  gl_Position = uLightViewProj * uModel * vec4(aPosition, 1.0);
}
`;

const SHADOW_FRAG = `#version 300 es
precision mediump float;
out vec4 fragColor;
void main() {
  fragColor = vec4(1.0);
}
`;

// ---- Main pass shader ----
const VERT_SRC = `#version 300 es

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;
layout(location = 3) in vec3 aTangent;

uniform mat4 uModel;
uniform mat4 uViewProj;
uniform mat4 uLightViewProj;

out vec2 vTexCoord;
out vec3 vWorldPos;
out mat3 vTBN;
out vec4 vLightSpacePos;

void main() {
  vec4 worldPos = uModel * vec4(aPosition, 1.0);
  vWorldPos = worldPos.xyz;

  mat3 modelMat3 = mat3(uModel);
  vec3 N = normalize(modelMat3 * aNormal);
  vec3 T = normalize(modelMat3 * aTangent);
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T);
  vTBN = mat3(T, B, N);

  vTexCoord = aTexCoord;
  vLightSpacePos = uLightViewProj * worldPos;
  gl_Position = uViewProj * worldPos;
}
`;

const FRAG_SRC = `#version 300 es
precision mediump float;

in vec2 vTexCoord;
in vec3 vWorldPos;
in mat3 vTBN;
in vec4 vLightSpacePos;

uniform sampler2D uTexture;
uniform sampler2D uNormalMap;
uniform sampler2D uShadowMap;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform bool uHasNormalMap;

out vec4 fragColor;

float shadowCalc(vec3 projCoords) {
  if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
      projCoords.y < 0.0 || projCoords.y > 1.0 ||
      projCoords.z > 1.0) return 1.0;

  float bias = 0.005;
  float shadow = 0.0;
  vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));

  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      float closestDepth = texture(uShadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
      shadow += (projCoords.z - bias > closestDepth) ? 0.0 : 1.0;
    }
  }
  return shadow / 9.0;
}

void main() {
  vec3 texColor = texture(uTexture, vTexCoord).rgb;

  vec3 N;
  if (uHasNormalMap) {
    vec3 mapNormal = texture(uNormalMap, vTexCoord).rgb * 2.0 - 1.0;
    N = normalize(vTBN * mapNormal);
  } else {
    N = normalize(vTBN[2]);
  }

  vec3 L = normalize(uLightPos - vWorldPos);
  vec3 V = normalize(uCameraPos - vWorldPos);
  vec3 R = reflect(-L, N);

  float ambient = 0.15;
  float diffuse = max(dot(N, L), 0.0);
  float specular = pow(max(dot(R, V), 0.0), 32.0);

  vec3 projCoords = vLightSpacePos.xyz / vLightSpacePos.w;
  projCoords = projCoords * 0.5 + 0.5;
  float shadow = shadowCalc(projCoords);

  vec3 color = texColor * ambient
             + texColor * diffuse * shadow
             + vec3(1.0) * specular * 0.5 * shadow;
  fragColor = vec4(color, 1.0);
}
`;

const SHADOW_SIZE = 1024;

export class WebGL2Renderer implements Renderer {
  private gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;

  // Main pass
  private program: WebGLProgram;
  private uModel: WebGLUniformLocation | null;
  private uViewProj: WebGLUniformLocation | null;
  private uTexture: WebGLUniformLocation | null;
  private uNormalMap: WebGLUniformLocation | null;
  private uShadowMap: WebGLUniformLocation | null;
  private uLightPos: WebGLUniformLocation | null;
  private uCameraPos: WebGLUniformLocation | null;
  private uHasNormalMap: WebGLUniformLocation | null;
  private uLightViewProj: WebGLUniformLocation | null;

  // Shadow pass
  private shadowProgram: WebGLProgram;
  private uShadowModel: WebGLUniformLocation | null;
  private uShadowLightVP: WebGLUniformLocation | null;
  private shadowFBO: WebGLFramebuffer;
  private shadowDepthTex: WebGLTexture;

  private flatNormalTex: WebGLTexture;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2");
    if (!gl) throw new Error("WebGL2 not supported");
    this.gl = gl;

    // Main pass
    this.program = this.buildProgram(VERT_SRC, FRAG_SRC);
    this.uModel = gl.getUniformLocation(this.program, "uModel");
    this.uViewProj = gl.getUniformLocation(this.program, "uViewProj");
    this.uTexture = gl.getUniformLocation(this.program, "uTexture");
    this.uNormalMap = gl.getUniformLocation(this.program, "uNormalMap");
    this.uShadowMap = gl.getUniformLocation(this.program, "uShadowMap");
    this.uLightPos = gl.getUniformLocation(this.program, "uLightPos");
    this.uCameraPos = gl.getUniformLocation(this.program, "uCameraPos");
    this.uHasNormalMap = gl.getUniformLocation(this.program, "uHasNormalMap");
    this.uLightViewProj = gl.getUniformLocation(this.program, "uLightViewProj");

    // Shadow pass
    this.shadowProgram = this.buildProgram(SHADOW_VERT, SHADOW_FRAG);
    this.uShadowModel = gl.getUniformLocation(this.shadowProgram, "uModel");
    this.uShadowLightVP = gl.getUniformLocation(this.shadowProgram, "uLightViewProj");

    // Shadow FBO
    this.shadowDepthTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.shadowDepthTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, SHADOW_SIZE, SHADOW_SIZE, 0,
      gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.shadowFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.shadowDepthTex, 0);
    gl.drawBuffers([gl.NONE]);
    gl.readBuffer(gl.NONE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    gl.enable(gl.DEPTH_TEST);

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
    }
  }

  renderFrame(drawCalls: DrawCall[], u: FrameUniforms): void {
    const gl = this.gl;

    // ---- Shadow pass (all objects) ----
    if (u.lightViewProj) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFBO);
      gl.viewport(0, 0, SHADOW_SIZE, SHADOW_SIZE);
      gl.clear(gl.DEPTH_BUFFER_BIT);

      gl.useProgram(this.shadowProgram);
      gl.uniformMatrix4fv(this.uShadowLightVP, false, u.lightViewProj);

      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.FRONT);
      for (const dc of drawCalls) {
        const mesh = dc.mesh as GL2MeshHandle;
        gl.uniformMatrix4fv(this.uShadowModel, false, dc.model);
        gl.bindVertexArray(mesh.vao);
        gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
      }
      gl.cullFace(gl.BACK);
      gl.disable(gl.CULL_FACE);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    // ---- Main pass (all objects) ----
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.08, 0.08, 0.12, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uViewProj, false, u.viewProj);
    gl.uniform3fv(this.uLightPos, u.lightPos as unknown as Float32Array);
    gl.uniform3fv(this.uCameraPos, u.cameraPos as unknown as Float32Array);

    if (u.lightViewProj) {
      gl.uniformMatrix4fv(this.uLightViewProj, false, u.lightViewProj);
    }

    // Shadow map on unit 2
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowDepthTex);
    gl.uniform1i(this.uShadowMap, 2);

    for (const dc of drawCalls) {
      const mesh = dc.mesh as GL2MeshHandle;
      const tex = dc.texture as GL2TextureHandle;

      gl.uniformMatrix4fv(this.uModel, false, dc.model);

      // Unit 0: albedo
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex.texture);
      gl.uniform1i(this.uTexture, 0);

      // Unit 1: normal map
      gl.activeTexture(gl.TEXTURE1);
      const hasNormal = !!dc.normalMap;
      gl.bindTexture(gl.TEXTURE_2D, hasNormal ? (dc.normalMap as GL2TextureHandle).texture : this.flatNormalTex);
      gl.uniform1i(this.uNormalMap, 1);
      gl.uniform1i(this.uHasNormalMap, hasNormal ? 1 : 0);

      gl.bindVertexArray(mesh.vao);
      gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
    }
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
