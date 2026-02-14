// WebGL2Renderer — implements the Renderer interface using WebGL2.
//
// Step 12: Deferred rendering.
//
// The forward renderer computed lighting per-fragment for every object. That
// works fine for a single light, but scales as O(objects × lights). Deferred
// rendering decouples geometry from lighting:
//
//   1. Shadow pass  — render depth from the light's point of view (unchanged).
//   2. G-Buffer pass — render ALL geometry into three textures (MRT):
//        attachment 0: world-space position  (RGBA16F)
//        attachment 1: world-space normal    (RGBA16F)
//        attachment 2: albedo color          (RGBA8)
//      No lighting math happens here — just material + geometry data.
//   3. Lighting pass — a fullscreen quad samples the G-Buffer textures and
//      the shadow map, then computes Phong lighting in screen space. This
//      runs once regardless of how many objects are in the scene.
//
// The key insight: lighting cost is now O(pixels × lights) instead of
// O(fragments × lights). Overlapping geometry doesn't re-shade — only the
// front-most fragment (written to the G-Buffer via depth test) gets lit.

import type { Renderer, VertexLayout, TextureDesc, MeshHandle, TextureHandle, DrawCall, FrameUniforms } from "./Renderer";

interface GL2MeshHandle extends MeshHandle {
  vao: WebGLVertexArrayObject;
}

interface GL2TextureHandle extends TextureHandle {
  texture: WebGLTexture;
}

// ---------------------------------------------------------------------------
// Shadow pass shader (depth only) — unchanged from forward renderer
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// G-Buffer geometry pass — writes position, normal, albedo to MRT
// ---------------------------------------------------------------------------

const GBUF_VERT = `#version 300 es

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
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T);
  vTBN = mat3(T, B, N);

  vTexCoord = aTexCoord;
  gl_Position = uViewProj * worldPos;
}
`;

// Fragment shader outputs to 3 render targets via layout qualifiers.
// No lighting math — just store geometry data for the lighting pass.
const GBUF_FRAG = `#version 300 es
precision highp float;

in vec2 vTexCoord;
in vec3 vWorldPos;
in mat3 vTBN;

uniform sampler2D uTexture;
uniform sampler2D uNormalMap;
uniform bool uHasNormalMap;

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedo;

void main() {
  gPosition = vec4(vWorldPos, 1.0);

  vec3 N;
  if (uHasNormalMap) {
    vec3 mapNormal = texture(uNormalMap, vTexCoord).rgb * 2.0 - 1.0;
    N = normalize(vTBN * mapNormal);
  } else {
    N = normalize(vTBN[2]);
  }
  gNormal = vec4(N, 1.0);

  gAlbedo = texture(uTexture, vTexCoord);
}
`;

// ---------------------------------------------------------------------------
// Lighting pass — fullscreen quad, reads G-Buffer + shadow map
// ---------------------------------------------------------------------------

const LIGHT_VERT = `#version 300 es

// Fullscreen triangle trick: 3 vertices, no VBO needed.
// gl_VertexID 0 → (-1,-1), 1 → (3,-1), 2 → (-1,3)
// This covers the entire screen with a single triangle.
out vec2 vUV;

void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x, y) * 0.5 + 0.5;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

const LIGHT_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;

uniform sampler2D uGPosition;
uniform sampler2D uGNormal;
uniform sampler2D uGAlbedo;
uniform sampler2D uShadowMap;

uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform mat4 uLightViewProj;

out vec4 fragColor;

float shadowCalc(vec3 worldPos) {
  vec4 lightSpacePos = uLightViewProj * vec4(worldPos, 1.0);
  vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
  projCoords = projCoords * 0.5 + 0.5;

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
  vec3 worldPos = texture(uGPosition, vUV).rgb;
  vec3 N = normalize(texture(uGNormal, vUV).rgb);
  vec3 albedo = texture(uGAlbedo, vUV).rgb;

  // Discard background pixels (position = 0,0,0 with alpha 0)
  if (texture(uGPosition, vUV).a == 0.0) {
    fragColor = vec4(0.08, 0.08, 0.12, 1.0);
    return;
  }

  vec3 L = normalize(uLightPos - worldPos);
  vec3 V = normalize(uCameraPos - worldPos);
  vec3 R = reflect(-L, N);

  float ambient = 0.15;
  float diffuse = max(dot(N, L), 0.0);
  float specular = pow(max(dot(R, V), 0.0), 32.0);

  float shadow = shadowCalc(worldPos);

  vec3 color = albedo * ambient
             + albedo * diffuse * shadow
             + vec3(1.0) * specular * 0.5 * shadow;

  fragColor = vec4(color, 1.0);
}
`;

const SHADOW_SIZE = 1024;

export class WebGL2Renderer implements Renderer {
  private gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;

  // G-Buffer geometry pass
  private gBufProgram: WebGLProgram;
  private gBufUModel: WebGLUniformLocation | null;
  private gBufUViewProj: WebGLUniformLocation | null;
  private gBufUTexture: WebGLUniformLocation | null;
  private gBufUNormalMap: WebGLUniformLocation | null;
  private gBufUHasNormalMap: WebGLUniformLocation | null;

  // G-Buffer FBO + textures
  private gBufFBO: WebGLFramebuffer;
  private gPositionTex: WebGLTexture;
  private gNormalTex: WebGLTexture;
  private gAlbedoTex: WebGLTexture;
  private gDepthRBO: WebGLRenderbuffer;
  private gBufWidth = 0;
  private gBufHeight = 0;

  // Lighting pass
  private lightProgram: WebGLProgram;
  private lightUGPosition: WebGLUniformLocation | null;
  private lightUGNormal: WebGLUniformLocation | null;
  private lightUGAlbedo: WebGLUniformLocation | null;
  private lightUShadowMap: WebGLUniformLocation | null;
  private lightULightPos: WebGLUniformLocation | null;
  private lightUCameraPos: WebGLUniformLocation | null;
  private lightULightViewProj: WebGLUniformLocation | null;
  private fullscreenVAO: WebGLVertexArrayObject;

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

    // Need EXT_color_buffer_float for RGBA16F render targets
    const cbfExt = gl.getExtension("EXT_color_buffer_float");
    if (!cbfExt) throw new Error("EXT_color_buffer_float not supported — required for deferred rendering");

    // ---- G-Buffer geometry pass ----
    this.gBufProgram = this.buildProgram(GBUF_VERT, GBUF_FRAG);
    this.gBufUModel = gl.getUniformLocation(this.gBufProgram, "uModel");
    this.gBufUViewProj = gl.getUniformLocation(this.gBufProgram, "uViewProj");
    this.gBufUTexture = gl.getUniformLocation(this.gBufProgram, "uTexture");
    this.gBufUNormalMap = gl.getUniformLocation(this.gBufProgram, "uNormalMap");
    this.gBufUHasNormalMap = gl.getUniformLocation(this.gBufProgram, "uHasNormalMap");

    // ---- G-Buffer FBO (created at correct size in resize/rebuildGBuffer) ----
    this.gBufFBO = gl.createFramebuffer()!;
    this.gPositionTex = gl.createTexture()!;
    this.gNormalTex = gl.createTexture()!;
    this.gAlbedoTex = gl.createTexture()!;
    this.gDepthRBO = gl.createRenderbuffer()!;

    // ---- Lighting pass ----
    this.lightProgram = this.buildProgram(LIGHT_VERT, LIGHT_FRAG);
    this.lightUGPosition = gl.getUniformLocation(this.lightProgram, "uGPosition");
    this.lightUGNormal = gl.getUniformLocation(this.lightProgram, "uGNormal");
    this.lightUGAlbedo = gl.getUniformLocation(this.lightProgram, "uGAlbedo");
    this.lightUShadowMap = gl.getUniformLocation(this.lightProgram, "uShadowMap");
    this.lightULightPos = gl.getUniformLocation(this.lightProgram, "uLightPos");
    this.lightUCameraPos = gl.getUniformLocation(this.lightProgram, "uCameraPos");
    this.lightULightViewProj = gl.getUniformLocation(this.lightProgram, "uLightViewProj");

    // Empty VAO for the fullscreen triangle (uses gl_VertexID, no attributes)
    this.fullscreenVAO = gl.createVertexArray()!;

    // ---- Shadow pass ----
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

    // 1x1 flat normal texture (tangent-space up)
    this.flatNormalTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.flatNormalTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([128, 128, 255, 255]));
  }

  // Rebuild G-Buffer textures when the canvas size changes.
  private rebuildGBuffer(w: number, h: number): void {
    const gl = this.gl;
    this.gBufWidth = w;
    this.gBufHeight = h;

    // Position — RGBA16F (world-space XYZ, alpha flags occupancy)
    gl.bindTexture(gl.TEXTURE_2D, this.gPositionTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Normal — RGBA16F (world-space normal)
    gl.bindTexture(gl.TEXTURE_2D, this.gNormalTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Albedo — RGBA8
    gl.bindTexture(gl.TEXTURE_2D, this.gAlbedoTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Depth renderbuffer
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.gDepthRBO);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);

    // Attach to FBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gBufFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.gPositionTex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.gNormalTex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, this.gAlbedoTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.gDepthRBO);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`G-Buffer FBO incomplete: 0x${status.toString(16)}`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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
    // Rebuild G-Buffer if canvas size changed
    if (this.gBufWidth !== w || this.gBufHeight !== h) {
      this.rebuildGBuffer(w, h);
    }
  }

  renderFrame(drawCalls: DrawCall[], u: FrameUniforms): void {
    const gl = this.gl;

    // ---- Pass 1: Shadow (all objects, depth only) ----
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

    // ---- Pass 2: G-Buffer geometry (all objects → MRT) ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gBufFBO);
    gl.viewport(0, 0, this.gBufWidth, this.gBufHeight);
    gl.clearColor(0.0, 0.0, 0.0, 0.0); // alpha=0 marks empty pixels
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.gBufProgram);
    gl.uniformMatrix4fv(this.gBufUViewProj, false, u.viewProj);

    for (const dc of drawCalls) {
      const mesh = dc.mesh as GL2MeshHandle;
      const tex = dc.texture as GL2TextureHandle;

      gl.uniformMatrix4fv(this.gBufUModel, false, dc.model);

      // Unit 0: albedo
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex.texture);
      gl.uniform1i(this.gBufUTexture, 0);

      // Unit 1: normal map
      gl.activeTexture(gl.TEXTURE1);
      const hasNormal = !!dc.normalMap;
      gl.bindTexture(gl.TEXTURE_2D, hasNormal ? (dc.normalMap as GL2TextureHandle).texture : this.flatNormalTex);
      gl.uniform1i(this.gBufUNormalMap, 1);
      gl.uniform1i(this.gBufUHasNormalMap, hasNormal ? 1 : 0);

      gl.bindVertexArray(mesh.vao);
      gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
    }

    // ---- Pass 3: Lighting (fullscreen quad, reads G-Buffer + shadow map) ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.lightProgram);

    // Bind G-Buffer textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.gPositionTex);
    gl.uniform1i(this.lightUGPosition, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.gNormalTex);
    gl.uniform1i(this.lightUGNormal, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.gAlbedoTex);
    gl.uniform1i(this.lightUGAlbedo, 2);

    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.shadowDepthTex);
    gl.uniform1i(this.lightUShadowMap, 3);

    // Lighting uniforms
    gl.uniform3fv(this.lightULightPos, u.lightPos as unknown as Float32Array);
    gl.uniform3fv(this.lightUCameraPos, u.cameraPos as unknown as Float32Array);
    if (u.lightViewProj) {
      gl.uniformMatrix4fv(this.lightULightViewProj, false, u.lightViewProj);
    }

    // Draw fullscreen triangle (3 vertices, empty VAO, shader uses gl_VertexID)
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(this.fullscreenVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.DEPTH_TEST);
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
