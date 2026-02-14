// WebGL2Renderer — implements the Renderer interface using WebGL2.
//
// Step 13: Post-processing — bloom and tone mapping.
//
// Building on the deferred pipeline from Step 12, we add a post-processing
// chain after the lighting pass:
//
//   1. Shadow pass    — depth from light (unchanged)
//   2. G-Buffer pass  — geometry → MRT (unchanged)
//   3. Lighting pass  — now renders to an HDR framebuffer (RGBA16F) instead
//                       of the screen. Specular is boosted so bright spots
//                       produce values > 1.0 — these feed the bloom.
//   4. Bloom extract  — threshold pass pulls pixels brighter than 1.0 into
//                       a separate texture.
//   5. Gaussian blur  — two-pass (horizontal + vertical) separable blur,
//                       repeated several times via ping-pong FBOs.
//   6. Composite      — combines the HDR scene with the blurred bloom and
//                       applies ACES filmic tone mapping to bring everything
//                       back into [0,1] for display.
//
// The bloom creates a soft glow around specular highlights. Tone mapping
// compresses the HDR range gracefully — bright areas saturate smoothly
// instead of clipping to white.

import type { Renderer, VertexLayout, TextureDesc, MeshHandle, TextureHandle, DrawCall, FrameUniforms, RenderOptions } from "./Renderer";

interface GL2MeshHandle extends MeshHandle {
  vao: WebGLVertexArrayObject;
}

interface GL2TextureHandle extends TextureHandle {
  texture: WebGLTexture;
}

// ---------------------------------------------------------------------------
// Shadow pass shader (depth only)
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
// Fullscreen triangle vertex shader — shared by all post-processing passes
// ---------------------------------------------------------------------------

const FULLSCREEN_VERT = `#version 300 es
out vec2 vUV;

void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUV = vec2(x, y) * 0.5 + 0.5;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Lighting pass — reads G-Buffer + shadow map, outputs HDR color
// ---------------------------------------------------------------------------

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

  if (texture(uGPosition, vUV).a == 0.0) {
    fragColor = vec4(0.08, 0.08, 0.12, 1.0);
    return;
  }

  vec3 L = normalize(uLightPos - worldPos);
  vec3 V = normalize(uCameraPos - worldPos);
  vec3 R = reflect(-L, N);

  float ambient = 0.15;
  float diffuse = max(dot(N, L), 0.0);
  // Boosted specular — values > 1.0 feed the bloom
  float specular = pow(max(dot(R, V), 0.0), 32.0) * 2.0;

  float shadow = shadowCalc(worldPos);

  vec3 color = albedo * ambient
             + albedo * diffuse * shadow
             + vec3(1.0) * specular * shadow;

  fragColor = vec4(color, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Bloom extraction — threshold bright pixels
// ---------------------------------------------------------------------------

const BLOOM_EXTRACT_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;
uniform sampler2D uScene;
uniform float uThreshold;
out vec4 fragColor;

void main() {
  vec3 color = texture(uScene, vUV).rgb;
  float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
  if (brightness > uThreshold) {
    fragColor = vec4(color - vec3(uThreshold), 1.0);
  } else {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
  }
}
`;

// ---------------------------------------------------------------------------
// Gaussian blur — separable (horizontal or vertical per pass)
// ---------------------------------------------------------------------------

const BLUR_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;
uniform sampler2D uImage;
uniform vec2 uDirection; // (1/w, 0) for horizontal, (0, 1/h) for vertical

out vec4 fragColor;

// 9-tap Gaussian weights (sigma ≈ 4)
const float weights[5] = float[](0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);

void main() {
  vec3 result = texture(uImage, vUV).rgb * weights[0];
  for (int i = 1; i < 5; i++) {
    vec2 offset = uDirection * float(i);
    result += texture(uImage, vUV + offset).rgb * weights[i];
    result += texture(uImage, vUV - offset).rgb * weights[i];
  }
  fragColor = vec4(result, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Composite + tone mapping — combine HDR scene with bloom, apply ACES
// ---------------------------------------------------------------------------

const COMPOSITE_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomStrength;
uniform float uExposure;
uniform bool uToneMap;

out vec4 fragColor;

// ACES filmic tone mapping (approximation by Krzysztof Narkowicz)
vec3 aces(vec3 x) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec3 scene = texture(uScene, vUV).rgb;
  vec3 bloom = texture(uBloom, vUV).rgb;

  vec3 hdr = scene + bloom * uBloomStrength;

  if (uToneMap) {
    vec3 mapped = aces(hdr * uExposure);
    mapped = pow(mapped, vec3(1.0 / 2.2));
    fragColor = vec4(mapped, 1.0);
  } else {
    fragColor = vec4(hdr, 1.0);
  }
}
`;

const SHADOW_SIZE = 1024;
const BLOOM_PASSES = 5; // number of horizontal+vertical blur iterations

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

  // Lighting pass (→ HDR FBO)
  private lightProgram: WebGLProgram;
  private lightUGPosition: WebGLUniformLocation | null;
  private lightUGNormal: WebGLUniformLocation | null;
  private lightUGAlbedo: WebGLUniformLocation | null;
  private lightUShadowMap: WebGLUniformLocation | null;
  private lightULightPos: WebGLUniformLocation | null;
  private lightUCameraPos: WebGLUniformLocation | null;
  private lightULightViewProj: WebGLUniformLocation | null;
  private hdrFBO: WebGLFramebuffer;
  private hdrTex: WebGLTexture;

  // Bloom extraction
  private bloomExtractProgram: WebGLProgram;
  private bloomExtractUScene: WebGLUniformLocation | null;
  private bloomExtractUThreshold: WebGLUniformLocation | null;

  // Gaussian blur (ping-pong)
  private blurProgram: WebGLProgram;
  private blurUImage: WebGLUniformLocation | null;
  private blurUDirection: WebGLUniformLocation | null;
  private bloomFBOs: [WebGLFramebuffer, WebGLFramebuffer];
  private bloomTextures: [WebGLTexture, WebGLTexture];

  // Composite + tone mapping
  private compositeProgram: WebGLProgram;
  private compositeUScene: WebGLUniformLocation | null;
  private compositeUBloom: WebGLUniformLocation | null;
  private compositeUBloomStrength: WebGLUniformLocation | null;
  private compositeUExposure: WebGLUniformLocation | null;
  private compositeUToneMap: WebGLUniformLocation | null;

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

    const cbfExt = gl.getExtension("EXT_color_buffer_float");
    if (!cbfExt) throw new Error("EXT_color_buffer_float not supported");

    // Empty VAO for fullscreen triangle passes
    this.fullscreenVAO = gl.createVertexArray()!;

    // ---- G-Buffer geometry pass ----
    this.gBufProgram = this.buildProgram(GBUF_VERT, GBUF_FRAG);
    this.gBufUModel = gl.getUniformLocation(this.gBufProgram, "uModel");
    this.gBufUViewProj = gl.getUniformLocation(this.gBufProgram, "uViewProj");
    this.gBufUTexture = gl.getUniformLocation(this.gBufProgram, "uTexture");
    this.gBufUNormalMap = gl.getUniformLocation(this.gBufProgram, "uNormalMap");
    this.gBufUHasNormalMap = gl.getUniformLocation(this.gBufProgram, "uHasNormalMap");

    this.gBufFBO = gl.createFramebuffer()!;
    this.gPositionTex = gl.createTexture()!;
    this.gNormalTex = gl.createTexture()!;
    this.gAlbedoTex = gl.createTexture()!;
    this.gDepthRBO = gl.createRenderbuffer()!;

    // ---- Lighting pass (renders to HDR FBO) ----
    this.lightProgram = this.buildProgram(FULLSCREEN_VERT, LIGHT_FRAG);
    this.lightUGPosition = gl.getUniformLocation(this.lightProgram, "uGPosition");
    this.lightUGNormal = gl.getUniformLocation(this.lightProgram, "uGNormal");
    this.lightUGAlbedo = gl.getUniformLocation(this.lightProgram, "uGAlbedo");
    this.lightUShadowMap = gl.getUniformLocation(this.lightProgram, "uShadowMap");
    this.lightULightPos = gl.getUniformLocation(this.lightProgram, "uLightPos");
    this.lightUCameraPos = gl.getUniformLocation(this.lightProgram, "uCameraPos");
    this.lightULightViewProj = gl.getUniformLocation(this.lightProgram, "uLightViewProj");
    this.hdrFBO = gl.createFramebuffer()!;
    this.hdrTex = gl.createTexture()!;

    // ---- Bloom extraction ----
    this.bloomExtractProgram = this.buildProgram(FULLSCREEN_VERT, BLOOM_EXTRACT_FRAG);
    this.bloomExtractUScene = gl.getUniformLocation(this.bloomExtractProgram, "uScene");
    this.bloomExtractUThreshold = gl.getUniformLocation(this.bloomExtractProgram, "uThreshold");

    // ---- Gaussian blur ----
    this.blurProgram = this.buildProgram(FULLSCREEN_VERT, BLUR_FRAG);
    this.blurUImage = gl.getUniformLocation(this.blurProgram, "uImage");
    this.blurUDirection = gl.getUniformLocation(this.blurProgram, "uDirection");
    this.bloomFBOs = [gl.createFramebuffer()!, gl.createFramebuffer()!];
    this.bloomTextures = [gl.createTexture()!, gl.createTexture()!];

    // ---- Composite + tone mapping ----
    this.compositeProgram = this.buildProgram(FULLSCREEN_VERT, COMPOSITE_FRAG);
    this.compositeUScene = gl.getUniformLocation(this.compositeProgram, "uScene");
    this.compositeUBloom = gl.getUniformLocation(this.compositeProgram, "uBloom");
    this.compositeUBloomStrength = gl.getUniformLocation(this.compositeProgram, "uBloomStrength");
    this.compositeUExposure = gl.getUniformLocation(this.compositeProgram, "uExposure");
    this.compositeUToneMap = gl.getUniformLocation(this.compositeProgram, "uToneMap");

    // ---- Shadow pass ----
    this.shadowProgram = this.buildProgram(SHADOW_VERT, SHADOW_FRAG);
    this.uShadowModel = gl.getUniformLocation(this.shadowProgram, "uModel");
    this.uShadowLightVP = gl.getUniformLocation(this.shadowProgram, "uLightViewProj");

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

  // Rebuild all screen-sized FBOs when canvas size changes.
  private rebuildScreenFBOs(w: number, h: number): void {
    const gl = this.gl;
    this.gBufWidth = w;
    this.gBufHeight = h;

    // --- G-Buffer ---
    this.initTexture(this.gPositionTex, gl.RGBA16F, gl.RGBA, gl.FLOAT, w, h);
    this.initTexture(this.gNormalTex, gl.RGBA16F, gl.RGBA, gl.FLOAT, w, h);
    this.initTexture(this.gAlbedoTex, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, w, h);

    gl.bindRenderbuffer(gl.RENDERBUFFER, this.gDepthRBO);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gBufFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.gPositionTex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.gNormalTex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, this.gAlbedoTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.gDepthRBO);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
    this.checkFBO("G-Buffer");

    // --- HDR scene FBO ---
    this.initTexture(this.hdrTex, gl.RGBA16F, gl.RGBA, gl.FLOAT, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.hdrFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.hdrTex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    this.checkFBO("HDR");

    // --- Bloom ping-pong FBOs (half resolution for cheaper blur) ---
    const bw = Math.max(1, w >> 1);
    const bh = Math.max(1, h >> 1);
    for (let i = 0; i < 2; i++) {
      this.initTexture(this.bloomTextures[i], gl.RGBA16F, gl.RGBA, gl.FLOAT, bw, bh);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFBOs[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.bloomTextures[i], 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      this.checkFBO(`Bloom ${i}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private initTexture(tex: WebGLTexture, internalFormat: GLenum, format: GLenum, type: GLenum, w: number, h: number): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  private checkFBO(name: string): void {
    const gl = this.gl;
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`${name} FBO incomplete: 0x${status.toString(16)}`);
    }
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
    if (this.gBufWidth !== w || this.gBufHeight !== h) {
      this.rebuildScreenFBOs(w, h);
    }
  }

  renderFrame(drawCalls: DrawCall[], u: FrameUniforms, opts: RenderOptions): void {
    const gl = this.gl;
    const w = this.gBufWidth;
    const h = this.gBufHeight;

    // ---- Pass 1: Shadow ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFBO);
    gl.viewport(0, 0, SHADOW_SIZE, SHADOW_SIZE);
    gl.clear(gl.DEPTH_BUFFER_BIT); // clears to 1.0 — "no shadow" everywhere

    if (opts.shadows && u.lightViewProj) {
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
    }

    // ---- Pass 2: G-Buffer ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gBufFBO);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.gBufProgram);
    gl.uniformMatrix4fv(this.gBufUViewProj, false, u.viewProj);

    for (const dc of drawCalls) {
      const mesh = dc.mesh as GL2MeshHandle;
      const tex = dc.texture as GL2TextureHandle;

      gl.uniformMatrix4fv(this.gBufUModel, false, dc.model);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex.texture);
      gl.uniform1i(this.gBufUTexture, 0);

      gl.activeTexture(gl.TEXTURE1);
      const hasNormal = opts.normalMaps && !!dc.normalMap;
      gl.bindTexture(gl.TEXTURE_2D, hasNormal ? (dc.normalMap as GL2TextureHandle).texture : this.flatNormalTex);
      gl.uniform1i(this.gBufUNormalMap, 1);
      gl.uniform1i(this.gBufUHasNormalMap, hasNormal ? 1 : 0);

      gl.bindVertexArray(mesh.vao);
      gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
    }

    // All remaining passes are fullscreen — no depth test needed
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(this.fullscreenVAO);

    // ---- Pass 3: Lighting → HDR FBO ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.hdrFBO);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.lightProgram);

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

    gl.uniform3fv(this.lightULightPos, u.lightPos as unknown as Float32Array);
    gl.uniform3fv(this.lightUCameraPos, u.cameraPos as unknown as Float32Array);
    if (u.lightViewProj) {
      gl.uniformMatrix4fv(this.lightULightViewProj, false, u.lightViewProj);
    }

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // ---- Passes 4–5: Bloom (only when post-processing is on) ----
    if (opts.postProcessing) {
      const bw = Math.max(1, w >> 1);
      const bh = Math.max(1, h >> 1);

      // Pass 4: Bloom extraction → bloom FBO 0
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFBOs[0]);
      gl.viewport(0, 0, bw, bh);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(this.bloomExtractProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.hdrTex);
      gl.uniform1i(this.bloomExtractUScene, 0);
      gl.uniform1f(this.bloomExtractUThreshold, 1.0);

      gl.drawArrays(gl.TRIANGLES, 0, 3);

      // Pass 5: Gaussian blur ping-pong
      gl.useProgram(this.blurProgram);
      gl.uniform1i(this.blurUImage, 0);

      for (let i = 0; i < BLOOM_PASSES; i++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFBOs[1]);
        gl.viewport(0, 0, bw, bh);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.bloomTextures[0]);
        gl.uniform2f(this.blurUDirection, 1.0 / bw, 0.0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFBOs[0]);
        gl.viewport(0, 0, bw, bh);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.bloomTextures[1]);
        gl.uniform2f(this.blurUDirection, 0.0, 1.0 / bh);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
    }

    // ---- Pass 6: Composite + tone mapping → screen ----
    // Always runs — converts HDR to screen. bloomStrength=0 when post-processing is off.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    gl.useProgram(this.compositeProgram);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.hdrTex);
    gl.uniform1i(this.compositeUScene, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomTextures[0]);
    gl.uniform1i(this.compositeUBloom, 1);

    gl.uniform1f(this.compositeUBloomStrength, opts.postProcessing ? 0.3 : 0.0);
    gl.uniform1f(this.compositeUExposure, 1.0);
    gl.uniform1i(this.compositeUToneMap, opts.postProcessing ? 1 : 0);

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
