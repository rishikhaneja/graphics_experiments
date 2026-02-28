// WebGPURenderer — implements the Renderer interface using the WebGPU API.
//
// Full feature parity with the WebGL2 backend:
//   1. Shadow pass    — depth-only from light's POV
//   2. G-Buffer pass  — geometry → MRT (position, normal, albedo)
//   3. Lighting pass  — fullscreen, reads G-Buffer + shadow map → HDR FBO
//   4. Bloom extract  — threshold bright pixels
//   5. Gaussian blur  — separable ping-pong at half resolution
//   6. Composite      — HDR + bloom, ACES tone mapping, gamma correction

import type { Renderer, VertexLayout, TextureDesc, MeshHandle, TextureHandle, DrawCall, FrameUniforms, RenderOptions } from "./Renderer";

interface GPUMeshHandle extends MeshHandle {
  buffer: GPUBuffer;
}

interface GPUTextureHandle extends TextureHandle {
  texture: GPUTexture;
  view: GPUTextureView;
}

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

// Shadow pass — depth only, no fragment output needed
const SHADOW_SHADER = /* wgsl */ `
struct ShadowUniforms {
  model: mat4x4f,
  lightViewProj: mat4x4f,
}
@group(0) @binding(0) var<uniform> u: ShadowUniforms;

@vertex
fn vs(@location(0) position: vec3f) -> @builtin(position) vec4f {
  var pos = u.lightViewProj * u.model * vec4f(position, 1.0);
  // gl-matrix produces z in [-1,1] (OpenGL); remap to [0,1] for WebGPU
  pos.z = pos.z * 0.5 + pos.w * 0.5;
  return pos;
}
`;

// G-Buffer geometry pass — writes position, normal, albedo to MRT
const GBUF_SHADER = /* wgsl */ `
struct GBufUniforms {
  model: mat4x4f,
  viewProj: mat4x4f,
  hasNormalMap: f32,
  emissive: f32,
}
@group(0) @binding(0) var<uniform> u: GBufUniforms;
@group(0) @binding(1) var texSampler: sampler;
@group(0) @binding(2) var albedoTex: texture_2d<f32>;
@group(0) @binding(3) var normalMapTex: texture_2d<f32>;

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) texCoord: vec2f,
  @location(1) worldPos: vec3f,
  @location(2) T: vec3f,
  @location(3) B: vec3f,
  @location(4) N: vec3f,
}

@vertex
fn vs(
  @location(0) position: vec3f,
  @location(1) texCoord: vec2f,
  @location(2) normal: vec3f,
  @location(3) tangent: vec3f,
) -> VertexOut {
  var out: VertexOut;
  let worldPos = u.model * vec4f(position, 1.0);
  out.worldPos = worldPos.xyz;

  let m3 = mat3x3f(u.model[0].xyz, u.model[1].xyz, u.model[2].xyz);
  let N = normalize(m3 * normal);
  var T = normalize(m3 * tangent);
  T = normalize(T - dot(T, N) * N);
  let B = cross(N, T);

  out.T = T;
  out.B = B;
  out.N = N;
  out.texCoord = texCoord;
  out.pos = u.viewProj * worldPos;
  return out;
}

struct GBufOut {
  @location(0) position: vec4f,
  @location(1) normal: vec4f,
  @location(2) albedo: vec4f,
}

@fragment
fn fs(in: VertexOut) -> GBufOut {
  var out: GBufOut;
  out.position = vec4f(in.worldPos, 1.0);

  var N: vec3f;
  if (u.hasNormalMap > 0.5) {
    let mapNormal = textureSample(normalMapTex, texSampler, in.texCoord).rgb * 2.0 - 1.0;
    let TBN = mat3x3f(in.T, in.B, in.N);
    N = normalize(TBN * mapNormal);
  } else {
    N = normalize(in.N);
  }
  // .w == 1.0 signals emissive — lighting pass bypasses Blinn-Phong for this pixel
  out.normal = vec4f(N, u.emissive);
  out.albedo = textureSample(albedoTex, texSampler, in.texCoord);
  return out;
}
`;

// Lighting pass — fullscreen, reads G-Buffer + shadow map, outputs HDR
const LIGHT_SHADER = /* wgsl */ `
struct LightUniforms {
  lightPos: vec3f,
  _pad0: f32,
  cameraPos: vec3f,
  _pad1: f32,
  lightViewProj: mat4x4f,
}
@group(0) @binding(0) var<uniform> u: LightUniforms;
@group(0) @binding(1) var gSampler: sampler;
@group(0) @binding(2) var gPosition: texture_2d<f32>;
@group(0) @binding(3) var gNormal: texture_2d<f32>;
@group(0) @binding(4) var gAlbedo: texture_2d<f32>;
@group(0) @binding(5) var shadowMap: texture_depth_2d;
@group(0) @binding(6) var shadowSampler: sampler_comparison;

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOut {
  var out: VertexOut;
  let x = f32((vid & 1u) << 2u) - 1.0;
  let y = f32((vid & 2u) << 1u) - 1.0;
  out.uv = vec2f(x, -y) * 0.5 + 0.5;
  out.pos = vec4f(x, y, 0.0, 1.0);
  return out;
}

fn shadowCalc(worldPos: vec3f) -> f32 {
  let lightSpacePos = u.lightViewProj * vec4f(worldPos, 1.0);
  let projCoords = lightSpacePos.xyz / lightSpacePos.w;
  let uv = projCoords.xy * 0.5 + 0.5;
  // gl-matrix produces OpenGL NDC z in [-1,1]; remap to WebGPU's [0,1]
  let depth = projCoords.z * 0.5 + 0.5;

  // Flip Y for WebGPU coordinate system
  let sampleUV = vec2f(uv.x, 1.0 - uv.y);
  let bias = 0.005;
  let texelSize = 1.0 / vec2f(textureDimensions(shadowMap));

  // PCF 3x3 — must stay in uniform control flow (no early return)
  var shadow = 0.0;
  for (var x = -1; x <= 1; x++) {
    for (var y = -1; y <= 1; y++) {
      shadow += textureSampleCompare(
        shadowMap, shadowSampler,
        sampleUV + vec2f(f32(x), f32(y)) * texelSize,
        depth - bias
      );
    }
  }
  shadow = shadow / 9.0;

  // Outside shadow frustum → fully lit
  let outOfBounds = uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth > 1.0;
  return select(shadow, 1.0, outOfBounds);
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let gPos = textureSample(gPosition, gSampler, in.uv);
  let worldPos = gPos.rgb;
  let gNorm = textureSample(gNormal, gSampler, in.uv);
  let N = normalize(gNorm.rgb);
  let albedo = textureSample(gAlbedo, gSampler, in.uv).rgb;

  let L = normalize(u.lightPos - worldPos);
  let V = normalize(u.cameraPos - worldPos);
  let R = reflect(-L, N);

  let ambient = 0.15;
  let diffuse = max(dot(N, L), 0.0);
  let specular = pow(max(dot(R, V), 0.0), 32.0) * 2.0;

  // shadowCalc must remain in uniform control flow (no early returns above)
  let shadow = shadowCalc(worldPos);

  let lit = albedo * ambient
          + albedo * diffuse * shadow
          + vec3f(1.0) * specular * shadow;

  // Emissive: skip lighting, output bright HDR albedo (feeds bloom)
  let afterEmissive = select(lit, albedo * 5.0, gNorm.w > 0.5);

  // Background pixels (alpha == 0) get a flat colour
  let color = select(afterEmissive, vec3f(0.0, 0.0, 0.0), gPos.a == 0.0);

  return vec4f(color, 1.0);
}
`;

// Bloom extraction — threshold bright pixels
const BLOOM_EXTRACT_SHADER = /* wgsl */ `
struct Params {
  threshold: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var sceneSampler: sampler;
@group(0) @binding(2) var sceneTex: texture_2d<f32>;

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOut {
  var out: VertexOut;
  let x = f32((vid & 1u) << 2u) - 1.0;
  let y = f32((vid & 2u) << 1u) - 1.0;
  out.uv = vec2f(x, -y) * 0.5 + 0.5;
  out.pos = vec4f(x, y, 0.0, 1.0);
  return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let color = textureSample(sceneTex, sceneSampler, in.uv).rgb;
  let brightness = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  if (brightness > params.threshold) {
    return vec4f(color - vec3f(params.threshold), 1.0);
  }
  return vec4f(0.0, 0.0, 0.0, 1.0);
}
`;

// Gaussian blur — separable
const BLUR_SHADER = /* wgsl */ `
struct Params {
  direction: vec2f,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var imgSampler: sampler;
@group(0) @binding(2) var imgTex: texture_2d<f32>;

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOut {
  var out: VertexOut;
  let x = f32((vid & 1u) << 2u) - 1.0;
  let y = f32((vid & 2u) << 1u) - 1.0;
  out.uv = vec2f(x, -y) * 0.5 + 0.5;
  out.pos = vec4f(x, y, 0.0, 1.0);
  return out;
}

const weights = array<f32, 5>(0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  var result = textureSample(imgTex, imgSampler, in.uv).rgb * weights[0];
  for (var i = 1; i < 5; i++) {
    let offset = params.direction * f32(i);
    result += textureSample(imgTex, imgSampler, in.uv + offset).rgb * weights[i];
    result += textureSample(imgTex, imgSampler, in.uv - offset).rgb * weights[i];
  }
  return vec4f(result, 1.0);
}
`;

// Composite + ACES tone mapping
const COMPOSITE_SHADER = /* wgsl */ `
struct Params {
  bloomStrength: f32,
  exposure: f32,
  toneMap: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var imgSampler: sampler;
@group(0) @binding(2) var sceneTex: texture_2d<f32>;
@group(0) @binding(3) var bloomTex: texture_2d<f32>;

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOut {
  var out: VertexOut;
  let x = f32((vid & 1u) << 2u) - 1.0;
  let y = f32((vid & 2u) << 1u) - 1.0;
  out.uv = vec2f(x, -y) * 0.5 + 0.5;
  out.pos = vec4f(x, y, 0.0, 1.0);
  return out;
}

fn aces(x: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let scene = textureSample(sceneTex, imgSampler, in.uv).rgb;
  let bloom = textureSample(bloomTex, imgSampler, in.uv).rgb;

  let hdr = scene + bloom * params.bloomStrength;
  var color = hdr;
  if (params.toneMap > 0.5) {
    color = aces(hdr * params.exposure);
    color = pow(color, vec3f(1.0 / 2.2));
  }

  return vec4f(color, 1.0);
}
`;

const SHADOW_SIZE = 1024;
const BLOOM_PASSES = 5;

export class WebGPURenderer implements Renderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private canvas: HTMLCanvasElement;

  // Shadow pass
  private shadowPipeline: GPURenderPipeline;
  private shadowBindGroupLayout: GPUBindGroupLayout;
  private shadowDepthTex!: GPUTexture;
  private shadowDepthView!: GPUTextureView;

  // G-Buffer pass
  private gBufPipeline: GPURenderPipeline;
  private gBufBindGroupLayout: GPUBindGroupLayout;
  private gPositionTex!: GPUTexture;
  private gPositionView!: GPUTextureView;
  private gNormalTex!: GPUTexture;
  private gNormalView!: GPUTextureView;
  private gAlbedoTex!: GPUTexture;
  private gAlbedoView!: GPUTextureView;
  private gDepthTex!: GPUTexture;
  private gDepthView!: GPUTextureView;

  // Lighting pass
  private lightPipeline: GPURenderPipeline;
  private lightBindGroupLayout: GPUBindGroupLayout;
  private lightUniformBuffer: GPUBuffer;
  private hdrTex!: GPUTexture;
  private hdrView!: GPUTextureView;

  // Bloom extract
  private bloomExtractPipeline: GPURenderPipeline;
  private bloomExtractBindGroupLayout: GPUBindGroupLayout;
  private bloomExtractUniformBuffer: GPUBuffer;

  // Blur
  private blurPipeline: GPURenderPipeline;
  private blurBindGroupLayout: GPUBindGroupLayout;
  private blurUniformBuffers: [GPUBuffer, GPUBuffer];
  private bloomTextures!: [GPUTexture, GPUTexture];
  private bloomViews!: [GPUTextureView, GPUTextureView];

  // Composite
  private compositePipeline: GPURenderPipeline;
  private compositeBindGroupLayout: GPUBindGroupLayout;
  private compositeUniformBuffer: GPUBuffer;

  // Shared
  private nearestSampler: GPUSampler;
  private linearSampler: GPUSampler;
  private shadowSampler: GPUSampler;
  private flatNormalView: GPUTextureView;
  private gBufUniformBuffers: GPUBuffer[] = [];
  private shadowUniformBuffers: GPUBuffer[] = [];

  private screenWidth = 0;
  private screenHeight = 0;

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    shadowPipeline: GPURenderPipeline,
    shadowBindGroupLayout: GPUBindGroupLayout,
    gBufPipeline: GPURenderPipeline,
    gBufBindGroupLayout: GPUBindGroupLayout,
    lightPipeline: GPURenderPipeline,
    lightBindGroupLayout: GPUBindGroupLayout,
    bloomExtractPipeline: GPURenderPipeline,
    bloomExtractBindGroupLayout: GPUBindGroupLayout,
    blurPipeline: GPURenderPipeline,
    blurBindGroupLayout: GPUBindGroupLayout,
    compositePipeline: GPURenderPipeline,
    compositeBindGroupLayout: GPUBindGroupLayout,
    nearestSampler: GPUSampler,
    linearSampler: GPUSampler,
    shadowSampler: GPUSampler,
    flatNormalView: GPUTextureView,
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.shadowPipeline = shadowPipeline;
    this.shadowBindGroupLayout = shadowBindGroupLayout;
    this.gBufPipeline = gBufPipeline;
    this.gBufBindGroupLayout = gBufBindGroupLayout;
    this.lightPipeline = lightPipeline;
    this.lightBindGroupLayout = lightBindGroupLayout;
    this.bloomExtractPipeline = bloomExtractPipeline;
    this.bloomExtractBindGroupLayout = bloomExtractBindGroupLayout;
    this.blurPipeline = blurPipeline;
    this.blurBindGroupLayout = blurBindGroupLayout;
    this.compositePipeline = compositePipeline;
    this.compositeBindGroupLayout = compositeBindGroupLayout;
    this.nearestSampler = nearestSampler;
    this.linearSampler = linearSampler;
    this.shadowSampler = shadowSampler;
    this.flatNormalView = flatNormalView;

    // Uniform buffers for fullscreen passes
    this.lightUniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bloomExtractUniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.blurUniformBuffers = [
      device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    ];
    this.compositeUniformBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Shadow depth texture (fixed size)
    this.shadowDepthTex = device.createTexture({
      size: [SHADOW_SIZE, SHADOW_SIZE],
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowDepthView = this.shadowDepthTex.createView();
  }

  static async create(canvas: HTMLCanvasElement): Promise<WebGPURenderer> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found");
    const device = await adapter.requestDevice();

    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Failed to get WebGPU context");

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    // ---- Samplers ----
    const nearestSampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
    const linearSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    const shadowSampler = device.createSampler({ compare: "less" });

    // ---- Flat normal texture ----
    const flatNormalTex = device.createTexture({
      size: [1, 1], format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: flatNormalTex }, new Uint8Array([128, 128, 255, 255]), { bytesPerRow: 4 }, [1, 1]);

    // ---- Shadow pipeline ----
    const shadowModule = device.createShaderModule({ code: SHADOW_SHADER });
    const shadowBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    });
    const shadowPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [shadowBindGroupLayout] }),
      vertex: {
        module: shadowModule, entryPoint: "vs",
        buffers: [{
          arrayStride: 44,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        }],
      },
      primitive: { topology: "triangle-list", cullMode: "front" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
    });

    // ---- G-Buffer pipeline ----
    const gBufModule = device.createShaderModule({ code: GBUF_SHADER });
    const gBufBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });
    const gBufPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [gBufBindGroupLayout] }),
      vertex: {
        module: gBufModule, entryPoint: "vs",
        buffers: [{
          arrayStride: 44,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x2" },
            { shaderLocation: 2, offset: 20, format: "float32x3" },
            { shaderLocation: 3, offset: 32, format: "float32x3" },
          ],
        }],
      },
      fragment: {
        module: gBufModule, entryPoint: "fs",
        targets: [
          { format: "rgba16float" },
          { format: "rgba16float" },
          { format: "rgba8unorm" },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    // ---- Lighting pipeline ----
    const lightModule = device.createShaderModule({ code: LIGHT_SHADER });
    const lightBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
      ],
    });
    const lightPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [lightBindGroupLayout] }),
      vertex: { module: lightModule, entryPoint: "vs" },
      fragment: { module: lightModule, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
      primitive: { topology: "triangle-list" },
    });

    // ---- Bloom extract pipeline ----
    const bloomExtractModule = device.createShaderModule({ code: BLOOM_EXTRACT_SHADER });
    const bloomExtractBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });
    const bloomExtractPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bloomExtractBindGroupLayout] }),
      vertex: { module: bloomExtractModule, entryPoint: "vs" },
      fragment: { module: bloomExtractModule, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
      primitive: { topology: "triangle-list" },
    });

    // ---- Blur pipeline ----
    const blurModule = device.createShaderModule({ code: BLUR_SHADER });
    const blurBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });
    const blurPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [blurBindGroupLayout] }),
      vertex: { module: blurModule, entryPoint: "vs" },
      fragment: { module: blurModule, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
      primitive: { topology: "triangle-list" },
    });

    // ---- Composite pipeline ----
    const compositeModule = device.createShaderModule({ code: COMPOSITE_SHADER });
    const compositeBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });
    const compositePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [compositeBindGroupLayout] }),
      vertex: { module: compositeModule, entryPoint: "vs" },
      fragment: { module: compositeModule, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });

    return new WebGPURenderer(
      canvas, device, context,
      shadowPipeline, shadowBindGroupLayout,
      gBufPipeline, gBufBindGroupLayout,
      lightPipeline, lightBindGroupLayout,
      bloomExtractPipeline, bloomExtractBindGroupLayout,
      blurPipeline, blurBindGroupLayout,
      compositePipeline, compositeBindGroupLayout,
      nearestSampler, linearSampler, shadowSampler,
      flatNormalTex.createView(),
    );
  }

  createMesh(data: Float32Array, layout: VertexLayout[]): GPUMeshHandle {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();

    const floatsPerVertex = layout.reduce((s, a) => s + a.size, 0);
    return { buffer, vertexCount: data.length / floatsPerVertex };
  }

  createTexture(desc: TextureDesc): GPUTextureHandle {
    const texture = this.device.createTexture({
      size: [desc.width, desc.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.device.queue.writeTexture(
      { texture },
      desc.data as unknown as ArrayBuffer,
      { bytesPerRow: desc.width * 4 },
      [desc.width, desc.height],
    );
    return { texture, view: texture.createView() };
  }

  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    if (this.screenWidth !== w || this.screenHeight !== h) {
      this.rebuildScreenTextures(w, h);
    }
  }

  private rebuildScreenTextures(w: number, h: number): void {
    this.screenWidth = w;
    this.screenHeight = h;
    const device = this.device;

    // G-Buffer textures
    this.gPositionTex?.destroy();
    this.gPositionTex = device.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.gPositionView = this.gPositionTex.createView();

    this.gNormalTex?.destroy();
    this.gNormalTex = device.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.gNormalView = this.gNormalTex.createView();

    this.gAlbedoTex?.destroy();
    this.gAlbedoTex = device.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.gAlbedoView = this.gAlbedoTex.createView();

    this.gDepthTex?.destroy();
    this.gDepthTex = device.createTexture({ size: [w, h], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.gDepthView = this.gDepthTex.createView();

    // HDR scene texture
    this.hdrTex?.destroy();
    this.hdrTex = device.createTexture({ size: [w, h], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.hdrView = this.hdrTex.createView();

    // Bloom ping-pong (half resolution)
    const bw = Math.max(1, w >> 1);
    const bh = Math.max(1, h >> 1);
    if (this.bloomTextures) {
      this.bloomTextures[0].destroy();
      this.bloomTextures[1].destroy();
    }
    const bt0 = device.createTexture({ size: [bw, bh], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const bt1 = device.createTexture({ size: [bw, bh], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.bloomTextures = [bt0, bt1];
    this.bloomViews = [bt0.createView(), bt1.createView()];
  }

  renderFrame(drawCalls: DrawCall[], u: FrameUniforms, opts: RenderOptions): void {
    const device = this.device;
    const encoder = device.createCommandEncoder();
    const w = this.screenWidth;
    const h = this.screenHeight;

    // Ensure enough per-draw uniform buffers
    while (this.shadowUniformBuffers.length < drawCalls.length) {
      this.shadowUniformBuffers.push(device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }
    while (this.gBufUniformBuffers.length < drawCalls.length) {
      this.gBufUniformBuffers.push(device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }

    // ---- Pass 1: Shadow ----
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowDepthView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });

      if (opts.shadows && u.lightViewProj) {
        pass.setPipeline(this.shadowPipeline);

        for (let i = 0; i < drawCalls.length; i++) {
          const dc = drawCalls[i];
          if (dc.emissive) continue; // light sources don't cast shadows
          const mesh = dc.mesh as GPUMeshHandle;

          const buf = new ArrayBuffer(128);
          const f32 = new Float32Array(buf);
          f32.set(dc.model as unknown as Float32Array, 0);
          f32.set(u.lightViewProj as unknown as Float32Array, 16);
          device.queue.writeBuffer(this.shadowUniformBuffers[i], 0, buf);

          const bindGroup = device.createBindGroup({
            layout: this.shadowBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.shadowUniformBuffers[i] } }],
          });
          pass.setBindGroup(0, bindGroup);
          pass.setVertexBuffer(0, mesh.buffer);
          pass.draw(mesh.vertexCount);
        }
      }
      // When shadows off, pass just clears to depth 1.0 (no shadow)
      pass.end();
    }

    // ---- Pass 2: G-Buffer ----
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.gPositionView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          { view: this.gNormalView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          { view: this.gAlbedoView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        ],
        depthStencilAttachment: {
          view: this.gDepthView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(this.gBufPipeline);

      for (let i = 0; i < drawCalls.length; i++) {
        const dc = drawCalls[i];
        const mesh = dc.mesh as GPUMeshHandle;
        const tex = dc.texture as GPUTextureHandle;
        const hasNormal = opts.normalMaps && !!dc.normalMap;
        const normalView = hasNormal ? (dc.normalMap as GPUTextureHandle).view : this.flatNormalView;

        const buf = new ArrayBuffer(144);
        const f32 = new Float32Array(buf);
        f32.set(dc.model as unknown as Float32Array, 0);
        f32.set(u.viewProj as unknown as Float32Array, 16);
        f32[32] = hasNormal ? 1.0 : 0.0;
        f32[33] = dc.emissive ? 1.0 : 0.0;
        device.queue.writeBuffer(this.gBufUniformBuffers[i], 0, buf);

        const bindGroup = device.createBindGroup({
          layout: this.gBufBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.gBufUniformBuffers[i] } },
            { binding: 1, resource: this.nearestSampler },
            { binding: 2, resource: tex.view },
            { binding: 3, resource: normalView },
          ],
        });
        pass.setBindGroup(0, bindGroup);
        pass.setVertexBuffer(0, mesh.buffer);
        pass.draw(mesh.vertexCount);
      }
      pass.end();
    }

    // ---- Pass 3: Lighting → HDR ----
    {
      const buf = new ArrayBuffer(96);
      const f32 = new Float32Array(buf);
      f32[0] = u.lightPos[0]; f32[1] = u.lightPos[1]; f32[2] = u.lightPos[2];
      f32[4] = u.cameraPos[0]; f32[5] = u.cameraPos[1]; f32[6] = u.cameraPos[2];
      if (u.lightViewProj) {
        f32.set(u.lightViewProj as unknown as Float32Array, 8);
      }
      device.queue.writeBuffer(this.lightUniformBuffer, 0, buf);

      const bindGroup = device.createBindGroup({
        layout: this.lightBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.lightUniformBuffer } },
          { binding: 1, resource: this.nearestSampler },
          { binding: 2, resource: this.gPositionView },
          { binding: 3, resource: this.gNormalView },
          { binding: 4, resource: this.gAlbedoView },
          { binding: 5, resource: this.shadowDepthView },
          { binding: 6, resource: this.shadowSampler },
        ],
      });

      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.hdrView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear", storeOp: "store",
        }],
      });
      pass.setPipeline(this.lightPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    }

    // ---- Passes 4–5: Bloom (only when post-processing is on) ----
    if (opts.postProcessing) {
      const bw = Math.max(1, w >> 1);
      const bh = Math.max(1, h >> 1);

      // Pass 4: Bloom extraction → bloom 0
      {
        device.queue.writeBuffer(this.bloomExtractUniformBuffer, 0, new Float32Array([1.0]));
        const bindGroup = device.createBindGroup({
          layout: this.bloomExtractBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.bloomExtractUniformBuffer } },
            { binding: 1, resource: this.linearSampler },
            { binding: 2, resource: this.hdrView },
          ],
        });
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: this.bloomViews[0],
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear", storeOp: "store",
          }],
        });
        pass.setPipeline(this.bloomExtractPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(3);
        pass.end();
      }

      // Pass 5: Gaussian blur ping-pong
      {
        device.queue.writeBuffer(this.blurUniformBuffers[0], 0, new Float32Array([1.0 / bw, 0.0]));
        device.queue.writeBuffer(this.blurUniformBuffers[1], 0, new Float32Array([0.0, 1.0 / bh]));

        for (let i = 0; i < BLOOM_PASSES; i++) {
          {
            const bg = device.createBindGroup({
              layout: this.blurBindGroupLayout,
              entries: [
                { binding: 0, resource: { buffer: this.blurUniformBuffers[0] } },
                { binding: 1, resource: this.linearSampler },
                { binding: 2, resource: this.bloomViews[0] },
              ],
            });
            const pass = encoder.beginRenderPass({
              colorAttachments: [{ view: this.bloomViews[1], loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
            });
            pass.setPipeline(this.blurPipeline);
            pass.setBindGroup(0, bg);
            pass.draw(3);
            pass.end();
          }
          {
            const bg = device.createBindGroup({
              layout: this.blurBindGroupLayout,
              entries: [
                { binding: 0, resource: { buffer: this.blurUniformBuffers[1] } },
                { binding: 1, resource: this.linearSampler },
                { binding: 2, resource: this.bloomViews[1] },
              ],
            });
            const pass = encoder.beginRenderPass({
              colorAttachments: [{ view: this.bloomViews[0], loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
            });
            pass.setPipeline(this.blurPipeline);
            pass.setBindGroup(0, bg);
            pass.draw(3);
            pass.end();
          }
        }
      }
    }

    // ---- Pass 6: Composite → screen ----
    // Always runs — converts HDR to screen. bloomStrength=0 when post-processing is off.
    {
      device.queue.writeBuffer(this.compositeUniformBuffer, 0,
        new Float32Array([opts.postProcessing ? 0.3 : 0.0, 1.0, opts.postProcessing ? 1.0 : 0.0]));
      const bindGroup = device.createBindGroup({
        layout: this.compositeBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.compositeUniformBuffer } },
          { binding: 1, resource: this.linearSampler },
          { binding: 2, resource: this.hdrView },
          { binding: 3, resource: this.bloomViews[0] },
        ],
      });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear", storeOp: "store",
        }],
      });
      pass.setPipeline(this.compositePipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    }

    device.queue.submit([encoder.finish()]);
  }

  get aspect(): number {
    return this.canvas.width / this.canvas.height;
  }
}
