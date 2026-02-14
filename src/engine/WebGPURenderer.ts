// WebGPURenderer — implements the Renderer interface using the WebGPU API.
//
// Step 9 adds normal mapping:
//   - Vertex shader now reads a tangent attribute and computes the TBN matrix.
//   - Fragment shader samples a normal map and transforms the sample from
//     tangent space to world space.
//   - A uHasNormalMap flag uniform controls fallback to the geometric normal.

import type { Renderer, VertexLayout, TextureDesc, MeshHandle, TextureHandle, DrawUniforms } from "./Renderer";

interface GPUMeshHandle extends MeshHandle {
  buffer: GPUBuffer;
}

interface GPUTextureHandle extends TextureHandle {
  texture: GPUTexture;
  view: GPUTextureView;
}

// WGSL shader with normal mapping support.
const SHADER_SRC = /* wgsl */ `
struct Uniforms {
  model: mat4x4f,
  viewProj: mat4x4f,
  lightPos: vec3f,
  _pad0: f32,
  cameraPos: vec3f,
  hasNormalMap: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var texSampler: sampler;
@group(0) @binding(2) var texData: texture_2d<f32>;
@group(0) @binding(3) var normalMapData: texture_2d<f32>;

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

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let texColor = textureSample(texData, texSampler, in.texCoord).rgb;

  var N: vec3f;
  if (u.hasNormalMap > 0.5) {
    let mapNormal = textureSample(normalMapData, texSampler, in.texCoord).rgb * 2.0 - 1.0;
    let TBN = mat3x3f(in.T, in.B, in.N);
    N = normalize(TBN * mapNormal);
  } else {
    N = normalize(in.N);
  }

  let L = normalize(u.lightPos - in.worldPos);
  let V = normalize(u.cameraPos - in.worldPos);
  let R = reflect(-L, N);

  let ambient = 0.15;
  let diffuse = max(dot(N, L), 0.0);
  let specular = pow(max(dot(R, V), 0.0), 32.0);

  let color = texColor * (ambient + diffuse) + vec3f(1.0) * specular * 0.5;
  return vec4f(color, 1.0);
}
`;

export class WebGPURenderer implements Renderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private pipeline: GPURenderPipeline;
  private depthTexture!: GPUTexture;
  private depthView!: GPUTextureView;
  private uniformBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;
  private canvas: HTMLCanvasElement;
  private sampler: GPUSampler;

  // Flat 1x1 normal map fallback (keep reference to prevent GC)
  private flatNormalView: GPUTextureView;

  // Uniform buffer size: 2 mat4 (128) + lightPos(12) + pad(4) + cameraPos(12) + hasNormalMap(4) = 160
  private static UNIFORM_SIZE = 160;

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    pipeline: GPURenderPipeline,
    uniformBuffer: GPUBuffer,
    bindGroupLayout: GPUBindGroupLayout,
    sampler: GPUSampler,
    flatNormalView: GPUTextureView,
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.pipeline = pipeline;
    this.uniformBuffer = uniformBuffer;
    this.bindGroupLayout = bindGroupLayout;
    this.sampler = sampler;
    this.flatNormalView = flatNormalView;
    this.createDepthTexture();
  }

  static async create(canvas: HTMLCanvasElement): Promise<WebGPURenderer> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found");
    const device = await adapter.requestDevice();

    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Failed to get WebGPU context");

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    const shaderModule = device.createShaderModule({ code: SHADER_SRC });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    const pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vs",
        buffers: [{
          arrayStride: 44, // 11 floats × 4 bytes (pos3 + uv2 + normal3 + tangent3)
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },   // position
            { shaderLocation: 1, offset: 12, format: "float32x2" },  // texcoord
            { shaderLocation: 2, offset: 20, format: "float32x3" },  // normal
            { shaderLocation: 3, offset: 32, format: "float32x3" },  // tangent
          ],
        }],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const uniformBuffer = device.createBuffer({
      size: WebGPURenderer.UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const sampler = device.createSampler({
      magFilter: "nearest",
      minFilter: "nearest",
    });

    // 1x1 flat normal texture (tangent-space up: 0,0,1 → 128,128,255)
    const flatNormalTex = device.createTexture({
      size: [1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: flatNormalTex },
      new Uint8Array([128, 128, 255, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    );

    return new WebGPURenderer(canvas, device, context, pipeline, uniformBuffer, bindGroupLayout, sampler, flatNormalTex.createView());
  }

  createMesh(data: Float32Array, _layout: VertexLayout[]): GPUMeshHandle {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();

    const floatsPerVertex = _layout.reduce((s, a) => s + a.size, 0);
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
      this.createDepthTexture();
    }
  }

  beginFrame(): void {
    // Command encoding happens in draw()
  }

  draw(mesh: GPUMeshHandle, texture: GPUTextureHandle, u: DrawUniforms): void {
    const hasNormal = !!u.normalMap;
    const normalView = hasNormal
      ? (u.normalMap as GPUTextureHandle).view
      : this.flatNormalView;

    // Create bind group with the correct textures
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: texture.view },
        { binding: 3, resource: normalView },
      ],
    });

    // Pack uniforms
    const buf = new ArrayBuffer(WebGPURenderer.UNIFORM_SIZE);
    const f32 = new Float32Array(buf);
    f32.set(u.model as unknown as Float32Array, 0);
    f32.set(u.viewProj as unknown as Float32Array, 16);
    f32[32] = u.lightPos[0];
    f32[33] = u.lightPos[1];
    f32[34] = u.lightPos[2];
    // f32[35] = padding
    f32[36] = u.cameraPos[0];
    f32[37] = u.cameraPos[1];
    f32[38] = u.cameraPos[2];
    f32[39] = hasNormal ? 1.0 : 0.0; // hasNormalMap flag
    this.device.queue.writeBuffer(this.uniformBuffer, 0, buf);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, mesh.buffer);
    pass.draw(mesh.vertexCount);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  get aspect(): number {
    return this.canvas.width / this.canvas.height;
  }

  private createDepthTexture(): void {
    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width || 1, this.canvas.height || 1],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthView = this.depthTexture.createView();
  }
}
