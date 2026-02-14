// WebGPURenderer — implements the Renderer interface using the WebGPU API.
//
// Step 11: updated to use renderFrame() with multiple DrawCalls.
// Shadow mapping is not yet implemented on the WebGPU backend — shadows
// are only rendered in WebGL2. The WebGPU path draws all objects with
// normal mapping but no shadow map.

import type { Renderer, VertexLayout, TextureDesc, MeshHandle, TextureHandle, DrawCall, FrameUniforms } from "./Renderer";

interface GPUMeshHandle extends MeshHandle {
  buffer: GPUBuffer;
}

interface GPUTextureHandle extends TextureHandle {
  texture: GPUTexture;
  view: GPUTextureView;
}

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
  private uniformBuffers: GPUBuffer[] = [];
  private bindGroupLayout: GPUBindGroupLayout;
  private canvas: HTMLCanvasElement;
  private sampler: GPUSampler;
  private flatNormalView: GPUTextureView;

  private static UNIFORM_SIZE = 160;

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    pipeline: GPURenderPipeline,
    bindGroupLayout: GPUBindGroupLayout,
    sampler: GPUSampler,
    flatNormalView: GPUTextureView,
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.pipeline = pipeline;
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
        module: shaderModule,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const sampler = device.createSampler({
      magFilter: "nearest",
      minFilter: "nearest",
    });

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

    return new WebGPURenderer(canvas, device, context, pipeline, bindGroupLayout, sampler, flatNormalTex.createView());
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

  renderFrame(drawCalls: DrawCall[], u: FrameUniforms): void {
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

    // Ensure we have enough uniform buffers for all draw calls
    while (this.uniformBuffers.length < drawCalls.length) {
      this.uniformBuffers.push(this.device.createBuffer({
        size: WebGPURenderer.UNIFORM_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
    }

    for (let i = 0; i < drawCalls.length; i++) {
      const dc = drawCalls[i];
      const mesh = dc.mesh as GPUMeshHandle;
      const tex = dc.texture as GPUTextureHandle;
      const hasNormal = !!dc.normalMap;
      const normalView = hasNormal ? (dc.normalMap as GPUTextureHandle).view : this.flatNormalView;

      // Pack uniforms into this draw call's own buffer
      const buf = new ArrayBuffer(WebGPURenderer.UNIFORM_SIZE);
      const f32 = new Float32Array(buf);
      f32.set(dc.model as unknown as Float32Array, 0);
      f32.set(u.viewProj as unknown as Float32Array, 16);
      f32[32] = u.lightPos[0];
      f32[33] = u.lightPos[1];
      f32[34] = u.lightPos[2];
      f32[36] = u.cameraPos[0];
      f32[37] = u.cameraPos[1];
      f32[38] = u.cameraPos[2];
      f32[39] = hasNormal ? 1.0 : 0.0;
      this.device.queue.writeBuffer(this.uniformBuffers[i], 0, buf);

      const bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffers[i] } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: tex.view },
          { binding: 3, resource: normalView },
        ],
      });

      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, mesh.buffer);
      pass.draw(mesh.vertexCount);
    }

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
