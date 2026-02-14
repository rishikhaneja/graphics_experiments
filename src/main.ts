// Roadmap:
//   [x] Step 0 — Project Scaffold
//   [x] Step 1 — Hello Triangle
//   [x] Step 2 — Transformations (rotating 3D cube)
//   [x] Step 3 — Interactive Camera
//   [x] Step 4 — Textures
//   [x] Step 5 — Lighting (Phong)
//   [x] Step 6 — Abstraction Layer
//   [x] Step 7 — WebGPU Backend
//   [ ] Step 8 — OBJ Mesh Loading
//   [ ] Step 9 — Normal Mapping
//   [ ] Step 10 — Shadow Mapping
//   [ ] Step 11 — Multiple Objects / Scene Graph
//   [ ] Step 12 — Deferred Rendering
//   [ ] Step 13 — Post-Processing (bloom, tone mapping)
//   [ ] Step 14 — Skeletal Animation

// Step 7: WebGPU Backend
//
// Step 6 extracted reusable abstractions, but they were still WebGL2-specific.
// Now we introduce a **Renderer interface** and two implementations:
//
//   - **WebGL2Renderer** — wraps the WebGL2 code from Steps 1–6.
//   - **WebGPURenderer** — a new backend using the WebGPU API with WGSL shaders.
//
// At startup we try WebGPU first (it's the future of web graphics), and fall
// back to WebGL2 if it's not available. The main loop is identical either way —
// it just calls renderer.beginFrame(), renderer.draw(), etc.
//
// Key differences in WebGPU:
//   - Shaders are written in **WGSL** instead of GLSL.
//   - State is baked into a **GPURenderPipeline** (no mutable GL state machine).
//   - Uniforms/textures are passed via **bind groups** (like Vulkan descriptors).
//   - Commands are recorded into a **GPUCommandEncoder**, then submitted as a batch.
//   - Initialization is **async** (requesting adapter and device returns Promises).

import { mat4, vec3 } from "gl-matrix";
import type { Renderer, MeshHandle, TextureHandle } from "./engine";
import { WebGPURenderer } from "./engine";
import { WebGL2Renderer } from "./engine";

// ---------------------------------------------------------------------------
// Initialize renderer — try WebGPU first, fall back to WebGL2
// ---------------------------------------------------------------------------

const canvas = document.getElementById("canvas") as HTMLCanvasElement;

async function createRenderer(): Promise<Renderer & { aspect: number }> {
  if (navigator.gpu) {
    try {
      const r = await WebGPURenderer.create(canvas);
      console.log("Using WebGPU backend");
      return r;
    } catch (e) {
      console.warn("WebGPU init failed, falling back to WebGL2:", e);
    }
  }
  console.log("Using WebGL2 backend");
  return new WebGL2Renderer(canvas);
}

// ---------------------------------------------------------------------------
// Cube geometry data — shared between both backends
// ---------------------------------------------------------------------------

function face(
  positions: number[][],
  normal: number[],
  uvs: number[][]
): number[] {
  const indices = [0, 1, 2, 0, 2, 3];
  const out: number[] = [];
  for (const i of indices) {
    out.push(...positions[i], ...uvs[i], ...normal);
  }
  return out;
}

// prettier-ignore
const CUBE_DATA = new Float32Array([
  ...face([[-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5]], [0,0,1], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5]], [0,0,-1], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[-0.5, 0.5, 0.5],[ 0.5, 0.5, 0.5],[ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5]], [0,1,0], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[-0.5,-0.5,-0.5],[ 0.5,-0.5,-0.5],[ 0.5,-0.5, 0.5],[-0.5,-0.5, 0.5]], [0,-1,0], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[ 0.5,-0.5, 0.5],[ 0.5,-0.5,-0.5],[ 0.5, 0.5,-0.5],[ 0.5, 0.5, 0.5]], [1,0,0], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[-0.5,-0.5,-0.5],[-0.5,-0.5, 0.5],[-0.5, 0.5, 0.5],[-0.5, 0.5,-0.5]], [-1,0,0], [[0,0],[1,0],[1,1],[0,1]]),
]);

const CUBE_LAYOUT = [
  { location: 0, size: 3 }, // position
  { location: 1, size: 2 }, // texcoord
  { location: 2, size: 3 }, // normal
];

// Checkerboard texture pixels.
const TEX_SIZE = 8;
const TEX_PIXELS = new Uint8Array(TEX_SIZE * TEX_SIZE * 4);
for (let row = 0; row < TEX_SIZE; row++) {
  for (let col = 0; col < TEX_SIZE; col++) {
    const i = (row * TEX_SIZE + col) * 4;
    const v = (row + col) % 2 === 0 ? 255 : 60;
    TEX_PIXELS[i] = v;
    TEX_PIXELS[i + 1] = v;
    TEX_PIXELS[i + 2] = v;
    TEX_PIXELS[i + 3] = 255;
  }
}

// ---------------------------------------------------------------------------
// Orbit camera
// ---------------------------------------------------------------------------

let camTheta = 0.5;
let camPhi = 1.0;
let camRadius = 3.0;

const CAM_PHI_MIN = 0.1;
const CAM_PHI_MAX = Math.PI - 0.1;
const CAM_RADIUS_MIN = 1.0;
const CAM_RADIUS_MAX = 20.0;

function cameraPosition(): vec3 {
  return vec3.fromValues(
    camRadius * Math.sin(camPhi) * Math.sin(camTheta),
    camRadius * Math.cos(camPhi),
    camRadius * Math.sin(camPhi) * Math.cos(camTheta)
  );
}

// ---------------------------------------------------------------------------
// Mouse controls
// ---------------------------------------------------------------------------

let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;

canvas.addEventListener("mousedown", (e) => {
  if (e.button === 0) {
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  }
});

window.addEventListener("mouseup", () => { isDragging = false; });

window.addEventListener("mousemove", (e) => {
  if (!isDragging) return;
  const dx = e.clientX - lastMouseX;
  const dy = e.clientY - lastMouseY;
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
  camTheta -= dx * 0.01;
  camPhi = Math.max(CAM_PHI_MIN, Math.min(CAM_PHI_MAX, camPhi + dy * 0.01));
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  camRadius = Math.max(CAM_RADIUS_MIN, Math.min(CAM_RADIUS_MAX, camRadius + e.deltaY * 0.01));
}, { passive: false });

// ---------------------------------------------------------------------------
// Main — async because WebGPU init is async
// ---------------------------------------------------------------------------

async function main() {
  const renderer = await createRenderer();

  const mesh: MeshHandle = renderer.createMesh(CUBE_DATA, CUBE_LAYOUT);
  const texture: TextureHandle = renderer.createTexture({
    width: TEX_SIZE,
    height: TEX_SIZE,
    data: TEX_PIXELS,
  });

  const model = mat4.create();
  const view = mat4.create();
  const proj = mat4.create();
  const viewProj = mat4.create();
  const lightPos = vec3.create();

  function frame(time: DOMHighResTimeStamp) {
    renderer.resize();
    renderer.beginFrame();

    const angle = time * 0.001;
    mat4.identity(model);
    mat4.rotateY(model, model, angle);
    mat4.rotateX(model, model, angle * 0.7);

    const eye = cameraPosition();
    mat4.lookAt(view, eye, [0, 0, 0], [0, 1, 0]);

    const aspect = renderer.aspect;
    mat4.perspective(proj, Math.PI / 4, aspect, 0.1, 100.0);
    mat4.multiply(viewProj, proj, view);

    const lightAngle = time * 0.0005;
    vec3.set(lightPos, 3.0 * Math.cos(lightAngle), 2.0, 3.0 * Math.sin(lightAngle));

    renderer.draw(mesh, texture, { model, viewProj, lightPos, cameraPos: eye });

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  document.body.innerHTML = `<h1 style="color:red;padding:2rem">${err.message}</h1>`;
  console.error(err);
});
