// Roadmap:
//   [x] Step 0 — Project Scaffold
//   [x] Step 1 — Hello Triangle
//   [x] Step 2 — Transformations (rotating 3D cube)
//   [x] Step 3 — Interactive Camera
//   [x] Step 4 — Textures
//   [x] Step 5 — Lighting (Phong)
//   [x] Step 6 — Abstraction Layer
//   [x] Step 7 — WebGPU Backend
//   [x] Step 8 — OBJ Mesh Loading
//   [x] Step 9 — Normal Mapping
//   [x] Step 10 — Shadow Mapping
//   [x] Step 11 — Multiple Objects / Scene Graph
//   [x] Step 12 — Deferred Rendering
//   [x] Step 13 — Post-Processing (bloom, tone mapping)
//   [ ] Step 14 — Skeletal Animation

// Step 13: Post-Processing — bloom and tone mapping
//
// Building on the deferred pipeline (Step 12), we add a post-processing
// chain after the lighting pass:
//
//   1. Lighting pass now renders to an HDR framebuffer (RGBA16F) with
//      boosted specular (values > 1.0) so bright spots feed the bloom.
//   2. Bloom extraction — threshold pass isolates bright pixels (> 1.0).
//   3. Gaussian blur — separable horizontal + vertical, ping-ponged 5×
//      at half resolution for a wide, soft glow.
//   4. Composite — combines HDR scene + blurred bloom, then applies
//      ACES filmic tone mapping and gamma correction.
//
// Both WebGL2 and WebGPU backends implement the full pipeline.

import { mat4, vec3 } from "gl-matrix";
import type { Renderer, MeshHandle, TextureHandle, DrawCall, RenderOptions } from "./engine";
import { WebGPURenderer } from "./engine";
import { WebGL2Renderer } from "./engine";
import { parseObj } from "./objParser";
import { computeTangents } from "./tangents";

// ---------------------------------------------------------------------------
// Initialize renderer
// ---------------------------------------------------------------------------

const canvas = document.getElementById("canvas") as HTMLCanvasElement;

// Backend selection via ?backend= query param, default auto-detect
const params = new URLSearchParams(window.location.search);
const backendParam = params.get("backend"); // "webgpu" | "webgl2" | null

// Sync the dropdown to the current selection and reload on change
const toggle = document.getElementById("backend-toggle") as HTMLSelectElement;
toggle.addEventListener("change", () => {
  params.set("backend", toggle.value);
  window.location.search = params.toString();
});

const shadowsToggle = document.getElementById("toggle-shadows") as HTMLInputElement;
const normalsToggle = document.getElementById("toggle-normals") as HTMLInputElement;
const postprocToggle = document.getElementById("toggle-postproc") as HTMLInputElement;

async function createRenderer(): Promise<Renderer & { aspect: number }> {
  if (backendParam !== "webgl2" && navigator.gpu) {
    try {
      const r = await WebGPURenderer.create(canvas);
      console.log("Using WebGPU backend");
      toggle.value = "webgpu";
      return r;
    } catch (e) {
      console.warn("WebGPU init failed, falling back to WebGL2:", e);
    }
  }
  console.log("Using WebGL2 backend");
  toggle.value = "webgl2";
  return new WebGL2Renderer(canvas);
}

// ---------------------------------------------------------------------------
// Load OBJ model
// ---------------------------------------------------------------------------

async function loadObj(url: string): Promise<Float32Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const text = await response.text();
  const result = parseObj(text);
  console.log(`Loaded ${url}: ${result.triangleCount} triangles`);
  return result.vertices;
}

// ---------------------------------------------------------------------------
// Vertex layout — pos + uv + normal + tangent
// ---------------------------------------------------------------------------

const VERTEX_LAYOUT = [
  { location: 0, size: 3 },
  { location: 1, size: 2 },
  { location: 2, size: 3 },
  { location: 3, size: 3 },
];

// ---------------------------------------------------------------------------
// Ground plane geometry (two triangles, with tangents)
// ---------------------------------------------------------------------------

function makeGroundPlane(size: number): Float32Array {
  // Interleaved: pos(3) + uv(2) + normal(3) + tangent(3) = 11 floats
  const s = size;
  // Two triangles forming a quad at y = -1 (CW winding so front face points up)
  // prettier-ignore
  return new Float32Array([
    // Triangle 1
    -s, -1, -s,  0, 0,  0, 1, 0,  1, 0, 0,
     s, -1,  s,  1, 1,  0, 1, 0,  1, 0, 0,
     s, -1, -s,  1, 0,  0, 1, 0,  1, 0, 0,
    // Triangle 2
    -s, -1, -s,  0, 0,  0, 1, 0,  1, 0, 0,
    -s, -1,  s,  0, 1,  0, 1, 0,  1, 0, 0,
     s, -1,  s,  1, 1,  0, 1, 0,  1, 0, 0,
  ]);
}

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------

const TEX_SIZE = 64;

// Checkerboard albedo
const TEX_PIXELS = new Uint8Array(TEX_SIZE * TEX_SIZE * 4);
for (let row = 0; row < TEX_SIZE; row++) {
  for (let col = 0; col < TEX_SIZE; col++) {
    const i = (row * TEX_SIZE + col) * 4;
    const tileSize = 8;
    const v = (Math.floor(row / tileSize) + Math.floor(col / tileSize)) % 2 === 0 ? 220 : 80;
    TEX_PIXELS[i] = v;
    TEX_PIXELS[i + 1] = v;
    TEX_PIXELS[i + 2] = v;
    TEX_PIXELS[i + 3] = 255;
  }
}

// Normal map — beveled tile edges
const NORMAL_MAP = new Uint8Array(TEX_SIZE * TEX_SIZE * 4);
{
  const tileSize = 8;
  const bevel = 2;
  for (let row = 0; row < TEX_SIZE; row++) {
    for (let col = 0; col < TEX_SIZE; col++) {
      const i = (row * TEX_SIZE + col) * 4;
      const tx = col % tileSize;
      const ty = row % tileSize;
      let nx = 0, ny = 0, nz = 1;
      if (tx < bevel) nx = -0.7;
      else if (tx >= tileSize - bevel) nx = 0.7;
      if (ty < bevel) ny = 0.7;
      else if (ty >= tileSize - bevel) ny = -0.7;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      nx /= len; ny /= len; nz /= len;
      NORMAL_MAP[i]     = Math.round((nx * 0.5 + 0.5) * 255);
      NORMAL_MAP[i + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      NORMAL_MAP[i + 2] = Math.round((nz * 0.5 + 0.5) * 255);
      NORMAL_MAP[i + 3] = 255;
    }
  }
}

// ---------------------------------------------------------------------------
// Orbit camera
// ---------------------------------------------------------------------------

let camTheta = 0.5;
let camPhi = 1.0;
let camRadius = 4.0;

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
// Main
// ---------------------------------------------------------------------------

async function main() {
  const [renderer, rawTorusData] = await Promise.all([
    createRenderer(),
    loadObj("models/torus.obj"),
  ]);

  // Create meshes
  const torusMeshData = computeTangents(rawTorusData);
  const torusMesh: MeshHandle = renderer.createMesh(torusMeshData, VERTEX_LAYOUT);

  const groundData = makeGroundPlane(3.0);
  const groundMesh: MeshHandle = renderer.createMesh(groundData, VERTEX_LAYOUT);

  // Create textures
  const texture: TextureHandle = renderer.createTexture({
    width: TEX_SIZE, height: TEX_SIZE, data: TEX_PIXELS,
  });
  const normalMap: TextureHandle = renderer.createTexture({
    width: TEX_SIZE, height: TEX_SIZE, data: NORMAL_MAP,
  });

  // Model matrices
  const torusModel = mat4.create();
  const groundModel = mat4.create(); // identity — ground stays put

  const view = mat4.create();
  const proj = mat4.create();
  const viewProj = mat4.create();
  const lightPos = vec3.create();
  const lightView = mat4.create();
  const lightProj = mat4.create();
  const lightViewProj = mat4.create();

  function frame(time: DOMHighResTimeStamp) {
    renderer.resize();

    // Animate torus
    const angle = time * 0.001;
    mat4.identity(torusModel);
    mat4.rotateY(torusModel, torusModel, angle);
    mat4.rotateX(torusModel, torusModel, angle * 0.7);

    // Camera
    const eye = cameraPosition();
    mat4.lookAt(view, eye, [0, 0, 0], [0, 1, 0]);
    mat4.perspective(proj, Math.PI / 4, renderer.aspect, 0.1, 100.0);
    mat4.multiply(viewProj, proj, view);

    // Light
    const lightAngle = time * 0.0005;
    vec3.set(lightPos, 3.0 * Math.cos(lightAngle), 3.0, 3.0 * Math.sin(lightAngle));
    mat4.lookAt(lightView, lightPos, [0, 0, 0], [0, 1, 0]);
    mat4.ortho(lightProj, -4, 4, -4, 4, 0.1, 12.0);
    mat4.multiply(lightViewProj, lightProj, lightView);

    // Build draw calls
    const drawCalls: DrawCall[] = [
      { mesh: torusMesh, texture, normalMap, model: torusModel },
      { mesh: groundMesh, texture, normalMap, model: groundModel },
    ];

    const options: RenderOptions = {
      shadows: shadowsToggle.checked,
      normalMaps: normalsToggle.checked,
      postProcessing: postprocToggle.checked,
    };

    renderer.renderFrame(drawCalls, {
      viewProj,
      lightPos,
      cameraPos: eye,
      lightViewProj,
    }, options);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  document.body.innerHTML = `<h1 style="color:red;padding:2rem">${err.message}</h1>`;
  console.error(err);
});
