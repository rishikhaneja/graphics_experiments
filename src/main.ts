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
//   [ ] Step 11 — Multiple Objects / Scene Graph
//   [ ] Step 12 — Deferred Rendering
//   [ ] Step 13 — Post-Processing (bloom, tone mapping)
//   [ ] Step 14 — Skeletal Animation

// Step 10: Shadow Mapping
//
// Shadows ground the scene — without them objects look like they float.
// We use **shadow mapping**, the most common real-time shadow technique:
//
//   1. **Shadow pass** — render the scene from the light's point of view into
//      a depth-only framebuffer (the "shadow map"). This records how far away
//      each surface is from the light.
//   2. **Main pass** — for every fragment, project it into light space and
//      compare its depth to the shadow map. If the fragment is further from
//      the light than the recorded depth, it's in shadow.
//
// Challenges solved here:
//   - **Shadow acne** — a small depth bias prevents surfaces from shadowing
//     themselves due to floating-point precision issues.
//   - **Peter panning** — we cull front faces in the shadow pass so the bias
//     doesn't push shadows away from the caster.
//   - **Soft edges** — PCF (percentage-closer filtering) averages 9 shadow
//     samples in a 3×3 kernel for smoother shadow boundaries.
//
// The light uses an orthographic projection (directional light), which gives
// uniform shadow quality across the scene.

import { mat4, vec3 } from "gl-matrix";
import type { Renderer, MeshHandle, TextureHandle } from "./engine";
import { WebGPURenderer } from "./engine";
import { WebGL2Renderer } from "./engine";
import { parseObj } from "./objParser";
import { computeTangents } from "./tangents";

// ---------------------------------------------------------------------------
// Initialize renderer
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
  { location: 0, size: 3 }, // position
  { location: 1, size: 2 }, // texcoord
  { location: 2, size: 3 }, // normal
  { location: 3, size: 3 }, // tangent
];

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
// Main
// ---------------------------------------------------------------------------

async function main() {
  const [renderer, rawMeshData] = await Promise.all([
    createRenderer(),
    loadObj("models/torus.obj"),
  ]);

  const meshData = computeTangents(rawMeshData);
  const mesh: MeshHandle = renderer.createMesh(meshData, VERTEX_LAYOUT);

  const texture: TextureHandle = renderer.createTexture({
    width: TEX_SIZE,
    height: TEX_SIZE,
    data: TEX_PIXELS,
  });

  const normalMap: TextureHandle = renderer.createTexture({
    width: TEX_SIZE,
    height: TEX_SIZE,
    data: NORMAL_MAP,
  });

  const model = mat4.create();
  const view = mat4.create();
  const proj = mat4.create();
  const viewProj = mat4.create();
  const lightPos = vec3.create();

  // Light-space matrices for shadow mapping
  const lightView = mat4.create();
  const lightProj = mat4.create();
  const lightViewProj = mat4.create();

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

    // Compute light VP (orthographic, looking at origin from light position)
    mat4.lookAt(lightView, lightPos, [0, 0, 0], [0, 1, 0]);
    mat4.ortho(lightProj, -3, 3, -3, 3, 0.1, 10.0);
    mat4.multiply(lightViewProj, lightProj, lightView);

    renderer.draw(mesh, texture, {
      model,
      viewProj,
      lightPos,
      cameraPos: eye,
      normalMap,
      lightViewProj,
    });

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  document.body.innerHTML = `<h1 style="color:red;padding:2rem">${err.message}</h1>`;
  console.error(err);
});
