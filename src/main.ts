// main.ts — entry point. Loads assets, wires up the engine, and starts the render loop.
//
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
//   [x] Step 14 — Game-Engine Refactor

// Step 14: Game-Engine Refactor
//
// Restructures the layer above Renderer into reusable engine modules:
//
//   Transform  — position/rotation(quat)/scale + parent-child hierarchy
//   Material   — bundles texture + normalMap
//   Entity     — transform + mesh + material + onUpdate callback
//   OrbitCamera — camera state + mouse/wheel input
//   Light      — position + shadow view-projection
//   Scene      — entity list + camera + light; builds DrawCall[]/FrameUniforms
//   Engine     — owns rAF loop with update/render separation
//
// main.ts now just wires things together: create renderer, load assets,
// build entities, assemble scene, start engine.

import { vec3, quat } from "gl-matrix";
import type { Renderer, MeshHandle, TextureHandle, RenderOptions, Material } from "./engine";
import { WebGPURenderer, WebGL2Renderer, Entity, OrbitCamera, Light, Scene, Engine } from "./engine";
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

async function createRenderer(): Promise<Renderer> {
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
// Sphere geometry (UV sphere with tangents)
// ---------------------------------------------------------------------------

function makeSphere(radius: number, stacks: number, slices: number): Float32Array {
  // Interleaved: pos(3) + uv(2) + normal(3) + tangent(3) = 11 floats per vertex
  const verts: number[] = [];

  function pushVertex(stack: number, slice: number) {
    const theta = (stack / stacks) * Math.PI;       // 0 (north) → π (south)
    const phi   = (slice / slices) * 2 * Math.PI;   // 0 → 2π around equator
    const sinT = Math.sin(theta), cosT = Math.cos(theta);
    const sinP = Math.sin(phi),   cosP = Math.cos(phi);
    // pos
    verts.push(radius * sinT * cosP, radius * cosT, radius * sinT * sinP);
    // uv
    verts.push(slice / slices, stack / stacks);
    // normal = normalised pos (outward on a unit sphere)
    verts.push(sinT * cosP, cosT, sinT * sinP);
    // tangent = longitude direction (-sinPhi, 0, cosPhi)
    verts.push(-sinP, 0, cosP);
  }

  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      // CCW winding → outward-facing normals (verified by cross product)
      pushVertex(i,     j);      pushVertex(i + 1, j + 1);  pushVertex(i + 1, j);
      pushVertex(i,     j);      pushVertex(i,     j + 1);  pushVertex(i + 1, j + 1);
    }
  }

  return new Float32Array(verts);
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

const camera = new OrbitCamera();
camera.attach(canvas);

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

  // Create textures & material
  const texture: TextureHandle = renderer.createTexture({
    width: TEX_SIZE, height: TEX_SIZE, data: TEX_PIXELS,
  });
  const normalMap: TextureHandle = renderer.createTexture({
    width: TEX_SIZE, height: TEX_SIZE, data: NORMAL_MAP,
  });
  const checkerMaterial: Material = { texture, normalMap };

  // Solid white texture for the light sphere (reads as emissive/unlit-looking)
  const whiteTex: TextureHandle = renderer.createTexture({
    width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]),
  });

  // Entities
  const torus = new Entity(torusMesh, checkerMaterial);
  torus.onUpdate = (time) => {
    const angle = time * 0.001;
    quat.identity(torus.transform.rotation);
    quat.rotateY(torus.transform.rotation, torus.transform.rotation, angle);
    quat.rotateX(torus.transform.rotation, torus.transform.rotation, angle * 0.7);
  };
  const ground = new Entity(groundMesh, checkerMaterial);

  // Light
  const light = new Light();
  light.onUpdate = (time) => {
    const a = time * 0.0005;
    vec3.set(light.position, 3.0 * Math.cos(a), 3.0, 3.0 * Math.sin(a));
  };

  // Light sphere — small white sphere that tracks the light position
  const lightSphereMesh: MeshHandle = renderer.createMesh(makeSphere(0.15, 12, 16), VERTEX_LAYOUT);
  const lightSphere = new Entity(lightSphereMesh, { texture: whiteTex, emissive: true });
  lightSphere.onUpdate = () => {
    vec3.copy(lightSphere.transform.position, light.position);
  };

  // Scene
  const scene = new Scene(camera, light);
  scene.add(torus);
  scene.add(ground);
  scene.add(lightSphere);

  // Engine
  const options: RenderOptions = {
    shadows: shadowsToggle.checked,
    normalMaps: normalsToggle.checked,
    postProcessing: postprocToggle.checked,
  };
  const engine = new Engine(renderer, scene, options);

  shadowsToggle.addEventListener("change", () => { engine.options.shadows = shadowsToggle.checked; });
  normalsToggle.addEventListener("change", () => { engine.options.normalMaps = normalsToggle.checked; });
  postprocToggle.addEventListener("change", () => { engine.options.postProcessing = postprocToggle.checked; });

  engine.start();
}

main().catch((err) => {
  document.body.innerHTML = `<h1 style="color:red;padding:2rem">${err.message}</h1>`;
  console.error(err);
});
