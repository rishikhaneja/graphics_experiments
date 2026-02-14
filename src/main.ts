// Roadmap:
//   [x] Step 0 — Project Scaffold
//   [x] Step 1 — Hello Triangle
//   [x] Step 2 — Transformations (rotating 3D cube)
//   [x] Step 3 — Interactive Camera
//   [x] Step 4 — Textures
//   [x] Step 5 — Lighting (Phong)
//   [x] Step 6 — Abstraction Layer
//   [ ] Step 7 — WebGPU Backend

// Step 6: Abstraction Layer
//
// Steps 1–5 built everything in one flat file with raw WebGL calls. Now we
// extract the repeating patterns into reusable classes:
//
//   - **ShaderProgram** — compiles/links GLSL, caches uniform locations,
//     provides typed setters (setMat4, setVec3, etc.).
//   - **Mesh** — takes a Float32Array and a vertex attribute layout, creates
//     the VBO + VAO automatically. Computes stride and offsets for you.
//   - **Texture** — wraps texture creation, parameter setup, and binding.
//
// The rendering logic in main.ts is now much shorter and focuses on the
// scene description (what to draw) rather than GPU plumbing (how to draw).
//
// These abstractions are deliberately thin — just enough to remove boilerplate,
// not so much that they hide what WebGL is doing.

import { mat4, vec3 } from "gl-matrix";
import { ShaderProgram, Mesh, Texture } from "./engine";

// ---------------------------------------------------------------------------
// Canvas + WebGL2 context
// ---------------------------------------------------------------------------

const canvas = document.getElementById("canvas") as HTMLCanvasElement;

const maybeGl = canvas.getContext("webgl2");
if (!maybeGl) {
  document.body.innerHTML =
    '<h1 style="color:white;padding:2rem">WebGL2 is not supported in this browser.</h1>';
  throw new Error("WebGL2 not supported");
}
const gl: WebGL2RenderingContext = maybeGl;

// ---------------------------------------------------------------------------
// Shaders (source is still inline — no build-time magic)
// ---------------------------------------------------------------------------

const vertexShaderSource = `#version 300 es

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uViewProj;

out vec2 vTexCoord;
out vec3 vWorldPos;
out vec3 vNormal;

void main() {
  vec4 worldPos = uModel * vec4(aPosition, 1.0);
  vWorldPos = worldPos.xyz;
  vNormal = normalize(mat3(uModel) * aNormal);
  vTexCoord = aTexCoord;
  gl_Position = uViewProj * worldPos;
}
`;

const fragmentShaderSource = `#version 300 es
precision mediump float;

in vec2 vTexCoord;
in vec3 vWorldPos;
in vec3 vNormal;

uniform sampler2D uTexture;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;

out vec4 fragColor;

void main() {
  vec3 texColor = texture(uTexture, vTexCoord).rgb;

  vec3 N = normalize(vNormal);
  vec3 L = normalize(uLightPos - vWorldPos);
  vec3 V = normalize(uCameraPos - vWorldPos);
  vec3 R = reflect(-L, N);

  float ambient = 0.15;
  float diffuse = max(dot(N, L), 0.0);
  float specular = pow(max(dot(R, V), 0.0), 32.0);

  vec3 color = texColor * (ambient + diffuse) + vec3(1.0) * specular * 0.5;
  fragColor = vec4(color, 1.0);
}
`;

const shader = new ShaderProgram(gl, vertexShaderSource, fragmentShaderSource);

// ---------------------------------------------------------------------------
// Cube geometry — using the Mesh abstraction
// ---------------------------------------------------------------------------

// Helper: generate 6 vertices (2 triangles) for one face.
// Each vertex: pos(3) + uv(2) + normal(3) = 8 floats.
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
const cubeData = new Float32Array([
  ...face([[-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5]], [0,0,1], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5]], [0,0,-1], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[-0.5, 0.5, 0.5],[ 0.5, 0.5, 0.5],[ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5]], [0,1,0], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[-0.5,-0.5,-0.5],[ 0.5,-0.5,-0.5],[ 0.5,-0.5, 0.5],[-0.5,-0.5, 0.5]], [0,-1,0], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[ 0.5,-0.5, 0.5],[ 0.5,-0.5,-0.5],[ 0.5, 0.5,-0.5],[ 0.5, 0.5, 0.5]], [1,0,0], [[0,0],[1,0],[1,1],[0,1]]),
  ...face([[-0.5,-0.5,-0.5],[-0.5,-0.5, 0.5],[-0.5, 0.5, 0.5],[-0.5, 0.5,-0.5]], [-1,0,0], [[0,0],[1,0],[1,1],[0,1]]),
]);

const cubeMesh = new Mesh(gl, cubeData, [
  { location: 0, size: 3 }, // position
  { location: 1, size: 2 }, // texcoord
  { location: 2, size: 3 }, // normal
]);

// ---------------------------------------------------------------------------
// Checkerboard texture
// ---------------------------------------------------------------------------

const TEX_SIZE = 8;
const texPixels = new Uint8Array(TEX_SIZE * TEX_SIZE * 4);
for (let row = 0; row < TEX_SIZE; row++) {
  for (let col = 0; col < TEX_SIZE; col++) {
    const i = (row * TEX_SIZE + col) * 4;
    const v = (row + col) % 2 === 0 ? 255 : 60;
    texPixels[i] = v;
    texPixels[i + 1] = v;
    texPixels[i + 2] = v;
    texPixels[i + 3] = 255;
  }
}

const checkerboard = new Texture(gl, {
  width: TEX_SIZE,
  height: TEX_SIZE,
  data: texPixels,
});

// ---------------------------------------------------------------------------
// GL state
// ---------------------------------------------------------------------------

gl.enable(gl.DEPTH_TEST);

// ---------------------------------------------------------------------------
// Canvas resizing
// ---------------------------------------------------------------------------

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    gl.viewport(0, 0, width, height);
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
// Matrices
// ---------------------------------------------------------------------------

const model = mat4.create();
const view = mat4.create();
const proj = mat4.create();
const viewProj = mat4.create();

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

function frame(time: DOMHighResTimeStamp) {
  resizeCanvas();

  gl.clearColor(0.08, 0.08, 0.12, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const angle = time * 0.001;
  mat4.identity(model);
  mat4.rotateY(model, model, angle);
  mat4.rotateX(model, model, angle * 0.7);

  const eye = cameraPosition();
  mat4.lookAt(view, eye, [0, 0, 0], [0, 1, 0]);

  const aspect = canvas.width / canvas.height;
  mat4.perspective(proj, Math.PI / 4, aspect, 0.1, 100.0);
  mat4.multiply(viewProj, proj, view);

  shader.use();
  shader.setMat4("uModel", model);
  shader.setMat4("uViewProj", viewProj);

  const lightAngle = time * 0.0005;
  shader.setVec3("uLightPos", 3.0 * Math.cos(lightAngle), 2.0, 3.0 * Math.sin(lightAngle));
  shader.setVec3v("uCameraPos", eye);

  checkerboard.bind(0);
  shader.setInt("uTexture", 0);

  cubeMesh.draw();

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
