// Roadmap:
//   [x] Step 0 — Project Scaffold
//   [x] Step 1 — Hello Triangle
//   [x] Step 2 — Transformations (rotating 3D cube)
//   [x] Step 3 — Interactive Camera
//   [ ] Step 4 — Textures
//   [ ] Step 5 — Lighting (Phong)
//   [ ] Step 6 — Abstraction Layer
//   [ ] Step 7 — WebGPU Backend

// Step 3: Interactive Camera
//
// Step 2 had a fixed camera at [0, 0, 3]. Now we let the user orbit around the
// cube with the mouse. The camera sits on a sphere centered at the origin:
//
//   x = r * sin(phi) * sin(theta)
//   y = r * cos(phi)
//   z = r * sin(phi) * cos(theta)
//
// where theta = horizontal angle, phi = vertical angle, r = distance.
//
// Controls:
//   - Left-click + drag → orbit (rotate theta/phi)
//   - Scroll wheel → zoom (change r)
//
// The cube still auto-rotates so there's something to look at from every angle.

import { mat4, vec3 } from "gl-matrix";

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
// Shader source code
// ---------------------------------------------------------------------------

const vertexShaderSource = `#version 300 es

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;

uniform mat4 uMVP;

out vec3 vColor;

void main() {
  vColor = aColor;
  gl_Position = uMVP * vec4(aPosition, 1.0);
}
`;

const fragmentShaderSource = `#version 300 es
precision mediump float;

in vec3 vColor;
out vec4 fragColor;

void main() {
  fragColor = vec4(vColor, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Compile a shader from source
// ---------------------------------------------------------------------------

function compileShader(type: GLenum, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to create shader");

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error:\n${log}`);
  }
  return shader;
}

// ---------------------------------------------------------------------------
// Link shaders into a program
// ---------------------------------------------------------------------------

function createProgram(
  vertexSource: string,
  fragmentSource: string
): WebGLProgram {
  const vs = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fs = compileShader(gl.FRAGMENT_SHADER, fragmentSource);

  const program = gl.createProgram();
  if (!program) throw new Error("Failed to create program");

  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error:\n${log}`);
  }

  gl.deleteShader(vs);
  gl.deleteShader(fs);

  return program;
}

// ---------------------------------------------------------------------------
// Create the shader program
// ---------------------------------------------------------------------------

const program = createProgram(vertexShaderSource, fragmentShaderSource);
const uMVPLoc = gl.getUniformLocation(program, "uMVP");

// ---------------------------------------------------------------------------
// Vertex data — a cube with 6 colored faces
// ---------------------------------------------------------------------------

// prettier-ignore
const vertices = new Float32Array([
  // Front face (z = +0.5) — red
  -0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
   0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
   0.5,  0.5,  0.5,   1.0, 0.0, 0.0,
  -0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
   0.5,  0.5,  0.5,   1.0, 0.0, 0.0,
  -0.5,  0.5,  0.5,   1.0, 0.0, 0.0,

  // Back face (z = -0.5) — cyan
  -0.5, -0.5, -0.5,   0.0, 1.0, 1.0,
  -0.5,  0.5, -0.5,   0.0, 1.0, 1.0,
   0.5,  0.5, -0.5,   0.0, 1.0, 1.0,
  -0.5, -0.5, -0.5,   0.0, 1.0, 1.0,
   0.5,  0.5, -0.5,   0.0, 1.0, 1.0,
   0.5, -0.5, -0.5,   0.0, 1.0, 1.0,

  // Top face (y = +0.5) — green
  -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
  -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
   0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
  -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
   0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
   0.5,  0.5, -0.5,   0.0, 1.0, 0.0,

  // Bottom face (y = -0.5) — magenta
  -0.5, -0.5, -0.5,   1.0, 0.0, 1.0,
   0.5, -0.5, -0.5,   1.0, 0.0, 1.0,
   0.5, -0.5,  0.5,   1.0, 0.0, 1.0,
  -0.5, -0.5, -0.5,   1.0, 0.0, 1.0,
   0.5, -0.5,  0.5,   1.0, 0.0, 1.0,
  -0.5, -0.5,  0.5,   1.0, 0.0, 1.0,

  // Right face (x = +0.5) — blue
   0.5, -0.5, -0.5,   0.0, 0.0, 1.0,
   0.5,  0.5, -0.5,   0.0, 0.0, 1.0,
   0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
   0.5, -0.5, -0.5,   0.0, 0.0, 1.0,
   0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
   0.5, -0.5,  0.5,   0.0, 0.0, 1.0,

  // Left face (x = -0.5) — yellow
  -0.5, -0.5, -0.5,   1.0, 1.0, 0.0,
  -0.5, -0.5,  0.5,   1.0, 1.0, 0.0,
  -0.5,  0.5,  0.5,   1.0, 1.0, 0.0,
  -0.5, -0.5, -0.5,   1.0, 1.0, 0.0,
  -0.5,  0.5,  0.5,   1.0, 1.0, 0.0,
  -0.5,  0.5, -0.5,   1.0, 1.0, 0.0,
]);

// ---------------------------------------------------------------------------
// Upload vertex data to GPU
// ---------------------------------------------------------------------------

const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

// ---------------------------------------------------------------------------
// Vertex Array Object (VAO) — describe the memory layout
// ---------------------------------------------------------------------------

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

const STRIDE = 6 * Float32Array.BYTES_PER_ELEMENT;

gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, STRIDE, 0);

gl.enableVertexAttribArray(1);
gl.vertexAttribPointer(1, 3, gl.FLOAT, false, STRIDE, 3 * Float32Array.BYTES_PER_ELEMENT);

gl.bindVertexArray(null);

// ---------------------------------------------------------------------------
// Enable depth testing
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
// Orbit camera state
// ---------------------------------------------------------------------------

// Spherical coordinates: theta (horizontal), phi (vertical), radius (distance).
// phi is clamped to avoid flipping at the poles.
let camTheta = 0.5;   // horizontal angle (radians)
let camPhi = 1.0;     // vertical angle — 0 = top pole, PI = bottom pole
let camRadius = 3.0;  // distance from origin

const CAM_PHI_MIN = 0.1;
const CAM_PHI_MAX = Math.PI - 0.1;
const CAM_RADIUS_MIN = 1.0;
const CAM_RADIUS_MAX = 20.0;

// Convert spherical → Cartesian for use in lookAt.
function cameraPosition(): vec3 {
  return vec3.fromValues(
    camRadius * Math.sin(camPhi) * Math.sin(camTheta),
    camRadius * Math.cos(camPhi),
    camRadius * Math.sin(camPhi) * Math.cos(camTheta)
  );
}

// ---------------------------------------------------------------------------
// Mouse controls — orbit on drag, zoom on scroll
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

window.addEventListener("mouseup", () => {
  isDragging = false;
});

window.addEventListener("mousemove", (e) => {
  if (!isDragging) return;

  const dx = e.clientX - lastMouseX;
  const dy = e.clientY - lastMouseY;
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;

  // Scale mouse pixels to radians. 300px ≈ 1 radian feels natural.
  camTheta -= dx * 0.01;
  camPhi += dy * 0.01;

  // Clamp phi so the camera can't flip over the poles.
  camPhi = Math.max(CAM_PHI_MIN, Math.min(CAM_PHI_MAX, camPhi));
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  camRadius += e.deltaY * 0.01;
  camRadius = Math.max(CAM_RADIUS_MIN, Math.min(CAM_RADIUS_MAX, camRadius));
}, { passive: false });

// ---------------------------------------------------------------------------
// MVP matrices — preallocate once, recompute each frame
// ---------------------------------------------------------------------------

const model = mat4.create();
const view = mat4.create();
const proj = mat4.create();
const mvp = mat4.create();

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

function frame(time: DOMHighResTimeStamp) {
  resizeCanvas();

  gl.clearColor(0.08, 0.08, 0.12, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // --- Model matrix: rotate the cube over time ---
  const angle = time * 0.001;
  mat4.identity(model);
  mat4.rotateY(model, model, angle);
  mat4.rotateX(model, model, angle * 0.7);

  // --- View matrix: camera on an orbit sphere ---
  const eye = cameraPosition();
  mat4.lookAt(view, eye, [0, 0, 0], [0, 1, 0]);

  // --- Projection matrix ---
  const aspect = canvas.width / canvas.height;
  mat4.perspective(proj, Math.PI / 4, aspect, 0.1, 100.0);

  // --- Combine: MVP = Projection × View × Model ---
  mat4.multiply(mvp, proj, view);
  mat4.multiply(mvp, mvp, model);

  // --- Draw ---
  gl.useProgram(program);
  gl.uniformMatrix4fv(uMVPLoc, false, mvp);

  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 36);

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
