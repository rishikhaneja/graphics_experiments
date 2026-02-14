// Roadmap:
//   [x] Step 0 — Project Scaffold
//   [x] Step 1 — Hello Triangle
//   [x] Step 2 — Transformations (rotating 3D cube)
//   [x] Step 3 — Interactive Camera
//   [x] Step 4 — Textures
//   [ ] Step 5 — Lighting (Phong)
//   [ ] Step 6 — Abstraction Layer
//   [ ] Step 7 — WebGPU Backend

// Step 4: Textures
//
// Step 3 colored each face with a flat color passed as a vertex attribute.
// Now we replace that color with a **texture lookup**. Each vertex gets a UV
// coordinate (2D position on the texture image, 0–1 range), and the fragment
// shader samples the texture at the interpolated UV.
//
// Key concepts:
//   - **UV coordinates** — 2D coordinates that map each vertex to a point on
//     the texture. (0,0) is bottom-left, (1,1) is top-right.
//   - **Texture unit** — a slot the GPU uses to hold a texture for sampling.
//     We bind our texture to unit 0 and tell the sampler uniform to use it.
//   - **sampler2D** — the GLSL type for a 2D texture sampler.
//
// We generate a checkerboard procedurally so we don't need any image files.

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

// Vertex shader — now passes UV coordinates instead of color.
const vertexShaderSource = `#version 300 es

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;

uniform mat4 uMVP;

out vec2 vTexCoord;

void main() {
  vTexCoord = aTexCoord;
  gl_Position = uMVP * vec4(aPosition, 1.0);
}
`;

// Fragment shader — samples a texture instead of using a vertex color.
const fragmentShaderSource = `#version 300 es
precision mediump float;

in vec2 vTexCoord;
uniform sampler2D uTexture;
out vec4 fragColor;

void main() {
  fragColor = texture(uTexture, vTexCoord);
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
const uTextureLoc = gl.getUniformLocation(program, "uTexture");

// ---------------------------------------------------------------------------
// Vertex data — a cube with UV coordinates per face
// ---------------------------------------------------------------------------

// Each vertex: x, y, z, u, v (5 floats = 20 bytes).
// Each face maps the full 0–1 UV range so the entire texture appears on each face.

// prettier-ignore
const vertices = new Float32Array([
  // Front face (z = +0.5)
  -0.5, -0.5,  0.5,   0.0, 0.0,
   0.5, -0.5,  0.5,   1.0, 0.0,
   0.5,  0.5,  0.5,   1.0, 1.0,
  -0.5, -0.5,  0.5,   0.0, 0.0,
   0.5,  0.5,  0.5,   1.0, 1.0,
  -0.5,  0.5,  0.5,   0.0, 1.0,

  // Back face (z = -0.5)
   0.5, -0.5, -0.5,   0.0, 0.0,
  -0.5, -0.5, -0.5,   1.0, 0.0,
  -0.5,  0.5, -0.5,   1.0, 1.0,
   0.5, -0.5, -0.5,   0.0, 0.0,
  -0.5,  0.5, -0.5,   1.0, 1.0,
   0.5,  0.5, -0.5,   0.0, 1.0,

  // Top face (y = +0.5)
  -0.5,  0.5,  0.5,   0.0, 0.0,
   0.5,  0.5,  0.5,   1.0, 0.0,
   0.5,  0.5, -0.5,   1.0, 1.0,
  -0.5,  0.5,  0.5,   0.0, 0.0,
   0.5,  0.5, -0.5,   1.0, 1.0,
  -0.5,  0.5, -0.5,   0.0, 1.0,

  // Bottom face (y = -0.5)
  -0.5, -0.5, -0.5,   0.0, 0.0,
   0.5, -0.5, -0.5,   1.0, 0.0,
   0.5, -0.5,  0.5,   1.0, 1.0,
  -0.5, -0.5, -0.5,   0.0, 0.0,
   0.5, -0.5,  0.5,   1.0, 1.0,
  -0.5, -0.5,  0.5,   0.0, 1.0,

  // Right face (x = +0.5)
   0.5, -0.5,  0.5,   0.0, 0.0,
   0.5, -0.5, -0.5,   1.0, 0.0,
   0.5,  0.5, -0.5,   1.0, 1.0,
   0.5, -0.5,  0.5,   0.0, 0.0,
   0.5,  0.5, -0.5,   1.0, 1.0,
   0.5,  0.5,  0.5,   0.0, 1.0,

  // Left face (x = -0.5)
  -0.5, -0.5, -0.5,   0.0, 0.0,
  -0.5, -0.5,  0.5,   1.0, 0.0,
  -0.5,  0.5,  0.5,   1.0, 1.0,
  -0.5, -0.5, -0.5,   0.0, 0.0,
  -0.5,  0.5,  0.5,   1.0, 1.0,
  -0.5,  0.5, -0.5,   0.0, 1.0,
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

// 5 floats per vertex (3 pos + 2 uv) = 20 bytes stride.
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

const STRIDE = 5 * Float32Array.BYTES_PER_ELEMENT; // 20 bytes

// Attribute 0: position (vec3)
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, STRIDE, 0);

// Attribute 1: texcoord (vec2) — starts at byte offset 12 (after 3 position floats).
gl.enableVertexAttribArray(1);
gl.vertexAttribPointer(1, 2, gl.FLOAT, false, STRIDE, 3 * Float32Array.BYTES_PER_ELEMENT);

gl.bindVertexArray(null);

// ---------------------------------------------------------------------------
// Generate a checkerboard texture procedurally
// ---------------------------------------------------------------------------

// 8×8 checkerboard, each cell is 1 pixel. We'll let GL_NEAREST filtering keep
// the hard pixel edges visible — this is the classic "programmer art" texture.
const TEX_SIZE = 8;
const texData = new Uint8Array(TEX_SIZE * TEX_SIZE * 4);

for (let row = 0; row < TEX_SIZE; row++) {
  for (let col = 0; col < TEX_SIZE; col++) {
    const i = (row * TEX_SIZE + col) * 4;
    const isWhite = (row + col) % 2 === 0;
    const v = isWhite ? 255 : 60;
    texData[i] = v;
    texData[i + 1] = v;
    texData[i + 2] = v;
    texData[i + 3] = 255;
  }
}

const texture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, texture);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, TEX_SIZE, TEX_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, texData);

// NEAREST filtering — no blurring between checkerboard cells.
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

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

  camTheta -= dx * 0.01;
  camPhi += dy * 0.01;
  camPhi = Math.max(CAM_PHI_MIN, Math.min(CAM_PHI_MAX, camPhi));
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  camRadius += e.deltaY * 0.01;
  camRadius = Math.max(CAM_RADIUS_MIN, Math.min(CAM_RADIUS_MAX, camRadius));
}, { passive: false });

// ---------------------------------------------------------------------------
// MVP matrices
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

  // Bind our checkerboard texture to texture unit 0 and point the sampler at it.
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(uTextureLoc, 0);

  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 36);

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
