// Roadmap:
//   [x] Step 0 — Project Scaffold
//   [x] Step 1 — Hello Triangle
//   [x] Step 2 — Transformations (rotating 3D cube)
//   [ ] Step 3 — Interactive Camera
//   [ ] Step 4 — Textures
//   [ ] Step 5 — Lighting (Phong)
//   [ ] Step 6 — Abstraction Layer
//   [ ] Step 7 — WebGPU Backend

// Step 2: Rotating Cube with MVP Transforms
//
// Step 1 drew a 2D triangle directly in clip space — no math involved.
// Now we introduce the **model-view-projection (MVP) pipeline**, the core of
// all 3D rendering. Every vertex goes through three transforms:
//
//   Model      — places the object in the world (rotation, translation, scale)
//   View       — moves the world so the camera is at the origin looking down -Z
//   Projection — squishes 3D into 2D with perspective (distant things look smaller)
//
// On the GPU: gl_Position = Projection * View * Model * vec4(position, 1.0)
//
// We also enable **depth testing** — without it, triangles drawn later would
// paint over closer triangles, and the cube would look inside-out.

import { mat4 } from "gl-matrix";

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

// Vertex shader — now takes a 3D position and a uniform MVP matrix.
//
// `uniform` variables are set once per draw call from the CPU. They don't
// change per-vertex (unlike `in` attributes). We use a uniform for the MVP
// matrix because every vertex of the cube shares the same transform.
const vertexShaderSource = `#version 300 es

layout(location = 0) in vec3 aPosition;   // 3D now (was vec2)
layout(location = 1) in vec3 aColor;

uniform mat4 uMVP;   // model-view-projection matrix, set from CPU each frame

out vec3 vColor;

void main() {
  vColor = aColor;
  // Multiply position by MVP to go from local object space → clip space.
  // vec4(..., 1.0): w=1 marks this as a position (not a direction vector).
  gl_Position = uMVP * vec4(aPosition, 1.0);
}
`;

// Fragment shader — unchanged from Step 1. Still receives interpolated color.
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

// Look up the uniform location for our MVP matrix. We'll upload a new matrix
// every frame as the cube rotates.
const uMVPLoc = gl.getUniformLocation(program, "uMVP");

// ---------------------------------------------------------------------------
// Vertex data — a cube with 6 colored faces
// ---------------------------------------------------------------------------

// A cube has 8 corners, but we can't share vertices across faces when each
// face has a different color. Each face is 2 triangles × 3 vertices = 6 verts,
// and 6 faces × 6 = 36 vertices total.
//
// Each vertex: x, y, z, r, g, b (6 floats = 24 bytes).
// The cube spans from -0.5 to +0.5 on each axis (unit cube centered at origin).

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

// Now 6 floats per vertex (3 pos + 3 color) = 24 bytes stride.
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

const STRIDE = 6 * Float32Array.BYTES_PER_ELEMENT; // 24 bytes

// Attribute 0: position (vec3) — starts at byte offset 0.
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(
  0,            // location
  3,            // 3 components (vec3, was vec2 in Step 1)
  gl.FLOAT,
  false,
  STRIDE,
  0
);

// Attribute 1: color (vec3) — starts at byte offset 12 (after 3 position floats).
gl.enableVertexAttribArray(1);
gl.vertexAttribPointer(
  1,            // location
  3,            // 3 components (vec3)
  gl.FLOAT,
  false,
  STRIDE,
  3 * Float32Array.BYTES_PER_ELEMENT // offset = 12 bytes
);

gl.bindVertexArray(null);

// ---------------------------------------------------------------------------
// Enable depth testing
// ---------------------------------------------------------------------------

// Without depth testing, triangles are painted in the order they appear in the
// buffer. A back face drawn after a front face would overwrite it. Depth testing
// keeps a per-pixel depth value and only lets closer fragments through.
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
// MVP matrices — preallocate once, recompute each frame
// ---------------------------------------------------------------------------

// We allocate these once to avoid GC pressure in the render loop.
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
  // time is in milliseconds. Multiply by 0.001 to get a smooth ~1 radian/sec.
  const angle = time * 0.001;
  mat4.identity(model);
  mat4.rotateY(model, model, angle);        // primary rotation around Y
  mat4.rotateX(model, model, angle * 0.7);  // slower tilt around X for tumble

  // --- View matrix: camera positioned at [0, 0, 3] looking at the origin ---
  mat4.lookAt(view, [0, 0, 3], [0, 0, 0], [0, 1, 0]);

  // --- Projection matrix: perspective with 45° FOV ---
  // Recalculate aspect ratio every frame so resizing doesn't stretch the cube.
  const aspect = canvas.width / canvas.height;
  mat4.perspective(proj, Math.PI / 4, aspect, 0.1, 100.0);

  // --- Combine: MVP = Projection × View × Model ---
  mat4.multiply(mvp, proj, view);   // mvp = proj * view
  mat4.multiply(mvp, mvp, model);   // mvp = (proj * view) * model

  // --- Draw ---
  gl.useProgram(program);

  // Upload the MVP matrix to the GPU. uniformMatrix4fv takes a Float32Array
  // (or array-like). gl-matrix mat4 is already a Float32Array.
  // The `false` parameter means "don't transpose" — gl-matrix uses column-major
  // order, which is what OpenGL/WebGL expects.
  gl.uniformMatrix4fv(uMVPLoc, false, mvp);

  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 36); // 36 vertices = 12 triangles = 6 faces

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
