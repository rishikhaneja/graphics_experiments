// Roadmap:
//   [x] Step 0 — Project Scaffold
//   [x] Step 1 — Hello Triangle
//   [ ] Step 2 — Transformations (rotating 3D cube)
//   [ ] Step 3 — Interactive Camera
//   [ ] Step 4 — Textures
//   [ ] Step 5 — Lighting (Phong)
//   [ ] Step 6 — Abstraction Layer
//   [ ] Step 7 — WebGPU Backend

// Step 1: Hello Triangle
//
// This is the "hello world" of GPU programming. We send three vertices to the
// GPU and tell it to fill the space between them with color.
//
// To get a single triangle on screen, we need to:
//   1. Write a vertex shader   — runs once per vertex, outputs clip-space position
//   2. Write a fragment shader — runs once per pixel inside the triangle, outputs color
//   3. Compile both shaders and link them into a "program" (the GPU pipeline)
//   4. Upload vertex data (positions + colors) into a GPU buffer
//   5. Describe the memory layout of that buffer with a Vertex Array Object (VAO)
//   6. In the render loop: bind the program + VAO, call gl.drawArrays()

// ---------------------------------------------------------------------------
// Canvas + WebGL2 context (same as Step 0)
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

// Vertex shader — GLSL ES 3.00 (the shading language for WebGL2 / OpenGL ES 3.0).
//
// `in` variables receive per-vertex data from the CPU (via a buffer).
// `out` variables are passed to the fragment shader, interpolated across the triangle.
// `gl_Position` is a built-in output: the vertex's position in *clip space*.
//
// Clip space is a coordinate system where:
//   x: -1 (left)  to +1 (right)
//   y: -1 (bottom) to +1 (top)
//   z: -1 (near)  to +1 (far)
// Anything outside this cube gets clipped (not drawn).
// For a 2D triangle we just set z=0 and w=1 (no perspective).
const vertexShaderSource = `#version 300 es

// Per-vertex inputs (attributes).
// location = 0 means "read from attribute slot 0 in the VAO".
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec3 aColor;

// Passed to the fragment shader. The GPU interpolates this value
// across the triangle's surface (barycentric interpolation), so
// each fragment gets a smoothly blended color.
out vec3 vColor;

void main() {
  vColor = aColor;
  // vec4(x, y, z, w). w=1.0 means "this is a point, not a direction".
  // With no projection matrix, clip space == our coordinate space.
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

// Fragment shader — runs once per pixel (fragment) inside the triangle.
//
// `in` variables are the interpolated outputs from the vertex shader.
// `out` declares what this shader writes. We write a single vec4 color (RGBA).
//
// precision mediump float — tells the GPU to use medium precision for floats.
// This is required in fragment shaders in GLSL ES; the vertex shader has a
// default but the fragment shader does not.
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

// This is boilerplate you'll write for every WebGL program.
// The GPU has its own compiler for GLSL. We send source as a string,
// ask it to compile, and check for errors.
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

// A "program" is a linked pair of vertex + fragment shaders.
// It represents the full GPU pipeline for drawing something.
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

  // Shaders are copied into the program at link time.
  // We can delete the individual shader objects now.
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  return program;
}

// ---------------------------------------------------------------------------
// Create the shader program
// ---------------------------------------------------------------------------

const program = createProgram(vertexShaderSource, fragmentShaderSource);

// ---------------------------------------------------------------------------
// Vertex data
// ---------------------------------------------------------------------------

// Three vertices, each with 2D position (x, y) and RGB color (r, g, b).
// We interleave position and color in a single buffer — this is the most
// common layout because it's cache-friendly (all data for one vertex is
// adjacent in memory).
//
// The triangle is in clip space (-1 to +1), centered on screen.
// prettier-ignore
const vertices = new Float32Array([
  // x      y      r    g    b
     0.0,   0.5,   1.0, 0.0, 0.0,  // top vertex    — red
    -0.5,  -0.5,   0.0, 1.0, 0.0,  // bottom-left   — green
     0.5,  -0.5,   0.0, 0.0, 1.0,  // bottom-right  — blue
]);

// ---------------------------------------------------------------------------
// Upload vertex data to GPU
// ---------------------------------------------------------------------------

// A Vertex Buffer Object (VBO) is a chunk of GPU memory that holds vertex data.
// We create one, bind it (make it "active"), and upload our Float32Array into it.
const vbo = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

// gl.bufferData copies data from CPU → GPU.
// STATIC_DRAW is a hint: we'll upload once and draw many times.
// (Other hints: DYNAMIC_DRAW for data that changes often, STREAM_DRAW for data
//  that changes every frame.)
gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

// ---------------------------------------------------------------------------
// Vertex Array Object (VAO) — describe the memory layout
// ---------------------------------------------------------------------------

// A VAO records *how* to read vertex attributes from buffers.
// Think of it as a "format descriptor" — it tells the GPU:
//   "attribute 0 is 2 floats starting at byte 0, stride 20 bytes"
//   "attribute 1 is 3 floats starting at byte 8, stride 20 bytes"
//
// In WebGL2, a VAO is required (WebGL1 had a default one).
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

// Each vertex is 5 floats = 20 bytes total (the "stride").
const STRIDE = 5 * Float32Array.BYTES_PER_ELEMENT; // 20

// Attribute 0: position (vec2) — starts at byte offset 0.
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(
  0,            // attribute location (matches layout(location = 0) in shader)
  2,            // number of components (vec2 = 2 floats)
  gl.FLOAT,     // data type
  false,        // don't normalize
  STRIDE,       // bytes between consecutive vertices
  0             // byte offset of this attribute within each vertex
);

// Attribute 1: color (vec3) — starts at byte offset 8 (after the 2 position floats).
gl.enableVertexAttribArray(1);
gl.vertexAttribPointer(
  1,            // attribute location (matches layout(location = 1) in shader)
  3,            // number of components (vec3 = 3 floats)
  gl.FLOAT,     // data type
  false,        // don't normalize
  STRIDE,       // bytes between consecutive vertices
  2 * Float32Array.BYTES_PER_ELEMENT // byte offset = 8 (skip 2 position floats)
);

// Unbind VAO so nothing else accidentally modifies it.
gl.bindVertexArray(null);

// ---------------------------------------------------------------------------
// Canvas resizing (same as Step 0)
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
// Render loop
// ---------------------------------------------------------------------------

function frame() {
  resizeCanvas();

  gl.clearColor(0.08, 0.08, 0.12, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  // Activate our shader program.
  gl.useProgram(program);

  // Bind the VAO — this restores all the attribute layout we configured above.
  gl.bindVertexArray(vao);

  // Draw! gl.drawArrays reads vertices sequentially from the buffer.
  //   TRIANGLES = interpret every 3 vertices as a triangle
  //   0         = start at vertex index 0
  //   3         = draw 3 vertices (1 triangle)
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  requestAnimationFrame(frame);
}

frame();
