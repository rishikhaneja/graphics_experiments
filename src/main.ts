// Roadmap:
//   [x] Step 0 — Project Scaffold
//   [x] Step 1 — Hello Triangle
//   [x] Step 2 — Transformations (rotating 3D cube)
//   [x] Step 3 — Interactive Camera
//   [x] Step 4 — Textures
//   [x] Step 5 — Lighting (Phong)
//   [ ] Step 6 — Abstraction Layer
//   [ ] Step 7 — WebGPU Backend

// Step 5: Phong Lighting
//
// Step 4 textured each face, but every face was equally bright regardless of
// its angle to the light. Now we add **Phong lighting**, the standard model
// for basic 3D shading:
//
//   color = ambient + diffuse + specular
//
// - **Ambient**: constant low-level illumination so nothing is pure black.
// - **Diffuse**: brightness proportional to cos(angle between normal and light).
//   Surfaces facing the light are bright; surfaces edge-on are dark.
// - **Specular**: bright highlight where the reflection vector aligns with the
//   view direction. Makes surfaces look shiny.
//
// New data needed per vertex: **normals** — the direction each face points.
// For a cube, all vertices on a face share the same normal (e.g. front face
// normal is [0, 0, 1]).
//
// We also need separate Model and View matrices in the shader (not just MVP)
// so we can transform normals and compute world-space lighting.

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

// Vertex shader — now outputs world-space position and normal for lighting.
// We pass the model matrix separately so we can transform the normal correctly.
// The **normal matrix** is the transpose of the inverse of the upper-left 3×3
// of the model matrix. For uniform scaling (our case), the model matrix itself
// works fine for normals — we just need to re-normalize.
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
  // Transform normal to world space. Normalize because the model matrix
  // may contain non-uniform scale (though ours doesn't currently).
  vNormal = normalize(mat3(uModel) * aNormal);
  vTexCoord = aTexCoord;
  gl_Position = uViewProj * worldPos;
}
`;

// Fragment shader — Phong lighting applied to the texture color.
const fragmentShaderSource = `#version 300 es
precision mediump float;

in vec2 vTexCoord;
in vec3 vWorldPos;
in vec3 vNormal;

uniform sampler2D uTexture;
uniform vec3 uLightPos;    // world-space light position
uniform vec3 uCameraPos;   // world-space camera position

out vec4 fragColor;

void main() {
  vec3 texColor = texture(uTexture, vTexCoord).rgb;

  // Normalize the interpolated normal (interpolation can de-normalize it).
  vec3 N = normalize(vNormal);
  vec3 L = normalize(uLightPos - vWorldPos);  // direction to light
  vec3 V = normalize(uCameraPos - vWorldPos); // direction to camera
  vec3 R = reflect(-L, N);                    // reflection of light around normal

  // Ambient — constant base light so back faces aren't pure black.
  float ambient = 0.15;

  // Diffuse — Lambert's cosine law: max(dot(N, L), 0).
  float diffuse = max(dot(N, L), 0.0);

  // Specular — Phong: pow(max(dot(R, V), 0), shininess).
  // Higher shininess = tighter, shinier highlight.
  float specular = pow(max(dot(R, V), 0.0), 32.0);

  vec3 color = texColor * (ambient + diffuse) + vec3(1.0) * specular * 0.5;
  fragColor = vec4(color, 1.0);
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
const uModelLoc = gl.getUniformLocation(program, "uModel");
const uViewProjLoc = gl.getUniformLocation(program, "uViewProj");
const uTextureLoc = gl.getUniformLocation(program, "uTexture");
const uLightPosLoc = gl.getUniformLocation(program, "uLightPos");
const uCameraPosLoc = gl.getUniformLocation(program, "uCameraPos");

// ---------------------------------------------------------------------------
// Vertex data — a cube with UVs and normals
// ---------------------------------------------------------------------------

// Each vertex: x, y, z, u, v, nx, ny, nz (8 floats = 32 bytes).
// Normals point outward from each face.

// Helper: generate 6 vertices (2 triangles) for one face.
function face(
  positions: number[][],   // 4 corners in CCW order
  normal: number[],
  uvs: number[][]          // 4 UV coords matching the corners
): number[] {
  // Two triangles: 0-1-2 and 0-2-3.
  const indices = [0, 1, 2, 0, 2, 3];
  const out: number[] = [];
  for (const i of indices) {
    out.push(...positions[i], ...uvs[i], ...normal);
  }
  return out;
}

// prettier-ignore
const vertices = new Float32Array([
  // Front face (z = +0.5), normal [0, 0, 1]
  ...face(
    [[-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5]],
    [0, 0, 1],
    [[0,0],[1,0],[1,1],[0,1]]
  ),
  // Back face (z = -0.5), normal [0, 0, -1]
  ...face(
    [[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5]],
    [0, 0, -1],
    [[0,0],[1,0],[1,1],[0,1]]
  ),
  // Top face (y = +0.5), normal [0, 1, 0]
  ...face(
    [[-0.5, 0.5, 0.5],[ 0.5, 0.5, 0.5],[ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5]],
    [0, 1, 0],
    [[0,0],[1,0],[1,1],[0,1]]
  ),
  // Bottom face (y = -0.5), normal [0, -1, 0]
  ...face(
    [[-0.5,-0.5,-0.5],[ 0.5,-0.5,-0.5],[ 0.5,-0.5, 0.5],[-0.5,-0.5, 0.5]],
    [0, -1, 0],
    [[0,0],[1,0],[1,1],[0,1]]
  ),
  // Right face (x = +0.5), normal [1, 0, 0]
  ...face(
    [[ 0.5,-0.5, 0.5],[ 0.5,-0.5,-0.5],[ 0.5, 0.5,-0.5],[ 0.5, 0.5, 0.5]],
    [1, 0, 0],
    [[0,0],[1,0],[1,1],[0,1]]
  ),
  // Left face (x = -0.5), normal [-1, 0, 0]
  ...face(
    [[-0.5,-0.5,-0.5],[-0.5,-0.5, 0.5],[-0.5, 0.5, 0.5],[-0.5, 0.5,-0.5]],
    [-1, 0, 0],
    [[0,0],[1,0],[1,1],[0,1]]
  ),
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

// 8 floats per vertex (3 pos + 2 uv + 3 normal) = 32 bytes stride.
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

const STRIDE = 8 * Float32Array.BYTES_PER_ELEMENT; // 32 bytes

// Attribute 0: position (vec3)
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, STRIDE, 0);

// Attribute 1: texcoord (vec2) — offset 12
gl.enableVertexAttribArray(1);
gl.vertexAttribPointer(1, 2, gl.FLOAT, false, STRIDE, 3 * Float32Array.BYTES_PER_ELEMENT);

// Attribute 2: normal (vec3) — offset 20
gl.enableVertexAttribArray(2);
gl.vertexAttribPointer(2, 3, gl.FLOAT, false, STRIDE, 5 * Float32Array.BYTES_PER_ELEMENT);

gl.bindVertexArray(null);

// ---------------------------------------------------------------------------
// Generate a checkerboard texture procedurally
// ---------------------------------------------------------------------------

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

  // --- Model matrix: rotate the cube over time ---
  const angle = time * 0.001;
  mat4.identity(model);
  mat4.rotateY(model, model, angle);
  mat4.rotateX(model, model, angle * 0.7);

  // --- View matrix ---
  const eye = cameraPosition();
  mat4.lookAt(view, eye, [0, 0, 0], [0, 1, 0]);

  // --- Projection matrix ---
  const aspect = canvas.width / canvas.height;
  mat4.perspective(proj, Math.PI / 4, aspect, 0.1, 100.0);

  // --- ViewProj = Projection × View (Model sent separately for lighting) ---
  mat4.multiply(viewProj, proj, view);

  // --- Draw ---
  gl.useProgram(program);
  gl.uniformMatrix4fv(uModelLoc, false, model);
  gl.uniformMatrix4fv(uViewProjLoc, false, viewProj);

  // Light orbits slowly so you can see the shading change.
  const lightAngle = time * 0.0005;
  gl.uniform3f(uLightPosLoc,
    3.0 * Math.cos(lightAngle),
    2.0,
    3.0 * Math.sin(lightAngle)
  );
  gl.uniform3fv(uCameraPosLoc, eye);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(uTextureLoc, 0);

  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 36);

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
