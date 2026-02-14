// Roadmap:
//   [x] Step 0 — Project Scaffold
//   [ ] Step 1 — Hello Triangle
//   [ ] Step 2 — Transformations (rotating 3D cube)
//   [ ] Step 3 — Interactive Camera
//   [ ] Step 4 — Textures
//   [ ] Step 5 — Lighting (Phong)
//   [ ] Step 6 — Abstraction Layer
//   [ ] Step 7 — WebGPU Backend

// Step 0: Get a WebGL2 context and clear the canvas to a color.
//
// This is the absolute minimum you need to talk to the GPU from a browser.
// We're not drawing anything yet — just proving we can control what color
// the canvas is, which means we have a working pipeline from JS → GPU.

// 1. Grab the <canvas> element from the DOM.
const canvas = document.getElementById("canvas") as HTMLCanvasElement;

// 2. Request a WebGL2 rendering context.
//
// WebGL2 is the browser's API for talking to the GPU via OpenGL ES 3.0.
// getContext("webgl2") returns a WebGL2RenderingContext — an object with
// methods that map (roughly) to OpenGL function calls.
//
// This can return null if the browser doesn't support WebGL2.
const maybeGl = canvas.getContext("webgl2");

if (!maybeGl) {
  document.body.innerHTML =
    '<h1 style="color:white;padding:2rem">WebGL2 is not supported in this browser.</h1>';
  throw new Error("WebGL2 not supported");
}

// After the null check + throw above, we know this is safe.
// We assign to a new const so TypeScript narrows the type to non-null
// for all code below (TS can't narrow across function boundaries otherwise).
const gl: WebGL2RenderingContext = maybeGl;

// 3. Handle canvas sizing.
//
// The CSS makes the canvas *appear* full-screen, but the canvas's internal
// pixel buffer (drawingBufferWidth/Height) is separate from its CSS size.
// If we don't sync them, everything will look blurry because the GPU is
// rendering at the wrong resolution.
function resizeCanvas() {
  // devicePixelRatio accounts for high-DPI displays (Retina, etc.).
  const dpr = window.devicePixelRatio || 1;
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;

    // Tell WebGL which rectangle of pixels to draw into.
    // (0, 0) is bottom-left in WebGL (not top-left like the DOM).
    gl.viewport(0, 0, width, height);
  }
}

// 4. The render loop.
//
// requestAnimationFrame calls our function once per display refresh
// (typically 60Hz). Even though we're just clearing for now, setting
// up the loop now means we're ready to animate on the next step.
function frame() {
  resizeCanvas();

  // clearColor sets the color that gl.clear() will fill with.
  // Arguments are (red, green, blue, alpha), each in the range [0, 1].
  // This dark blue-gray is easy on the eyes and proves it's working.
  gl.clearColor(0.08, 0.08, 0.12, 1.0);

  // gl.clear() fills the specified buffers with their clear values.
  // COLOR_BUFFER_BIT = the color buffer (what you see on screen).
  // Later we'll also clear DEPTH_BUFFER_BIT for 3D rendering.
  gl.clear(gl.COLOR_BUFFER_BIT);

  requestAnimationFrame(frame);
}

// Kick off the loop.
frame();
