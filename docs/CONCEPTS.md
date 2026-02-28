# Concepts

---

## Albedo

The base color of a surface, before any lighting is applied. Like the paint color of a wall, before accounting for shadows or highlights. If a wall is painted red, red is its albedo — it doesn't matter whether the light is bright or dim.

In this project: the checkerboard texture (`TEX_PIXELS` in `src/main.ts`) is the albedo. It's stored in the G-Buffer's third render target (RT2, RGBA8) and read during the lighting pass.

---

## Bloom

A glow effect around bright objects. Because we render into an HDR framebuffer, bright highlights can exceed 1.0. Bloom extracts those bright pixels, blurs them, then adds them back to the final image.

Steps in this project:
1. Pass 4 extracts pixels brighter than 1.0 into a half-resolution texture
2. Pass 5 blurs that texture repeatedly (separable Gaussian, horizontal + vertical, 5 iterations)
3. Pass 6 adds the blurred result on top of the HDR scene

Toggled by the **Post-Processing** checkbox in the UI.

---

## Deferred Rendering

A strategy that separates "what is the geometry?" from "how should it be lit?"

In **forward rendering** (the naive approach): for each light, draw every object lit by it. With 10 lights and 1000 objects, that's up to 10,000 passes.

In **deferred rendering**:
1. Draw all objects once, storing geometry info into textures (the G-Buffer)
2. Run lighting once per light on a fullscreen quad, reading the G-Buffer

Lighting cost no longer multiplies with mesh count. Adding more objects doesn't slow down lighting.

In this project: passes 2–3 of the pipeline. See also: **G-Buffer**.

---

## Fullscreen Triangle

A rendering trick used for post-processing passes. Instead of drawing a mesh, the vertex shader generates a single oversized triangle (bigger than the screen) from the vertex index — no vertex buffer needed. The GPU clips it to the viewport, effectively covering every pixel on screen.

In this project: used by the lighting, bloom extract, blur, and composite passes in both renderers. See `FULLSCREEN_VERT` (WebGL2) and `FULLSCREEN_SHADER` (WebGPU).

---

## G-Buffer

Short for "Geometry Buffer." A set of textures that store surface information for every visible pixel — computed during the G-Buffer pass and read during the lighting pass.

In this project, three render targets:
| Target | Format | Stores |
|--------|--------|--------|
| RT0 | RGBA16F | World-space position |
| RT1 | RGBA16F | World-space normal |
| RT2 | RGBA8 | Albedo (surface color) |

These are written in pass 2 and read in pass 3 (lighting). See `gBufFBO` in both renderers.

---

## Gamma Correction

Monitors don't display colors linearly — they apply a power curve, making mid-range values look brighter than they physically are. To compensate, we apply the inverse correction (raise each color channel to the power 1/2.2) before outputting to screen.

Without it: colors look washed out and too bright.

In this project: applied at the end of pass 6 (composite shader), after tone mapping.

---

## HDR (High Dynamic Range)

Standard framebuffers store color values from 0.0 to 1.0. HDR framebuffers use floating-point textures (RGBA16F) and allow values above 1.0. This lets us represent very bright highlights (e.g., a specular reflection of the sun could be 50.0, not just 1.0).

HDR is necessary for bloom to work — pixels that are "too bright to display" are the ones that should glow.

In this project: the lighting pass outputs to an RGBA16F texture (`hdrFBO`). Values are compressed back to [0, 1] by tone mapping in pass 6.

---

## Normal

A vector perpendicular to a surface, pointing outward. It tells the lighting shader which way the surface "faces." A surface facing the light source receives maximum light; one facing away receives none.

Each vertex has a normal. The GPU interpolates normals across the triangle so the lighting varies smoothly rather than being flat per-triangle.

In this project: stored at vertex layout location 2, and written to G-Buffer RT1 during the G-Buffer pass.

---

## Normal Map

A texture that encodes a surface direction per pixel, used to fake fine surface detail (bumps, grooves, cracks) without adding geometry.

Each pixel stores a direction as an RGB color:
- Red = X component (left-right)
- Green = Y component (up-down)
- Blue = Z component (in-out)

A flat surface uses `(128, 128, 255)` = "pointing straight out." An edge would use a tilted direction.

The lighting shader reads this texture and uses the encoded direction instead of the mesh's geometric normal — making a flat quad look bumpy.

In this project: `NORMAL_MAP` (beveled tile edges) in `src/main.ts`. Toggled by the **Normal Maps** checkbox. When off, a flat 1×1 `(128, 128, 255)` texture is used.

See also: **Tangent**, **TBN Matrix**.

---

## PCF (Percentage Closer Filtering)

A technique to soften the edges of shadows. Without it, shadows have hard pixel-wide edges (shadow acne and aliasing). PCF samples the shadow map multiple times in a small area around each pixel, then averages the results.

In this project: a 3×3 grid of samples is taken around each pixel during the lighting pass. See the lighting shader in both renderers.

---

## Shadow Map

A depth texture captured from the light's point of view. It stores "how far is the closest surface from the light?" for every direction the light can see.

During the lighting pass, each pixel checks: "is the distance from this pixel to the light greater than what the shadow map says is the closest surface?" If yes, something is blocking the light — the pixel is in shadow.

In this project: 1024×1024 depth texture, rendered in pass 1 with front-face culling (to avoid self-shadowing). See `shadowFBO` in both renderers. Toggled by the **Shadows** checkbox.

See also: **PCF**.

---

## Tangent

A vector that lies *along* the surface, aligned with the texture's horizontal direction (the U axis of the UV coordinates). Together with the normal and the bitangent (perpendicular to both), it forms a coordinate system on the surface.

This coordinate system is needed for normal mapping: the directions in a normal map texture are in "texture space" (relative to the texture), but the shader needs them in "world space" (relative to the scene). The tangent tells you which 3D world direction corresponds to the texture's horizontal axis.

In this project: computed per-triangle in `src/tangents.ts`, stored at vertex layout location 3. Also included directly in `makeGroundPlane()` as `(1, 0, 0)` (pointing along X, which matches the ground's UV layout).

See also: **TBN Matrix**, **Normal Map**.

---

## TBN Matrix

A 3×3 matrix built from three vectors: **T**angent, **B**itangent, and **N**ormal. It's a coordinate frame on the surface.

Its job: convert a direction from "texture space" (as stored in a normal map) into "world space" (usable by the lighting shader). Without TBN, normal maps would only work correctly on surfaces that happen to face the right way.

The bitangent (B) is not stored in the vertex buffer — it's computed in the shader as the cross product of the tangent and normal.

In this project: constructed in the G-Buffer vertex shader, passed to the fragment shader, and applied to the normal map sample.

---

## Tone Mapping (ACES)

HDR rendering produces brightness values from 0.0 into the tens or hundreds. Screens can only display 0.0–1.0. Tone mapping compresses the HDR range into displayable range in a visually pleasing way.

ACES (Academy Color Encoding System) is a filmic curve — it preserves detail in the highlights and shadows rather than simply clamping. Bright areas roll off gently instead of blowing out to white.

Without tone mapping: anything above 1.0 displays as pure white.

In this project: applied in pass 6 (composite shader) using the approximation:
```
(x * (2.51x + 0.03)) / (x * (2.43x + 0.59) + 0.14)
```

See also: **Gamma Correction**, **HDR**.

---

## UVs (Texture Coordinates)

A pair of numbers (U, V) stored at each vertex that say: "which part of the texture image should appear at this point on the mesh?" Both range from 0.0 to 1.0, where (0, 0) is the bottom-left of the texture and (1, 1) is the top-right.

The GPU interpolates UVs across the triangle, so every pixel gets a smooth UV and the texture maps seamlessly across the surface.

In this project: stored at vertex layout location 1. The `objParser.ts` reads UVs from the OBJ file; `makeGroundPlane()` assigns them manually so the checkerboard tiles once across the quad.

---

## Vertex Layout

The description of what data is packed into each vertex and in what order. A vertex is just a sequence of floats — the "layout" tells the renderer how to interpret them (which bytes are position, which are UVs, etc.).

In this project, 11 floats per vertex (44 bytes):

| Offset | Attribute | Size |
|--------|-----------|------|
| 0 bytes | position (x, y, z) | 3 floats |
| 12 bytes | texCoord (u, v) | 2 floats |
| 20 bytes | normal (nx, ny, nz) | 3 floats |
| 32 bytes | tangent (tx, ty, tz) | 3 floats |

In this project: `VERTEX_LAYOUT` in `src/main.ts:94`, passed to `renderer.createMesh()`. Both `computeTangents()` and `makeGroundPlane()` produce data that follows this layout exactly.
