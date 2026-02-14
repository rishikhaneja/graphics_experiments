// OBJ Parser — parses Wavefront .obj text into an interleaved Float32Array.
//
// Supported directives:
//   v  x y z       — vertex position
//   vt u v         — texture coordinate
//   vn x y z       — vertex normal
//   f  v/vt/vn ... — face (triangulated via fan if > 3 vertices)
//
// The output is a flat, interleaved array: [pos.x, pos.y, pos.z, uv.u, uv.v, norm.x, norm.y, norm.z, ...]
// This matches the vertex layout our renderers expect (location 0=pos, 1=uv, 2=normal).
//
// Limitations (intentional — we add complexity only when we need it):
//   - No material (.mtl) support yet
//   - No groups/objects — everything goes into one mesh
//   - Assumes triangulated or convex polygon faces (fan triangulation)

export interface ObjParseResult {
  /** Interleaved vertex data: [px,py,pz, u,v, nx,ny,nz, ...] */
  vertices: Float32Array;
  /** Number of triangles */
  triangleCount: number;
}

export function parseObj(text: string): ObjParseResult {
  const positions: number[][] = [];
  const texcoords: number[][] = [];
  const normals: number[][] = [];
  const vertexData: number[] = [];

  const lines = text.split("\n");

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (line.length === 0 || line[0] === "#") continue;

    const parts = line.split(/\s+/);
    const keyword = parts[0];

    if (keyword === "v") {
      positions.push([
        parseFloat(parts[1]),
        parseFloat(parts[2]),
        parseFloat(parts[3]),
      ]);
    } else if (keyword === "vt") {
      texcoords.push([
        parseFloat(parts[1]),
        parseFloat(parts[2]),
      ]);
    } else if (keyword === "vn") {
      normals.push([
        parseFloat(parts[1]),
        parseFloat(parts[2]),
        parseFloat(parts[3]),
      ]);
    } else if (keyword === "f") {
      // Parse face vertices. Each face vertex is "v/vt/vn" or "v//vn" or "v/vt" or "v".
      // OBJ indices are 1-based; negative indices count from the end.
      const faceVerts: number[][] = [];

      for (let i = 1; i < parts.length; i++) {
        const indices = parts[i].split("/");
        const vi = resolveIndex(indices[0], positions.length);
        const ti = indices[1] ? resolveIndex(indices[1], texcoords.length) : -1;
        const ni = indices[2] ? resolveIndex(indices[2], normals.length) : -1;

        const pos = positions[vi];
        const uv = ti >= 0 ? texcoords[ti] : [0, 0];
        const norm = ni >= 0 ? normals[ni] : [0, 0, 0];

        faceVerts.push([...pos, ...uv, ...norm]);
      }

      // Fan triangulation: vertex 0 is the hub.
      for (let i = 1; i < faceVerts.length - 1; i++) {
        vertexData.push(...faceVerts[0]);
        vertexData.push(...faceVerts[i]);
        vertexData.push(...faceVerts[i + 1]);
      }
    }
    // Silently ignore other directives (mtllib, usemtl, s, g, o, etc.)
  }

  const vertices = new Float32Array(vertexData);
  const floatsPerVertex = 8; // 3 pos + 2 uv + 3 normal
  const triangleCount = vertices.length / floatsPerVertex / 3;

  return { vertices, triangleCount };
}

/** Resolve a 1-based (or negative) OBJ index to a 0-based array index. */
function resolveIndex(s: string, count: number): number {
  const idx = parseInt(s, 10);
  return idx < 0 ? count + idx : idx - 1;
}
