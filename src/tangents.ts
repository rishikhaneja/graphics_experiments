// Tangent calculation — computes per-vertex tangent vectors for normal mapping.
//
// Normal mapping requires a TBN (Tangent, Bitangent, Normal) matrix at each
// vertex to transform the normal map sample from tangent space to world space.
//
// The tangent vector is derived from the triangle's edge vectors and UV deltas
// using the standard MikkTSpace-like formula:
//   T = (deltaPos1 * deltaUV2.v - deltaPos2 * deltaUV1.v) / det
//
// Input:  interleaved [px,py,pz, u,v, nx,ny,nz, ...] (8 floats/vertex)
// Output: interleaved [px,py,pz, u,v, nx,ny,nz, tx,ty,tz, ...] (11 floats/vertex)

export function computeTangents(vertices: Float32Array): Float32Array {
  const inStride = 8;  // 3 pos + 2 uv + 3 normal
  const outStride = 11; // 3 pos + 2 uv + 3 normal + 3 tangent
  const vertexCount = vertices.length / inStride;
  const out = new Float32Array(vertexCount * outStride);

  // Accumulate tangents per-vertex (triangles share vertices via averaging)
  const tangents = new Float32Array(vertexCount * 3); // tx,ty,tz per vertex

  for (let i = 0; i < vertexCount; i += 3) {
    const i0 = i * inStride;
    const i1 = (i + 1) * inStride;
    const i2 = (i + 2) * inStride;

    // Positions
    const p0x = vertices[i0], p0y = vertices[i0 + 1], p0z = vertices[i0 + 2];
    const p1x = vertices[i1], p1y = vertices[i1 + 1], p1z = vertices[i1 + 2];
    const p2x = vertices[i2], p2y = vertices[i2 + 1], p2z = vertices[i2 + 2];

    // UVs
    const u0 = vertices[i0 + 3], v0 = vertices[i0 + 4];
    const u1 = vertices[i1 + 3], v1 = vertices[i1 + 4];
    const u2 = vertices[i2 + 3], v2 = vertices[i2 + 4];

    // Edge vectors
    const e1x = p1x - p0x, e1y = p1y - p0y, e1z = p1z - p0z;
    const e2x = p2x - p0x, e2y = p2y - p0y, e2z = p2z - p0z;

    // UV deltas
    const du1 = u1 - u0, dv1 = v1 - v0;
    const du2 = u2 - u0, dv2 = v2 - v0;

    const det = du1 * dv2 - du2 * dv1;
    const invDet = Math.abs(det) > 1e-8 ? 1.0 / det : 0.0;

    // Tangent
    const tx = (e1x * dv2 - e2x * dv1) * invDet;
    const ty = (e1y * dv2 - e2y * dv1) * invDet;
    const tz = (e1z * dv2 - e2z * dv1) * invDet;

    // Accumulate for all 3 vertices of this triangle
    for (let v = 0; v < 3; v++) {
      const ti = (i + v) * 3;
      tangents[ti] += tx;
      tangents[ti + 1] += ty;
      tangents[ti + 2] += tz;
    }
  }

  // Write output: copy original data + normalized tangent
  for (let i = 0; i < vertexCount; i++) {
    const inOff = i * inStride;
    const outOff = i * outStride;

    // Copy pos, uv, normal
    for (let j = 0; j < 8; j++) {
      out[outOff + j] = vertices[inOff + j];
    }

    // Normalize tangent, then Gram-Schmidt orthogonalize against normal
    const ti = i * 3;
    let tx = tangents[ti], ty = tangents[ti + 1], tz = tangents[ti + 2];
    const nx = vertices[inOff + 5], ny = vertices[inOff + 6], nz = vertices[inOff + 7];

    // T = T - N * dot(N, T)
    const dot = nx * tx + ny * ty + nz * tz;
    tx -= nx * dot;
    ty -= ny * dot;
    tz -= nz * dot;

    // Normalize
    const len = Math.sqrt(tx * tx + ty * ty + tz * tz);
    if (len > 1e-8) {
      tx /= len;
      ty /= len;
      tz /= len;
    }

    out[outOff + 8] = tx;
    out[outOff + 9] = ty;
    out[outOff + 10] = tz;
  }

  return out;
}
