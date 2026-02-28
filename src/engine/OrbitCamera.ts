import { mat4, vec3 } from "gl-matrix";

export class OrbitCamera {
  theta = 0.5;
  phi = 1.0;
  radius = 4.0;

  readonly phiMin = 0.1;
  readonly phiMax = Math.PI - 0.1;
  readonly radiusMin = 1.0;
  readonly radiusMax = 20.0;

  private _isDragging = false;
  private _lastX = 0;
  private _lastY = 0;

  /** Bind mouse/wheel listeners for orbit control. */
  attach(canvas: HTMLCanvasElement): void {
    canvas.addEventListener("mousedown", (e) => {
      if (e.button === 0) {
        this._isDragging = true;
        this._lastX = e.clientX;
        this._lastY = e.clientY;
      }
    });
    window.addEventListener("mouseup", () => { this._isDragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!this._isDragging) return;
      const dx = e.clientX - this._lastX;
      const dy = e.clientY - this._lastY;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      this.theta -= dx * 0.01;
      this.phi = Math.max(this.phiMin, Math.min(this.phiMax, this.phi + dy * 0.01));
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.radius = Math.max(this.radiusMin, Math.min(this.radiusMax, this.radius + e.deltaY * 0.01));
    }, { passive: false });
  }

  /** Current eye position in world space. */
  position(): vec3 {
    return vec3.fromValues(
      this.radius * Math.sin(this.phi) * Math.sin(this.theta),
      this.radius * Math.cos(this.phi),
      this.radius * Math.sin(this.phi) * Math.cos(this.theta),
    );
  }

  /** Builds the view-projection matrix. */
  viewProjection(out: mat4, aspect: number): mat4 {
    const eye = this.position();
    const view = mat4.create();
    const proj = mat4.create();
    mat4.lookAt(view, eye, [0, 0, 0], [0, 1, 0]);
    mat4.perspective(proj, Math.PI / 4, aspect, 0.1, 100.0);
    mat4.multiply(out, proj, view);
    return out;
  }
}
