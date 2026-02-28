// Engine — owns the requestAnimationFrame loop, calling update then render each frame.

import type { Renderer, RenderOptions } from "./Renderer";
import type { Scene } from "./Scene";

export class Engine {
  renderer: Renderer;
  scene: Scene;
  options: RenderOptions;
  private _rafId = 0;

  constructor(renderer: Renderer, scene: Scene, options: RenderOptions) {
    this.renderer = renderer;
    this.scene = scene;
    this.options = options;
  }

  start(): void {
    const loop = (time: DOMHighResTimeStamp) => {
      this.renderer.resize();
      this.scene.update(time);
      const { drawCalls, uniforms } = this.scene.buildFrame(this.renderer.aspect);
      this.renderer.renderFrame(drawCalls, uniforms, this.options);
      this._rafId = requestAnimationFrame(loop);
    };
    this._rafId = requestAnimationFrame(loop);
  }

  stop(): void {
    cancelAnimationFrame(this._rafId);
  }
}
