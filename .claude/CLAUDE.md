# Graphics Experiments

A from-scratch learning project for 3D graphics, game engine design, and shaders.

## Philosophy

- **No black boxes.** We write our own shaders and abstractions. Every line is understood and documented.
- **Small increments.** Each step builds on the previous one. No big-bang architecture.
- **Abstractions emerge from pain.** We don't create wrappers until we've written enough raw code to feel why we need them.
- **Document in code.** Concepts and decisions are explained via comments in the source files. Git history tracks the evolution.

## Stack

- **Language:** TypeScript (strict mode)
- **Bundler:** Vite (dev dependency only)
- **Graphics APIs:** WebGL2 now, WebGPU later
- **Math:** gl-matrix (vectors, matrices, transforms)
- **Runtime dependencies:** gl-matrix only. Everything else we build ourselves.

## Conventions

- Shaders are written as inline strings or in `src/shaders/` — no build-time shader magic.
- Use gl-matrix for math (vec3, mat4, etc.). No other external graphics libraries.
- Each learning step is a git commit. Keep commit messages short (e.g. "Step 1: hello triangle").
- Do not add Co-Authored-By lines to commits.
- Roadmap lives at the top of `src/main.ts`.

## After Every Change

1. `npx tsc --noEmit` — verify no type errors
2. `npm run dev` — confirm it runs and renders correctly
3. Update the roadmap
4. **Report any debt.** If you had to cut corners, report it and ask for recommendations.

## Environment

- Shell: Git Bash (MINGW64) on Windows 11
- Use MINGW-style paths in Bash commands (`/d/code/...`), not Windows backslashes
- Windows `cmd` syntax does not work in this shell
