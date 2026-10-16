# librw-metal

Fork of [librw](https://github.com/aap/librw) (MIT, a RenderWare reimplementation by aap) with a Metal backend for macOS.

- `src/mtl/mtlgl_impl.cpp`: implements the OpenGL 3.3 subset that the GL3 backend uses on top of Metal
  (buffers, textures, framebuffers, programs, render state, draws)
- `src/mtl/mtlshadertrans.cpp`: translates the GLSL shaders to Metal Shading Language at run time
  (glslang and SPIRV-Cross)
- GL3 backend additions: render scale with FSR1-style upscaling, temporal resolve, per-object motion vectors,
  hooks for Metal ray tracing and MetalFX

Build with `-DLIBRW_METALGL=ON` (needs `brew install glslang spirv-cross`). This repository contains no game code or data.
It is used by [regta-upstate-metal](https://github.com/mindingulove/regta-upstate-metal).
