#pragma once
// OpenGL 3.3 core subset on top of Metal, for the engine's OpenGL backend (see mtlgl.mm).
// The backend's own glad loader is fed the entry points from mtlglGetProcAddress, so the rest of the engine
// keeps calling gl* functions unchanged while the work is done by Metal.

#ifdef __cplusplus
extern "C" {
#endif

// Creates the Metal device and the layer of the given NSWindow (GLFW window created with GLFW_NO_API).
// widthPx/heightPx: the framebuffer size in pixels. Returns 0 on failure.
int mtlglInit(void *nsWindow, int widthPx, int heightPx);
void mtlglShutdown(void);
void *mtlglGetProcAddress(const char *name);

// Finishes the frame: draws the default framebuffer to the window and presents it.
void mtlglPresent(void);
void mtlglSetVsync(int interval);
// The size of the default framebuffer changed (window resized / moved to another screen).
void mtlglResize(int widthPx, int heightPx);

// Interop for the Metal compute passes (ray tracing, MetalFX): the Metal objects behind GL object names.
void *mtlglDevice(void);
void *mtlglCommandQueue(void);
void *mtlglTexture(unsigned int glName);	// id<MTLTexture> (bridged, not retained); creates it when needed
void mtlglFlush(void);	// ends the current pass and submits the work recorded so far

#ifdef __cplusplus
}
#endif
