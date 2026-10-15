// OpenGL 3.3 core subset implemented on Metal. See mtlgl.h.
//
// Conventions (so that the engine's GL code, shaders and render-to-texture passes keep working unchanged):
//  * every vertex shader gets its clip space y flipped, so a texture's row 0 is GL's bottom row (GL convention) for all targets;
//    the default framebuffer is an offscreen texture that is flipped once when it is presented
//  * clip space depth is converted from -1..1 to 0..1 by the translator
//  * uniform values live in per-program shadow blocks that are uploaded when a draw needs them
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Cocoa/Cocoa.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <atomic>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>

#include "../gl/glad/glad.h"
#include "mtlgl.h"
#include "mtlshadertrans.h"

#ifndef GL_TEXTURE_RECTANGLE
#define GL_TEXTURE_RECTANGLE 0x84F5
#endif

namespace {

using rw::mtl::TransProgram;

static void logf_(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	FILE *f = fopen("metalgl.log", "a");
	if(f){ vfprintf(f, fmt, ap); fputc('\n', f); fclose(f); }
	va_end(ap);
}

// ---------------------------------------------------------------------------------------------------------
// objects

struct Page { id<MTLBuffer> buf; size_t used; size_t size; uint64_t serial; };

struct Buf {
	id<MTLBuffer> buf = nil;	// current storage (own, or a page of the ring)
	size_t off = 0;		// offset of the data inside buf
	size_t size = 0;	// size given by glBufferData
	size_t valid = 0;	// bytes that hold defined data
	bool ring = false;	// storage is a slice of the ring (frame lifetime)
	GLenum usage = GL_STATIC_DRAW;
};

struct Level { int w = 0, h = 0; std::vector<uint8_t> data; bool has = false; };

struct Tex {
	GLenum target = GL_TEXTURE_2D;
	id<MTLTexture> mtl = nil;
	MTLPixelFormat fmt = MTLPixelFormatInvalid;
	int w = 0, h = 0;
	GLenum ifmt = 0;
	bool compressed = false, depth = false;
	int maxLevel = 1000;
	bool genMips = false;
	std::vector<Level> lv;
	GLenum minF = GL_NEAREST_MIPMAP_LINEAR, magF = GL_LINEAR, wrapS = GL_REPEAT, wrapT = GL_REPEAT;
	float aniso = 1.0f;
	uint64_t lastUse = 0;
	int created = 0;	// levels of mtl
};

struct Rb { id<MTLTexture> mtl = nil; int w = 0, h = 0; GLenum fmt = 0; };

struct Attach { Tex *tex = nullptr; Rb *rb = nullptr; int level = 0; };

struct Fbo {
	Attach color[4];
	Attach depth;	// depth/stencil
	GLenum drawBuf[4] = { GL_COLOR_ATTACHMENT0, GL_NONE, GL_NONE, GL_NONE };
	bool hasDrawBufSet = false;
	// pending clear, applied when the next pass for this framebuffer begins
	bool clrColor[4] = {}; float clrVal[4][4] = {};
	bool clrDepth = false; float clrD = 1.0f;
	bool clrStencil = false; uint8_t clrS = 0;
	bool isDefault = false;
};

struct Shader { GLenum type; std::string src; };

struct Prog {
	std::vector<std::string> vsSrcParts, fsSrcParts;	// not kept after link
	bool linked = false;
	TransProgram tr;
	id<MTLLibrary> vsLib = nil, fsLib = nil;
	id<MTLFunction> vsFn = nil, fsFn = nil;
	std::string log;
	// uniforms
	std::vector<uint8_t> vsBlock, fsBlock;
	struct U { std::string name; int vsOff = -1, fsOff = -1; int size = 0; int sampler = -1; };
	std::vector<U> uniforms;
	std::vector<int> samplerUnit;	// per translator sampler: texture unit
	uint64_t version = 1;		// bumped when a uniform changes
	std::unordered_map<std::string, int> byName;
	// compiled lazily
	bool compiled = false, compileFailed = false;
};

struct Attrib {
	bool enabled = false;
	GLuint buffer = 0;
	int size = 4; GLenum type = GL_FLOAT; bool normalized = false;
	int stride = 0; size_t offset = 0;
};

struct Vao { Attrib a[8]; GLuint ebo = 0; };

}  // namespace

namespace {

// ---------------------------------------------------------------------------------------------------------
// context

struct PsoKey {
	uint64_t prog;
	uint32_t colorFmt[4];
	uint8_t colorMask[4];
	uint8_t blendOn[4];
	uint8_t srcRGB, dstRGB, srcA, dstA;
	uint32_t depthFmt, stencilFmt;
	uint8_t flip;
	uint8_t nAttr;
	uint8_t pad[2];
	struct A { uint8_t fmt, slot; uint16_t off; } a[8];
	uint16_t stride[8];
	bool operator==(const PsoKey &o) const { return memcmp(this, &o, sizeof(*this)) == 0; }
};
struct PsoHash { size_t operator()(const PsoKey &k) const {
	const uint8_t *p = (const uint8_t*)&k; uint64_t h = 1469598103934665603ull;
	for(size_t i = 0; i < sizeof(k); i++){ h ^= p[i]; h *= 1099511628211ull; }
	return (size_t)h; } };

struct DsKey {
	uint8_t depthTest, depthFunc, depthMask, stencilOn, sFunc, sFail, sZFail, sPass, sMask, sWrite, pad0, pad1;
	bool operator==(const DsKey &o) const { return memcmp(this, &o, sizeof(*this)) == 0; }
};
struct DsHash { size_t operator()(const DsKey &k) const {
	const uint8_t *p = (const uint8_t*)&k; uint64_t h = 1469598103934665603ull;
	for(size_t i = 0; i < sizeof(k); i++){ h ^= p[i]; h *= 1099511628211ull; }
	return (size_t)h; } };

struct Ctx {
	id<MTLDevice> dev = nil;
	id<MTLCommandQueue> queue = nil;
	CAMetalLayer *layer = nil;
	NSWindow *window = nil;
	int vsync = 1;

	id<MTLCommandBuffer> cb = nil;
	id<MTLRenderCommandEncoder> enc = nil;
	std::atomic<uint64_t> completed{0};
	uint64_t serial = 1;	// serial of the command buffer being recorded

	// object tables (index = GL name)
	std::vector<Buf*> bufs{ nullptr };
	std::vector<Tex*> texs{ nullptr };
	std::vector<Rb*> rbs{ nullptr };
	std::vector<Fbo*> fbos{ nullptr };
	std::vector<Shader*> shaders{ nullptr };
	std::vector<Prog*> progs{ nullptr };
	std::vector<Vao*> vaos{ nullptr };

	// default framebuffer
	Fbo *defFbo = nullptr;
	Tex *defColor = nullptr, *defDepth = nullptr;
	int fbW = 0, fbH = 0;

	// bindings
	GLuint arrayBuf = 0, uniformBuf = 0;
	GLuint vao = 0;
	GLuint drawFbo = 0, readFbo = 0, rbo = 0;
	GLuint prog = 0;
	int activeUnit = 0;
	GLuint unitTex2D[16] = {}, unitTexRect[16] = {};
	GLenum readBuffer = GL_COLOR_ATTACHMENT0;

	// render state
	bool blend = false; bool blendOff[4] = {};
	GLenum bSrcRGB = GL_ONE, bDstRGB = GL_ZERO, bSrcA = GL_ONE, bDstA = GL_ZERO;
	bool depthTest = false, depthMask = true; GLenum depthFunc = GL_LESS;
	bool cull = false; GLenum cullFace = GL_BACK;
	bool scissor = false; int sc[4] = {};
	int vp[4] = {};
	bool colorMask[4] = { true, true, true, true };
	bool stencilTest = false;
	GLenum sFunc = GL_ALWAYS, sFail = GL_KEEP, sZFail = GL_KEEP, sPass = GL_KEEP;
	int sRef = 0; uint32_t sMask = 0xFF, sWrite = 0xFF;
	float clearColor[4] = {};
	int unpackAlign = 4, packAlign = 4;

	// ring for streamed data
	std::vector<Page> pages;
	size_t curPage = 0;

	// caches
	std::unordered_map<PsoKey, id<MTLRenderPipelineState>, PsoHash> psos;
	std::unordered_map<DsKey, id<MTLDepthStencilState>, DsHash> dss;
	std::unordered_map<uint32_t, id<MTLSamplerState>> samplers;
	id<MTLTexture> blackTex = nil;

	// current encoder state
	struct EncState {
		Fbo *fbo = nullptr;
		id<MTLTexture> color[4] = {}, depth = nil;
		bool draws = false;
		id<MTLRenderPipelineState> pso = nil;
		id<MTLDepthStencilState> ds = nil;
		int cullMode = -1, winding = -1;
		bool vpSet = false; int vp[4] = {};
		bool scSet = false; int sc[4] = {};
		int stencilRef = -1;
		uint64_t progVer[2] = { 0, 0 }; GLuint progId[2] = { 0, 0 };
		id<MTLBuffer> vbuf[16] = {}; size_t voff[16] = {};
		id<MTLTexture> ftex[16] = {}, vtex[16] = {};
		id<MTLSamplerState> fsamp[16] = {};
	} es;

	// internal shaders (clear, blit)
	id<MTLLibrary> utilLib = nil;
	std::unordered_map<uint64_t, id<MTLRenderPipelineState>> utilPsos;
};

static Ctx *G = nullptr;

static inline Tex *texOf(GLuint n){ return n < G->texs.size() ? G->texs[n] : nullptr; }
static inline Buf *bufOf(GLuint n){ return n < G->bufs.size() ? G->bufs[n] : nullptr; }
static inline Fbo *fboOf(GLuint n){ return n == 0 ? G->defFbo : (n < G->fbos.size() ? G->fbos[n] : nullptr); }

// ---------------------------------------------------------------------------------------------------------
// command buffer / encoder

static id<MTLCommandBuffer>
cmd(void)
{
	if(!G->cb){
		G->cb = [G->queue commandBuffer];
		uint64_t s = G->serial;
		[G->cb addCompletedHandler:^(id<MTLCommandBuffer> b){
			uint64_t cur = G->completed.load();
			while(cur < s && !G->completed.compare_exchange_weak(cur, s)) {}
		}];
	}
	return G->cb;
}

static void
endEncoder(void)
{
	if(G->enc){
		[G->enc endEncoding];
		G->enc = nil;
	}
	G->es = Ctx::EncState();
}

static void
submit(bool wait)
{
	endEncoder();
	if(G->cb){
		id<MTLCommandBuffer> cb = G->cb;
		G->cb = nil;
		[cb commit];
		G->serial++;
		if(wait) [cb waitUntilCompleted];
	}
}

// ---------------------------------------------------------------------------------------------------------
// ring for streamed buffers and texture uploads

static bool
ringAlloc(size_t n, id<MTLBuffer> *buf, size_t *off)
{
	n = (n + 255) & ~(size_t)255;
	uint64_t done = G->completed.load();
	if(!(G->curPage < G->pages.size() && G->pages[G->curPage].size - G->pages[G->curPage].used >= n)){
		bool found = false;
		for(size_t i = 0; i < G->pages.size(); i++){
			Page &p = G->pages[i];
			if(i != G->curPage && p.serial <= done && p.size >= n){
				p.used = 0; G->curPage = i; found = true; break;
			}
		}
		if(!found){
			Page p;
			p.size = n > (16u << 20) ? n : (16u << 20);
			p.buf = [G->dev newBufferWithLength:p.size options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
			if(!p.buf) return false;
			p.used = 0;
			p.serial = 0;
			G->pages.push_back(p);
			G->curPage = G->pages.size() - 1;
		}
	}
	Page &p = G->pages[G->curPage];
	*buf = p.buf;
	*off = p.used;
	p.used += n;
	p.serial = G->serial;
	return true;
}

// ---------------------------------------------------------------------------------------------------------
// pixel formats

static MTLPixelFormat
mtlFormatFor(GLenum ifmt, bool *depth, bool *compressed)
{
	*depth = false;
	*compressed = false;
	switch(ifmt){
	case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
	case GL_COMPRESSED_RGB_S3TC_DXT1_EXT: *compressed = true; return MTLPixelFormatBC1_RGBA;
	case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT: *compressed = true; return MTLPixelFormatBC2_RGBA;
	case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: *compressed = true; return MTLPixelFormatBC3_RGBA;
	case GL_R8: case GL_RED: return MTLPixelFormatR8Unorm;
	case GL_RG8: return MTLPixelFormatRG8Unorm;
	case GL_R32F: return MTLPixelFormatR32Float;
	case GL_R16F: return MTLPixelFormatR16Float;
	case GL_RG16F: return MTLPixelFormatRG16Float;
	case GL_RGBA16F: return MTLPixelFormatRGBA16Float;
	case GL_RGBA32F: return MTLPixelFormatRGBA32Float;
	case GL_DEPTH24_STENCIL8: case GL_DEPTH_STENCIL: case GL_DEPTH32F_STENCIL8:
		*depth = true; return MTLPixelFormatDepth32Float_Stencil8;
	case GL_DEPTH_COMPONENT: case GL_DEPTH_COMPONENT16: case GL_DEPTH_COMPONENT24: case GL_DEPTH_COMPONENT32F:
		*depth = true; return MTLPixelFormatDepth32Float;
	case GL_BGRA: return MTLPixelFormatBGRA8Unorm;
	default: return MTLPixelFormatRGBA8Unorm;	// RGBA8, RGB8, RGBA, RGB, RGB5_A1, RGBA4, 4 ...
	}
}

static size_t
levelBytes(MTLPixelFormat f, int w, int h, size_t *rowBytes)
{
	size_t bw = 1, bb = 4;
	switch(f){
	case MTLPixelFormatBC1_RGBA: bw = 4; bb = 8; break;
	case MTLPixelFormatBC2_RGBA: case MTLPixelFormatBC3_RGBA: bw = 4; bb = 16; break;
	case MTLPixelFormatR8Unorm: bb = 1; break;
	case MTLPixelFormatRG8Unorm: case MTLPixelFormatR16Float: bb = 2; break;
	case MTLPixelFormatR32Float: case MTLPixelFormatRG16Float: bb = 4; break;
	case MTLPixelFormatRGBA16Float: bb = 8; break;
	case MTLPixelFormatRGBA32Float: bb = 16; break;
	default: break;
	}
	size_t bx = (w + bw - 1)/bw, by = (h + bw - 1)/bw;
	*rowBytes = bx*bb;
	return bx*by*bb;
}

// Converts client pixels to the Metal format of the texture.
static void
convertPixels(MTLPixelFormat mf, GLenum format, GLenum type, int w, int h, const void *src, std::vector<uint8_t> &out)
{
	size_t rowBytes;
	size_t total = levelBytes(mf, w, h, &rowBytes);
	out.assign(total, 0);
	int comps = format == GL_RGBA || format == GL_BGRA ? 4 : format == GL_RGB || format == GL_BGR ? 3 :
	            format == GL_RG ? 2 : 1;
	size_t bpp;
	if(type == GL_UNSIGNED_SHORT_5_5_5_1 || type == GL_UNSIGNED_SHORT_5_6_5 || type == GL_UNSIGNED_SHORT_4_4_4_4 ||
	   type == GL_UNSIGNED_SHORT_1_5_5_5_REV || type == GL_UNSIGNED_SHORT_4_4_4_4_REV) bpp = 2;
	else if(type == GL_UNSIGNED_INT_8_8_8_8_REV || type == GL_UNSIGNED_INT_8_8_8_8) bpp = 4;
	else if(type == GL_FLOAT) bpp = 4*comps;
	else if(type == GL_HALF_FLOAT) bpp = 2*comps;
	else bpp = comps;
	size_t srcRow = ((size_t)w*bpp + G->unpackAlign - 1)/G->unpackAlign*G->unpackAlign;
	const uint8_t *s = (const uint8_t*)src;
	bool toRGBA8 = mf == MTLPixelFormatRGBA8Unorm || mf == MTLPixelFormatBGRA8Unorm;
	if(!toRGBA8){
		// same layout as the source (R8, RG8, floats, half floats)
		size_t n = rowBytes < srcRow ? rowBytes : srcRow;
		for(int y = 0; y < h; y++)
			memcpy(out.data() + y*rowBytes, s + y*srcRow, n);
		return;
	}
	bool bgra = mf == MTLPixelFormatBGRA8Unorm;
	for(int y = 0; y < h; y++){
		const uint8_t *r = s + y*srcRow;
		uint8_t *d = out.data() + y*rowBytes;
		for(int x = 0; x < w; x++, d += 4){
			uint8_t R, Gc, B, A = 255;
			if(type == GL_UNSIGNED_SHORT_5_5_5_1){
				uint16_t v = ((const uint16_t*)r)[x];
				R = ((v >> 11) & 31)*255/31; Gc = ((v >> 6) & 31)*255/31; B = ((v >> 1) & 31)*255/31; A = (v & 1) ? 255 : 0;
			}else if(type == GL_UNSIGNED_SHORT_5_6_5){
				uint16_t v = ((const uint16_t*)r)[x];
				R = ((v >> 11) & 31)*255/31; Gc = ((v >> 5) & 63)*255/63; B = (v & 31)*255/31;
			}else if(type == GL_UNSIGNED_SHORT_4_4_4_4){
				uint16_t v = ((const uint16_t*)r)[x];
				R = ((v >> 12) & 15)*17; Gc = ((v >> 8) & 15)*17; B = ((v >> 4) & 15)*17; A = (v & 15)*17;
			}else if(type == GL_UNSIGNED_SHORT_1_5_5_5_REV){
				uint16_t v = ((const uint16_t*)r)[x];
				B = (v & 31)*255/31; Gc = ((v >> 5) & 31)*255/31; R = ((v >> 10) & 31)*255/31; A = (v & 0x8000) ? 255 : 0;
			}else if(type == GL_UNSIGNED_INT_8_8_8_8_REV){
				const uint8_t *p = r + x*4;
				if(format == GL_BGRA){ B = p[0]; Gc = p[1]; R = p[2]; A = p[3]; }
				else { R = p[0]; Gc = p[1]; B = p[2]; A = p[3]; }
			}else{
				const uint8_t *p = r + x*bpp;
				if(comps == 4){
					if(format == GL_BGRA){ B = p[0]; Gc = p[1]; R = p[2]; A = p[3]; }
					else { R = p[0]; Gc = p[1]; B = p[2]; A = p[3]; }
				}else if(comps == 3){
					if(format == GL_BGR){ B = p[0]; Gc = p[1]; R = p[2]; }
					else { R = p[0]; Gc = p[1]; B = p[2]; }
				}else if(comps == 2){ R = Gc = B = p[0]; A = p[1]; }
				else { R = Gc = B = p[0]; }
			}
			if(bgra){ d[0] = B; d[1] = Gc; d[2] = R; d[3] = A; }
			else { d[0] = R; d[1] = Gc; d[2] = B; d[3] = A; }
		}
	}
}

// ---------------------------------------------------------------------------------------------------------
// textures

static int
fullLevels(int w, int h)
{
	int n = 1;
	while(w > 1 || h > 1){ n++; w = w > 1 ? w/2 : 1; h = h > 1 ? h/2 : 1; }
	return n;
}

static void
uploadLevel(Tex *t, int level, const void *bytes, size_t rowBytes, int w, int h)
{
	if(t->lastUse != 0 && t->lastUse > G->completed.load()){
		// the GPU may still read this texture: go through a staging buffer and a blit
		size_t total = rowBytes*(size_t)(t->compressed ? (h + 3)/4 : h);
		id<MTLBuffer> sb; size_t so;
		if(ringAlloc(total, &sb, &so)){
			memcpy((uint8_t*)sb.contents + so, bytes, total);
			endEncoder();
			id<MTLBlitCommandEncoder> bl = [cmd() blitCommandEncoder];
			[bl copyFromBuffer:sb sourceOffset:so sourceBytesPerRow:rowBytes sourceBytesPerImage:total
			        sourceSize:MTLSizeMake(w, h, 1) toTexture:t->mtl destinationSlice:0 destinationLevel:level
			        destinationOrigin:MTLOriginMake(0, 0, 0)];
			[bl endEncoding];
			t->lastUse = G->serial;
			return;
		}
	}
	[t->mtl replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:level withBytes:bytes bytesPerRow:rowBytes];
}

// Creates the Metal texture of a GL texture from what the engine specified so far.
static bool
ensureTex(Tex *t)
{
	if(t->mtl) return true;
	if(t->w <= 0 || t->h <= 0 || t->fmt == MTLPixelFormatInvalid) return false;
	int specified = 0;
	while(specified < (int)t->lv.size() && t->lv[specified].w > 0) specified++;
	int full = fullLevels(t->w, t->h);
	int n = specified > 0 ? specified : 1;
	if(n > t->maxLevel + 1) n = t->maxLevel + 1;
	if(n > full) n = full;
	if(t->genMips) n = full;
	if(n < 1) n = 1;
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t->fmt width:t->w height:t->h mipmapped:NO];
	d.mipmapLevelCount = n;
	d.storageMode = MTLStorageModeShared;
	d.usage = MTLTextureUsageShaderRead | (t->compressed ? 0 : (MTLTextureUsageRenderTarget | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView));
	if(t->depth){ d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget; }
	t->mtl = [G->dev newTextureWithDescriptor:d];
	if(!t->mtl){ logf_("texture creation failed %dx%d fmt %d", t->w, t->h, (int)t->fmt); return false; }
	t->created = n;
	t->lastUse = 0;
	if(!t->depth){
		bool any = false;
		for(int i = 0; i < n && i < (int)t->lv.size(); i++){
			Level &l = t->lv[i];
			if(l.w > 0 && !l.data.empty()){
				size_t rb;
				levelBytes(t->fmt, l.w, l.h, &rb);
				[t->mtl replaceRegion:MTLRegionMake2D(0, 0, l.w, l.h) mipmapLevel:i withBytes:l.data.data() bytesPerRow:rb];
				std::vector<uint8_t>().swap(l.data);
				if(i == 0) any = true;
			}
		}
		if(t->genMips && any && !t->compressed){
			endEncoder();
			id<MTLBlitCommandEncoder> bl = [cmd() blitCommandEncoder];
			[bl generateMipmapsForTexture:t->mtl];
			[bl endEncoding];
			t->lastUse = G->serial;
		}
	}
	return true;
}

static uint32_t
samplerKey(Tex *t)
{
	auto filt = [](GLenum f){ return (f == GL_NEAREST || f == GL_NEAREST_MIPMAP_NEAREST || f == GL_NEAREST_MIPMAP_LINEAR) ? 0 : 1; };
	auto mip = [](GLenum f){ return (f == GL_NEAREST || f == GL_LINEAR) ? 0 : (f == GL_NEAREST_MIPMAP_NEAREST || f == GL_LINEAR_MIPMAP_NEAREST) ? 1 : 2; };
	auto wrap = [](GLenum w){ return w == GL_CLAMP_TO_EDGE ? 1 : w == GL_MIRRORED_REPEAT ? 2 : w == GL_CLAMP_TO_BORDER ? 3 : 0; };
	int an = (int)t->aniso; if(an < 1) an = 1; if(an > 16) an = 16;
	uint32_t k = filt(t->minF) | filt(t->magF) << 1 | mip(t->minF) << 2 | wrap(t->wrapS) << 4 | wrap(t->wrapT) << 6 | an << 8;
	if(t->created <= 1) k |= 1u << 16;
	return k;
}

static id<MTLSamplerState>
samplerFor(Tex *t)
{
	uint32_t k = samplerKey(t);
	auto it = G->samplers.find(k);
	if(it != G->samplers.end()) return it->second;
	MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
	d.minFilter = (k & 1) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
	d.magFilter = (k & 2) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
	int mip = (k >> 2) & 3;
	d.mipFilter = ((k & (1u << 16)) || mip == 0) ? MTLSamplerMipFilterNotMipmapped : mip == 1 ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
	auto addr = [](int w){ return w == 1 ? MTLSamplerAddressModeClampToEdge : w == 2 ? MTLSamplerAddressModeMirrorRepeat :
	                              w == 3 ? MTLSamplerAddressModeClampToBorderColor : MTLSamplerAddressModeRepeat; };
	d.sAddressMode = addr((k >> 4) & 3);
	d.tAddressMode = addr((k >> 6) & 3);
	d.maxAnisotropy = (k >> 8) & 31;
	id<MTLSamplerState> s = [G->dev newSamplerStateWithDescriptor:d];
	G->samplers[k] = s;
	return s;
}

}  // namespace

namespace {

// ---------------------------------------------------------------------------------------------------------
// render passes

static inline MTLScissorRect scissorRect(int x, int y, int w, int h){ MTLScissorRect r = { (NSUInteger)x, (NSUInteger)y, (NSUInteger)w, (NSUInteger)h }; return r; }

static id<MTLTexture>
attachTexture(Attach &a)
{
	if(a.tex){
		if(!ensureTex(a.tex)) return nil;
		a.tex->lastUse = G->serial;
		return a.tex->mtl;
	}
	if(a.rb) return a.rb->mtl;
	return nil;
}

static void
fboTargets(Fbo *f, __strong id<MTLTexture> *color, __strong id<MTLTexture> *depth)
{
	for(int i = 0; i < 4; i++) color[i] = attachTexture(f->color[i]);
	*depth = attachTexture(f->depth);
}

static bool
fboHasPending(Fbo *f)
{
	return f->clrColor[0] || f->clrColor[1] || f->clrColor[2] || f->clrColor[3] || f->clrDepth || f->clrStencil;
}

// Starts (or keeps) the render pass for a framebuffer.
static bool
beginPass(Fbo *f)
{
	id<MTLTexture> color[4], depth;
	fboTargets(f, color, &depth);
	if(!color[0] && !color[1] && !color[2] && !color[3] && !depth) return false;
	if(G->enc && G->es.fbo == f && G->es.depth == depth && !fboHasPending(f)){
		bool same = true;
		for(int i = 0; i < 4; i++) if(G->es.color[i] != color[i]) same = false;
		if(same) return true;
	}
	endEncoder();
	MTLRenderPassDescriptor *pd = [MTLRenderPassDescriptor renderPassDescriptor];
	for(int i = 0; i < 4; i++){
		if(!color[i]) continue;
		pd.colorAttachments[i].texture = color[i];
		pd.colorAttachments[i].storeAction = MTLStoreActionStore;
		if(f->clrColor[i]){
			pd.colorAttachments[i].loadAction = MTLLoadActionClear;
			pd.colorAttachments[i].clearColor = MTLClearColorMake(f->clrVal[i][0], f->clrVal[i][1], f->clrVal[i][2], f->clrVal[i][3]);
		}else
			pd.colorAttachments[i].loadAction = MTLLoadActionLoad;
		f->clrColor[i] = false;
	}
	if(depth){
		pd.depthAttachment.texture = depth;
		pd.depthAttachment.storeAction = MTLStoreActionStore;
		pd.depthAttachment.loadAction = f->clrDepth ? MTLLoadActionClear : MTLLoadActionLoad;
		pd.depthAttachment.clearDepth = f->clrD;
		if(depth.pixelFormat == MTLPixelFormatDepth32Float_Stencil8){
			pd.stencilAttachment.texture = depth;
			pd.stencilAttachment.storeAction = MTLStoreActionStore;
			pd.stencilAttachment.loadAction = f->clrStencil ? MTLLoadActionClear : MTLLoadActionLoad;
			pd.stencilAttachment.clearStencil = f->clrS;
		}
	}
	f->clrDepth = f->clrStencil = false;
	G->enc = [cmd() renderCommandEncoderWithDescriptor:pd];
	G->es = Ctx::EncState();
	G->es.fbo = f;
	for(int i = 0; i < 4; i++) G->es.color[i] = color[i];
	G->es.depth = depth;
	return true;
}

// Applies pending clears of a framebuffer that is about to stop being the draw target (or is read from).
static void
flushPending(Fbo *f)
{
	if(!f || !fboHasPending(f)) return;
	if(beginPass(f)) endEncoder();
}

// ---------------------------------------------------------------------------------------------------------
// internal pipelines (clear, blit, present)

static const char *utilSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float2 uv; };
struct UParams { float4 srcRect; float4 misc; };
vertex VOut util_vs(uint vid [[vertex_id]], constant UParams &u [[buffer(0)]]) {
	float2 p = float2(float((vid << 1) & 2), float(vid & 2));
	VOut o;
	float yy = u.misc.x > 0.5 ? -(p.y*2.0 - 1.0) : (p.y*2.0 - 1.0);
	o.pos = float4(p.x*2.0 - 1.0, yy, u.misc.y, 1.0);
	o.uv = u.srcRect.xy + p*(u.srcRect.zw - u.srcRect.xy);
	return o;
}
struct COut { float4 c0 [[color(0)]]; float4 c1 [[color(1)]]; float4 c2 [[color(2)]]; float4 c3 [[color(3)]]; };
fragment COut util_clear_fs(VOut in [[stage_in]], constant float4 *c [[buffer(0)]]) {
	COut o; o.c0 = c[0]; o.c1 = c[1]; o.c2 = c[2]; o.c3 = c[3]; return o;
}
fragment float4 util_blit_fs(VOut in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {
	return t.sample(s, in.uv);
}
)MSL";

static id<MTLRenderPipelineState>
utilPso(bool blit, MTLPixelFormat color[4], MTLPixelFormat depth, MTLColorWriteMask masks[4])
{
	uint64_t key = blit ? 1 : 0;
	for(int i = 0; i < 4; i++) key = key*131 + (uint64_t)color[i]*7 + masks[i];
	key = key*131 + (uint64_t)depth;
	auto it = G->utilPsos.find(key);
	if(it != G->utilPsos.end()) return it->second;
	if(!G->utilLib){
		NSError *e = nil;
		G->utilLib = [G->dev newLibraryWithSource:[NSString stringWithUTF8String:utilSource] options:nil error:&e];
		if(!G->utilLib){ logf_("util library: %s", e.localizedDescription.UTF8String); return nil; }
	}
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	d.vertexFunction = [G->utilLib newFunctionWithName:@"util_vs"];
	d.fragmentFunction = [G->utilLib newFunctionWithName:blit ? @"util_blit_fs" : @"util_clear_fs"];
	for(int i = 0; i < 4; i++){
		d.colorAttachments[i].pixelFormat = color[i];
		d.colorAttachments[i].writeMask = masks[i];
	}
	d.depthAttachmentPixelFormat = depth;
	if(depth == MTLPixelFormatDepth32Float_Stencil8) d.stencilAttachmentPixelFormat = depth;
	NSError *e = nil;
	id<MTLRenderPipelineState> p = [G->dev newRenderPipelineStateWithDescriptor:d error:&e];
	if(!p) logf_("util pipeline: %s", e.localizedDescription.UTF8String);
	G->utilPsos[key] = p;
	return p;
}

// ---------------------------------------------------------------------------------------------------------
// programs

static bool
compileProg(Prog *p)
{
	if(p->compiled) return true;
	if(p->compileFailed) return false;
	NSError *e = nil;
	MTLCompileOptions *opt = [MTLCompileOptions new];
	p->vsLib = [G->dev newLibraryWithSource:[NSString stringWithUTF8String:p->tr.vsMsl.c_str()] options:opt error:&e];
	if(!p->vsLib){ logf_("vertex MSL: %s", e.localizedDescription.UTF8String); p->compileFailed = true; return false; }
	p->fsLib = [G->dev newLibraryWithSource:[NSString stringWithUTF8String:p->tr.fsMsl.c_str()] options:opt error:&e];
	if(!p->fsLib){ logf_("fragment MSL: %s", e.localizedDescription.UTF8String); p->compileFailed = true; return false; }
	p->fsFn = [p->fsLib newFunctionWithName:[NSString stringWithUTF8String:p->tr.fsEntry.c_str()]];
	p->compiled = p->fsFn != nil;
	return p->compiled;
}

static GLenum
blendFactor(GLenum f, MTLBlendFactor *out)
{
	switch(f){
	case GL_ZERO: *out = MTLBlendFactorZero; break;
	case GL_ONE: *out = MTLBlendFactorOne; break;
	case GL_SRC_COLOR: *out = MTLBlendFactorSourceColor; break;
	case GL_ONE_MINUS_SRC_COLOR: *out = MTLBlendFactorOneMinusSourceColor; break;
	case GL_DST_COLOR: *out = MTLBlendFactorDestinationColor; break;
	case GL_ONE_MINUS_DST_COLOR: *out = MTLBlendFactorOneMinusDestinationColor; break;
	case GL_SRC_ALPHA: *out = MTLBlendFactorSourceAlpha; break;
	case GL_ONE_MINUS_SRC_ALPHA: *out = MTLBlendFactorOneMinusSourceAlpha; break;
	case GL_DST_ALPHA: *out = MTLBlendFactorDestinationAlpha; break;
	case GL_ONE_MINUS_DST_ALPHA: *out = MTLBlendFactorOneMinusDestinationAlpha; break;
	case GL_SRC_ALPHA_SATURATE: *out = MTLBlendFactorSourceAlphaSaturated; break;
	default: *out = MTLBlendFactorOne; break;
	}
	return f;
}

static MTLVertexFormat
vertexFormat(GLenum type, int size, bool norm)
{
	if(size < 1) size = 1; if(size > 4) size = 4;
	switch(type){
	case GL_FLOAT: { static const MTLVertexFormat f[] = { MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4 }; return f[size - 1]; }
	case GL_UNSIGNED_BYTE:
		if(norm){ static const MTLVertexFormat f[] = { MTLVertexFormatUCharNormalized, MTLVertexFormatUChar2Normalized, MTLVertexFormatUChar3Normalized, MTLVertexFormatUChar4Normalized }; return f[size - 1]; }
		else { static const MTLVertexFormat f[] = { MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4 }; return f[size - 1]; }
	case GL_BYTE:
		if(norm){ static const MTLVertexFormat f[] = { MTLVertexFormatCharNormalized, MTLVertexFormatChar2Normalized, MTLVertexFormatChar3Normalized, MTLVertexFormatChar4Normalized }; return f[size - 1]; }
		else { static const MTLVertexFormat f[] = { MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4 }; return f[size - 1]; }
	case GL_UNSIGNED_SHORT:
		if(norm){ static const MTLVertexFormat f[] = { MTLVertexFormatUShortNormalized, MTLVertexFormatUShort2Normalized, MTLVertexFormatUShort3Normalized, MTLVertexFormatUShort4Normalized }; return f[size - 1]; }
		else { static const MTLVertexFormat f[] = { MTLVertexFormatUShort, MTLVertexFormatUShort2, MTLVertexFormatUShort3, MTLVertexFormatUShort4 }; return f[size - 1]; }
	case GL_SHORT:
		if(norm){ static const MTLVertexFormat f[] = { MTLVertexFormatShortNormalized, MTLVertexFormatShort2Normalized, MTLVertexFormatShort3Normalized, MTLVertexFormatShort4Normalized }; return f[size - 1]; }
		else { static const MTLVertexFormat f[] = { MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4 }; return f[size - 1]; }
	default: return MTLVertexFormatFloat4;
	}
}

static int
typeSize(GLenum t)
{
	switch(t){ case GL_FLOAT: case GL_INT: case GL_UNSIGNED_INT: return 4; case GL_SHORT: case GL_UNSIGNED_SHORT: return 2; default: return 1; }
}

// ---------------------------------------------------------------------------------------------------------
// draw

static id<MTLBuffer> zeroAttribBuf = nil;

static id<MTLDepthStencilState>
depthStencilState(bool hasDepth, bool hasStencil)
{
	DsKey k = {};
	k.depthTest = hasDepth && G->depthTest;
	k.depthFunc = (uint8_t)(G->depthFunc - GL_NEVER);
	k.depthMask = hasDepth && G->depthTest && G->depthMask;
	k.stencilOn = hasStencil && G->stencilTest;
	k.sFunc = (uint8_t)(G->sFunc - GL_NEVER);
	k.sFail = (uint8_t)(G->sFail & 0xff); k.sZFail = (uint8_t)(G->sZFail & 0xff); k.sPass = (uint8_t)(G->sPass & 0xff);
	k.sMask = (uint8_t)G->sMask; k.sWrite = (uint8_t)G->sWrite;
	auto it = G->dss.find(k);
	if(it != G->dss.end()) return it->second;
	auto cmp = [](GLenum f){
		switch(f){ case GL_NEVER: return MTLCompareFunctionNever; case GL_LESS: return MTLCompareFunctionLess; case GL_EQUAL: return MTLCompareFunctionEqual;
		case GL_LEQUAL: return MTLCompareFunctionLessEqual; case GL_GREATER: return MTLCompareFunctionGreater; case GL_NOTEQUAL: return MTLCompareFunctionNotEqual;
		case GL_GEQUAL: return MTLCompareFunctionGreaterEqual; default: return MTLCompareFunctionAlways; } };
	auto sop = [](GLenum o){
		switch(o){ case GL_ZERO: return MTLStencilOperationZero; case GL_REPLACE: return MTLStencilOperationReplace; case GL_INCR: return MTLStencilOperationIncrementClamp;
		case GL_DECR: return MTLStencilOperationDecrementClamp; case GL_INVERT: return MTLStencilOperationInvert; case GL_INCR_WRAP: return MTLStencilOperationIncrementWrap;
		case GL_DECR_WRAP: return MTLStencilOperationDecrementWrap; default: return MTLStencilOperationKeep; } };
	MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
	d.depthCompareFunction = k.depthTest ? cmp(G->depthFunc) : MTLCompareFunctionAlways;
	d.depthWriteEnabled = k.depthMask;
	if(k.stencilOn){
		MTLStencilDescriptor *s = [MTLStencilDescriptor new];
		s.stencilCompareFunction = cmp(G->sFunc);
		s.stencilFailureOperation = sop(G->sFail);
		s.depthFailureOperation = sop(G->sZFail);
		s.depthStencilPassOperation = sop(G->sPass);
		s.readMask = G->sMask; s.writeMask = G->sWrite;
		d.frontFaceStencil = s; d.backFaceStencil = s;
	}
	id<MTLDepthStencilState> st = [G->dev newDepthStencilStateWithDescriptor:d];
	G->dss[k] = st;
	return st;
}

static void
drawCommon(GLenum mode, GLsizei count, GLenum idxType, size_t idxOff, bool indexed, GLint first)
{
	Prog *p = G->prog < G->progs.size() ? G->progs[G->prog] : nullptr;
	if(!p || !p->linked || count <= 0) return;
	if(!compileProg(p)) return;
	Fbo *f = fboOf(G->drawFbo);
	if(!f) return;
	if(!beginPass(f)) return;
	Vao *vao = G->vao < G->vaos.size() ? G->vaos[G->vao] : nullptr;
	if(!vao) return;

	// ---- vertex layout
	struct Slot { Buf *b; int stride; size_t base; };
	Slot slots[8]; int nSlots = 0;
	PsoKey key;
	memset(&key, 0, sizeof(key));
	key.prog = (uint64_t)G->prog;
	bool useAttr[8] = {};
	for(int loc : p->tr.vsAttribs) if(loc >= 0 && loc < 8) useAttr[loc] = true;
	bool needZero = false;
	for(int i = 0; i < 8; i++){
		if(!useAttr[i]) continue;
		Attrib &a = vao->a[i];
		Buf *b = a.enabled ? bufOf(a.buffer) : nullptr;
		if(!b || !b->buf){
			// disabled attribute: OpenGL supplies (0, 0, 0, 1)
			key.a[key.nAttr++] = { (uint8_t)MTLVertexFormatFloat4, 15, 0 };
			needZero = true;
			continue;
		}
		int stride = a.stride ? a.stride : a.size*typeSize(a.type);
		int s = -1;
		for(int k = 0; k < nSlots; k++)
			if(slots[k].b == b && slots[k].stride == stride && a.offset >= slots[k].base && a.offset - slots[k].base < (size_t)stride){ s = k; break; }
		if(s < 0){
			if(nSlots >= 8) return;
			s = nSlots++;
			slots[s] = { b, stride, a.offset };
			// the base must be the smallest offset of all attributes using this buffer layout
		}
		if(a.offset < slots[s].base) slots[s].base = a.offset;
		key.a[key.nAttr++] = { (uint8_t)vertexFormat(a.type, a.size, a.normalized), (uint8_t)(s + 1), (uint16_t)0 };
	}
	// second pass: offsets relative to the slot base
	{
		int n = 0;
		for(int i = 0; i < 8; i++){
			if(!useAttr[i]) continue;
			Attrib &a = vao->a[i];
			Buf *b = a.enabled ? bufOf(a.buffer) : nullptr;
			if(!b || !b->buf){ n++; continue; }
			int slot = key.a[n].slot - 1;
			key.a[n].off = (uint16_t)(a.offset - slots[slot].base);
			key.stride[slot + 1] = (uint16_t)slots[slot].stride;
			n++;
		}
	}
	// attribute index of each entry (needed to build the descriptor)
	int attrIdx[8]; { int n = 0; for(int i = 0; i < 8; i++) if(useAttr[i]) attrIdx[n++] = i; }

	// ---- pipeline state
	__strong id<MTLTexture> *ct = G->es.color;
	for(int i = 0; i < 4; i++){
		if(!ct[i]) continue;
		key.colorFmt[i] = (uint32_t)ct[i].pixelFormat;
		bool active = f->drawBuf[i] == (GLenum)(GL_COLOR_ATTACHMENT0 + i);
		uint8_t mask = 0;
		if(active) mask = (G->colorMask[0] ? 8 : 0) | (G->colorMask[1] ? 4 : 0) | (G->colorMask[2] ? 2 : 0) | (G->colorMask[3] ? 1 : 0);
		key.colorMask[i] = mask;
		key.blendOn[i] = G->blend && !G->blendOff[i];
	}
	key.srcRGB = (uint8_t)(G->bSrcRGB & 0xff); key.dstRGB = (uint8_t)(G->bDstRGB & 0xff);
	key.srcA = (uint8_t)(G->bSrcA & 0xff); key.dstA = (uint8_t)(G->bDstA & 0xff);
	// GL constants of the blend factors differ in the high byte only for a few values: keep it exact
	key.pad[0] = (uint8_t)(((G->bSrcRGB >> 8) & 1) | ((G->bDstRGB >> 8) & 1) << 1 | ((G->bSrcA >> 8) & 1) << 2 | ((G->bDstA >> 8) & 1) << 3);
	id<MTLTexture> dt = G->es.depth;
	key.depthFmt = dt ? (uint32_t)dt.pixelFormat : 0;
	key.stencilFmt = (dt && dt.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) ? (uint32_t)dt.pixelFormat : 0;
	key.flip = 1;

	id<MTLRenderPipelineState> pso;
	auto it = G->psos.find(key);
	if(it != G->psos.end()) pso = it->second;
	else{
		MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
		bool flip = true;
		[fc setConstantValue:&flip type:MTLDataTypeBool atIndex:0];
		NSError *e = nil;
		id<MTLFunction> vfn = [p->vsLib newFunctionWithName:[NSString stringWithUTF8String:p->tr.vsEntry.c_str()] constantValues:fc error:&e];
		if(!vfn){ logf_("vertex function: %s", e.localizedDescription.UTF8String); G->psos[key] = nil; return; }
		MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
		d.vertexFunction = vfn;
		d.fragmentFunction = p->fsFn;
		MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
		for(int i = 0; i < key.nAttr; i++){
			vd.attributes[attrIdx[i]].format = (MTLVertexFormat)key.a[i].fmt;
			vd.attributes[attrIdx[i]].bufferIndex = key.a[i].slot;
			vd.attributes[attrIdx[i]].offset = key.a[i].off;
		}
		for(int s = 1; s <= nSlots; s++){
			vd.layouts[s].stride = key.stride[s];
			vd.layouts[s].stepFunction = MTLVertexStepFunctionPerVertex;
		}
		vd.layouts[15].stride = 0;
		vd.layouts[15].stepFunction = MTLVertexStepFunctionConstant;
		d.vertexDescriptor = vd;
		for(int i = 0; i < 4; i++){
			if(!key.colorFmt[i]) continue;
			MTLRenderPipelineColorAttachmentDescriptor *ca = d.colorAttachments[i];
			ca.pixelFormat = (MTLPixelFormat)key.colorFmt[i];
			ca.writeMask = (MTLColorWriteMask)key.colorMask[i];
			if(key.blendOn[i]){
				MTLBlendFactor sr, dr, sa, da;
				blendFactor(G->bSrcRGB, &sr); blendFactor(G->bDstRGB, &dr); blendFactor(G->bSrcA, &sa); blendFactor(G->bDstA, &da);
				ca.blendingEnabled = YES;
				ca.sourceRGBBlendFactor = sr; ca.destinationRGBBlendFactor = dr;
				ca.sourceAlphaBlendFactor = sa; ca.destinationAlphaBlendFactor = da;
			}
		}
		if(key.depthFmt) d.depthAttachmentPixelFormat = (MTLPixelFormat)key.depthFmt;
		if(key.stencilFmt) d.stencilAttachmentPixelFormat = (MTLPixelFormat)key.stencilFmt;
		pso = [G->dev newRenderPipelineStateWithDescriptor:d error:&e];
		if(!pso) logf_("pipeline state: %s", e.localizedDescription.UTF8String);
		G->psos[key] = pso;
	}
	if(!pso) return;

	id<MTLRenderCommandEncoder> enc = G->enc;
	Ctx::EncState &es = G->es;
	if(es.pso != pso){ [enc setRenderPipelineState:pso]; es.pso = pso; }
	id<MTLDepthStencilState> ds = depthStencilState(dt != nil, key.stencilFmt != 0);
	if(es.ds != ds){ [enc setDepthStencilState:ds]; es.ds = ds; }
	if(G->stencilTest && es.stencilRef != G->sRef){ [enc setStencilReferenceValue:G->sRef]; es.stencilRef = G->sRef; }
	int cullMode = G->cull ? (G->cullFace == GL_FRONT ? 1 : G->cullFace == GL_BACK ? 2 : 3) : 0;
	if(es.cullMode != cullMode){
		[enc setCullMode:cullMode == 0 ? MTLCullModeNone : cullMode == 1 ? MTLCullModeFront : MTLCullModeBack];
		es.cullMode = cullMode;
	}
	if(es.winding != 1){ [enc setFrontFacingWinding:MTLWindingClockwise]; es.winding = 1; }

	int tw = 0, th = 0;
	for(int i = 0; i < 4; i++) if(ct[i]){ tw = (int)ct[i].width; th = (int)ct[i].height; break; }
	if(!tw && dt){ tw = (int)dt.width; th = (int)dt.height; }
	if(!es.vpSet || memcmp(es.vp, G->vp, sizeof(G->vp)) != 0){
		MTLViewport v = { (double)G->vp[0], (double)G->vp[1], (double)G->vp[2], (double)G->vp[3], 0.0, 1.0 };
		[enc setViewport:v];
		memcpy(es.vp, G->vp, sizeof(G->vp)); es.vpSet = true;
	}
	{
		int s[4] = { 0, 0, tw, th };
		if(G->scissor){
			int x0 = G->sc[0], y0 = G->sc[1], x1 = G->sc[0] + G->sc[2], y1 = G->sc[1] + G->sc[3];
			if(x0 < 0) x0 = 0; if(y0 < 0) y0 = 0; if(x1 > tw) x1 = tw; if(y1 > th) y1 = th;
			if(x1 < x0) x1 = x0; if(y1 < y0) y1 = y0;
			s[0] = x0; s[1] = y0; s[2] = x1 - x0; s[3] = y1 - y0;
		}
		if(!es.scSet || memcmp(es.sc, s, sizeof(s)) != 0){
			[enc setScissorRect:scissorRect(s[0], s[1], s[2], s[3])];
			memcpy(es.sc, s, sizeof(s)); es.scSet = true;
		}
	}

	// ---- vertex buffers
	for(int s = 0; s < nSlots; s++){
		Buf *b = slots[s].b;
		size_t off = b->off + slots[s].base;
		if(es.vbuf[s + 1] != b->buf || es.voff[s + 1] != off){
			[enc setVertexBuffer:b->buf offset:off atIndex:s + 1];
			es.vbuf[s + 1] = b->buf; es.voff[s + 1] = off;
		}
	}
	if(needZero){
		if(!zeroAttribBuf){
			float z[4] = { 0, 0, 0, 1 };
			zeroAttribBuf = [G->dev newBufferWithBytes:z length:sizeof(z) options:MTLResourceStorageModeShared];
		}
		[enc setVertexBuffer:zeroAttribBuf offset:0 atIndex:15];
		es.vbuf[15] = nil;
	}

	// ---- uniforms
	if(!p->vsBlock.empty() && (es.progId[0] != G->prog || es.progVer[0] != p->version)){
		if(p->vsBlock.size() <= 4096)
			[enc setVertexBytes:p->vsBlock.data() length:p->vsBlock.size() atIndex:rw::mtl::TRANS_VS_UNIFORM_BUFFER];
		else{
			id<MTLBuffer> rb; size_t ro;
			if(ringAlloc(p->vsBlock.size(), &rb, &ro)){
				memcpy((uint8_t*)rb.contents + ro, p->vsBlock.data(), p->vsBlock.size());
				[enc setVertexBuffer:rb offset:ro atIndex:rw::mtl::TRANS_VS_UNIFORM_BUFFER];
			}
		}
		es.progId[0] = G->prog; es.progVer[0] = p->version;
	}
	if(!p->fsBlock.empty() && (es.progId[1] != G->prog || es.progVer[1] != p->version)){
		[enc setFragmentBytes:p->fsBlock.data() length:p->fsBlock.size() atIndex:rw::mtl::TRANS_FS_UNIFORM_BUFFER];
		es.progId[1] = G->prog; es.progVer[1] = p->version;
	}

	// ---- textures
	for(size_t i = 0; i < p->tr.samplers.size(); i++){
		auto &sm = p->tr.samplers[i];
		int unit = p->samplerUnit[i];
		if(unit < 0 || unit > 15) unit = 0;
		Tex *t = texOf(G->unitTex2D[unit]);
		if(!t) t = texOf(G->unitTexRect[unit]);
		id<MTLTexture> mt = nil; id<MTLSamplerState> ss = nil;
		if(t && ensureTex(t)){ mt = t->mtl; t->lastUse = G->serial; ss = samplerFor(t); }
		if(!mt){
			if(!G->blackTex){
				MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:1 height:1 mipmapped:NO];
				G->blackTex = [G->dev newTextureWithDescriptor:d];
				uint8_t px[4] = { 0, 0, 0, 255 };
				[G->blackTex replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:px bytesPerRow:4];
			}
			mt = G->blackTex;
			Tex dummy; ss = samplerFor(&dummy);
		}
		if(sm.stage == 1){
			if(es.ftex[sm.slot] != mt){ [enc setFragmentTexture:mt atIndex:sm.slot]; es.ftex[sm.slot] = mt; }
			if(es.fsamp[sm.slot] != ss){ [enc setFragmentSamplerState:ss atIndex:sm.slot]; es.fsamp[sm.slot] = ss; }
		}else{
			[enc setVertexTexture:mt atIndex:sm.slot];
			[enc setVertexSamplerState:ss atIndex:sm.slot];
		}
	}

	// ---- draw
	MTLPrimitiveType prim;
	switch(mode){
	case GL_POINTS: prim = MTLPrimitiveTypePoint; break;
	case GL_LINES: prim = MTLPrimitiveTypeLine; break;
	case GL_LINE_STRIP: prim = MTLPrimitiveTypeLineStrip; break;
	case GL_TRIANGLE_STRIP: prim = MTLPrimitiveTypeTriangleStrip; break;
	case GL_TRIANGLES: prim = MTLPrimitiveTypeTriangle; break;
	case GL_TRIANGLE_FAN: prim = MTLPrimitiveTypeTriangle; break;
	default: return;
	}
	es.draws = true;
	if(mode == GL_TRIANGLE_FAN && count >= 3){
		// no fans in Metal: expand to a triangle list
		int tris = count - 2;
		id<MTLBuffer> ib; size_t io;
		if(!ringAlloc((size_t)tris*3*4, &ib, &io)) return;
		uint32_t *out = (uint32_t*)((uint8_t*)ib.contents + io);
		if(indexed){
			Buf *eb = bufOf(vao->ebo);
			if(!eb || !eb->buf) return;
			const uint8_t *src = (const uint8_t*)eb->buf.contents + eb->off + idxOff;
			auto idx = [&](int k) -> uint32_t { return idxType == GL_UNSIGNED_INT ? ((const uint32_t*)src)[k] : idxType == GL_UNSIGNED_SHORT ? ((const uint16_t*)src)[k] : src[k]; };
			for(int k = 0; k < tris; k++){ out[k*3] = idx(0); out[k*3 + 1] = idx(k + 1); out[k*3 + 2] = idx(k + 2); }
		}else
			for(int k = 0; k < tris; k++){ out[k*3] = first; out[k*3 + 1] = first + k + 1; out[k*3 + 2] = first + k + 2; }
		[enc drawIndexedPrimitives:prim indexCount:tris*3 indexType:MTLIndexTypeUInt32 indexBuffer:ib indexBufferOffset:io];
		return;
	}
	if(indexed){
		Buf *eb = bufOf(vao->ebo);
		if(!eb || !eb->buf) return;
		MTLIndexType it2 = idxType == GL_UNSIGNED_INT ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
		[enc drawIndexedPrimitives:prim indexCount:count indexType:it2 indexBuffer:eb->buf indexBufferOffset:eb->off + idxOff];
	}else
		[enc drawPrimitives:prim vertexStart:first vertexCount:count];
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// the GL entry points

namespace {
#define MGL static

template<class T> static GLuint addObj(std::vector<T*> &v, T *o){ v.push_back(o); return (GLuint)(v.size() - 1); }

MGL const GLubyte *mgl_glGetString(GLenum name)
{
	static std::string renderer;
	switch(name){
	case GL_VERSION: return (const GLubyte*)"3.3 Metal";
	case GL_VENDOR: return (const GLubyte*)"Apple";
	case GL_RENDERER: renderer = std::string("Metal: ") + [G->dev.name UTF8String]; return (const GLubyte*)renderer.c_str();
	case GL_SHADING_LANGUAGE_VERSION: return (const GLubyte*)"3.30";
	case GL_EXTENSIONS: return (const GLubyte*)"GL_EXT_texture_compression_s3tc GL_EXT_texture_filter_anisotropic GL_ARB_texture_rectangle";
	}
	return (const GLubyte*)"";
}

static const char *kExtensions[] = { "GL_EXT_texture_compression_s3tc", "GL_EXT_texture_filter_anisotropic", "GL_ARB_texture_rectangle", "GL_ARB_framebuffer_object" };

MGL const GLubyte *mgl_glGetStringi(GLenum name, GLuint i)
{
	if(name == GL_EXTENSIONS && i < sizeof(kExtensions)/sizeof(kExtensions[0])) return (const GLubyte*)kExtensions[i];
	return (const GLubyte*)"";
}

MGL GLenum mgl_glGetError(void){ return 0; }

MGL void mgl_glGetIntegerv(GLenum pname, GLint *v)
{
	switch(pname){
	case GL_TEXTURE_BINDING_2D: *v = G->unitTex2D[G->activeUnit]; break;
	case GL_TEXTURE_BINDING_RECTANGLE: *v = G->unitTexRect[G->activeUnit]; break;
	case GL_FRAMEBUFFER_BINDING: *v = G->drawFbo; break;
	case GL_READ_FRAMEBUFFER_BINDING: *v = G->readFbo; break;
	case GL_ACTIVE_TEXTURE: *v = GL_TEXTURE0 + G->activeUnit; break;
	case GL_CURRENT_PROGRAM: *v = G->prog; break;
	case GL_VERTEX_ARRAY_BINDING: *v = G->vao; break;
	case GL_ARRAY_BUFFER_BINDING: *v = G->arrayBuf; break;
	case GL_MAX_SAMPLES: *v = 1; break;
	case GL_NUM_EXTENSIONS: *v = (GLint)(sizeof(kExtensions)/sizeof(kExtensions[0])); break;
	case GL_MAJOR_VERSION: *v = 3; break;
	case GL_MINOR_VERSION: *v = 3; break;
	case GL_BLEND_SRC_RGB: *v = G->bSrcRGB; break;
	case GL_BLEND_DST_RGB: *v = G->bDstRGB; break;
	case GL_BLEND_SRC_ALPHA: *v = G->bSrcA; break;
	case GL_BLEND_DST_ALPHA: *v = G->bDstA; break;
	case GL_MAX_TEXTURE_SIZE: *v = 16384; break;
	case GL_MAX_TEXTURE_IMAGE_UNITS: *v = 16; break;
	case GL_VIEWPORT: memcpy(v, G->vp, sizeof(G->vp)); break;
	default: *v = 0; break;
	}
}

MGL void mgl_glGetFloatv(GLenum pname, GLfloat *v)
{
	if(pname == GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT) *v = 16.0f;
	else *v = 0.0f;
}

MGL void mgl_glGetBooleanv(GLenum pname, GLboolean *v)
{
	if(pname == GL_COLOR_WRITEMASK) for(int i = 0; i < 4; i++) v[i] = G->colorMask[i];
	else *v = 0;
}

static bool *enableFlag(GLenum cap)
{
	switch(cap){
	case GL_BLEND: return &G->blend;
	case GL_DEPTH_TEST: return &G->depthTest;
	case GL_CULL_FACE: return &G->cull;
	case GL_SCISSOR_TEST: return &G->scissor;
	case GL_STENCIL_TEST: return &G->stencilTest;
	}
	return nullptr;
}
MGL void mgl_glEnable(GLenum cap){ if(bool *f = enableFlag(cap)) *f = true; }
MGL void mgl_glDisable(GLenum cap){ if(bool *f = enableFlag(cap)) *f = false; }
MGL GLboolean mgl_glIsEnabled(GLenum cap){ bool *f = enableFlag(cap); return f && *f; }
MGL void mgl_glEnablei(GLenum cap, GLuint i){ if(cap == GL_BLEND && i < 4) G->blendOff[i] = false; }
MGL void mgl_glDisablei(GLenum cap, GLuint i){ if(cap == GL_BLEND && i < 4) G->blendOff[i] = true; }

MGL void mgl_glBlendFunc(GLenum s, GLenum d){ G->bSrcRGB = G->bSrcA = s; G->bDstRGB = G->bDstA = d; }
MGL void mgl_glBlendFuncSeparate(GLenum a, GLenum b, GLenum c, GLenum d){ G->bSrcRGB = a; G->bDstRGB = b; G->bSrcA = c; G->bDstA = d; }
MGL void mgl_glDepthFunc(GLenum f){ G->depthFunc = f; }
MGL void mgl_glDepthMask(GLboolean m){ G->depthMask = m != 0; }
MGL void mgl_glColorMask(GLboolean r, GLboolean g, GLboolean b, GLboolean a){ G->colorMask[0] = r; G->colorMask[1] = g; G->colorMask[2] = b; G->colorMask[3] = a; }
MGL void mgl_glCullFace(GLenum m){ G->cullFace = m; }
MGL void mgl_glFrontFace(GLenum){ }
MGL void mgl_glStencilFunc(GLenum f, GLint ref, GLuint mask){ G->sFunc = f; G->sRef = ref; G->sMask = mask; }
MGL void mgl_glStencilOp(GLenum a, GLenum b, GLenum c){ G->sFail = a; G->sZFail = b; G->sPass = c; }
MGL void mgl_glStencilMask(GLuint m){ G->sWrite = m; }
MGL void mgl_glScissor(GLint x, GLint y, GLsizei w, GLsizei h){ G->sc[0] = x; G->sc[1] = y; G->sc[2] = w; G->sc[3] = h; }
MGL void mgl_glViewport(GLint x, GLint y, GLsizei w, GLsizei h){ G->vp[0] = x; G->vp[1] = y; G->vp[2] = w; G->vp[3] = h; }
MGL void mgl_glClearColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a){ G->clearColor[0] = r; G->clearColor[1] = g; G->clearColor[2] = b; G->clearColor[3] = a; }
MGL void mgl_glPixelStorei(GLenum p, GLint v){ if(p == GL_UNPACK_ALIGNMENT) G->unpackAlign = v; else if(p == GL_PACK_ALIGNMENT) G->packAlign = v; }
MGL void mgl_glFinish(void){ submit(true); }
MGL void mgl_glFlush(void){ }

// ---- buffers
MGL void mgl_glGenBuffers(GLsizei n, GLuint *ids){ for(int i = 0; i < n; i++) ids[i] = addObj(G->bufs, new Buf()); }
MGL void mgl_glDeleteBuffers(GLsizei n, const GLuint *ids){
	for(int i = 0; i < n; i++) if(ids[i] && ids[i] < G->bufs.size() && G->bufs[ids[i]]){ delete G->bufs[ids[i]]; G->bufs[ids[i]] = nullptr; }
}
MGL void mgl_glBindBuffer(GLenum target, GLuint id)
{
	if(target == GL_ARRAY_BUFFER) G->arrayBuf = id;
	else if(target == GL_ELEMENT_ARRAY_BUFFER){ if(Vao *v = G->vao ? G->vaos[G->vao] : nullptr) v->ebo = id; }
	else if(target == GL_UNIFORM_BUFFER) G->uniformBuf = id;
}
static GLuint *targetBinding(GLenum target, Vao **vao)
{
	if(target == GL_ARRAY_BUFFER) return &G->arrayBuf;
	if(target == GL_ELEMENT_ARRAY_BUFFER){ *vao = G->vao ? G->vaos[G->vao] : nullptr; return *vao ? &(*vao)->ebo : nullptr; }
	if(target == GL_UNIFORM_BUFFER) return &G->uniformBuf;
	return nullptr;
}
static Buf *boundBuf(GLenum target)
{
	Vao *v = nullptr;
	GLuint *b = targetBinding(target, &v);
	return b ? bufOf(*b) : nullptr;
}
MGL void mgl_glBufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage)
{
	Buf *b = boundBuf(target);
	if(!b) return;
	b->size = (size_t)size;
	b->usage = usage;
	b->valid = data ? (size_t)size : 0;
	bool stream = usage == GL_STREAM_DRAW || usage == GL_DYNAMIC_DRAW || usage == GL_STREAM_READ || usage == GL_STREAM_COPY;
	if(stream){
		// storage is made when data arrives (glBufferSubData), from the ring
		b->buf = nil; b->off = 0; b->ring = true;
		if(data && size > 0){
			id<MTLBuffer> rb; size_t ro;
			if(ringAlloc((size_t)size, &rb, &ro)){ memcpy((uint8_t*)rb.contents + ro, data, (size_t)size); b->buf = rb; b->off = ro; }
		}
	}else{
		b->ring = false; b->off = 0;
		b->buf = size > 0 ? (data ? [G->dev newBufferWithBytes:data length:(NSUInteger)size options:MTLResourceStorageModeShared]
		                          : [G->dev newBufferWithLength:(NSUInteger)size options:MTLResourceStorageModeShared]) : nil;
	}
}
MGL void mgl_glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data)
{
	Buf *b = boundBuf(target);
	if(!b || size <= 0) return;
	size_t end = (size_t)offset + (size_t)size;
	if(b->ring || b->usage == GL_STREAM_DRAW || b->usage == GL_DYNAMIC_DRAW){
		// new ring block: keep what was valid before, apply the update (earlier draws still read the old block)
		size_t newValid = b->valid > end ? b->valid : end;
		id<MTLBuffer> rb; size_t ro;
		if(!ringAlloc(newValid, &rb, &ro)) return;
		uint8_t *dst = (uint8_t*)rb.contents + ro;
		if(b->buf && b->valid && ((size_t)offset > 0 || end < b->valid))
			memcpy(dst, (uint8_t*)b->buf.contents + b->off, b->valid);
		memcpy(dst + offset, data, (size_t)size);
		b->buf = rb; b->off = ro; b->valid = newValid; b->ring = true;
	}else if(b->buf){
		// static storage: if the GPU may still read it, move to a fresh copy
		endEncoder();
		id<MTLBuffer> nb = [G->dev newBufferWithBytes:(uint8_t*)b->buf.contents + b->off length:b->buf.length - b->off options:MTLResourceStorageModeShared];
		memcpy((uint8_t*)nb.contents + offset, data, (size_t)size);
		b->buf = nb; b->off = 0;
	}
}
MGL void mgl_glBindBufferBase(GLenum, GLuint, GLuint){ }

// ---- vertex arrays
MGL void mgl_glGenVertexArrays(GLsizei n, GLuint *ids){ for(int i = 0; i < n; i++) ids[i] = addObj(G->vaos, new Vao()); }
MGL void mgl_glDeleteVertexArrays(GLsizei n, const GLuint *ids){
	for(int i = 0; i < n; i++) if(ids[i] && ids[i] < G->vaos.size() && G->vaos[ids[i]]){ delete G->vaos[ids[i]]; G->vaos[ids[i]] = nullptr; }
}
MGL void mgl_glBindVertexArray(GLuint id){ G->vao = id; }
MGL void mgl_glEnableVertexAttribArray(GLuint i){ if(Vao *v = G->vao ? G->vaos[G->vao] : nullptr) if(i < 8) v->a[i].enabled = true; }
MGL void mgl_glDisableVertexAttribArray(GLuint i){ if(Vao *v = G->vao ? G->vaos[G->vao] : nullptr) if(i < 8) v->a[i].enabled = false; }
MGL void mgl_glVertexAttribPointer(GLuint i, GLint size, GLenum type, GLboolean norm, GLsizei stride, const void *ptr)
{
	Vao *v = G->vao ? G->vaos[G->vao] : nullptr;
	if(!v || i >= 8) return;
	Attrib &a = v->a[i];
	a.buffer = G->arrayBuf; a.size = size; a.type = type; a.normalized = norm != 0; a.stride = stride; a.offset = (size_t)ptr;
}

// ---- textures
MGL void mgl_glGenTextures(GLsizei n, GLuint *ids){ for(int i = 0; i < n; i++) ids[i] = addObj(G->texs, new Tex()); }
MGL void mgl_glDeleteTextures(GLsizei n, const GLuint *ids){
	for(int i = 0; i < n; i++) if(ids[i] && ids[i] < G->texs.size() && G->texs[ids[i]]){
		for(int u = 0; u < 16; u++){ if(G->unitTex2D[u] == ids[i]) G->unitTex2D[u] = 0; if(G->unitTexRect[u] == ids[i]) G->unitTexRect[u] = 0; }
		delete G->texs[ids[i]]; G->texs[ids[i]] = nullptr;
	}
}
MGL void mgl_glActiveTexture(GLenum unit){ int u = (int)unit - GL_TEXTURE0; if(u >= 0 && u < 16) G->activeUnit = u; }
MGL void mgl_glBindTexture(GLenum target, GLuint id)
{
	if(Tex *t = texOf(id)) t->target = target;
	if(target == GL_TEXTURE_2D) G->unitTex2D[G->activeUnit] = id;
	else if(target == GL_TEXTURE_RECTANGLE) G->unitTexRect[G->activeUnit] = id;
}
static Tex *boundTex(GLenum target)
{
	return texOf(target == GL_TEXTURE_RECTANGLE ? G->unitTexRect[G->activeUnit] : G->unitTex2D[G->activeUnit]);
}
MGL void mgl_glTexParameteri(GLenum target, GLenum pname, GLint v)
{
	Tex *t = boundTex(target);
	if(!t) return;
	switch(pname){
	case GL_TEXTURE_MIN_FILTER: t->minF = v; break;
	case GL_TEXTURE_MAG_FILTER: t->magF = v; break;
	case GL_TEXTURE_WRAP_S: t->wrapS = v; break;
	case GL_TEXTURE_WRAP_T: t->wrapT = v; break;
	case GL_TEXTURE_MAX_LEVEL: t->maxLevel = v; break;
	}
}
MGL void mgl_glTexParameterf(GLenum target, GLenum pname, GLfloat v)
{
	if(pname == GL_TEXTURE_MAX_ANISOTROPY_EXT){ if(Tex *t = boundTex(target)) t->aniso = v; return; }
	mgl_glTexParameteri(target, pname, (GLint)v);
}
// (re)specifies a level. data may be null (storage only).
static void texImage(Tex *t, int level, GLenum ifmt, int w, int h, GLenum format, GLenum type, const void *data, size_t compressedSize)
{
	if(level == 0){
		bool d, c;
		MTLPixelFormat f = mtlFormatFor(ifmt, &d, &c);
		if(t->mtl && (t->w != w || t->h != h || t->fmt != f)){ t->mtl = nil; t->created = 0; }
		t->w = w; t->h = h; t->fmt = f; t->depth = d; t->compressed = c; t->ifmt = ifmt;
		if(!t->mtl) t->lv.clear();
	}
	if((int)t->lv.size() <= level) t->lv.resize(level + 1);
	Level &l = t->lv[level];
	l.w = w; l.h = h;
	if(!data){ l.data.clear(); return; }
	std::vector<uint8_t> px;
	if(compressedSize) px.assign((const uint8_t*)data, (const uint8_t*)data + compressedSize);
	else convertPixels(t->fmt, format, type, w, h, data, px);
	if(t->mtl && level < t->created){
		size_t rb;
		levelBytes(t->fmt, w, h, &rb);
		uploadLevel(t, level, px.data(), rb, w, h);
	}else
		l.data = std::move(px);
}
MGL void mgl_glTexImage2D(GLenum target, GLint level, GLint ifmt, GLsizei w, GLsizei h, GLint border, GLenum format, GLenum type, const void *data)
{
	Tex *t = boundTex(target);
	if(t) texImage(t, level, (GLenum)ifmt, w, h, format, type, data, 0);
}
MGL void mgl_glCompressedTexImage2D(GLenum target, GLint level, GLenum ifmt, GLsizei w, GLsizei h, GLint border, GLsizei size, const void *data)
{
	Tex *t = boundTex(target);
	if(t) texImage(t, level, ifmt, w, h, 0, 0, data, (size_t)size);
}
MGL void mgl_glTexSubImage2D(GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, const void *data)
{
	Tex *t = boundTex(target);
	if(!t || !data || !ensureTex(t)) return;
	std::vector<uint8_t> px;
	convertPixels(t->fmt, format, type, w, h, data, px);
	size_t rb; levelBytes(t->fmt, w, h, &rb);
	if(t->lastUse != 0 && t->lastUse > G->completed.load()){
		id<MTLBuffer> sb; size_t so;
		if(ringAlloc(px.size(), &sb, &so)){
			memcpy((uint8_t*)sb.contents + so, px.data(), px.size());
			endEncoder();
			id<MTLBlitCommandEncoder> bl = [cmd() blitCommandEncoder];
			[bl copyFromBuffer:sb sourceOffset:so sourceBytesPerRow:rb sourceBytesPerImage:px.size() sourceSize:MTLSizeMake(w, h, 1)
			         toTexture:t->mtl destinationSlice:0 destinationLevel:level destinationOrigin:MTLOriginMake(x, y, 0)];
			[bl endEncoding];
			t->lastUse = G->serial;
			return;
		}
	}
	[t->mtl replaceRegion:MTLRegionMake2D(x, y, w, h) mipmapLevel:level withBytes:px.data() bytesPerRow:rb];
}
MGL void mgl_glGenerateMipmap(GLenum target)
{
	Tex *t = boundTex(target);
	if(!t) return;
	if(!t->mtl){ t->genMips = true; return; }
	if(t->created > 1 && !t->compressed){
		endEncoder();
		id<MTLBlitCommandEncoder> bl = [cmd() blitCommandEncoder];
		[bl generateMipmapsForTexture:t->mtl];
		[bl endEncoding];
		t->lastUse = G->serial;
	}
}
MGL void mgl_glGetTexLevelParameteriv(GLenum target, GLint level, GLenum pname, GLint *v)
{
	Tex *t = boundTex(target);
	*v = 0;
	if(!t) return;
	int w = t->w >> level, h = t->h >> level;
	if(w < 1) w = 1; if(h < 1) h = 1;
	if(pname == GL_TEXTURE_WIDTH) *v = w;
	else if(pname == GL_TEXTURE_HEIGHT) *v = h;
}

}  // namespace

namespace {

// ---- renderbuffers and framebuffers
MGL void mgl_glGenRenderbuffers(GLsizei n, GLuint *ids){ for(int i = 0; i < n; i++) ids[i] = addObj(G->rbs, new Rb()); }
MGL void mgl_glDeleteRenderbuffers(GLsizei n, const GLuint *ids){
	for(int i = 0; i < n; i++) if(ids[i] && ids[i] < G->rbs.size() && G->rbs[ids[i]]){ delete G->rbs[ids[i]]; G->rbs[ids[i]] = nullptr; }
}
MGL void mgl_glBindRenderbuffer(GLenum, GLuint id){ G->rbo = id; }
MGL void mgl_glRenderbufferStorage(GLenum, GLenum fmt, GLsizei w, GLsizei h)
{
	Rb *r = G->rbo < G->rbs.size() ? G->rbs[G->rbo] : nullptr;
	if(!r) return;
	bool d, c;
	MTLPixelFormat f = mtlFormatFor(fmt, &d, &c);
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	td.storageMode = MTLStorageModePrivate;
	r->mtl = [G->dev newTextureWithDescriptor:td];
	r->w = w; r->h = h; r->fmt = fmt;
}
MGL void mgl_glGenFramebuffers(GLsizei n, GLuint *ids){ for(int i = 0; i < n; i++) ids[i] = addObj(G->fbos, new Fbo()); }
MGL void mgl_glDeleteFramebuffers(GLsizei n, const GLuint *ids){
	for(int i = 0; i < n; i++) if(ids[i] && ids[i] < G->fbos.size() && G->fbos[ids[i]]){
		if(G->drawFbo == ids[i]) G->drawFbo = 0;
		if(G->readFbo == ids[i]) G->readFbo = 0;
		if(G->es.fbo == G->fbos[ids[i]]) endEncoder();
		delete G->fbos[ids[i]]; G->fbos[ids[i]] = nullptr;
	}
}
MGL void mgl_glBindFramebuffer(GLenum target, GLuint id)
{
	if(target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER){
		if(id != G->drawFbo) flushPending(fboOf(G->drawFbo));
		G->drawFbo = id;
	}
	if(target == GL_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER) G->readFbo = id;
}
MGL void mgl_glFramebufferTexture2D(GLenum target, GLenum att, GLenum textarget, GLuint tex, GLint level)
{
	GLuint id = target == GL_READ_FRAMEBUFFER ? G->readFbo : G->drawFbo;
	Fbo *f = id ? fboOf(id) : nullptr;
	if(!f) return;
	Attach a;
	a.tex = texOf(tex); a.level = level;
	if(att >= GL_COLOR_ATTACHMENT0 && att < GL_COLOR_ATTACHMENT0 + 4) f->color[att - GL_COLOR_ATTACHMENT0] = a;
	else if(att == GL_DEPTH_ATTACHMENT || att == GL_DEPTH_STENCIL_ATTACHMENT || att == GL_STENCIL_ATTACHMENT) f->depth = a;
}
MGL void mgl_glFramebufferRenderbuffer(GLenum target, GLenum att, GLenum, GLuint rb)
{
	GLuint id = target == GL_READ_FRAMEBUFFER ? G->readFbo : G->drawFbo;
	Fbo *f = id ? fboOf(id) : nullptr;
	if(!f) return;
	Attach a;
	a.rb = rb < G->rbs.size() ? G->rbs[rb] : nullptr;
	if(att >= GL_COLOR_ATTACHMENT0 && att < GL_COLOR_ATTACHMENT0 + 4) f->color[att - GL_COLOR_ATTACHMENT0] = a;
	else f->depth = a;
}
MGL GLenum mgl_glCheckFramebufferStatus(GLenum target)
{
	Fbo *f = fboOf(target == GL_READ_FRAMEBUFFER ? G->readFbo : G->drawFbo);
	if(!f) return GL_FRAMEBUFFER_UNDEFINED;
	bool any = f->depth.tex || f->depth.rb;
	for(int i = 0; i < 4; i++) if(f->color[i].tex || f->color[i].rb) any = true;
	return any || f->isDefault ? GL_FRAMEBUFFER_COMPLETE : GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
}
MGL void mgl_glDrawBuffers(GLsizei n, const GLenum *bufs)
{
	Fbo *f = fboOf(G->drawFbo);
	if(!f || f->isDefault) return;
	for(int i = 0; i < 4; i++) f->drawBuf[i] = i < n ? bufs[i] : GL_NONE;
}
MGL void mgl_glReadBuffer(GLenum b){ G->readBuffer = b; }

// ---- clear
static void
clearByDraw(Fbo *f, GLbitfield mask, const float *colorVals /* [4][4] or null */, const bool *colorSel)
{
	if(!beginPass(f)) return;
	MTLPixelFormat cf[4]; MTLColorWriteMask masks[4];
	int tw = 0, th = 0;
	for(int i = 0; i < 4; i++){
		cf[i] = G->es.color[i] ? G->es.color[i].pixelFormat : MTLPixelFormatInvalid;
		masks[i] = MTLColorWriteMaskNone;
		if(G->es.color[i]){
			if(!tw){ tw = (int)G->es.color[i].width; th = (int)G->es.color[i].height; }
			if(colorSel && colorSel[i]){
				masks[i] = (G->colorMask[0] ? MTLColorWriteMaskRed : 0) | (G->colorMask[1] ? MTLColorWriteMaskGreen : 0) |
				           (G->colorMask[2] ? MTLColorWriteMaskBlue : 0) | (G->colorMask[3] ? MTLColorWriteMaskAlpha : 0);
			}
		}
	}
	MTLPixelFormat df = G->es.depth ? G->es.depth.pixelFormat : MTLPixelFormatInvalid;
	if(!tw && G->es.depth){ tw = (int)G->es.depth.width; th = (int)G->es.depth.height; }
	id<MTLRenderPipelineState> pso = utilPso(false, cf, df, masks);
	if(!pso) return;
	id<MTLRenderCommandEncoder> enc = G->enc;
	[enc setRenderPipelineState:pso];
	MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
	dd.depthCompareFunction = MTLCompareFunctionAlways;
	dd.depthWriteEnabled = (mask & GL_DEPTH_BUFFER_BIT) && df != MTLPixelFormatInvalid;
	static id<MTLDepthStencilState> dsClear[2];
	int di = dd.depthWriteEnabled ? 1 : 0;
	if(!dsClear[di]) dsClear[di] = [G->dev newDepthStencilStateWithDescriptor:dd];
	[enc setDepthStencilState:dsClear[di]];
	[enc setCullMode:MTLCullModeNone];
	[enc setViewport:(MTLViewport){ 0, 0, (double)tw, (double)th, 0, 1 }];
	int s[4] = { 0, 0, tw, th };
	if(G->scissor){
		int x0 = G->sc[0], y0 = G->sc[1], x1 = G->sc[0] + G->sc[2], y1 = G->sc[1] + G->sc[3];
		if(x0 < 0) x0 = 0; if(y0 < 0) y0 = 0; if(x1 > tw) x1 = tw; if(y1 > th) y1 = th;
		if(x1 < x0) x1 = x0; if(y1 < y0) y1 = y0;
		s[0] = x0; s[1] = y0; s[2] = x1 - x0; s[3] = y1 - y0;
	}
	[enc setScissorRect:scissorRect(s[0], s[1], s[2], s[3])];
	float params[8] = { 0, 0, 1, 1, 0, 1.0f /* depth value set below */, 0, 0 };
	params[4] = 1.0f;	// flip
	params[5] = 1.0f;	// depth
	[enc setVertexBytes:params length:sizeof(params) atIndex:0];
	float cols[16] = {};
	if(colorVals) memcpy(cols, colorVals, sizeof(cols));
	[enc setFragmentBytes:cols length:sizeof(cols) atIndex:0];
	[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	// everything the encoder state cache knew is now stale
	G->es.pso = nil; G->es.ds = nil; G->es.cullMode = -1; G->es.vpSet = false; G->es.scSet = false;
	G->es.progId[0] = G->es.progId[1] = 0;
	for(int i = 0; i < 16; i++){ G->es.vbuf[i] = nil; G->es.ftex[i] = nil; G->es.fsamp[i] = nil; }
	G->es.draws = true;
}

MGL void mgl_glClear(GLbitfield mask)
{
	Fbo *f = fboOf(G->drawFbo);
	if(!f) return;
	bool allMask = G->colorMask[0] && G->colorMask[1] && G->colorMask[2] && G->colorMask[3];
	if(!G->scissor && allMask && (!(mask & GL_DEPTH_BUFFER_BIT) || G->depthMask)){
		if(mask & GL_COLOR_BUFFER_BIT)
			for(int i = 0; i < 4; i++) if(f->drawBuf[i] == (GLenum)(GL_COLOR_ATTACHMENT0 + i)){
				f->clrColor[i] = true; memcpy(f->clrVal[i], G->clearColor, sizeof(G->clearColor));
			}
		if(mask & GL_DEPTH_BUFFER_BIT){ f->clrDepth = true; f->clrD = 1.0f; }
		if(mask & GL_STENCIL_BUFFER_BIT){ f->clrStencil = true; f->clrS = 0; }
		if(G->es.fbo == f) endEncoder();
		return;
	}
	bool sel[4]; float vals[16];
	for(int i = 0; i < 4; i++){ sel[i] = (mask & GL_COLOR_BUFFER_BIT) && f->drawBuf[i] == (GLenum)(GL_COLOR_ATTACHMENT0 + i); memcpy(vals + i*4, G->clearColor, 16); }
	clearByDraw(f, mask, vals, sel);
}
MGL void mgl_glClearBufferfv(GLenum buffer, GLint drawbuffer, const GLfloat *v)
{
	Fbo *f = fboOf(G->drawFbo);
	if(!f || drawbuffer < 0 || drawbuffer > 3) return;
	if(buffer == GL_COLOR){
		f->clrColor[drawbuffer] = true; memcpy(f->clrVal[drawbuffer], v, 16);
		if(G->es.fbo == f) endEncoder();
	}else if(buffer == GL_DEPTH){
		f->clrDepth = true; f->clrD = v[0];
		if(G->es.fbo == f) endEncoder();
	}
}

// ---- copies
static id<MTLTexture>
readTarget(Fbo *f)
{
	if(!f) return nil;
	flushPending(f);
	Attach &a = f->color[(G->readBuffer >= GL_COLOR_ATTACHMENT0 && G->readBuffer < GL_COLOR_ATTACHMENT0 + 4) ? G->readBuffer - GL_COLOR_ATTACHMENT0 : 0];
	return attachTexture(a);
}

MGL void mgl_glCopyTexSubImage2D(GLenum target, GLint level, GLint xo, GLint yo, GLint x, GLint y, GLsizei w, GLsizei h)
{
	Tex *t = boundTex(target);
	if(!t || !ensureTex(t)) return;
	id<MTLTexture> src = readTarget(fboOf(G->readFbo));
	if(!src) return;
	endEncoder();
	id<MTLBlitCommandEncoder> bl = [cmd() blitCommandEncoder];
	[bl copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x, y, 0) sourceSize:MTLSizeMake(w, h, 1)
	          toTexture:t->mtl destinationSlice:0 destinationLevel:level destinationOrigin:MTLOriginMake(xo, yo, 0)];
	[bl endEncoding];
	t->lastUse = G->serial;
}

MGL void mgl_glBlitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield mask, GLenum filter)
{
	Fbo *src = fboOf(G->readFbo), *dst = fboOf(G->drawFbo);
	if(!src || !dst) return;
	flushPending(src);
	if(src != dst) flushPending(dst);
	int sw = sx1 - sx0, sh = sy1 - sy0, dw = dx1 - dx0, dh = dy1 - dy0;
	bool same = sw == dw && sh == dh;
	if(mask & GL_COLOR_BUFFER_BIT){
		id<MTLTexture> st = attachTexture(src->color[0]);
		id<MTLTexture> dt = attachTexture(dst->color[0]);
		if(st && dt){
			if(same && st.pixelFormat == dt.pixelFormat){
				endEncoder();
				id<MTLBlitCommandEncoder> bl = [cmd() blitCommandEncoder];
				[bl copyFromTexture:st sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(sx0, sy0, 0) sourceSize:MTLSizeMake(sw, sh, 1)
				          toTexture:dt destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(dx0, dy0, 0)];
				[bl endEncoding];
			}else{
				// scaled copy: draw the source into the destination rectangle
				Fbo tmp; tmp.color[0] = dst->color[0];
				tmp.drawBuf[0] = GL_COLOR_ATTACHMENT0; tmp.drawBuf[1] = tmp.drawBuf[2] = tmp.drawBuf[3] = GL_NONE;
				endEncoder();
				MTLRenderPassDescriptor *pd = [MTLRenderPassDescriptor renderPassDescriptor];
				pd.colorAttachments[0].texture = dt;
				pd.colorAttachments[0].loadAction = MTLLoadActionLoad;
				pd.colorAttachments[0].storeAction = MTLStoreActionStore;
				id<MTLRenderCommandEncoder> enc = [cmd() renderCommandEncoderWithDescriptor:pd];
				MTLPixelFormat cf[4] = { dt.pixelFormat, MTLPixelFormatInvalid, MTLPixelFormatInvalid, MTLPixelFormatInvalid };
				MTLColorWriteMask mk[4] = { MTLColorWriteMaskAll, MTLColorWriteMaskNone, MTLColorWriteMaskNone, MTLColorWriteMaskNone };
				id<MTLRenderPipelineState> pso = utilPso(true, cf, MTLPixelFormatInvalid, mk);
				if(pso){
					[enc setRenderPipelineState:pso];
					[enc setCullMode:MTLCullModeNone];
					int vx = dx0 < dx1 ? dx0 : dx1, vy = dy0 < dy1 ? dy0 : dy1;
					[enc setViewport:(MTLViewport){ (double)vx, (double)vy, (double)abs(dw), (double)abs(dh), 0, 1 }];
					[enc setScissorRect:scissorRect(vx < 0 ? 0 : vx, vy < 0 ? 0 : vy,
						(NSUInteger)(vx + abs(dw) > (int)dt.width ? (int)dt.width - vx : abs(dw)), (NSUInteger)(vy + abs(dh) > (int)dt.height ? (int)dt.height - vy : abs(dh)))];
					float tw = (float)st.width, th = (float)st.height;
					float u0 = sx0/tw, v0 = sy0/th, u1 = sx1/tw, v1 = sy1/th;
					if(dx0 > dx1){ float t = u0; u0 = u1; u1 = t; }
					if(dy0 > dy1){ float t = v0; v0 = v1; v1 = t; }
					float params[8] = { u0, v0, u1, v1, 1.0f, 0.0f, 0, 0 };
					[enc setVertexBytes:params length:sizeof(params) atIndex:0];
					Tex dummy; dummy.minF = dummy.magF = filter == GL_LINEAR ? GL_LINEAR : GL_NEAREST; dummy.wrapS = dummy.wrapT = GL_CLAMP_TO_EDGE; dummy.created = 1;
					[enc setFragmentSamplerState:samplerFor(&dummy) atIndex:0];
					[enc setFragmentTexture:st atIndex:0];
					[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
				}
				[enc endEncoding];
			}
		}
	}
	if(mask & GL_DEPTH_BUFFER_BIT){
		id<MTLTexture> st = attachTexture(src->depth), dt = attachTexture(dst->depth);
		if(st && dt && same && st.pixelFormat == dt.pixelFormat){
			endEncoder();
			id<MTLBlitCommandEncoder> bl = [cmd() blitCommandEncoder];
			[bl copyFromTexture:st sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(sx0, sy0, 0) sourceSize:MTLSizeMake(sw, sh, 1)
			          toTexture:dt destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(dx0, dy0, 0)];
			[bl endEncoding];
		}
	}
}

// ---- read back
MGL void mgl_glReadPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, void *out)
{
	id<MTLTexture> src = readTarget(fboOf(G->readFbo));
	if(!src) return;
	submit(true);
	size_t bpp = src.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : src.pixelFormat == MTLPixelFormatR8Unorm ? 1 : src.pixelFormat == MTLPixelFormatR32Float ? 4 : 4;
	std::vector<uint8_t> raw((size_t)w*h*bpp);
	[src getBytes:raw.data() bytesPerRow:(size_t)w*bpp fromRegion:MTLRegionMake2D(x, y, w, h) mipmapLevel:0];
	int comps = format == GL_RGBA || format == GL_BGRA ? 4 : format == GL_RGB || format == GL_BGR ? 3 : 1;
	bool bgraSrc = src.pixelFormat == MTLPixelFormatBGRA8Unorm;
	size_t dstRow = ((size_t)w*comps + G->packAlign - 1)/G->packAlign*G->packAlign;
	uint8_t *o = (uint8_t*)out;
	if(src.pixelFormat == MTLPixelFormatRGBA8Unorm || bgraSrc){
		for(int yy = 0; yy < h; yy++) for(int xx = 0; xx < w; xx++){
			const uint8_t *p = &raw[((size_t)yy*w + xx)*4];
			uint8_t r = bgraSrc ? p[2] : p[0], g = p[1], b = bgraSrc ? p[0] : p[2], a = p[3];
			uint8_t *d = o + yy*dstRow + (size_t)xx*comps;
			if(format == GL_BGRA){ d[0] = b; d[1] = g; d[2] = r; d[3] = a; }
			else if(format == GL_BGR){ d[0] = b; d[1] = g; d[2] = r; }
			else if(comps == 4){ d[0] = r; d[1] = g; d[2] = b; d[3] = a; }
			else if(comps == 3){ d[0] = r; d[1] = g; d[2] = b; }
			else d[0] = r;
		}
	}else{
		for(int yy = 0; yy < h; yy++) memcpy(o + yy*dstRow, &raw[(size_t)yy*w*bpp], (size_t)w*bpp < dstRow ? (size_t)w*bpp : dstRow);
	}
}
MGL void mgl_glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void *out)
{
	Tex *t = boundTex(target);
	if(!t || !ensureTex(t)) return;
	submit(true);
	int w = t->w >> level, h = t->h >> level; if(w < 1) w = 1; if(h < 1) h = 1;
	size_t rb; size_t total = levelBytes(t->fmt, w, h, &rb);
	std::vector<uint8_t> raw(total);
	[t->mtl getBytes:raw.data() bytesPerRow:rb fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:level];
	int comps = format == GL_RGBA || format == GL_BGRA ? 4 : format == GL_RGB ? 3 : 1;
	uint8_t *o = (uint8_t*)out;
	size_t dstRow = ((size_t)w*comps + G->packAlign - 1)/G->packAlign*G->packAlign;
	if(t->fmt == MTLPixelFormatRGBA8Unorm || t->fmt == MTLPixelFormatBGRA8Unorm){
		bool bgraSrc = t->fmt == MTLPixelFormatBGRA8Unorm;
		for(int yy = 0; yy < h; yy++) for(int xx = 0; xx < w; xx++){
			const uint8_t *p = &raw[yy*rb + (size_t)xx*4];
			uint8_t r = bgraSrc ? p[2] : p[0], g = p[1], b = bgraSrc ? p[0] : p[2], a = p[3];
			uint8_t *d = o + yy*dstRow + (size_t)xx*comps;
			if(format == GL_BGRA){ d[0] = b; d[1] = g; d[2] = r; d[3] = a; }
			else if(comps == 4){ d[0] = r; d[1] = g; d[2] = b; d[3] = a; }
			else if(comps == 3){ d[0] = r; d[1] = g; d[2] = b; }
			else d[0] = r;
		}
	}else
		memcpy(out, raw.data(), total);
}
MGL void mgl_glGetCompressedTexImage(GLenum target, GLint level, void *out)
{
	Tex *t = boundTex(target);
	if(!t || !ensureTex(t)) return;
	submit(true);
	int w = t->w >> level, h = t->h >> level; if(w < 1) w = 1; if(h < 1) h = 1;
	size_t rb; levelBytes(t->fmt, w, h, &rb);
	[t->mtl getBytes:out bytesPerRow:rb fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:level];
}

// ---- shaders and programs
MGL GLuint mgl_glCreateShader(GLenum type){ Shader *s = new Shader(); s->type = type; return addObj(G->shaders, s); }
MGL void mgl_glShaderSource(GLuint id, GLsizei n, const GLchar *const *strs, const GLint *lens)
{
	if(id >= G->shaders.size() || !G->shaders[id]) return;
	std::string &src = G->shaders[id]->src;
	src.clear();
	for(int i = 0; i < n; i++) src += (lens && lens[i] >= 0) ? std::string(strs[i], (size_t)lens[i]) : std::string(strs[i]);
}
MGL void mgl_glCompileShader(GLuint){ }
MGL void mgl_glGetShaderiv(GLuint, GLenum pname, GLint *v){ *v = pname == GL_COMPILE_STATUS ? 1 : 0; }
MGL void mgl_glGetShaderInfoLog(GLuint, GLsizei max, GLsizei *len, GLchar *log){ if(len) *len = 0; if(max > 0) log[0] = 0; }
MGL void mgl_glDeleteShader(GLuint id){ if(id < G->shaders.size() && G->shaders[id]){ delete G->shaders[id]; G->shaders[id] = nullptr; } }
MGL GLuint mgl_glCreateProgram(void){ return addObj(G->progs, new Prog()); }
MGL void mgl_glAttachShader(GLuint prog, GLuint sh)
{
	if(prog >= G->progs.size() || !G->progs[prog] || sh >= G->shaders.size() || !G->shaders[sh]) return;
	Prog *p = G->progs[prog];
	(G->shaders[sh]->type == GL_VERTEX_SHADER ? p->vsSrcParts : p->fsSrcParts).push_back(G->shaders[sh]->src);
}
MGL void mgl_glLinkProgram(GLuint id)
{
	if(id >= G->progs.size() || !G->progs[id]) return;
	Prog *p = G->progs[id];
	if(p->vsSrcParts.empty() || p->fsSrcParts.empty()){ p->log = "missing shader"; return; }
	const char *vs[2] = { p->vsSrcParts.back().c_str(), nullptr };
	const char *fs[2] = { p->fsSrcParts.back().c_str(), nullptr };
	p->tr = TransProgram();
	if(!rw::mtl::translateProgram(vs, fs, p->tr)){
		p->log = p->tr.error;
		logf_("shader translation failed: %s", p->tr.error.c_str());
		return;
	}
	p->vsBlock.assign((size_t)p->tr.vsBlockSize, 0);
	p->fsBlock.assign((size_t)p->tr.fsBlockSize, 0);
	p->uniforms.clear(); p->byName.clear();
	for(auto &u : p->tr.uniforms){
		auto it = p->byName.find(u.name);
		if(it == p->byName.end()){ it = p->byName.insert({ u.name, (int)p->uniforms.size() }).first; p->uniforms.push_back(Prog::U()); p->uniforms.back().name = u.name; }
		Prog::U &pu = p->uniforms[it->second];
		(u.stage == 0 ? pu.vsOff : pu.fsOff) = u.offset;
		pu.size = u.size;
	}
	for(auto &s : p->tr.samplers){
		auto it = p->byName.find(s.name);
		if(it == p->byName.end()){ it = p->byName.insert({ s.name, (int)p->uniforms.size() }).first; p->uniforms.push_back(Prog::U()); p->uniforms.back().name = s.name; }
		p->uniforms[it->second].sampler = 1;
	}
	p->samplerUnit.assign(p->tr.samplers.size(), 0);
	p->linked = true;
	p->vsSrcParts.clear(); p->fsSrcParts.clear();
}
MGL void mgl_glGetProgramiv(GLuint id, GLenum pname, GLint *v)
{
	Prog *p = id < G->progs.size() ? G->progs[id] : nullptr;
	if(pname == GL_LINK_STATUS) *v = p && p->linked;
	else if(pname == GL_INFO_LOG_LENGTH) *v = p ? (GLint)p->log.size() + 1 : 0;
	else *v = 0;
}
MGL void mgl_glGetProgramInfoLog(GLuint id, GLsizei max, GLsizei *len, GLchar *log)
{
	Prog *p = id < G->progs.size() ? G->progs[id] : nullptr;
	std::string s = p ? p->log : "";
	if(max <= 0) return;
	size_t n = s.size() < (size_t)max - 1 ? s.size() : (size_t)max - 1;
	memcpy(log, s.data(), n); log[n] = 0;
	if(len) *len = (GLsizei)n;
}
MGL void mgl_glUseProgram(GLuint id){ G->prog = id; }
MGL void mgl_glDeleteProgram(GLuint id){
	if(id < G->progs.size() && G->progs[id]){ if(G->prog == id) G->prog = 0; delete G->progs[id]; G->progs[id] = nullptr; }
}
MGL void mgl_glBindAttribLocation(GLuint, GLuint, const GLchar*){ }
MGL GLint mgl_glGetAttribLocation(GLuint, const GLchar*){ return -1; }
MGL GLuint mgl_glGetUniformBlockIndex(GLuint, const GLchar*){ return GL_INVALID_INDEX; }
MGL void mgl_glUniformBlockBinding(GLuint, GLuint, GLuint){ }
MGL GLint mgl_glGetUniformLocation(GLuint id, const GLchar *name)
{
	Prog *p = id < G->progs.size() ? G->progs[id] : nullptr;
	if(!p || !p->linked) return -1;
	std::string n(name);
	size_t br = n.find('[');
	if(br != std::string::npos) n.resize(br);
	auto it = p->byName.find(n);
	return it == p->byName.end() ? -1 : it->second + 1;
}

static void
setUniformBytes(GLint loc, const void *src, size_t bytes)
{
	Prog *p = G->prog < G->progs.size() ? G->progs[G->prog] : nullptr;
	if(!p || loc <= 0 || loc > (GLint)p->uniforms.size()) return;
	Prog::U &u = p->uniforms[loc - 1];
	size_t n = bytes < (size_t)u.size ? bytes : (size_t)u.size;
	bool changed = false;
	if(u.vsOff >= 0 && (size_t)u.vsOff + n <= p->vsBlock.size() && memcmp(&p->vsBlock[u.vsOff], src, n)){ memcpy(&p->vsBlock[u.vsOff], src, n); changed = true; }
	if(u.fsOff >= 0 && (size_t)u.fsOff + n <= p->fsBlock.size() && memcmp(&p->fsBlock[u.fsOff], src, n)){ memcpy(&p->fsBlock[u.fsOff], src, n); changed = true; }
	if(changed) p->version++;
}
MGL void mgl_glUniform1i(GLint loc, GLint v)
{
	Prog *p = G->prog < G->progs.size() ? G->progs[G->prog] : nullptr;
	if(!p || loc <= 0 || loc > (GLint)p->uniforms.size()) return;
	Prog::U &u = p->uniforms[loc - 1];
	if(u.sampler >= 0){
		for(size_t i = 0; i < p->tr.samplers.size(); i++) if(p->tr.samplers[i].name == u.name) p->samplerUnit[i] = v;
		return;
	}
	setUniformBytes(loc, &v, 4);
}
MGL void mgl_glUniform1f(GLint loc, GLfloat a){ setUniformBytes(loc, &a, 4); }
MGL void mgl_glUniform2f(GLint loc, GLfloat a, GLfloat b){ float v[2] = { a, b }; setUniformBytes(loc, v, 8); }
MGL void mgl_glUniform3f(GLint loc, GLfloat a, GLfloat b, GLfloat c){ float v[3] = { a, b, c }; setUniformBytes(loc, v, 12); }
MGL void mgl_glUniform4f(GLint loc, GLfloat a, GLfloat b, GLfloat c, GLfloat d){ float v[4] = { a, b, c, d }; setUniformBytes(loc, v, 16); }
MGL void mgl_glUniform1fv(GLint loc, GLsizei n, const GLfloat *v){ setUniformBytes(loc, v, (size_t)n*4); }
MGL void mgl_glUniform2fv(GLint loc, GLsizei n, const GLfloat *v){ setUniformBytes(loc, v, (size_t)n*8); }
MGL void mgl_glUniform3fv(GLint loc, GLsizei n, const GLfloat *v){ setUniformBytes(loc, v, (size_t)n*12); }
MGL void mgl_glUniform4fv(GLint loc, GLsizei n, const GLfloat *v){ setUniformBytes(loc, v, (size_t)n*16); }
MGL void mgl_glUniform4iv(GLint loc, GLsizei n, const GLint *v){ setUniformBytes(loc, v, (size_t)n*16); }
MGL void mgl_glUniformMatrix4fv(GLint loc, GLsizei n, GLboolean transpose, const GLfloat *v)
{
	if(!transpose){ setUniformBytes(loc, v, (size_t)n*64); return; }
	float t[16];
	for(int i = 0; i < 4; i++) for(int j = 0; j < 4; j++) t[i*4 + j] = v[j*4 + i];
	setUniformBytes(loc, t, 64);
}

// ---- draw
MGL void mgl_glDrawArrays(GLenum mode, GLint first, GLsizei count){ drawCommon(mode, count, 0, 0, false, first); }
MGL void mgl_glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *offset){ drawCommon(mode, count, type, (size_t)offset, true, 0); }

#define ENTRY(n) { #n, (void*)mgl_##n }
static const struct { const char *name; void *fn; } kEntries[] = {
	ENTRY(glGetString), ENTRY(glGetStringi), ENTRY(glGetError), ENTRY(glGetIntegerv), ENTRY(glGetFloatv), ENTRY(glGetBooleanv),
	ENTRY(glEnable), ENTRY(glDisable), ENTRY(glIsEnabled), ENTRY(glEnablei), ENTRY(glDisablei),
	ENTRY(glBlendFunc), ENTRY(glBlendFuncSeparate), ENTRY(glDepthFunc), ENTRY(glDepthMask), ENTRY(glColorMask), ENTRY(glCullFace),
	ENTRY(glFrontFace), ENTRY(glStencilFunc), ENTRY(glStencilOp), ENTRY(glStencilMask), ENTRY(glScissor), ENTRY(glViewport),
	ENTRY(glClearColor), ENTRY(glClear), ENTRY(glClearBufferfv), ENTRY(glPixelStorei), ENTRY(glFinish), ENTRY(glFlush),
	ENTRY(glGenBuffers), ENTRY(glDeleteBuffers), ENTRY(glBindBuffer), ENTRY(glBufferData), ENTRY(glBufferSubData), ENTRY(glBindBufferBase),
	ENTRY(glGenVertexArrays), ENTRY(glDeleteVertexArrays), ENTRY(glBindVertexArray), ENTRY(glEnableVertexAttribArray),
	ENTRY(glDisableVertexAttribArray), ENTRY(glVertexAttribPointer),
	ENTRY(glGenTextures), ENTRY(glDeleteTextures), ENTRY(glActiveTexture), ENTRY(glBindTexture), ENTRY(glTexParameteri),
	ENTRY(glTexParameterf), ENTRY(glTexImage2D), ENTRY(glCompressedTexImage2D), ENTRY(glTexSubImage2D), ENTRY(glGenerateMipmap),
	ENTRY(glGetTexLevelParameteriv), ENTRY(glGetTexImage), ENTRY(glGetCompressedTexImage), ENTRY(glReadPixels), ENTRY(glCopyTexSubImage2D),
	ENTRY(glGenRenderbuffers), ENTRY(glDeleteRenderbuffers), ENTRY(glBindRenderbuffer), ENTRY(glRenderbufferStorage),
	ENTRY(glGenFramebuffers), ENTRY(glDeleteFramebuffers), ENTRY(glBindFramebuffer), ENTRY(glFramebufferTexture2D),
	ENTRY(glFramebufferRenderbuffer), ENTRY(glCheckFramebufferStatus), ENTRY(glDrawBuffers), ENTRY(glReadBuffer), ENTRY(glBlitFramebuffer),
	ENTRY(glCreateShader), ENTRY(glShaderSource), ENTRY(glCompileShader), ENTRY(glGetShaderiv), ENTRY(glGetShaderInfoLog), ENTRY(glDeleteShader),
	ENTRY(glCreateProgram), ENTRY(glAttachShader), ENTRY(glLinkProgram), ENTRY(glGetProgramiv), ENTRY(glGetProgramInfoLog), ENTRY(glUseProgram),
	ENTRY(glDeleteProgram), ENTRY(glBindAttribLocation), ENTRY(glGetAttribLocation), ENTRY(glGetUniformBlockIndex), ENTRY(glUniformBlockBinding),
	ENTRY(glGetUniformLocation), ENTRY(glUniform1i), ENTRY(glUniform1f), ENTRY(glUniform2f), ENTRY(glUniform3f), ENTRY(glUniform4f),
	ENTRY(glUniform1fv), ENTRY(glUniform2fv), ENTRY(glUniform3fv), ENTRY(glUniform4fv), ENTRY(glUniform4iv), ENTRY(glUniformMatrix4fv),
	ENTRY(glDrawArrays), ENTRY(glDrawElements),
};

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// setup, present, interop

namespace {

static void
makeDefaultTargets(int w, int h)
{
	if(w < 1) w = 1; if(h < 1) h = 1;
	if(!G->defFbo){
		G->defFbo = new Fbo();
		G->defFbo->isDefault = true;
		G->defColor = new Tex();
		G->defDepth = new Tex();
		G->defFbo->color[0].tex = G->defColor;
		G->defFbo->depth.tex = G->defDepth;
	}
	endEncoder();
	MTLTextureDescriptor *cd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
	cd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	cd.storageMode = MTLStorageModeShared;
	G->defColor->mtl = [G->dev newTextureWithDescriptor:cd];
	G->defColor->w = w; G->defColor->h = h; G->defColor->fmt = MTLPixelFormatRGBA8Unorm; G->defColor->created = 1;
	MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:w height:h mipmapped:NO];
	dd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	dd.storageMode = MTLStorageModePrivate;
	G->defDepth->mtl = [G->dev newTextureWithDescriptor:dd];
	G->defDepth->w = w; G->defDepth->h = h; G->defDepth->fmt = MTLPixelFormatDepth32Float_Stencil8; G->defDepth->depth = true; G->defDepth->created = 1;
	G->fbW = w; G->fbH = h;
	G->defFbo->clrColor[0] = true;	// start with black
	memset(G->defFbo->clrVal[0], 0, 16); G->defFbo->clrVal[0][3] = 1.0f;
	G->defFbo->clrDepth = true; G->defFbo->clrD = 1.0f;
}

}  // namespace

extern "C" int
mtlglInit(void *nsWindow, int w, int h)
{
	if(G) return 1;
	@autoreleasepool {
		G = new Ctx();
		G->dev = MTLCreateSystemDefaultDevice();
		if(!G->dev){ delete G; G = nullptr; return 0; }
		G->queue = [G->dev newCommandQueue];
		NSWindow *win = (__bridge NSWindow*)nsWindow;
		NSView *view = win.contentView;
		G->window = win;
		G->layer = [CAMetalLayer layer];
		G->layer.device = G->dev;
		G->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
		G->layer.framebufferOnly = YES;
		G->layer.contentsScale = win.backingScaleFactor;
		G->layer.drawableSize = CGSizeMake(w, h);
		G->layer.maximumDrawableCount = 3;
		view.wantsLayer = YES;
		view.layer = G->layer;
		makeDefaultTargets(w, h);
		G->vp[2] = w; G->vp[3] = h;
		FILE *f = fopen("metalgl.log", "w"); if(f){ fprintf(f, "Metal GL layer: %s\n", G->dev.name.UTF8String); fclose(f); }
	}
	return 1;
}

extern "C" void
mtlglShutdown(void)
{
	if(!G) return;
	submit(true);
	// objects are left to process exit (Metal objects must not be released during exit handlers)
	G = nullptr;
}

extern "C" void *
mtlglGetProcAddress(const char *name)
{
	static std::unordered_map<std::string, void*> table;
	if(table.empty()) for(auto &e : kEntries) table[e.name] = e.fn;
	auto it = table.find(name);
	return it == table.end() ? nullptr : it->second;
}

extern "C" void
mtlglSetVsync(int interval)
{
	if(!G) return;
	G->vsync = interval;
	G->layer.displaySyncEnabled = interval != 0;
}

extern "C" void
mtlglResize(int w, int h)
{
	if(!G || (w == G->fbW && h == G->fbH)) return;
	@autoreleasepool {
		submit(true);
		G->layer.drawableSize = CGSizeMake(w, h);
		makeDefaultTargets(w, h);
	}
}

extern "C" void
mtlglPresent(void)
{
	if(!G) return;
	@autoreleasepool {
		flushPending(G->defFbo);
		endEncoder();
		id<CAMetalDrawable> dr = [G->layer nextDrawable];
		if(dr){
			MTLRenderPassDescriptor *pd = [MTLRenderPassDescriptor renderPassDescriptor];
			pd.colorAttachments[0].texture = dr.texture;
			pd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
			pd.colorAttachments[0].storeAction = MTLStoreActionStore;
			id<MTLRenderCommandEncoder> enc = [cmd() renderCommandEncoderWithDescriptor:pd];
			MTLPixelFormat cf[4] = { dr.texture.pixelFormat, MTLPixelFormatInvalid, MTLPixelFormatInvalid, MTLPixelFormatInvalid };
			MTLColorWriteMask mk[4] = { MTLColorWriteMaskAll, MTLColorWriteMaskNone, MTLColorWriteMaskNone, MTLColorWriteMaskNone };
			id<MTLRenderPipelineState> pso = utilPso(true, cf, MTLPixelFormatInvalid, mk);
			if(pso){
				[enc setRenderPipelineState:pso];
				[enc setCullMode:MTLCullModeNone];
				float params[8] = { 0, 0, 1, 1, 0, 0, 0, 0 };
				[enc setVertexBytes:params length:sizeof(params) atIndex:0];
				Tex dummy; dummy.minF = dummy.magF = GL_LINEAR; dummy.wrapS = dummy.wrapT = GL_CLAMP_TO_EDGE; dummy.created = 1;
				[enc setFragmentSamplerState:samplerFor(&dummy) atIndex:0];
				[enc setFragmentTexture:G->defColor->mtl atIndex:0];
				[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
			}
			[enc endEncoding];
			[cmd() presentDrawable:dr];
		}
		submit(false);
		// follow the window: a changed backing size means a new default framebuffer
		NSView *v = G->window.contentView;
		NSRect b = [v convertRectToBacking:v.bounds];
		int nw = (int)b.size.width, nh = (int)b.size.height;
		if(nw > 0 && nh > 0 && (nw != G->fbW || nh != G->fbH)) mtlglResize(nw, nh);
	}
}

extern "C" void *mtlglDevice(void){ return G ? (__bridge void*)G->dev : nullptr; }
extern "C" void *mtlglCommandQueue(void){ return G ? (__bridge void*)G->queue : nullptr; }
extern "C" void *mtlglTexture(unsigned int name)
{
	Tex *t = G ? texOf(name) : nullptr;
	if(!t || !ensureTex(t)) return nullptr;
	return (__bridge void*)t->mtl;
}
extern "C" void mtlglFlush(void){ if(G) submit(false); }
