#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <array>

#include "../rwbase.h"
#include "../rwerror.h"
#include "../rwplg.h"
#include "../rwrender.h"
#include "../rwengine.h"
#include "../rwpipeline.h"
#include "../rwobjects.h"
#include "../rwanim.h"
#include "../rwplugins.h"
#ifdef RW_OPENGL

#include "rwgl3.h"
#include "rwgl3shader.h"
#include "rwgl3impl.h"

#define PLUGIN_ID 0

namespace rw {
namespace gl3 {

GlGlobals glGlobals;

Gl3Caps gl3Caps;
// terrible hack for GLES
bool32 needToReadBackTextures;

int32   alphaFunc;
float32 alphaRef;

struct UniformState
{
	float32 alphaRefLow;
	float32 alphaRefHigh;
	int32   pad[2];

	float32 fogStart;
	float32 fogEnd;
	float32 fogRange;
	float32 fogDisable;

	RGBAf   fogColor;
};

struct UniformScene
{
	float32 proj[16];
	float32 view[16];
};

#define MAX_LIGHTS 8

struct UniformObject
{
	RawMatrix    world;
	RGBAf        ambLight;
	struct {
		float type;
		float radius;
		float minusCosAngle;
		float hardSpot;
	} lightParams[MAX_LIGHTS];
	V4d lightPosition[MAX_LIGHTS];
	V4d lightDirection[MAX_LIGHTS];
	RGBAf lightColor[MAX_LIGHTS];
};

const char *shaderDecl120 =
"#version 120\n"
"#define GL2\n"
"#define texture texture2D\n"
"#define VSIN(index) attribute\n"
"#define VSOUT varying\n"
"#define FSIN varying\n"
"#define FRAGCOLOR(c) (gl_FragColor = c)\n";
const char *shaderDecl330 =
"#version 330\n"
"#define HAS_VEL_OUT\n"
"#define VSIN(index) layout(location = index) in\n"
"#define VSOUT out\n"
"#define FSIN in\n"
"#define FRAGCOLOR(c) (fragColor = c)\n";
const char *shaderDecl100es =
"#version 100\n"
"#define GL2\n"
"#define texture texture2D\n"
"#define VSIN(index) attribute\n"
"#define VSOUT varying\n"
"#define FSIN varying\n"
"#define FRAGCOLOR(c) (gl_FragColor = c)\n"
"precision highp float;\n"
"precision highp int;\n";
const char *shaderDecl310es =
"#version 310 es\n"
"#define VSIN(index) layout(location = index) in\n"
"#define VSOUT out\n"
"#define FSIN in\n"
"#define FRAGCOLOR(c) (fragColor = c)\n"
"precision highp float;\n"
"precision highp int;\n";

const char *shaderDecl;

// this needs a define in the shaders as well!
//#define RW_GL_USE_UBOS

static GLuint vao;
#ifdef RW_GL_USE_UBOS
static GLuint ubo_state, ubo_scene, ubo_object;
#endif
static GLuint whitetex;
static UniformState uniformState;
static UniformScene uniformScene;
static UniformObject uniformObject;

#ifndef RW_GL_USE_UBOS
// State
int32 u_alphaRef;
int32 u_fogData;
int32 u_fogColor;

// Scene
int32 u_proj;
int32 u_view;

// Object
int32 u_world;
int32 u_ambLight;
int32 u_lightParams;
int32 u_lightPosition;
int32 u_lightDirection;
int32 u_lightColor;
#endif

int32 u_matColor;
int32 u_surfProps;

Shader *defaultShader, *defaultShader_noAT;
Shader *defaultShader_fullLight, *defaultShader_fullLight_noAT;

static bool32 stateDirty = 1;
static bool32 sceneDirty = 1;
static bool32 objectDirty = 1;

struct RwRasterStateCache {
	Raster *raster;
	Texture::Addressing addressingU;
	Texture::Addressing addressingV;
	Texture::FilterMode filter;
};

#define MAXNUMSTAGES 8

// cached RW render states
struct RwStateCache {
	bool32 vertexAlpha;
	uint32 alphaTestEnable;
	uint32 alphaFunc;
	bool32 textureAlpha;
	bool32 blendEnable;
	uint32 srcblend, destblend;
	uint32 zwrite;
	uint32 ztest;
	uint32 cullmode;
	uint32 stencilenable;
	uint32 stencilpass;
	uint32 stencilfail;
	uint32 stencilzfail;
	uint32 stencilfunc;
	uint32 stencilref;
	uint32 stencilmask;
	uint32 stencilwritemask;
	uint32 fogEnable;
	float32 fogStart;
	float32 fogEnd;

	// emulation of PS2 GS
	bool32 gsalpha;
	uint32 gsalpharef;

	RwRasterStateCache texstage[MAXNUMSTAGES];
};
static RwStateCache rwStateCache;

enum
{
	// actual gl states
	RWGL_BLEND,
	RWGL_SRCBLEND,
	RWGL_DESTBLEND,
	RWGL_DEPTHTEST,
	RWGL_DEPTHFUNC,
	RWGL_DEPTHMASK,
	RWGL_CULL,
	RWGL_CULLFACE,
	RWGL_STENCIL,
	RWGL_STENCILFUNC,
	RWGL_STENCILFAIL,
	RWGL_STENCILZFAIL,
	RWGL_STENCILPASS,
	RWGL_STENCILREF,
	RWGL_STENCILMASK,
	RWGL_STENCILWRITEMASK,

	// uniforms
	RWGL_ALPHAFUNC,
	RWGL_ALPHAREF,
	RWGL_FOG,
	RWGL_FOGSTART,
	RWGL_FOGEND,
	RWGL_FOGCOLOR,

	RWGL_NUM_STATES
};
static bool uniformStateDirty[RWGL_NUM_STATES];

struct GlState {
	bool32 blendEnable;
	uint32 srcblend, destblend;

	bool32 depthTest;
	uint32 depthFunc;

	uint32 depthMask;

	bool32 cullEnable;
	uint32 cullFace;

	bool32 stencilEnable;
	// glStencilFunc
	uint32 stencilFunc;
	uint32 stencilRef;
	uint32 stencilMask;
	// glStencilOp
	uint32 stencilPass;
	uint32 stencilFail;
	uint32 stencilZFail;
	// glStencilMask
	uint32 stencilWriteMask;
};
static GlState curGlState, oldGlState;

static int32 activeTexture;
static uint32 boundTexture[MAXNUMSTAGES];

// re3.ini [Graphics] RenderScale (0.5-1.0): the 3D scene is drawn at this fraction of the window's
// pixel size and upscaled (FSR 1 style) before the HUD is drawn
static float gRenderScale = 1.0f;
static bool gScaledFrame = true;	// currently drawing into the reduced-size target
// re3.ini [Graphics] RenderTemporal=1: sub-pixel camera jitter + temporal accumulation (upscaler / anti-aliasing)
static bool gTemporal = false;
static GLuint rsFbo, rsVelTex, rsReflTex;	// reduced-size scene target, motion vector attachment, reflection direction attachment
static int gVelWant = 1, gVelBoth = 1;	// draw buffers the selected shader wants (1 colour, 2 +motion, 3 +reflection) / enabled
static float gRTStrength = 0.85f, gRTDistance = 250.0f;
static bool gRT = false;
static bool gRTShadows = false;			// RenderRTShadows=1: sun shadows traced from the same scene
static int gRTDiv = 2;				// ray tracing resolution divisor (2 = half resolution, RenderRTHalfRes=0 for full)		// re3.ini [Graphics] RenderRT=1: ray traced reflections on vehicles (Metal ray tracing)
static bool gNeedVel = false;		// motion vector attachment needed (temporal upscaling or ray tracing)
static int gVelObj, gSceneSerial;
static float gVelPrevWorld[16];
static float gVelOn;
static bool gRTRoadsOn = true;
static float gRoadWet, gVelRoadWet, gRTRoadStrength = 0.7f;
// which ray traced effects are wanted right now: car reflections, shadows and wet road reflections work on their own
static inline bool rtWetRoads(void){ return gRTRoadsOn && gRoadWet > 0.05f; }
static inline bool rtWantsRefl(void){ return gRT || rtWetRoads(); }
static inline bool rtActive(void){ return gRT || gRTShadows || rtWetRoads(); }	// wetness of the roads (0..1), value for the object being drawn
extern "C" { int regtaVelocityDynamic; int regtaRoadDraw; }	// set by the game while it draws a moving object / a road
extern "C" void regtaRTSetWetness(float wet) { gRoadWet = wet; }	// set by the game while it draws a moving object
struct VelEntry { float cur[16], prev[16]; int frame; };
static std::unordered_map<const void*, VelEntry> gVelMap;
// Graphics options the game's menu changes while running (the menu stores them in re3.ini [Graphics] under the same keys).
// Index into kScaleTable for the render scale (restart needed for that one).
extern "C" { int8_t regtaGfxScaleIdx, regtaGfxTemporal, regtaGfxMetalFX, regtaGfxSSR, regtaGfxRT, regtaGfxRTRoads, regtaGfxRTShadows, regtaGfxPuddles = 1; }
static const float kScaleTable[6] = { 1.0f, 0.9f, 0.8f, 0.75f, 0.67f, 0.5f };
// frame cost of the extra graphics passes (milliseconds, smoothed): ray tracing, MetalFX, whole upscale/resolve step
static float gPerfMs[3];
static inline double perfNow(void){ return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
extern "C" void regtaPerfGet(float *ms){ ms[0] = gPerfMs[0]; ms[1] = gPerfMs[1]; ms[2] = gPerfMs[2]; }
static bool gMetalFX = false;		// re3.ini [Graphics] RenderMetalFX=1: MetalFX temporal scaler instead of the GLSL resolve
static float gMfxJitterSignX = 1.0f, gMfxJitterSignY = 1.0f;
static bool gSSR = false;		// vehicles may sample a copy of the scene (screen space reflections)
static bool gUseTarget = false;		// 3D scene goes through the offscreen target (render scale < 1 or temporal)
static float gJit[2];			// current jitter in target pixels
static int gFrameIdx;
static bool gJitterApplied;		// the main camera was jittered this frame
static bool gHistValid;
static float gCurVP[16], gPrevVP[16], gInvVP[16];

static void
mat4mul(float *r, const float *a, const float *b)	// column-major r = a*b
{
	for(int c = 0; c < 4; c++)
		for(int rw = 0; rw < 4; rw++){
			float v = 0;
			for(int k = 0; k < 4; k++) v += a[k*4+rw]*b[c*4+k];
			r[c*4+rw] = v;
		}
}

static bool
mat4inv(float *out, const float *m)
{
	float inv[16];
	inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
	inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
	inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
	inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
	inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
	inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
	inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
	inv[13] = m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
	inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
	inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
	inv[10] = m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
	inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
	inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
	inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
	inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
	inv[15] = m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
	float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
	if(det == 0.0f) return false;
	det = 1.0f/det;
	for(int i = 0; i < 16; i++) out[i] = inv[i]*det;
	return true;
}

static float
halton(int i, int b)
{
	float f = 1.0f, r = 0.0f;
	while(i > 0){ f /= b; r += f*(i % b); i /= b; }
	return r;
}

static uint32 currentFramebuffer;

static void
applyVelBuffers(void)
{
	if(!rsVelTex || currentFramebuffer != rsFbo) return;
	int want = gVelWant;
	if(want > 2 && (!rtWantsRefl() || !rsReflTex)) want = 2;
	if(want == 2 && !gTemporal) want = 1;		// motion vectors only matter for the temporal upscaler
	if(want == gVelBoth) return;
	static const GLenum bufs[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
	gVelBoth = want;
	glDrawBuffers(gVelBoth, bufs);
}

static uint32 blendMap[] = {
	GL_ZERO,	// actually invalid
	GL_ZERO,
	GL_ONE,
	GL_SRC_COLOR,
	GL_ONE_MINUS_SRC_COLOR,
	GL_SRC_ALPHA,
	GL_ONE_MINUS_SRC_ALPHA,
	GL_DST_ALPHA,
	GL_ONE_MINUS_DST_ALPHA,
	GL_DST_COLOR,
	GL_ONE_MINUS_DST_COLOR,
	GL_SRC_ALPHA_SATURATE,
};

static uint32 stencilOpMap[] = {
	GL_KEEP,	// actually invalid
	GL_KEEP,
	GL_ZERO,
	GL_REPLACE,
	GL_INCR,
	GL_DECR,
	GL_INVERT,
	GL_INCR_WRAP,
	GL_DECR_WRAP
};

static uint32 stencilFuncMap[] = {
	GL_NEVER,	// actually invalid
	GL_NEVER,
	GL_LESS,
	GL_EQUAL,
	GL_LEQUAL,
	GL_GREATER,
	GL_NOTEQUAL,
	GL_GEQUAL,
	GL_ALWAYS
};

static float maxAnisotropy;

/*
 * GL state cache
 */

void
setGlRenderState(uint32 state, uint32 value)
{
	switch(state){
	case RWGL_BLEND: curGlState.blendEnable = value; break;
	case RWGL_SRCBLEND: curGlState.srcblend = value; break;
	case RWGL_DESTBLEND: curGlState.destblend = value; break;
	case RWGL_DEPTHTEST: curGlState.depthTest = value; break;
	case RWGL_DEPTHFUNC: curGlState.depthFunc = value; break;
	case RWGL_DEPTHMASK: curGlState.depthMask = value; break;
	case RWGL_CULL: curGlState.cullEnable = value; break;
	case RWGL_CULLFACE: curGlState.cullFace = value; break;
	case RWGL_STENCIL: curGlState.stencilEnable = value; break;
	case RWGL_STENCILFUNC: curGlState.stencilFunc = value; break;
	case RWGL_STENCILFAIL: curGlState.stencilFail = value; break;
	case RWGL_STENCILZFAIL: curGlState.stencilZFail = value; break;
	case RWGL_STENCILPASS: curGlState.stencilPass = value; break;
	case RWGL_STENCILREF: curGlState.stencilRef = value; break;
	case RWGL_STENCILMASK: curGlState.stencilMask = value; break;
	case RWGL_STENCILWRITEMASK: curGlState.stencilWriteMask = value; break;
	}
}

void
flushGlRenderState(void)
{
	if(oldGlState.blendEnable != curGlState.blendEnable){
		oldGlState.blendEnable = curGlState.blendEnable;
		(oldGlState.blendEnable ? glEnable : glDisable)(GL_BLEND);
		if(oldGlState.blendEnable && gNeedVel){	// motion vectors and reflection directions are never blended
			glDisablei(GL_BLEND, 1);
			if(rsReflTex) glDisablei(GL_BLEND, 2);
		}
	}

	if(oldGlState.srcblend != curGlState.srcblend ||
	   oldGlState.destblend != curGlState.destblend){
		oldGlState.srcblend = curGlState.srcblend;
		oldGlState.destblend = curGlState.destblend;
		glBlendFunc(oldGlState.srcblend, oldGlState.destblend);
	}

	if(oldGlState.depthTest != curGlState.depthTest){
		oldGlState.depthTest = curGlState.depthTest;
		(oldGlState.depthTest ? glEnable : glDisable)(GL_DEPTH_TEST);
	}
	if(oldGlState.depthFunc != curGlState.depthFunc){
		oldGlState.depthFunc = curGlState.depthFunc;
		glDepthFunc(oldGlState.depthFunc);
	}
	if(oldGlState.depthMask != curGlState.depthMask){
		oldGlState.depthMask = curGlState.depthMask;
		glDepthMask(oldGlState.depthMask);
	}

	if(oldGlState.stencilEnable != curGlState.stencilEnable){
		oldGlState.stencilEnable = curGlState.stencilEnable;
		(oldGlState.stencilEnable ? glEnable : glDisable)(GL_STENCIL_TEST);
	}
	if(oldGlState.stencilFunc != curGlState.stencilFunc ||
	   oldGlState.stencilRef != curGlState.stencilRef ||
	   oldGlState.stencilMask != curGlState.stencilMask){
		oldGlState.stencilFunc = curGlState.stencilFunc;
		oldGlState.stencilRef = curGlState.stencilRef;
		oldGlState.stencilMask = curGlState.stencilMask;
		glStencilFunc(oldGlState.stencilFunc, oldGlState.stencilRef, oldGlState.stencilMask);
	}
	if(oldGlState.stencilPass != curGlState.stencilPass ||
	   oldGlState.stencilFail != curGlState.stencilFail ||
	   oldGlState.stencilZFail != curGlState.stencilZFail){
		oldGlState.stencilPass = curGlState.stencilPass;
		oldGlState.stencilFail = curGlState.stencilFail;
		oldGlState.stencilZFail = curGlState.stencilZFail;
		glStencilOp(oldGlState.stencilFail, oldGlState.stencilZFail, oldGlState.stencilPass);
	}
	if(oldGlState.stencilWriteMask != curGlState.stencilWriteMask){
		oldGlState.stencilWriteMask = curGlState.stencilWriteMask;
		glStencilMask(oldGlState.stencilWriteMask);
	}

	if(oldGlState.cullEnable != curGlState.cullEnable){
		oldGlState.cullEnable = curGlState.cullEnable;
		(oldGlState.cullEnable ? glEnable : glDisable)(GL_CULL_FACE);
	}
	if(oldGlState.cullFace != curGlState.cullFace){
		oldGlState.cullFace = curGlState.cullFace;
		glCullFace(oldGlState.cullFace);
	}
}



void
setAlphaBlend(bool32 enable)
{
	if(rwStateCache.blendEnable != enable){
		rwStateCache.blendEnable = enable;
		setGlRenderState(RWGL_BLEND, enable);
	}
}

bool32
getAlphaBlend(void)
{
	return rwStateCache.blendEnable;
}

bool32 getAlphaTest(void) { return rwStateCache.alphaTestEnable; }

static void
setDepthTest(bool32 enable)
{
	if(rwStateCache.ztest != enable){
		rwStateCache.ztest = enable;
		if(rwStateCache.zwrite && !enable){
			// If we still want to write, enable but set mode to always
			setGlRenderState(RWGL_DEPTHTEST, true);
			setGlRenderState(RWGL_DEPTHFUNC, GL_ALWAYS);
		}else{
			setGlRenderState(RWGL_DEPTHTEST, rwStateCache.ztest);
			setGlRenderState(RWGL_DEPTHFUNC, GL_LEQUAL);
		}
	}
}

static void
setDepthWrite(bool32 enable)
{
	enable = enable ? GL_TRUE : GL_FALSE;
	if(rwStateCache.zwrite != enable){
		rwStateCache.zwrite = enable;
		if(enable && !rwStateCache.ztest){
			// Have to switch on ztest so writing can work
			setGlRenderState(RWGL_DEPTHTEST, true);
			setGlRenderState(RWGL_DEPTHFUNC, GL_ALWAYS);
		}
		setGlRenderState(RWGL_DEPTHMASK, rwStateCache.zwrite);
	}
}

static void
setAlphaTest(bool32 enable)
{
	uint32 shaderfunc;
	if(rwStateCache.alphaTestEnable != enable){
		rwStateCache.alphaTestEnable = enable;
		shaderfunc = rwStateCache.alphaTestEnable ? rwStateCache.alphaFunc : ALPHAALWAYS;
		if(alphaFunc != shaderfunc){
			alphaFunc = shaderfunc;
			uniformStateDirty[RWGL_ALPHAFUNC] = true;
			stateDirty = 1;
		}
	}
}

static void
setAlphaTestFunction(uint32 function)
{
	uint32 shaderfunc;
	if(rwStateCache.alphaFunc != function){
		rwStateCache.alphaFunc = function;
		shaderfunc = rwStateCache.alphaTestEnable ? rwStateCache.alphaFunc : ALPHAALWAYS;
		if(alphaFunc != shaderfunc){
			alphaFunc = shaderfunc;
			uniformStateDirty[RWGL_ALPHAFUNC] = true;
			stateDirty = 1;
		}
	}
}

static void
setVertexAlpha(bool32 enable)
{
	if(rwStateCache.vertexAlpha != enable){
		if(!rwStateCache.textureAlpha){
			setAlphaBlend(enable);
			setAlphaTest(enable);
		}
		rwStateCache.vertexAlpha = enable;
	}
}

static void
setActiveTexture(int32 n)
{
	if(activeTexture != n){
		activeTexture = n;
		glActiveTexture(GL_TEXTURE0+n);
	}
}

uint32
bindTexture(uint32 texid)
{
	uint32 prev = boundTexture[activeTexture];
	if(prev != texid){
		boundTexture[activeTexture] = texid;
		glBindTexture(GL_TEXTURE_2D, texid);
	}
	return prev;
}

void
bindFramebuffer(uint32 fbo)
{
	if(currentFramebuffer != fbo){
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		currentFramebuffer = fbo;
		applyVelBuffers();
	}
}

static GLint filterConvMap_NoMIP[] = {
	0, GL_NEAREST, GL_LINEAR,
	   GL_NEAREST, GL_LINEAR,
	   GL_NEAREST, GL_LINEAR
};
static GLint filterConvMap_MIP[] = {
	0, GL_NEAREST, GL_LINEAR,
	   GL_NEAREST_MIPMAP_NEAREST, GL_LINEAR_MIPMAP_NEAREST,
	   GL_NEAREST_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_LINEAR
};

static GLint addressConvMap[] = {
	0, GL_REPEAT, GL_MIRRORED_REPEAT,
	GL_CLAMP_TO_EDGE, GL_CLAMP_TO_BORDER
};

static void
setFilterMode(uint32 stage, int32 filter, int32 maxAniso = 1)
{
	if(rwStateCache.texstage[stage].filter != (Texture::FilterMode)filter){
		rwStateCache.texstage[stage].filter = (Texture::FilterMode)filter;
		Raster *raster = rwStateCache.texstage[stage].raster;
		if(raster){
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, rwStateCache.texstage[stage].raster, nativeRasterOffset);
			if(natras->filterMode != filter){
				setActiveTexture(stage);
				if(natras->autogenMipmap || natras->numLevels > 1){
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_MIP[filter]);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterConvMap_NoMIP[filter]);
				}else{
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_NoMIP[filter]);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterConvMap_NoMIP[filter]);
				}
				natras->filterMode = filter;
			}
			if(natras->maxAnisotropy != maxAniso){
				setActiveTexture(stage);
				glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, (float)maxAniso);
				natras->maxAnisotropy = maxAniso;
			}
		}
	}
}

static void
setAddressU(uint32 stage, int32 addressing)
{
	if(rwStateCache.texstage[stage].addressingU != (Texture::Addressing)addressing){
		rwStateCache.texstage[stage].addressingU = (Texture::Addressing)addressing;
		Raster *raster = rwStateCache.texstage[stage].raster;
		if(raster){
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, raster, nativeRasterOffset);
			if(natras->addressU == addressing){
				setActiveTexture(stage);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, addressConvMap[addressing]);
				natras->addressU = addressing;
			}
		}
	}
}

static void
setAddressV(uint32 stage, int32 addressing)
{
	if(rwStateCache.texstage[stage].addressingV != (Texture::Addressing)addressing){
		rwStateCache.texstage[stage].addressingV = (Texture::Addressing)addressing;
		Raster *raster = rwStateCache.texstage[stage].raster;
		if(raster){
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, rwStateCache.texstage[stage].raster, nativeRasterOffset);
			if(natras->addressV == addressing){
				setActiveTexture(stage);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, addressConvMap[addressing]);
				natras->addressV = addressing;
			}
		}
	}
}

static void
setRasterStageOnly(uint32 stage, Raster *raster)
{
	bool32 alpha;
	if(raster != rwStateCache.texstage[stage].raster){
		rwStateCache.texstage[stage].raster = raster;
		setActiveTexture(stage);
		if(raster){
			assert(raster->platform == PLATFORM_GL3);
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, raster, nativeRasterOffset);
			bindTexture(natras->texid);

			rwStateCache.texstage[stage].filter = (rw::Texture::FilterMode)natras->filterMode;
			rwStateCache.texstage[stage].addressingU = (rw::Texture::Addressing)natras->addressU;
			rwStateCache.texstage[stage].addressingV = (rw::Texture::Addressing)natras->addressV;

			alpha = natras->hasAlpha;
		}else{
			bindTexture(whitetex);
			alpha = 0;
		}

		if(stage == 0){
			if(alpha != rwStateCache.textureAlpha){
				rwStateCache.textureAlpha = alpha;
				if(!rwStateCache.vertexAlpha){
					setAlphaBlend(alpha);
					setAlphaTest(alpha);
				}
			}
		}
	}
}

static void
setRasterStage(uint32 stage, Raster *raster)
{
	bool32 alpha;
	if(raster != rwStateCache.texstage[stage].raster){
		rwStateCache.texstage[stage].raster = raster;
		setActiveTexture(stage);
		if(raster){
			assert(raster->platform == PLATFORM_GL3);
			Gl3Raster *natras = PLUGINOFFSET(Gl3Raster, raster, nativeRasterOffset);
			bindTexture(natras->texid);
			uint32 filter = rwStateCache.texstage[stage].filter;
			uint32 addrU = rwStateCache.texstage[stage].addressingU;
			uint32 addrV = rwStateCache.texstage[stage].addressingV;
			if(natras->filterMode != filter){
				if(natras->autogenMipmap || natras->numLevels > 1){
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_MIP[filter]);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterConvMap_NoMIP[filter]);
				}else{
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterConvMap_NoMIP[filter]);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterConvMap_NoMIP[filter]);
				}
				natras->filterMode = filter;
			}
			if(natras->addressU != addrU){
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, addressConvMap[addrU]);
				natras->addressU = addrU;
			}
			if(natras->addressV != addrV){
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, addressConvMap[addrV]);
				natras->addressV = addrV;
			}
			alpha = natras->hasAlpha;
		}else{
			bindTexture(whitetex);
			alpha = 0;
		}

		if(stage == 0){
			if(alpha != rwStateCache.textureAlpha){
				rwStateCache.textureAlpha = alpha;
				if(!rwStateCache.vertexAlpha){
					setAlphaBlend(alpha);
					setAlphaTest(alpha);
				}
			}
		}
	}
}

void
evictRaster(Raster *raster)
{
	int i;
	for(i = 0; i < MAXNUMSTAGES; i++){
		//assert(rwStateCache.texstage[i].raster != raster);
		if(rwStateCache.texstage[i].raster != raster)
			continue;
		setRasterStage(i, nil);
	}
}

void
setTexture(int32 stage, Texture *tex)
{
	if(tex == nil || tex->raster == nil){
		setRasterStage(stage, nil);
		return;
	}
	setRasterStageOnly(stage, tex->raster);
	setFilterMode(stage, tex->getFilter(), tex->getMaxAnisotropy());
	setAddressU(stage, tex->getAddressU());
	setAddressV(stage, tex->getAddressV());
}

static void
setRenderState(int32 state, void *pvalue)
{
	uint32 value = (uint32)(uintptr)pvalue;
	switch(state){
	case TEXTURERASTER:
		setRasterStage(0, (Raster*)pvalue);
		break;
	case TEXTUREADDRESS:
		setAddressU(0, value);
		setAddressV(0, value);
		break;
	case TEXTUREADDRESSU:
		setAddressU(0, value);
		break;
	case TEXTUREADDRESSV:
		setAddressV(0, value);
		break;
	case TEXTUREFILTER:
		setFilterMode(0, value);
		break;
	case VERTEXALPHA:
		setVertexAlpha(value);
		break;
	case SRCBLEND:
		if(rwStateCache.srcblend != value){
			rwStateCache.srcblend = value;
			setGlRenderState(RWGL_SRCBLEND, blendMap[rwStateCache.srcblend]);
		}
		break;
	case DESTBLEND:
		if(rwStateCache.destblend != value){
			rwStateCache.destblend = value;
			setGlRenderState(RWGL_DESTBLEND, blendMap[rwStateCache.destblend]);
		}
		break;
	case ZTESTENABLE:
		setDepthTest(value);
		break;
	case ZWRITEENABLE:
		setDepthWrite(value);
		break;
	case FOGENABLE:
		if(rwStateCache.fogEnable != value){
			rwStateCache.fogEnable = value;
			uniformStateDirty[RWGL_FOG] = true;
			stateDirty = 1;
		}
		break;
	case FOGCOLOR:
		// no cache check here...too lazy
		RGBA c;
		c.red = value;
		c.green = value>>8;
		c.blue = value>>16;
		c.alpha = value>>24;
		convColor(&uniformState.fogColor, &c);
		uniformStateDirty[RWGL_FOGCOLOR] = true;
		stateDirty = 1;
		break;
	case CULLMODE:
		if(rwStateCache.cullmode != value){
			rwStateCache.cullmode = value;
			if(rwStateCache.cullmode == CULLNONE)
				setGlRenderState(RWGL_CULL, false);
			else{
				setGlRenderState(RWGL_CULL, true);
				setGlRenderState(RWGL_CULLFACE, rwStateCache.cullmode == CULLBACK ? GL_BACK : GL_FRONT);
			}
		}
		break;

	case STENCILENABLE:
		if(rwStateCache.stencilenable != value){
			rwStateCache.stencilenable = value;
			setGlRenderState(RWGL_STENCIL, value);
		}
		break;
	case STENCILFAIL:
		if(rwStateCache.stencilfail != value){
			rwStateCache.stencilfail = value;
			setGlRenderState(RWGL_STENCILFAIL, stencilOpMap[value]);
		}
		break;
	case STENCILZFAIL:
		if(rwStateCache.stencilzfail != value){
			rwStateCache.stencilzfail = value;
			setGlRenderState(RWGL_STENCILZFAIL, stencilOpMap[value]);
		}
		break;
	case STENCILPASS:
		if(rwStateCache.stencilpass != value){
			rwStateCache.stencilpass = value;
			setGlRenderState(RWGL_STENCILPASS, stencilOpMap[value]);
		}
		break;
	case STENCILFUNCTION:
		if(rwStateCache.stencilfunc != value){
			rwStateCache.stencilfunc = value;
			setGlRenderState(RWGL_STENCILFUNC, stencilFuncMap[value]);
		}
		break;
	case STENCILFUNCTIONREF:
		if(rwStateCache.stencilref != value){
			rwStateCache.stencilref = value;
			setGlRenderState(RWGL_STENCILREF, value);
		}
		break;
	case STENCILFUNCTIONMASK:
		if(rwStateCache.stencilmask != value){
			rwStateCache.stencilmask = value;
			setGlRenderState(RWGL_STENCILMASK, value);
		}
		break;
	case STENCILFUNCTIONWRITEMASK:
		if(rwStateCache.stencilwritemask != value){
			rwStateCache.stencilwritemask = value;
			setGlRenderState(RWGL_STENCILWRITEMASK, value);
		}
		break;

	case ALPHATESTFUNC:
		setAlphaTestFunction(value);
		break;
	case ALPHATESTREF:
		if(alphaRef != value/255.0f){
			alphaRef = value/255.0f;
			uniformStateDirty[RWGL_ALPHAREF] = true;
			stateDirty = 1;
		}
		break;
	case GSALPHATEST:
		rwStateCache.gsalpha = value;
		break;
	case GSALPHATESTREF:
		rwStateCache.gsalpharef = value;
	}
}

static void*
getRenderState(int32 state)
{
	uint32 val;
	RGBA rgba;
	switch(state){
	case TEXTURERASTER:
		return rwStateCache.texstage[0].raster;
	case TEXTUREADDRESS:
		if(rwStateCache.texstage[0].addressingU == rwStateCache.texstage[0].addressingV)
			val = rwStateCache.texstage[0].addressingU;
		else
			val = 0;	// invalid
		break;
	case TEXTUREADDRESSU:
		val = rwStateCache.texstage[0].addressingU;
		break;
	case TEXTUREADDRESSV:
		val = rwStateCache.texstage[0].addressingV;
		break;
	case TEXTUREFILTER:
		val = rwStateCache.texstage[0].filter;
		break;

	case VERTEXALPHA:
		val = rwStateCache.vertexAlpha;
		break;
	case SRCBLEND:
		val = rwStateCache.srcblend;
		break;
	case DESTBLEND:
		val = rwStateCache.destblend;
		break;
	case ZTESTENABLE:
		val = rwStateCache.ztest;
		break;
	case ZWRITEENABLE:
		val = rwStateCache.zwrite;
		break;
	case FOGENABLE:
		val = rwStateCache.fogEnable;
		break;
	case FOGCOLOR:
		convColor(&rgba, &uniformState.fogColor);
		val = RWRGBAINT(rgba.red, rgba.green, rgba.blue, rgba.alpha);
		break;
	case CULLMODE:
		val = rwStateCache.cullmode;
		break;

	case STENCILENABLE:
		val = rwStateCache.stencilenable;
		break;
	case STENCILFAIL:
		val = rwStateCache.stencilfail;
		break;
	case STENCILZFAIL:
		val = rwStateCache.stencilzfail;
		break;
	case STENCILPASS:
		val = rwStateCache.stencilpass;
		break;
	case STENCILFUNCTION:
		val = rwStateCache.stencilfunc;
		break;
	case STENCILFUNCTIONREF:
		val = rwStateCache.stencilref;
		break;
	case STENCILFUNCTIONMASK:
		val = rwStateCache.stencilmask;
		break;
	case STENCILFUNCTIONWRITEMASK:
		val = rwStateCache.stencilwritemask;
		break;

	case ALPHATESTFUNC:
		val = rwStateCache.alphaFunc;
		break;
	case ALPHATESTREF:
		val = (uint32)(alphaRef*255.0f);
		break;
	case GSALPHATEST:
		val = rwStateCache.gsalpha;
		break;
	case GSALPHATESTREF:
		val = rwStateCache.gsalpharef;
		break;
	default:
		val = 0;
	}
	return (void*)(uintptr)val;
}

static void
resetRenderState(void)
{	
	rwStateCache.alphaFunc = ALPHAGREATEREQUAL;
	alphaFunc = 0;
	alphaRef = 10.0f/255.0f;
	uniformStateDirty[RWGL_ALPHAREF] = true;
	uniformState.fogDisable = 1.0f;
	uniformState.fogStart = 0.0f;
	uniformState.fogEnd = 0.0f;
	uniformState.fogRange = 0.0f;
	uniformState.fogColor = { 1.0f, 1.0f, 1.0f, 1.0f };
	rwStateCache.gsalpha = 0;
	rwStateCache.gsalpharef = 128;
	stateDirty = 1;

	rwStateCache.vertexAlpha = 0;
	rwStateCache.textureAlpha = 0;
	rwStateCache.alphaTestEnable = 0;

	memset(&oldGlState, 0xFE, sizeof(oldGlState));

	rwStateCache.blendEnable = 0;
	setGlRenderState(RWGL_BLEND, false);
	rwStateCache.srcblend = BLENDSRCALPHA;
	rwStateCache.destblend = BLENDINVSRCALPHA;
	setGlRenderState(RWGL_SRCBLEND, blendMap[rwStateCache.srcblend]);
	setGlRenderState(RWGL_DESTBLEND, blendMap[rwStateCache.destblend]);

	rwStateCache.zwrite = GL_TRUE;
	setGlRenderState(RWGL_DEPTHMASK, rwStateCache.zwrite);

	rwStateCache.ztest = 1;
	setGlRenderState(RWGL_DEPTHTEST, true);
	setGlRenderState(RWGL_DEPTHFUNC, GL_LEQUAL);

	rwStateCache.cullmode = CULLNONE;
	setGlRenderState(RWGL_CULL, false);
	setGlRenderState(RWGL_CULLFACE, GL_BACK);

	rwStateCache.stencilenable = 0;
	setGlRenderState(RWGL_STENCIL, GL_FALSE);
	rwStateCache.stencilfail = STENCILKEEP;
	setGlRenderState(RWGL_STENCILFAIL, GL_KEEP);
	rwStateCache.stencilzfail = STENCILKEEP;
	setGlRenderState(RWGL_STENCILZFAIL, GL_KEEP);
	rwStateCache.stencilpass = STENCILKEEP;
	setGlRenderState(RWGL_STENCILPASS, GL_KEEP);
	rwStateCache.stencilfunc = STENCILALWAYS;
	setGlRenderState(RWGL_STENCILFUNC, GL_ALWAYS);
	rwStateCache.stencilref = 0;
	setGlRenderState(RWGL_STENCILREF, 0);
	rwStateCache.stencilmask = 0xFFFFFFFF;
	setGlRenderState(RWGL_STENCILMASK, 0xFFFFFFFF);
	rwStateCache.stencilwritemask = 0xFFFFFFFF;
	setGlRenderState(RWGL_STENCILWRITEMASK, 0xFFFFFFFF);

	activeTexture = -1;
	for(int i = 0; i < MAXNUMSTAGES; i++){
		setActiveTexture(i);
		bindTexture(whitetex);
	}
	setActiveTexture(0);
}

void
setWorldMatrix(Matrix *mat)
{
	convMatrix(&uniformObject.world, mat);
	setUniform(u_world, &uniformObject.world);
	objectDirty = 1;

	// previous-frame world matrix of moving objects (for motion vectors)
	gVelObj++;
	// everything static can get wet: the shader picks the upward facing grey surfaces (roads, pavements)
	gVelRoadWet = (!regtaVelocityDynamic && gRTRoadsOn) ? gRoadWet*gRTRoadStrength : 0.0f;
	if((gTemporal || gRT) && gUseTarget && regtaVelocityDynamic){
		if(gVelMap.size() > 30000) gVelMap.clear();
		VelEntry &e = gVelMap[mat];
		float *w = (float*)&uniformObject.world;
		if(e.frame != gFrameIdx){
			if(e.frame == gFrameIdx-1) memcpy(e.prev, e.cur, sizeof(e.prev));
			else memcpy(e.prev, w, sizeof(e.prev));
			e.frame = gFrameIdx;
		}
		memcpy(e.cur, w, sizeof(e.cur));
		memcpy(gVelPrevWorld, e.prev, sizeof(gVelPrevWorld));
		gVelOn = 1.0f;
	}else{
		memcpy(gVelPrevWorld, &uniformObject.world, sizeof(gVelPrevWorld));
		gVelOn = 0.0f;
	}
}

void
velocityShaderSelected(bool v, bool refl)
{
	gVelWant = refl ? 3 : v ? 2 : 1;
	applyVelBuffers();
}

void
uploadVelocityUniforms(Shader *s)
{
	if(s->velLoc[0] == -2){
		s->velLoc[0] = glGetUniformLocation(s->program, "u_vpCur");
		s->velLoc[1] = glGetUniformLocation(s->program, "u_vpPrev");
		s->velLoc[2] = glGetUniformLocation(s->program, "u_prevWorld");
		s->velLoc[3] = glGetUniformLocation(s->program, "u_velOn");
		s->roadLoc = glGetUniformLocation(s->program, "u_roadWet");
		s->puddleLoc = glGetUniformLocation(s->program, "u_roadPuddles");
	}
	if(s->velFrame != gFrameIdx || s->velScene != gSceneSerial){
		if(s->velLoc[0] >= 0) glUniformMatrix4fv(s->velLoc[0], 1, GL_FALSE, gCurVP);
		if(s->velLoc[1] >= 0) glUniformMatrix4fv(s->velLoc[1], 1, GL_FALSE, gPrevVP);
		s->velFrame = gFrameIdx;
		s->velScene = gSceneSerial;
	}
	if(s->velObj != gVelObj){
		if(s->velLoc[2] >= 0) glUniformMatrix4fv(s->velLoc[2], 1, GL_FALSE, gVelPrevWorld);
		if(s->velLoc[3] >= 0) glUniform1f(s->velLoc[3], gVelOn);
		if(s->roadLoc >= 0) glUniform1f(s->roadLoc, gVelRoadWet);
		if(s->puddleLoc >= 0) glUniform1f(s->puddleLoc, regtaGfxPuddles ? 1.0f : 0.0f);
		s->velObj = gVelObj;
	}
}

int32
setLights(WorldLights *lightData)
{
	int i, n;
	Light *l;
	int32 bits;

	uniformObject.ambLight = lightData->ambient;

	bits = 0;

	if(lightData->numAmbients)
		bits |= VSLIGHT_AMBIENT;

	n = 0;
	for(i = 0; i < lightData->numDirectionals && i < 8; i++){
		l = lightData->directionals[i];
		uniformObject.lightParams[n].type = 1.0f;
		uniformObject.lightColor[n] = l->color;
		memcpy(&uniformObject.lightDirection[n], &l->getFrame()->getLTM()->at, sizeof(V3d));
		bits |= VSLIGHT_DIRECT;
		n++;
		if(n >= MAX_LIGHTS)
			goto out;
	}

	for(i = 0; i < lightData->numLocals; i++){
		Light *l = lightData->locals[i];

		switch(l->getType()){
		case Light::POINT:
			uniformObject.lightParams[n].type = 2.0f;
			uniformObject.lightParams[n].radius = l->radius;
			uniformObject.lightColor[n] = l->color;
			memcpy(&uniformObject.lightPosition[n], &l->getFrame()->getLTM()->pos, sizeof(V3d));
			bits |= VSLIGHT_POINT;
			n++;
			if(n >= MAX_LIGHTS)
				goto out;
			break;
		case Light::SPOT:
		case Light::SOFTSPOT:
			uniformObject.lightParams[n].type = 3.0f;
			uniformObject.lightParams[n].minusCosAngle = l->minusCosAngle;
			uniformObject.lightParams[n].radius = l->radius;
			uniformObject.lightColor[n] = l->color;
			memcpy(&uniformObject.lightPosition[n], &l->getFrame()->getLTM()->pos, sizeof(V3d));
			memcpy(&uniformObject.lightDirection[n], &l->getFrame()->getLTM()->at, sizeof(V3d));
			// lower bound of falloff
			if(l->getType() == Light::SOFTSPOT)
				uniformObject.lightParams[n].hardSpot = 0.0f;
			else
				uniformObject.lightParams[n].hardSpot = 1.0f;
			bits |= VSLIGHT_SPOT;
			n++;
			if(n >= MAX_LIGHTS)
				goto out;
			break;
		}
	}

	uniformObject.lightParams[n].type = 0.0f;

	setUniform(u_ambLight, &uniformObject.ambLight);
	setUniform(u_lightParams, uniformObject.lightParams);
	setUniform(u_lightPosition, uniformObject.lightPosition);
	setUniform(u_lightDirection, uniformObject.lightDirection);
	setUniform(u_lightColor, uniformObject.lightColor);
out:
	objectDirty = 1;
	return bits;
}

void
setProjectionMatrix(float32 *mat)
{
	memcpy(&uniformScene.proj, mat, 64);
	setUniform(u_proj, uniformScene.proj);
	sceneDirty = 1;
}

void
setViewMatrix(float32 *mat)
{
	memcpy(&uniformScene.view, mat, 64);
	setUniform(u_view, uniformScene.view);
	sceneDirty = 1;
}

Shader *lastShaderUploaded;

void
setMaterial(const RGBA &color, const SurfaceProperties &surfaceprops, float extraSurfProp)
{
	rw::RGBAf col;
	convColor(&col, &color);
	setUniform(u_matColor, &col);

	float surfProps[4];
	surfProps[0] = surfaceprops.ambient;
	surfProps[1] = surfaceprops.specular;
	surfProps[2] = surfaceprops.diffuse;
	surfProps[3] = extraSurfProp;
	setUniform(u_surfProps, surfProps);
}

void
flushCache(void)
{
	flushGlRenderState();

#ifndef RW_GL_USE_UBOS

	// what's this doing here??
	uniformState.fogDisable = rwStateCache.fogEnable ? 0.0f : 1.0f;
	uniformState.fogStart = rwStateCache.fogStart;
	uniformState.fogEnd = rwStateCache.fogEnd;
	uniformState.fogRange = 1.0f/(rwStateCache.fogStart - rwStateCache.fogEnd);

	if(uniformStateDirty[RWGL_ALPHAFUNC] || uniformStateDirty[RWGL_ALPHAREF]){
		float alphaTest[4] = {};
		switch(alphaFunc){
		case ALPHAALWAYS:
		default:
			alphaTest[0] = -1000.0f;
			alphaTest[1] = 1000.0f;
			break;
		case ALPHAGREATEREQUAL:
			alphaTest[0] = alphaRef;
			alphaTest[1] = 1000.0f;
			break;
		case ALPHALESS:
			alphaTest[0] = -1000.0f;
			alphaTest[1] = alphaRef;
			break;
		}
		setUniform(u_alphaRef, alphaTest);
		uniformStateDirty[RWGL_ALPHAFUNC] = false;
		uniformStateDirty[RWGL_ALPHAREF] = false;
	}

	if(uniformStateDirty[RWGL_FOG] ||
	   uniformStateDirty[RWGL_FOGSTART] ||
	   uniformStateDirty[RWGL_FOGEND]){
		float fogData[4] = {
			uniformState.fogStart,
			uniformState.fogEnd,
			uniformState.fogRange,
			uniformState.fogDisable
		};
		setUniform(u_fogData, fogData);
		uniformStateDirty[RWGL_FOG] = false;
		uniformStateDirty[RWGL_FOGSTART] = false;
		uniformStateDirty[RWGL_FOGEND] = false;
	}

	if(uniformStateDirty[RWGL_FOGCOLOR]){
		setUniform(u_fogColor, &uniformState.fogColor);
		uniformStateDirty[RWGL_FOGCOLOR] = false;
	}

#else
	if(objectDirty){
		glBindBuffer(GL_UNIFORM_BUFFER, ubo_object);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformObject), nil, GL_STREAM_DRAW);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformObject), &uniformObject, GL_STREAM_DRAW);
		objectDirty = 0;
	}
	if(sceneDirty){
		glBindBuffer(GL_UNIFORM_BUFFER, ubo_scene);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformScene), nil, GL_STREAM_DRAW);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformScene), &uniformScene, GL_STREAM_DRAW);
		sceneDirty = 0;
	}
	if(stateDirty){
		switch(alphaFunc){
		case ALPHAALWAYS:
		default:
			uniformState.alphaRefLow = -1000.0f;
			uniformState.alphaRefHigh = 1000.0f;
			break;
		case ALPHAGREATEREQUAL:
			uniformState.alphaRefLow = alphaRef;
			uniformState.alphaRefHigh = 1000.0f;
			break;
		case ALPHALESS:
			uniformState.alphaRefLow = -1000.0f;
			uniformState.alphaRefHigh = alphaRef;
			break;
		}
		uniformState.fogDisable = rwStateCache.fogEnable ? 0.0f : 1.0f;
		uniformState.fogStart = rwStateCache.fogStart;
		uniformState.fogEnd = rwStateCache.fogEnd;
		uniformState.fogRange = 1.0f/(rwStateCache.fogStart - rwStateCache.fogEnd);
		glBindBuffer(GL_UNIFORM_BUFFER, ubo_state);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformState), nil, GL_STREAM_DRAW);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformState), &uniformState, GL_STREAM_DRAW);
		stateDirty = 0;
	}
#endif
	flushUniforms();
}

// ---- render scale: reduced-size colour target for the 3D scene, resolved by an FSR 1 style pass ----
static GLuint rsTex, rsUpFbo, rsUpTex;
static GLuint rsHistFbo[2], rsHistTex[2];
static int rsHistCur;
static int rsW, rsH, rsOutW, rsOutH;
static Camera *rsCamera;

static void
rsMakeTex(GLuint *fbo, GLuint *tex, int w, int h)
{
	if(*tex == 0) glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_2D, *tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nil);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	if(*fbo == 0) glGenFramebuffers(1, fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *tex, 0);
}

// (re)creates the targets when the window size changed; leaves bindings to the caller
static void
rsEnsureTargets(Raster *fbuf)
{
	float sc = getHidpiScale();
	int w = (int)(fbuf->width*sc + .5f), h = (int)(fbuf->height*sc + .5f);
	int fw = 0, fh = 0;
	glfwGetFramebufferSize(glGlobals.window, &fw, &fh);
	if(rsFbo && w == rsW && h == rsH && fw == rsOutW && fh == rsOutH) return;
	GLint prevTex, prevFbo;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex);
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
	rsMakeTex(&rsFbo, &rsTex, w, h);
	rsMakeTex(&rsUpFbo, &rsUpTex, fw, fh);
	if(gNeedVel){
		if(rsVelTex == 0) glGenTextures(1, &rsVelTex);
		glBindTexture(GL_TEXTURE_2D, rsVelTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_HALF_FLOAT, nil);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindFramebuffer(GL_FRAMEBUFFER, rsFbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, rsVelTex, 0);
		if(gNeedVel){
			if(rsReflTex == 0) glGenTextures(1, &rsReflTex);
			glBindTexture(GL_TEXTURE_2D, rsReflTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_HALF_FLOAT, nil);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, rsReflTex, 0);
		}
		gVelBoth = 1;
		{
			// new framebuffer object: only attachment 0 is drawn to until a shader asks for more
			static const GLenum b0[1] = { GL_COLOR_ATTACHMENT0 };
			glDrawBuffers(1, b0);
		}
	}
	if(gNeedVel){
		for(int i = 0; i < 2; i++){
			if(rsHistTex[i] == 0) glGenTextures(1, &rsHistTex[i]);
			glBindTexture(GL_TEXTURE_2D, rsHistTex[i]);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, fw, fh, 0, GL_RGBA, GL_HALF_FLOAT, nil);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			if(rsHistFbo[i] == 0) glGenFramebuffers(1, &rsHistFbo[i]);
			glBindFramebuffer(GL_FRAMEBUFFER, rsHistFbo[i]);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rsHistTex[i], 0);
		}
		gHistValid = false;
	}
	glBindTexture(GL_TEXTURE_2D, prevTex);
	glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
	rsW = w; rsH = h; rsOutW = fw; rsOutH = fh;
}

static void
setFrameBuffer(Camera *cam)
{
	Raster *fbuf = cam->frameBuffer->parent;
	Raster *zbuf = cam->zBuffer->parent;
	assert(fbuf);

	Gl3Raster *natfb = PLUGINOFFSET(Gl3Raster, fbuf, nativeRasterOffset);
	Gl3Raster *natzb = PLUGINOFFSET(Gl3Raster, zbuf, nativeRasterOffset);
	assert(fbuf->type == Raster::CAMERA || fbuf->type == Raster::CAMERATEXTURE);

	if(gUseTarget && fbuf->type == Raster::CAMERA){
		rsCamera = cam;
		if(gScaledFrame){
			rsEnsureTargets(fbuf);
			natfb->fbo = rsFbo;
		}else
			natfb->fbo = 0;
	}

	// Have to make sure depth buffer is attached to FB's fbo
	bindFramebuffer(natfb->fbo);
	if(zbuf){
		if(natfb->fboMate == zbuf){
			// all good
			assert(natzb->fboMate == fbuf);
		}else{
			if(natzb->fboMate){
				// have to detatch from fbo first!
				Gl3Raster *oldfb = PLUGINOFFSET(Gl3Raster, natzb->fboMate, nativeRasterOffset);
				if(oldfb->fbo){
					bindFramebuffer(oldfb->fbo);
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
					bindFramebuffer(natfb->fbo);
				}
				oldfb->fboMate = nil;
			}
			natfb->fboMate = zbuf;
			natzb->fboMate = fbuf;
			if(natfb->fbo){
				if(gl3Caps.gles)
					glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, natzb->texid);
				else
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, natzb->texid, 0);
			}
		}
	}else{
		// remove z-buffer
		if(natfb->fboMate && natfb->fbo)
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
		natfb->fboMate = nil;
	}
}

static float
nativeHidpiScale(void)
{
#ifdef LIBRW_GLFW
	if(glGlobals.window){
		int fw = 0, fh = 0, ww = 0, wh = 0;
		glfwGetFramebufferSize(glGlobals.window, &fw, &fh);
		glfwGetWindowSize(glGlobals.window, &ww, &wh);
		if(ww > 0 && fw > ww) return (float)fw / (float)ww;
	}
#endif
	return 1.0f;
}

float
getHidpiScale(void)
{
	return nativeHidpiScale() * gRenderScale;
}

float
hidpiScaleFor(int32 w, int32 h)
{
	float s = getHidpiScale();
	if(s == 1.0f) return 1.0f;
#ifdef LIBRW_GLFW
	int ww = 0, wh = 0;
	glfwGetWindowSize(glGlobals.window, &ww, &wh);
	if(w >= ww / 2 && h >= wh / 2) return s;
#endif
	return 1.0f;
}

static Rect
getFramebufferRect(Raster *frameBuffer)
{
	Rect r;
	Raster *fb = frameBuffer->parent;
	if(fb->type == Raster::CAMERA){
#ifdef LIBRW_SDL2
		SDL_GetWindowSize(glGlobals.window, &r.w, &r.h);
#elif defined(LIBRW_SDL3)
		SDL_GetWindowSize(glGlobals.window, &r.w, &r.h);
#elif defined(LIBRW_GLFW)
		glfwGetFramebufferSize(glGlobals.window, &r.w, &r.h);
		if(gUseTarget && gScaledFrame){
			r.w = (int32)(r.w * gRenderScale + 0.5f);
			r.h = (int32)(r.h * gRenderScale + 0.5f);
		}
#else
		missing implementation
#endif
	}else{
		float tsc = PLUGINOFFSET(Gl3Raster, fb, nativeRasterOffset)->scale;
		if(tsc <= 0.0f) tsc = 1.0f;
		r.w = (int32)(fb->width * tsc + 0.5f);
		r.h = (int32)(fb->height * tsc + 0.5f);
	}
	r.x = 0;
	r.y = 0;

	// Got a subraster
	if(frameBuffer != fb){
		// raster sizes are in window points, the default framebuffer is in pixels
		// (they differ on HiDPI screens), so scale the viewport up
		float sx = 1.0f, sy = 1.0f;
		if(fb->type != Raster::CAMERA){
			float tsc = PLUGINOFFSET(Gl3Raster, fb, nativeRasterOffset)->scale;
			if(tsc > 0.0f) sx = sy = tsc;
		}
#ifdef LIBRW_GLFW
		if(fb->type == Raster::CAMERA){
			int ww, wh;
			glfwGetWindowSize(glGlobals.window, &ww, &wh);
			if(ww > 0 && wh > 0){
				sx = (float)r.w / ww;
				sy = (float)r.h / wh;
			}
		}
#endif
		r.x = (int32)(frameBuffer->offsetX * sx + 0.5f);
		int32 h = (int32)(frameBuffer->height * sy + 0.5f);
		// GL y offset is from bottom
		r.y = r.h - h - (int32)(frameBuffer->offsetY * sy + 0.5f);
		r.w = (int32)(frameBuffer->width * sx + 0.5f);
		r.h = h;
	}

	return r;
}

static void
setViewport(Raster *frameBuffer)
{
	Rect r = getFramebufferRect(frameBuffer);
	if(r.w != glGlobals.presentWidth || r.h != glGlobals.presentHeight ||
	   r.x != glGlobals.presentOffX || r.y != glGlobals.presentOffY){
		glViewport(r.x, r.y, r.w, r.h);
		glGlobals.presentWidth = r.w;
		glGlobals.presentHeight = r.h;
		glGlobals.presentOffX = r.x;
		glGlobals.presentOffY = r.y;
	}
}

// FSR 1 style upscaler: edge adaptive spatial upsampling (12 taps, directional Lanczos-like kernel)
// followed by robust contrast adaptive sharpening
static const char *rsVertSrc =
"#version 330\n"
"out vec2 uv;\n"
"void main(){\n"
"	vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
"	uv = p;\n"
"	gl_Position = vec4(p*2.0-1.0, 0.0, 1.0);\n"
"}\n";

static const char *rsEasuSrc =
"#version 330\n"
"uniform sampler2D tex;\n"
"uniform vec2 inSize;\n"
"uniform vec2 outSize;\n"
"in vec2 uv;\n"
"out vec4 fragColor;\n"
"vec3 fetch(ivec2 p){ return texelFetch(tex, clamp(p, ivec2(0), ivec2(inSize)-1), 0).rgb; }\n"
"float lum(vec3 c){ return c.b*0.5 + (c.r*0.5 + c.g); }\n"
"void easuSet(inout vec2 dir, inout float len, vec2 pp, bool biS, bool biT, bool biU, bool biV,\n"
"             float lA, float lB, float lC, float lD, float lE){\n"
"	float w = 0.0;\n"
"	if(biS) w = (1.0-pp.x)*(1.0-pp.y);\n"
"	if(biT) w = pp.x*(1.0-pp.y);\n"
"	if(biU) w = (1.0-pp.x)*pp.y;\n"
"	if(biV) w = pp.x*pp.y;\n"
"	float dc = lD-lC, cb = lC-lB;\n"
"	float lenX = max(abs(dc), abs(cb));\n"
"	lenX = lenX > 0.0 ? 1.0/lenX : 0.0;\n"
"	float dirX = lD-lB;\n"
"	dir.x += dirX*w;\n"
"	lenX = clamp(abs(dirX)*lenX, 0.0, 1.0);\n"
"	lenX *= lenX;\n"
"	len += lenX*w;\n"
"	float ec = lE-lC, ca = lC-lA;\n"
"	float lenY = max(abs(ec), abs(ca));\n"
"	lenY = lenY > 0.0 ? 1.0/lenY : 0.0;\n"
"	float dirY = lE-lA;\n"
"	dir.y += dirY*w;\n"
"	lenY = clamp(abs(dirY)*lenY, 0.0, 1.0);\n"
"	lenY *= lenY;\n"
"	len += lenY*w;\n"
"}\n"
"void easuTap(inout vec3 aC, inout float aW, vec2 off, vec2 dir, vec2 len, float lob, float clp, vec3 c){\n"
"	vec2 v = vec2(off.x*dir.x + off.y*dir.y, off.x*(-dir.y) + off.y*dir.x);\n"
"	v *= len;\n"
"	float d2 = min(v.x*v.x + v.y*v.y, clp);\n"
"	float wB = 2.0/5.0*d2 - 1.0;\n"
"	float wA = lob*d2 - 1.0;\n"
"	wB *= wB; wA *= wA;\n"
"	wB = 25.0/16.0*wB - (25.0/16.0 - 1.0);\n"
"	float w = wB*wA;\n"
"	aC += c*w; aW += w;\n"
"}\n"
"void main(){\n"
"	vec2 outPix = uv*outSize;\n"
"	vec2 pp = outPix*(inSize/outSize) - 0.5;\n"
"	vec2 fp = floor(pp);\n"
"	pp -= fp;\n"
"	ivec2 ip = ivec2(fp);\n"
"	vec3 b = fetch(ip+ivec2(0,-1)), c = fetch(ip+ivec2(1,-1));\n"
"	vec3 e = fetch(ip+ivec2(-1,0)), f = fetch(ip), g = fetch(ip+ivec2(1,0)), h = fetch(ip+ivec2(2,0));\n"
"	vec3 i = fetch(ip+ivec2(-1,1)), j = fetch(ip+ivec2(0,1)), k = fetch(ip+ivec2(1,1)), l = fetch(ip+ivec2(2,1));\n"
"	vec3 n = fetch(ip+ivec2(0,2)), o = fetch(ip+ivec2(1,2));\n"
"	float bL=lum(b), cL=lum(c), eL=lum(e), fL=lum(f), gL=lum(g), hL=lum(h);\n"
"	float iL=lum(i), jL=lum(j), kL=lum(k), lL=lum(l), nL=lum(n), oL=lum(o);\n"
"	vec2 dir = vec2(0.0); float len = 0.0;\n"
"	easuSet(dir, len, pp, true,false,false,false, bL,eL,fL,gL,jL);\n"
"	easuSet(dir, len, pp, false,true,false,false, cL,fL,gL,hL,kL);\n"
"	easuSet(dir, len, pp, false,false,true,false, fL,iL,jL,kL,nL);\n"
"	easuSet(dir, len, pp, false,false,false,true, gL,jL,kL,lL,oL);\n"
"	float dir2 = dir.x*dir.x + dir.y*dir.y;\n"
"	bool zro = dir2 < (1.0/32768.0);\n"
"	dir2 = inversesqrt(max(dir2, 1.0/32768.0));\n"
"	dir2 = zro ? 1.0 : dir2;\n"
"	dir.x = zro ? 1.0 : dir.x;\n"
"	dir *= dir2;\n"
"	len = len*0.5; len *= len;\n"
"	float stretch = (dir.x*dir.x + dir.y*dir.y) / max(abs(dir.x), abs(dir.y));\n"
"	vec2 len2 = vec2(1.0 + (stretch-1.0)*len, 1.0 - 0.5*len);\n"
"	float lob = 0.5 + ((1.0/4.0 - 0.04) - 0.5)*len;\n"
"	float clp = 1.0/lob;\n"
"	vec3 aC = vec3(0.0); float aW = 0.0;\n"
"	easuTap(aC,aW, vec2( 0,-1)-pp, dir,len2,lob,clp, b);\n"
"	easuTap(aC,aW, vec2( 1,-1)-pp, dir,len2,lob,clp, c);\n"
"	easuTap(aC,aW, vec2(-1, 1)-pp, dir,len2,lob,clp, i);\n"
"	easuTap(aC,aW, vec2( 0, 1)-pp, dir,len2,lob,clp, j);\n"
"	easuTap(aC,aW, vec2( 0, 0)-pp, dir,len2,lob,clp, f);\n"
"	easuTap(aC,aW, vec2(-1, 0)-pp, dir,len2,lob,clp, e);\n"
"	easuTap(aC,aW, vec2( 1, 1)-pp, dir,len2,lob,clp, k);\n"
"	easuTap(aC,aW, vec2( 2, 1)-pp, dir,len2,lob,clp, l);\n"
"	easuTap(aC,aW, vec2( 2, 0)-pp, dir,len2,lob,clp, h);\n"
"	easuTap(aC,aW, vec2( 1, 0)-pp, dir,len2,lob,clp, g);\n"
"	easuTap(aC,aW, vec2( 1, 2)-pp, dir,len2,lob,clp, o);\n"
"	easuTap(aC,aW, vec2( 0, 2)-pp, dir,len2,lob,clp, n);\n"
"	vec3 res = aC/aW;\n"
"	vec3 mn = min(min(f,g), min(j,k)), mx = max(max(f,g), max(j,k));\n"
"	fragColor = vec4(clamp(res, mn, mx), 1.0);\n"
"}\n";

static const char *rsRcasSrc =
"#version 330\n"
"uniform sampler2D tex;\n"
"uniform float sharp;\n"
"in vec2 uv;\n"
"out vec4 fragColor;\n"
"void main(){\n"
"	ivec2 sz = textureSize(tex, 0);\n"
"	ivec2 p = ivec2(uv*vec2(sz));\n"
"	vec3 b = texelFetch(tex, clamp(p+ivec2(0,-1), ivec2(0), sz-1), 0).rgb;\n"
"	vec3 d = texelFetch(tex, clamp(p+ivec2(-1,0), ivec2(0), sz-1), 0).rgb;\n"
"	vec3 e = texelFetch(tex, clamp(p, ivec2(0), sz-1), 0).rgb;\n"
"	vec3 f = texelFetch(tex, clamp(p+ivec2(1,0), ivec2(0), sz-1), 0).rgb;\n"
"	vec3 h = texelFetch(tex, clamp(p+ivec2(0,1), ivec2(0), sz-1), 0).rgb;\n"
"	float bL = b.b*0.5 + (b.r*0.5 + b.g), dL = d.b*0.5 + (d.r*0.5 + d.g), eL = e.b*0.5 + (e.r*0.5 + e.g);\n"
"	float fL = f.b*0.5 + (f.r*0.5 + f.g), hL = h.b*0.5 + (h.r*0.5 + h.g);\n"
"	float nz = 0.25*(bL+dL+fL+hL) - eL;\n"
"	nz = clamp(abs(nz) / (max(max(bL,dL), max(eL, max(fL,hL))) - min(min(bL,dL), min(eL, min(fL,hL))) + 1e-5), 0.0, 1.0);\n"
"	nz = -0.5*nz + 1.0;\n"
"	vec3 mn4 = min(min(b,d), min(f,h)), mx4 = max(max(b,d), max(f,h));\n"
"	vec3 hitMin = min(mn4, e) / (4.0*mx4 + 1e-5);\n"
"	vec3 hitMax = (vec3(1.0) - max(mx4, e)) / (4.0*mn4 - vec3(4.0) + 1e-5);\n"
"	vec3 lobeRGB = max(-hitMin, hitMax);\n"
"	float lobe = max(-0.1875, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * exp2(-sharp);\n"
"	lobe *= nz;\n"
"	float rcpL = 1.0/(4.0*lobe + 1.0);\n"
"	fragColor = vec4((lobe*(b+d+f+h) + e)*rcpL, 1.0);\n"
"}\n";


static const char *rsTemporalSrc =
"#version 330\n"
"uniform sampler2D colorTex;\n"
"uniform sampler2D depthTex;\n"
"uniform sampler2D histTex;\n"
"uniform sampler2D velTex;\n"
"uniform vec2 lowSize;\n"
"uniform vec2 outSize;\n"
"uniform vec2 jitter;\n"
"uniform mat4 invVP;\n"
"uniform mat4 prevVP;\n"
"uniform float histValid;\n"
"out vec4 fragColor;\n"
"vec3 fetchC(ivec2 p){ return texelFetch(colorTex, clamp(p, ivec2(0), ivec2(lowSize)-1), 0).rgb; }\n"
"float fetchD(ivec2 p){ return texelFetch(depthTex, clamp(p, ivec2(0), ivec2(lowSize)-1), 0).r; }\n"
"vec3 sampleHist(vec2 uv){\n"
"	// Catmull-Rom through 5 bilinear taps\n"
"	vec2 pos = uv*outSize;\n"
"	vec2 c = floor(pos - 0.5) + 0.5;\n"
"	vec2 f = pos - c;\n"
"	vec2 w0 = f*(-0.5 + f*(1.0 - 0.5*f));\n"
"	vec2 w1 = 1.0 + f*f*(-2.5 + 1.5*f);\n"
"	vec2 w2 = f*(0.5 + f*(2.0 - 1.5*f));\n"
"	vec2 w3 = f*f*(-0.5 + 0.5*f);\n"
"	vec2 w12 = w1 + w2;\n"
"	vec2 t0 = (c - 1.0)/outSize, t3 = (c + 2.0)/outSize, t12 = (c + w2/w12)/outSize;\n"
"	vec3 r = texture(histTex, vec2(t12.x, t0.y)).rgb*(w12.x*w0.y)\n"
"	       + texture(histTex, vec2(t0.x, t12.y)).rgb*(w0.x*w12.y)\n"
"	       + texture(histTex, vec2(t12.x, t12.y)).rgb*(w12.x*w12.y)\n"
"	       + texture(histTex, vec2(t3.x, t12.y)).rgb*(w3.x*w12.y)\n"
"	       + texture(histTex, vec2(t12.x, t3.y)).rgb*(w12.x*w3.y);\n"
"	return max(r, vec3(0.0));\n"
"}\n"
"void main(){\n"
"	vec2 outUV = gl_FragCoord.xy/outSize;\n"
"	vec2 q = outUV*lowSize + jitter;\n"
"	ivec2 ci = ivec2(floor(q));\n"
"	vec3 sum = vec3(0.0), mn = vec3(1e4), mx = vec3(-1e4);\n"
"	float wsum = 0.0, dmin = 1.0;\n"
"	ivec2 dTex = ci;\n"
"	for(int y = -1; y <= 1; y++)\n"
"	for(int x = -1; x <= 1; x++){\n"
"		ivec2 t = ci + ivec2(x, y);\n"
"		vec3 c = fetchC(t);\n"
"		vec2 d = vec2(t) + 0.5 - q;\n"
"		float w = exp(-2.3*dot(d, d));\n"
"		sum += c*w; wsum += w;\n"
"		mn = min(mn, c); mx = max(mx, c);\n"
"		float dd = fetchD(t);\n"
"		if(dd < dmin){ dmin = dd; dTex = t; }\n"
"	}\n"
"	vec3 cur = sum/wsum;\n"
"	vec4 vel = texelFetch(velTex, clamp(dTex, ivec2(0), ivec2(lowSize)-1), 0);\n"
"	vec2 puv;\n"
"	bool ok;\n"
"	if(vel.a > 0.5){\n"
"		// moving object: its own motion vector\n"
"		puv = outUV - vel.xy;\n"
"		ok = histValid > 0.5;\n"
"	}else{\n"
"		vec4 wp = invVP*vec4(outUV*2.0-1.0, dmin*2.0-1.0, 1.0);\n"
"		vec4 pc = prevVP*vec4(wp.xyz/wp.w, 1.0);\n"
"		puv = pc.xy/pc.w*0.5 + 0.5;\n"
"		ok = histValid > 0.5 && pc.w > 0.0;\n"
"	}\n"
"	ok = ok && all(greaterThan(puv, vec2(0.0))) && all(lessThan(puv, vec2(1.0)));\n"
"	vec3 res = cur;\n"
"	if(ok){\n"
"		vec3 h = sampleHist(puv);\n"
"		vec3 hc = clamp(h, mn - 0.01, mx + 0.01);\n"
"		float a = clamp(0.1 + length(h - hc)*3.0, 0.1, 1.0);\n"
"		res = mix(hc, cur, a);\n"
"	}\n"
"	fragColor = vec4(res, 1.0);\n"
"}\n";

static GLuint rsEasuProg, rsRcasProg, rsVao, rsTemporalProg;
static float gRenderSharpness = 0.2f;	// RCAS sharpening in stops (0 = max, 2 = none)

static GLuint
rsCompile(const char *vs, const char *fs)
{
	GLuint prog = glCreateProgram();
	GLuint sh[2];
	const char *src[2] = { vs, fs };
	GLenum ty[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
	for(int i = 0; i < 2; i++){
		sh[i] = glCreateShader(ty[i]);
		glShaderSource(sh[i], 1, &src[i], nil);
		glCompileShader(sh[i]);
		GLint ok = 0;
		glGetShaderiv(sh[i], GL_COMPILE_STATUS, &ok);
		if(!ok){
			char log[1024];
			glGetShaderInfoLog(sh[i], sizeof(log), nil, log);
			fprintf(stderr, "render scale shader error: %s\n", log);
			return 0;
		}
		glAttachShader(prog, sh[i]);
	}
	glLinkProgram(prog);
	GLint ok = 0;
	glGetProgramiv(prog, GL_LINK_STATUS, &ok);
	if(!ok){
		fprintf(stderr, "render scale shader link failed\n");
		return 0;
	}
	glDeleteShader(sh[0]);
	glDeleteShader(sh[1]);
	return prog;
}


// ---- MetalFX temporal scaler (macOS): GL converts the scene into shared IOSurface textures, Metal upscales ----
extern "C" {
int mfxSetup(int inW, int inH, int outW, int outH, unsigned *glTex);
int mfxRun(float jitterX, float jitterY, int reset);
void mfxShutdown(void);
}

static const char *rsMfxPrepSrc =
"#version 330\n"
"uniform sampler2D colorTex;\n"
"uniform sampler2D depthTex;\n"
"uniform sampler2D velTex;\n"
"uniform vec2 lowSize;\n"
"uniform mat4 invVP;\n"
"uniform mat4 prevVP;\n"
"layout(location = 0) out vec4 oColor;\n"
"layout(location = 1) out vec4 oDepth;\n"
"layout(location = 2) out vec4 oMotion;\n"
"void main(){\n"
"	ivec2 p = ivec2(gl_FragCoord.xy);\n"
"	oColor = vec4(texelFetch(colorTex, p, 0).rgb, 1.0);\n"
"	float d = texelFetch(depthTex, p, 0).r;\n"
"	oDepth = vec4(d, 0.0, 0.0, 1.0);\n"
"	vec2 uv = gl_FragCoord.xy/lowSize;\n"
"	vec4 vel = texelFetch(velTex, p, 0);\n"
"	vec2 v;\n"
"	if(vel.a > 0.5){\n"
"		v = vel.xy;\n"
"	}else{\n"
"		vec4 wp = invVP*vec4(uv*2.0-1.0, d*2.0-1.0, 1.0);\n"
"		vec4 pc = prevVP*vec4(wp.xyz/wp.w, 1.0);\n"
"		v = uv - (pc.xy/pc.w*0.5 + 0.5);\n"
"	}\n"
"	// MetalFX: pixels from the current position to the previous one (rows run the other way in the shared memory)\n"
"	oMotion = vec4(-v.x*lowSize.x, -v.y*lowSize.y, 0.0, 1.0);\n"
"}\n";

static const char *rsMfxCopySrc =
"#version 330\n"
"uniform sampler2DRect src;\n"
"out vec4 fragColor;\n"
"void main(){\n"
"	fragColor = vec4(texelFetch(src, ivec2(gl_FragCoord.xy)).rgb, 1.0);\n"
"}\n";

static GLuint rsMfxPrepProg, rsMfxCopyProg, rsMfxFbo;
static unsigned rsMfxTex[4];
static int rsMfxIn[2], rsMfxOut[2];
static bool rsMfxReady, rsMfxFailed;

static void
rsMfxLog(const char *msg)
{
	fprintf(stderr, "MetalFX: %s\n", msg);
	if(FILE *f = fopen("metalfx.log", "a")){
		fprintf(f, "%s\n", msg);
		fclose(f);
	}
}

static bool
rsMfxEnsure(void)
{
	if(rsMfxFailed) return false;
	if(rsMfxReady && rsMfxIn[0] == rsW && rsMfxIn[1] == rsH && rsMfxOut[0] == rsOutW && rsMfxOut[1] == rsOutH)
		return true;
	rsMfxReady = false;
	if(!rsMfxPrepProg){
		rsMfxPrepProg = rsCompile(rsVertSrc, rsMfxPrepSrc);
		rsMfxCopyProg = rsCompile(rsVertSrc, rsMfxCopySrc);
		if(!rsMfxPrepProg || !rsMfxCopyProg){ rsMfxLog("shader compile failed, using the GLSL resolve"); rsMfxFailed = true; return false; }
	}
	if(!mfxSetup(rsW, rsH, rsOutW, rsOutH, rsMfxTex)){
		rsMfxLog("setup failed, using the GLSL resolve");
		rsMfxFailed = true;
		return false;
	}
	GLint prevFbo;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
	if(!rsMfxFbo) glGenFramebuffers(1, &rsMfxFbo);
	glBindFramebuffer(GL_FRAMEBUFFER, rsMfxFbo);
	for(int i = 0; i < 3; i++)
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_RECTANGLE, rsMfxTex[i], 0);
	static const GLenum bufs[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(3, bufs);
	GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
	if(st != GL_FRAMEBUFFER_COMPLETE){
		char b[96];
		snprintf(b, sizeof(b), "input framebuffer incomplete (0x%x), using the GLSL resolve", st);
		rsMfxLog(b);
		mfxShutdown();
		rsMfxFailed = true;
		return false;
	}
	rsMfxIn[0] = rsW; rsMfxIn[1] = rsH; rsMfxOut[0] = rsOutW; rsMfxOut[1] = rsOutH;
	rsMfxReady = true;
	return true;
}

// Runs the scaler for this frame and leaves the result in the history texture rsHistTex[rsHistCur].
static bool
rsMetalFXPass(void)
{
	if(!rsMfxEnsure()) return false;
	Gl3Raster *natzb = PLUGINOFFSET(Gl3Raster, rsCamera->zBuffer->parent, nativeRasterOffset);
	GLint prevTex0, prevTex1, prevTex3, prevRect0;
	glActiveTexture(GL_TEXTURE0); glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0); glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &prevRect0);
	glActiveTexture(GL_TEXTURE1); glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex1);
	glActiveTexture(GL_TEXTURE3); glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex3);

	// convert: colour, depth and motion vectors into the shared textures
	glBindFramebuffer(GL_FRAMEBUFFER, rsMfxFbo);
	glViewport(0, 0, rsW, rsH);
	glUseProgram(rsMfxPrepProg);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, natzb->texid);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, rsVelTex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rsTex);
	glUniform1i(glGetUniformLocation(rsMfxPrepProg, "colorTex"), 0);
	glUniform1i(glGetUniformLocation(rsMfxPrepProg, "depthTex"), 1);
	glUniform1i(glGetUniformLocation(rsMfxPrepProg, "velTex"), 3);
	glUniform2f(glGetUniformLocation(rsMfxPrepProg, "lowSize"), (float)rsW, (float)rsH);
	glUniformMatrix4fv(glGetUniformLocation(rsMfxPrepProg, "invVP"), 1, GL_FALSE, gInvVP);
	glUniformMatrix4fv(glGetUniformLocation(rsMfxPrepProg, "prevVP"), 1, GL_FALSE, gPrevVP);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glFinish();	// the shared textures must be complete before Metal reads them

	bool ok = mfxRun(gJit[0]*gMfxJitterSignX, gJit[1]*gMfxJitterSignY, gHistValid ? 0 : 1) != 0;
	if(ok){
		// result -> history texture (so the sharpening pass and the next frame find it where they expect it)
		int cur = rsHistCur ^ 1;
		glBindFramebuffer(GL_FRAMEBUFFER, rsHistFbo[cur]);
		glViewport(0, 0, rsOutW, rsOutH);
		glUseProgram(rsMfxCopyProg);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_RECTANGLE, rsMfxTex[3]);
		glUniform1i(glGetUniformLocation(rsMfxCopyProg, "src"), 0);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		rsHistCur = cur;
	}else{
		rsMfxLog("run failed, using the GLSL resolve");
		rsMfxFailed = true;
		mfxShutdown();
		rsMfxReady = false;
	}
	glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, prevTex3);
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, prevTex1);
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, prevTex0); glBindTexture(GL_TEXTURE_RECTANGLE, prevRect0);
	return ok;
}


// ---- ray traced reflections (Metal ray tracing through shared IOSurface textures) ----
extern "C" {
struct RtEnv {
	float skyTop[3], skyHorizon[3], ambient[3], sunDirection[3], sunColor[3];
	float maxDistance, fogDistance;
	float toSun[3];
	float sunVisibility;
	float shadowDistance;
	int shadows;
};
int mrtSetup(int width, int height, unsigned *glTex);
void mrtShutdown(void);
int mrtReady(void);
int mrtAddGeometry(const float *positions, int numVerts, int numMeshes, const uint32_t *const *indices, const int *indexCount, const float *colours, const int *meshCasts);
void mrtFrameBegin(void);
void mrtAddInstance(int geometry, const float *world, uintptr_t key);
int mrtTrace(const float *invViewProj, const RtEnv *env);
}

static RtEnv gRTEnv = { {0.3f,0.5f,0.9f}, {0.7f,0.8f,0.9f}, {0.3f,0.3f,0.3f}, {0,0,-1}, {0.6f,0.6f,0.5f}, 250.0f, 200.0f, {0,0,1}, 0.0f, 120.0f, 0 };
static float gRTShadowStrength = 0.55f, gRTShadowDistance = 120.0f;

// called by the game every frame: unit vector to the sun and how strongly the sun lights the scene (0 at night)
extern "C" void
regtaRTSetSun(const float *toSun, float visibility)
{
	memcpy(gRTEnv.toSun, toSun, 12);
	gRTEnv.sunVisibility = visibility;
}

// called by the game every frame with the time-of-day colours (0..1)
extern "C" void
regtaRTSetEnvironment(const float *skyTop, const float *skyBottom, const float *ambient, const float *directional)
{
	memcpy(gRTEnv.skyTop, skyTop, 12);
	memcpy(gRTEnv.skyHorizon, skyBottom, 12);
	memcpy(gRTEnv.ambient, ambient, 12);
	memcpy(gRTEnv.sunColor, directional, 12);
}

struct RtGeom { int id; int numVerts; uint32 numIdx; };
static std::unordered_map<const void*, RtGeom> gRTGeom;
static std::unordered_map<uint32, std::array<float,3> > gRTTexColour;
static uint32 gRTLastFrame = 0xFFFFFFFFu;

// average colour of a texture (its smallest mip level)
static void
rtTextureColour(Texture *tex, float *rgb)
{
	rgb[0] = rgb[1] = rgb[2] = 0.5f;
	if(!tex || !tex->raster) return;
	Raster *ras = tex->raster;
	Gl3Raster *nat = PLUGINOFFSET(Gl3Raster, ras, nativeRasterOffset);
	if(!nat->texid) return;
	auto it = gRTTexColour.find(nat->texid);
	if(it != gRTTexColour.end()){ memcpy(rgb, it->second.data(), 12); return; }
	std::array<float,3> c = { 0.5f, 0.5f, 0.5f };
	if(nat->numLevels > 1){
		uint32 prev = bindTexture(nat->texid);
		int lvl = nat->numLevels - 1;
		GLint w = 0, h = 0;
		glGetTexLevelParameteriv(GL_TEXTURE_2D, lvl, GL_TEXTURE_WIDTH, &w);
		glGetTexLevelParameteriv(GL_TEXTURE_2D, lvl, GL_TEXTURE_HEIGHT, &h);
		if(w > 0 && h > 0 && w*h <= 64){
			uint8 px[64*4];
			glGetTexImage(GL_TEXTURE_2D, lvl, GL_RGBA, GL_UNSIGNED_BYTE, px);
			float sum[3] = { 0, 0, 0 };
			for(int i = 0; i < w*h; i++) for(int k = 0; k < 3; k++) sum[k] += px[i*4 + k];
			for(int k = 0; k < 3; k++) c[k] = sum[k]/(w*h*255.0f);
		}
		bindTexture(prev);
		if(glGetError() != GL_NO_ERROR) c = { 0.5f, 0.5f, 0.5f };
	}
	gRTTexColour[nat->texid] = c;
	memcpy(rgb, c.data(), 12);
}

// Called by the render callbacks for every atomic that is drawn: registers its geometry for ray tracing (once) and
// its placement (every frame). Static objects persist for a while after they leave the screen.
void
rtCollect(Atomic *atomic, InstanceDataHeader*)
{
	if(!rtActive() || !gScaledFrame || !rsCamera || engine->currentCamera != rsCamera) return;
	if(gRTLastFrame != (uint32)gFrameIdx){
		gRTLastFrame = (uint32)gFrameIdx;
		if(mrtReady()) mrtFrameBegin();
	}
	if(!mrtReady()) return;
	Geometry *g = atomic->geometry;
	if(!g || !g->meshHeader || g->numMorphTargets < 1 || !g->morphTargets[0].vertices || g->numVertices <= 0) return;
	if(Skin::get(g)) return;
	MeshHeader *mh = g->meshHeader;
	RtGeom &e = gRTGeom[g];
	if(e.id == 0 && e.numVerts == 0){ e.id = -3; }	// new entry
	if(e.id >= 0 && (e.numVerts != g->numVertices || e.numIdx != mh->totalIndices)) e.id = -3;
	if(e.id == -3 || e.id == -2){
		std::vector<std::vector<uint32_t> > lists(mh->numMeshes);
		std::vector<const uint32_t*> ptrs(mh->numMeshes);
		std::vector<int> counts(mh->numMeshes);
		std::vector<float> colours(mh->numMeshes*6);
		std::vector<int> meshCasts(mh->numMeshes, 1);	// alpha tested / blended meshes (leaves, fences, glass) would cast solid shadows
		bool groundLike = false;
		Mesh *meshes = mh->getMeshes();
		bool bad = false;
		for(uint32 m = 0; m < mh->numMeshes && !bad; m++){
			Mesh *me = &meshes[m];
			std::vector<uint32_t> &L = lists[m];
			if(mh->flags & MeshHeader::TRISTRIP){
				for(uint32 i = 2; i < me->numIndices; i++){
					uint32 a = me->indices[i-2], b = me->indices[i-1], c = me->indices[i];
					if(a == b || b == c || a == c) continue;
					if(i & 1) std::swap(a, b);
					L.push_back(a); L.push_back(b); L.push_back(c);
				}
			}else{
				for(uint32 i = 0; i + 2 < me->numIndices; i += 3){
					L.push_back(me->indices[i]); L.push_back(me->indices[i+1]); L.push_back(me->indices[i+2]);
				}
			}
			for(uint32_t v : L) if(v >= (uint32_t)g->numVertices){ bad = true; break; }
			ptrs[m] = L.data();
			counts[m] = (int)L.size();
			float tc[3];
			rtTextureColour(me->material ? me->material->texture : nil, tc);
			RGBA mc = me->material ? me->material->color : RGBA{255,255,255,255};
			if(mc.alpha < 255) meshCasts[m] = 0;
			if(me->material && me->material->texture && me->material->texture->raster &&
			   PLUGINOFFSET(Gl3Raster, me->material->texture->raster, nativeRasterOffset)->hasAlpha) meshCasts[m] = 0;
			colours[m*6+0] = tc[0]*mc.red/255.0f;
			colours[m*6+1] = tc[1]*mc.green/255.0f;
			colours[m*6+2] = tc[2]*mc.blue/255.0f;
			// average prelit vertex light of the mesh (baked day light of map models); non-prelit geometry is lit by the game's lights
			float pl[3] = { 0.4f, 0.4f, 0.4f };
			if((g->flags & Geometry::PRELIT) && g->colors && L.size() >= 3){
				double sum[3] = { 0, 0, 0 };
				for(uint32_t v : L){ sum[0] += g->colors[v].red; sum[1] += g->colors[v].green; sum[2] += g->colors[v].blue; }
				for(int k = 0; k < 3; k++) pl[k] = (float)(sum[k]/(L.size()*255.0));
			}
			colours[m*6+3] = pl[0]; colours[m*6+4] = pl[1]; colours[m*6+5] = pl[2];
		}
		if(!bad && !regtaVelocityDynamic){
			// ground-like meshes (roads, pavements, grass, terrain) must not cast shadows: they lie on or just under the surfaces
			// that receive them, so rays hit them by accident and leave large dark patches. Skip flat meshes and meshes that are
			// mostly horizontal, upward facing surface.
			const V3d *vert = g->morphTargets[0].vertices;
			float zmin = 1e30f, zmax = -1e30f, areaAll = 0.0f, areaUp = 0.0f;
			for(int v = 0; v < g->numVertices; v++){ if(vert[v].z < zmin) zmin = vert[v].z; if(vert[v].z > zmax) zmax = vert[v].z; }
			for(uint32 m = 0; m < mh->numMeshes; m++){
				const std::vector<uint32_t> &L = lists[m];
				for(size_t i = 0; i + 2 < L.size(); i += 3){
					V3d a = vert[L[i]], b = vert[L[i+1]], c = vert[L[i+2]];
					float e1x = b.x-a.x, e1y = b.y-a.y, e1z = b.z-a.z, e2x = c.x-a.x, e2y = c.y-a.y, e2z = c.z-a.z;
					float nx = e1y*e2z - e1z*e2y, ny = e1z*e2x - e1x*e2z, nz = e1x*e2y - e1y*e2x;
					float len = sqrtf(nx*nx + ny*ny + nz*nz);
					areaAll += len;
					if(len > 0.0f && fabsf(nz)/len > 0.7f) areaUp += len;
				}
			}
			if(zmax - zmin < 0.35f || (areaAll > 0.0f && areaUp/areaAll > 0.6f)) groundLike = true;
		}
		if(groundLike) for(auto &c : meshCasts) c = 0;
		if(bad){
			e.id = -1;
		}else{
			int id = mrtAddGeometry((const float*)g->morphTargets[0].vertices, g->numVertices, (int)mh->numMeshes, ptrs.data(), counts.data(), colours.data(), meshCasts.data());
			e.id = id;
			e.numVerts = g->numVertices;
			e.numIdx = mh->totalIndices;
		}
	}
	if(e.id >= 0)
		mrtAddInstance(e.id, (const float*)&uniformObject.world, regtaVelocityDynamic ? 0 : (uintptr_t)atomic->getFrame()->getLTM());
}

static const char *rsRTPrepSrc =
"#version 330\n"
"uniform sampler2D depthTex;\n"
"uniform sampler2D reflTex;\n"
"uniform int div;\n"
"layout(location = 0) out vec4 oDepth;\n"
"layout(location = 1) out vec4 oDir;\n"
"void main(){\n"
"	ivec2 hi = textureSize(depthTex, 0) - 1;\n"
"	ivec2 base = ivec2(gl_FragCoord.xy)*div;\n"
"	// one sample per div x div block: keep the pixel with the strongest reflection (thin marked parts must not vanish)\n"
"	vec4 best = vec4(0.0);\n"
"	ivec2 bp = clamp(base, ivec2(0), hi);\n"
"	for(int y = 0; y < div; y++)\n"
"	for(int x = 0; x < div; x++){\n"
"		ivec2 q = clamp(base + ivec2(x, y), ivec2(0), hi);\n"
"		vec4 d = texelFetch(reflTex, q, 0);\n"
"		if(d.w > best.w){ best = d; bp = q; }\n"
"	}\n"
"	oDepth = vec4(texelFetch(depthTex, bp, 0).r, 0.0, 0.0, 1.0);\n"
"	oDir = best;\n"
"}\n";

static const char *rsRTCompositeSrc =
"#version 330\n"
"uniform sampler2DRect rtTex;\n"
"uniform sampler2D reflTex;\n"
"uniform float strength;\n"
"uniform int div;\n"
"out vec4 fragColor;\n"
"void main(){\n"
"	ivec2 p = ivec2(gl_FragCoord.xy);\n"
"	float w = texelFetch(reflTex, p, 0).a;\n"
"	if(w <= 0.001) discard;\n"
"	// soften the traced image a little (ripples, and the reflected colours are only per-material averages)\n"
"	ivec2 hi = textureSize(rtTex) - 1;\n"
"	ivec2 c = clamp(p/div, ivec2(0), hi);\n"
"	int o = max(1, 2/div);\n"
"	vec4 r = texelFetch(rtTex, c)*0.36;\n"
"	r += texelFetch(rtTex, clamp(c + ivec2(o, 0), ivec2(0), hi))*0.16 + texelFetch(rtTex, clamp(c - ivec2(o, 0), ivec2(0), hi))*0.16;\n"
"	r += texelFetch(rtTex, clamp(c + ivec2(0, o), ivec2(0), hi))*0.16 + texelFetch(rtTex, clamp(c - ivec2(0, o), ivec2(0), hi))*0.16;\n"
"	fragColor = vec4(r.rgb, clamp(w*strength, 0.0, 1.0));\n"
"}\n";

static const char *rsRTShadowSrc =
"#version 330\n"
"uniform sampler2DRect shadowTex;\n"
"uniform sampler2DRect depthRect;\n"
"uniform sampler2D depthFull;\n"
"uniform float strength;\n"
"uniform vec2 nearFar;\n"
"uniform int div;\n"
"out vec4 fragColor;\n"
"float lin(float d){\n"
"	float z = d*2.0 - 1.0;\n"
"	return 2.0*nearFar.x*nearFar.y/(nearFar.y + nearFar.x - z*(nearFar.y - nearFar.x));\n"
"}\n"
"void main(){\n"
"	ivec2 p = ivec2(gl_FragCoord.xy);\n"
"	ivec2 hi = textureSize(depthRect) - 1;\n"
"	float dc = texelFetch(depthFull, p, 0).r;\n"
"	if(dc >= 0.99999) discard;\n"
"	float lc = lin(dc);\n"
"	// depth aware blur: the shadow rays are noisy (one ray per pixel), so smooth between pixels of the same surface\n"
"	float sum = 0.0, ws = 0.0;\n"
"	ivec2 c = p/div;\n"
"	float sigma = div > 1 ? 0.45 : 0.2;\n"
"	for(int y = -2; y <= 2; y++)\n"
"	for(int x = -2; x <= 2; x++){\n"
"		ivec2 q = clamp(c + ivec2(x, y), ivec2(0), hi);\n"
"		float lq = lin(texelFetch(depthRect, q).r);\n"
"		float w = exp(-float(x*x + y*y)*sigma)*exp(-abs(lq - lc)/(0.03*lc + 0.05));\n"
"		sum += texelFetch(shadowTex, q).r*w;\n"
"		ws += w;\n"
"	}\n"
"	float s = sum/max(ws, 1e-4);\n"
"	if(s <= 0.004) discard;\n"
"	fragColor = vec4(vec3(1.0 - s*strength), 1.0);\n"
"}\n";

static GLuint rsRTShadowProg;
static GLuint rsRTPrepProg, rsRTCompProg, rsRTFbo;
static unsigned rsRTTex[4];
static int rsRTSize[2];
static bool rsRTFailed;

static void
rsRTLog(const char *msg)
{
	fprintf(stderr, "RayTracing: %s\n", msg);
	if(FILE *f = fopen("metalfx.log", "a")){ fprintf(f, "%s\n", msg); fclose(f); }
}

// Traces reflections for the pixels the vehicle shader marked and blends the result into the scene colour.
static void
rsRayTracePass(void)
{
	if(!rtActive() || rsRTFailed || !rsReflTex || !rsFbo || !rsCamera || !rsCamera->zBuffer) return;
	if(!mrtReady() || rsRTSize[0] != rsW || rsRTSize[1] != rsH){
		if(!rsRTPrepProg){
			rsRTPrepProg = rsCompile(rsVertSrc, rsRTPrepSrc);
			rsRTCompProg = rsCompile(rsVertSrc, rsRTCompositeSrc);
			rsRTShadowProg = rsCompile(rsVertSrc, rsRTShadowSrc);
			if(!rsRTPrepProg || !rsRTCompProg || !rsRTShadowProg){ rsRTLog("shader compile failed, ray tracing off"); rsRTFailed = true; return; }
		}
		gRTGeom.clear();
		if(!mrtSetup((rsW + gRTDiv - 1)/gRTDiv, (rsH + gRTDiv - 1)/gRTDiv, rsRTTex)){ rsRTLog("setup failed, ray tracing off"); rsRTFailed = true; return; }
		GLint prevFbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
		if(!rsRTFbo) glGenFramebuffers(1, &rsRTFbo);
		glBindFramebuffer(GL_FRAMEBUFFER, rsRTFbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, rsRTTex[0], 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, rsRTTex[1], 0);
		static const GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
		glDrawBuffers(2, bufs);
		GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
		if(st != GL_FRAMEBUFFER_COMPLETE){ rsRTLog("input framebuffer incomplete, ray tracing off"); mrtShutdown(); rsRTFailed = true; return; }
		rsRTSize[0] = rsW; rsRTSize[1] = rsH;
		return;	// geometry is registered from now on; first traced frame is the next one
	}
	Gl3Raster *natzb = PLUGINOFFSET(Gl3Raster, rsCamera->zBuffer->parent, nativeRasterOffset);
	GLint prevTex0, prevTex1, prevRect0, prevRect1, prevTex2 = -1;
	glActiveTexture(GL_TEXTURE0); glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0); glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &prevRect0);
	glActiveTexture(GL_TEXTURE1); glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex1); glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &prevRect1);

	glBindFramebuffer(GL_FRAMEBUFFER, rsRTFbo);
	glViewport(0, 0, (rsW + gRTDiv - 1)/gRTDiv, (rsH + gRTDiv - 1)/gRTDiv);
	glUseProgram(rsRTPrepProg);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, natzb->texid);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, rsReflTex);
	glUniform1i(glGetUniformLocation(rsRTPrepProg, "depthTex"), 0);
	glUniform1i(glGetUniformLocation(rsRTPrepProg, "reflTex"), 1);
	glUniform1i(glGetUniformLocation(rsRTPrepProg, "div"), gRTDiv);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glFinish();

	gRTEnv.maxDistance = gRTDistance;
	gRTEnv.fogDistance = gRTDistance*0.8f;
	gRTEnv.shadowDistance = gRTShadowDistance;
	gRTEnv.shadows = gRTShadows ? 1 : 0;
	bool ok = mrtTrace(gInvVP, &gRTEnv) != 0;
	if(ok){
		// blend the traced colour over the paint where the vehicle shader marked a reflection
		glBindFramebuffer(GL_FRAMEBUFFER, rsFbo);
		static const GLenum b0[1] = { GL_COLOR_ATTACHMENT0 };
		glDrawBuffers(1, b0);
		gVelBoth = 1;
		glViewport(0, 0, rsW, rsH);
		glEnable(GL_BLEND);
		if(rtWantsRefl()){
			glUseProgram(rsRTCompProg);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_RECTANGLE, rsRTTex[2]);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, rsReflTex);
			glUniform1i(glGetUniformLocation(rsRTCompProg, "rtTex"), 0);
			glUniform1i(glGetUniformLocation(rsRTCompProg, "reflTex"), 1);
			glUniform1i(glGetUniformLocation(rsRTCompProg, "div"), gRTDiv);
			glUniform1f(glGetUniformLocation(rsRTCompProg, "strength"), gRTStrength);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glDrawArrays(GL_TRIANGLES, 0, 3);
		}
		if(gRTShadows && gRTEnv.sunVisibility > 0.01f){
			// darken the scene where the sun rays were blocked
			glUseProgram(rsRTShadowProg);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_RECTANGLE, rsRTTex[3]);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_RECTANGLE, rsRTTex[0]);
			glActiveTexture(GL_TEXTURE2);
			glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex2);
			glBindTexture(GL_TEXTURE_2D, natzb->texid);
			glActiveTexture(GL_TEXTURE0);
			glUniform1i(glGetUniformLocation(rsRTShadowProg, "depthFull"), 2);
			glUniform1i(glGetUniformLocation(rsRTShadowProg, "div"), gRTDiv);
			glUniform1i(glGetUniformLocation(rsRTShadowProg, "shadowTex"), 0);
			glUniform1i(glGetUniformLocation(rsRTShadowProg, "depthRect"), 1);
			glUniform2f(glGetUniformLocation(rsRTShadowProg, "nearFar"), rsCamera->nearPlane, rsCamera->farPlane);
			glUniform1f(glGetUniformLocation(rsRTShadowProg, "strength"), gRTShadowStrength*gRTEnv.sunVisibility);
			glBlendFunc(GL_ZERO, GL_SRC_COLOR);
			glDrawArrays(GL_TRIANGLES, 0, 3);
		}
		glDisable(GL_BLEND);
	}
	if(prevTex2 >= 0){ glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, prevTex2); }
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, prevTex1); glBindTexture(GL_TEXTURE_RECTANGLE, prevRect1);
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, prevTex0); glBindTexture(GL_TEXTURE_RECTANGLE, prevRect0);
}

// Upscales the reduced-size scene to the window. Called before the HUD is drawn (and by showRaster
// when nobody did), everything drawn afterwards goes to the window at native resolution.
static void
resolveRenderScale(void)
{
	if(!gUseTarget || !gScaledFrame || !rsFbo || !rsCamera) return;
	gScaledFrame = false;
	double resolveT0 = perfNow();
	if(!rsEasuProg){
		rsEasuProg = rsCompile(rsVertSrc, rsEasuSrc);
		rsRcasProg = rsCompile(rsVertSrc, rsRcasSrc);
		rsTemporalProg = rsCompile(rsVertSrc, rsTemporalSrc);
		glGenVertexArrays(1, &rsVao);
	}
	Gl3Raster *natfb = PLUGINOFFSET(Gl3Raster, rsCamera->frameBuffer->parent, nativeRasterOffset);
	natfb->fbo = 0;
	if(!rsEasuProg || !rsRcasProg){
		// shaders unusable: plain blit
		glBindFramebuffer(GL_READ_FRAMEBUFFER, rsFbo);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBlitFramebuffer(0, 0, rsW, rsH, 0, 0, rsOutW, rsOutH, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		bindFramebuffer(0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glGlobals.presentWidth = 0;
		setViewport(rsCamera->frameBuffer);
		return;
	}

	GLint prevVao, prevActive, prevTex0, prevProg;
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVao);
	glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActive);
	glActiveTexture(GL_TEXTURE0);
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex0);
	glGetIntegerv(GL_CURRENT_PROGRAM, &prevProg);
	GLboolean blend = glIsEnabled(GL_BLEND), depth = glIsEnabled(GL_DEPTH_TEST),
	          cull = glIsEnabled(GL_CULL_FACE), scissor = glIsEnabled(GL_SCISSOR_TEST);
	GLboolean cmask[4];
	glGetBooleanv(GL_COLOR_WRITEMASK, cmask);
	GLint blendSrcRGB, blendDstRGB, blendSrcA, blendDstA;	// the passes below change the blend function: the HUD must find the game's
	glGetIntegerv(GL_BLEND_SRC_RGB, &blendSrcRGB); glGetIntegerv(GL_BLEND_DST_RGB, &blendDstRGB);
	glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrcA); glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDstA);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_SCISSOR_TEST);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glBindVertexArray(rsVao);

	{
		double t0 = perfNow();
		rsRayTracePass();
		gPerfMs[0] += ((float)(perfNow() - t0) - gPerfMs[0])*0.05f;
	}

	GLuint sharpenSrc = rsUpTex;
	GLint prevTex1 = 0, prevTex2 = 0, prevTex3 = 0;
	bool temporal = gTemporal && gJitterApplied && rsTemporalProg && rsHistTex[0] && rsVelTex && rsCamera->zBuffer;
	double mfxT0 = perfNow();
	bool mfxDone = temporal && gMetalFX && rsMetalFXPass();
	gPerfMs[1] += ((mfxDone ? (float)(perfNow() - mfxT0) : 0.0f) - gPerfMs[1])*0.05f;
	if(mfxDone){
		gHistValid = true;
		sharpenSrc = rsHistTex[rsHistCur];
	}else if(temporal){
		// pass 1: jittered samples + reprojected history -> new history
		int cur = rsHistCur ^ 1;
		Gl3Raster *natzb = PLUGINOFFSET(Gl3Raster, rsCamera->zBuffer->parent, nativeRasterOffset);
		glBindFramebuffer(GL_FRAMEBUFFER, rsHistFbo[cur]);
		glViewport(0, 0, rsOutW, rsOutH);
		glUseProgram(rsTemporalProg);
		glActiveTexture(GL_TEXTURE1);
		glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex1);
		glBindTexture(GL_TEXTURE_2D, natzb->texid);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glActiveTexture(GL_TEXTURE2);
		glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex2);
		glBindTexture(GL_TEXTURE_2D, rsHistTex[rsHistCur]);
		glActiveTexture(GL_TEXTURE3);
		glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex3);
		glBindTexture(GL_TEXTURE_2D, rsVelTex);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, rsTex);
		glUniform1i(glGetUniformLocation(rsTemporalProg, "velTex"), 3);
		glUniform1i(glGetUniformLocation(rsTemporalProg, "colorTex"), 0);
		glUniform1i(glGetUniformLocation(rsTemporalProg, "depthTex"), 1);
		glUniform1i(glGetUniformLocation(rsTemporalProg, "histTex"), 2);
		glUniform2f(glGetUniformLocation(rsTemporalProg, "lowSize"), (float)rsW, (float)rsH);
		glUniform2f(glGetUniformLocation(rsTemporalProg, "outSize"), (float)rsOutW, (float)rsOutH);
		glUniform2f(glGetUniformLocation(rsTemporalProg, "jitter"), gJit[0], gJit[1]);
		glUniformMatrix4fv(glGetUniformLocation(rsTemporalProg, "invVP"), 1, GL_FALSE, gInvVP);
		glUniformMatrix4fv(glGetUniformLocation(rsTemporalProg, "prevVP"), 1, GL_FALSE, gPrevVP);
		glUniform1f(glGetUniformLocation(rsTemporalProg, "histValid"), gHistValid ? 1.0f : 0.0f);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, prevTex3);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, prevTex2);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, prevTex1);
		glActiveTexture(GL_TEXTURE0);
		rsHistCur = cur;
		gHistValid = true;
		sharpenSrc = rsHistTex[cur];
	}else{
		gHistValid = false;
		// pass 1: edge adaptive upsampling into the native-size texture
		glBindFramebuffer(GL_FRAMEBUFFER, rsUpFbo);
		glViewport(0, 0, rsOutW, rsOutH);
		glUseProgram(rsEasuProg);
		glBindTexture(GL_TEXTURE_2D, rsTex);
		glUniform1i(glGetUniformLocation(rsEasuProg, "tex"), 0);
		glUniform2f(glGetUniformLocation(rsEasuProg, "inSize"), (float)rsW, (float)rsH);
		glUniform2f(glGetUniformLocation(rsEasuProg, "outSize"), (float)rsOutW, (float)rsOutH);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	// pass 2: sharpening into the window
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, rsOutW, rsOutH);
	glUseProgram(rsRcasProg);
	glBindTexture(GL_TEXTURE_2D, sharpenSrc);
	glUniform1i(glGetUniformLocation(rsRcasProg, "tex"), 0);
	glUniform1f(glGetUniformLocation(rsRcasProg, "sharp"), gRenderSharpness);
	glDrawArrays(GL_TRIANGLES, 0, 3);

	glBindVertexArray(prevVao);
	glBindTexture(GL_TEXTURE_2D, prevTex0);
	glActiveTexture(prevActive);
	glUseProgram(prevProg);
	glBlendFuncSeparate(blendSrcRGB, blendDstRGB, blendSrcA, blendDstA);
	if(blend) glEnable(GL_BLEND);
	if(depth) glEnable(GL_DEPTH_TEST);
	if(cull) glEnable(GL_CULL_FACE);
	if(scissor) glEnable(GL_SCISSOR_TEST);
	glColorMask(cmask[0], cmask[1], cmask[2], cmask[3]);
	currentFramebuffer = 0;
	glGlobals.presentWidth = 0;
	setViewport(rsCamera->frameBuffer);
	gPerfMs[2] += ((float)(perfNow() - resolveT0) - gPerfMs[2])*0.05f;
}

extern "C" void regtaResolveRenderScale(void) { resolveRenderScale(); }


// ---- scene capture for screen space reflections ----
// A copy of the colour and depth rendered so far; taken the first time a vehicle is drawn in a frame.
static GLuint capFbo, capColor, capDepth;
static int capW, capH;
static bool capTaken;
static int capCount;	// captures taken this frame (one per vehicle, so cars can reflect each other)
static const int capMaxPerFrame = 8;

// Called after a vehicle has been drawn: the next vehicle takes a fresh copy that includes it.
extern "C" void
regtaSceneCaptureInvalidate(void)
{
	if(capCount < capMaxPerFrame)
		capTaken = false;
}

static void
capMakeTargets(int w, int h)
{
	GLint prevTex;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex);
	if(!capFbo){
		glGenFramebuffers(1, &capFbo);
		glGenTextures(1, &capColor);
		glGenTextures(1, &capDepth);
	}
	glBindTexture(GL_TEXTURE_2D, capColor);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nil);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, capDepth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, w, h, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, nil);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, prevTex);
	glBindFramebuffer(GL_FRAMEBUFFER, capFbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, capColor, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, capDepth, 0);
	capW = w; capH = h;
}

// Returns 1 and fills the info when a copy of the main scene is available.
extern "C" int
regtaSceneCapture(unsigned *colorTex, unsigned *depthTex, float *texSize, float *viewport, float *proj, float *view)
{
	if(!gSSR || !gUseTarget || !rsFbo || !rsCamera || rw::engine->currentCamera != rsCamera || !gScaledFrame)
		return 0;
	if(!capTaken){
		if(capW != rsW || capH != rsH || !capFbo)
			capMakeTargets(rsW, rsH);
		GLboolean scissor = glIsEnabled(GL_SCISSOR_TEST);
		glDisable(GL_SCISSOR_TEST);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, rsFbo);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, capFbo);
		glBlitFramebuffer(0, 0, rsW, rsH, 0, 0, rsW, rsH, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_FRAMEBUFFER, currentFramebuffer);
		if(scissor) glEnable(GL_SCISSOR_TEST);
		capTaken = true;
		capCount++;
	}
	*colorTex = capColor;
	*depthTex = capDepth;
	texSize[0] = (float)capW; texSize[1] = (float)capH;
	viewport[0] = (float)glGlobals.presentOffX; viewport[1] = (float)glGlobals.presentOffY;
	viewport[2] = (float)glGlobals.presentWidth; viewport[3] = (float)glGlobals.presentHeight;
	memcpy(proj, uniformScene.proj, 64);
	memcpy(view, uniformScene.view, 64);
	return 1;
}

static void
syncGfxSettings(void)
{
	gTemporal = regtaGfxTemporal != 0;
	gMetalFX = regtaGfxMetalFX != 0;
	gSSR = regtaGfxSSR != 0;
	gRT = regtaGfxRT != 0;
	gRTShadows = regtaGfxRTShadows != 0;
	gRTRoadsOn = regtaGfxRTRoads != 0;
}

static void
beginUpdate(Camera *cam)
{
	float view[16], proj[16];
	syncGfxSettings();
	// View Matrix
	Matrix inv;
	Matrix::invert(&inv, cam->getFrame()->getLTM());
	// Since we're looking into positive Z,
	// flip X to ge a left handed view space.
	view[0]  = -inv.right.x;
	view[1]  =  inv.right.y;
	view[2]  =  inv.right.z;
	view[3]  =  0.0f;
	view[4]  = -inv.up.x;
	view[5]  =  inv.up.y;
	view[6]  =  inv.up.z;
	view[7]  =  0.0f;
	view[8]  =  -inv.at.x;
	view[9]  =   inv.at.y;
	view[10] =  inv.at.z;
	view[11] =  0.0f;
	view[12] = -inv.pos.x;
	view[13] =  inv.pos.y;
	view[14] =  inv.pos.z;
	view[15] =  1.0f;
	memcpy(&cam->devView, &view, sizeof(RawMatrix));
	setViewMatrix(view);

	// Projection Matrix
	float32 invwx = 1.0f/cam->viewWindow.x;
	float32 invwy = 1.0f/cam->viewWindow.y;
	float32 invz = 1.0f/(cam->farPlane-cam->nearPlane);

	proj[0] = invwx;
	proj[1] = 0.0f;
	proj[2] = 0.0f;
	proj[3] = 0.0f;

	proj[4] = 0.0f;
	proj[5] = invwy;
	proj[6] = 0.0f;
	proj[7] = 0.0f;

	proj[8] = cam->viewOffset.x*invwx;
	proj[9] = cam->viewOffset.y*invwy;
	proj[12] = -proj[8];
	proj[13] = -proj[9];
	if(cam->projection == Camera::PERSPECTIVE){
		proj[10] = (cam->farPlane+cam->nearPlane)*invz;
		proj[11] = 1.0f;

		proj[14] = -2.0f*cam->nearPlane*cam->farPlane*invz;
		proj[15] = 0.0f;
	}else{
		proj[10] = 2.0f*invz;
		proj[11] = 0.0f;

		proj[14] = -(cam->farPlane+cam->nearPlane)*invz;
		proj[15] = 1.0f;
	}
	memcpy(&cam->devProj, &proj, sizeof(RawMatrix));
	if(gUseTarget && (gTemporal || gRT) && gScaledFrame && cam->frameBuffer->parent->type == Raster::CAMERA &&
	   cam->projection == Camera::PERSPECTIVE){
		// unjittered view-projection for reprojection
		float vp[16];
		mat4mul(vp, proj, view);
		memcpy(gCurVP, vp, sizeof(vp));
		gSceneSerial++;
		if(mat4inv(gInvVP, vp) && gTemporal){
			// sub-pixel jitter (Halton 2,3) of the projection, in target pixels
			float sc = getHidpiScale();
			float tw = cam->frameBuffer->parent->width*sc, th = cam->frameBuffer->parent->height*sc;
			gJit[0] = halton(gFrameIdx % 16 + 1, 2) - 0.5f;
			gJit[1] = halton(gFrameIdx % 16 + 1, 3) - 0.5f;
			proj[8] += 2.0f*gJit[0]/tw;
			proj[9] += 2.0f*gJit[1]/th;
			gJitterApplied = true;
		}
	}
	setProjectionMatrix(proj);

	if(rwStateCache.fogStart != cam->fogPlane){
		rwStateCache.fogStart = cam->fogPlane;
		uniformStateDirty[RWGL_FOGSTART] = true;
		stateDirty = 1;
	}
	if(rwStateCache.fogEnd != cam->farPlane){
		rwStateCache.fogEnd = cam->farPlane;
		uniformStateDirty[RWGL_FOGEND] = true;
		stateDirty = 1;
	}

	setFrameBuffer(cam);

	setViewport(cam->frameBuffer);
}

static void
endUpdate(Camera *cam)
{
}

static void
clearCamera(Camera *cam, RGBA *col, uint32 mode)
{
	RGBAf colf;
	GLbitfield mask;

	setFrameBuffer(cam);

	// make sure we're only clearing the part of the framebuffer
	// that is subrastered
	bool setScissor = cam->frameBuffer != cam->frameBuffer->parent;
	if(setScissor){
		Rect r = getFramebufferRect(cam->frameBuffer);
		glScissor(r.x, r.y, r.w, r.h);
		glEnable(GL_SCISSOR_TEST);
	}

	convColor(&colf, col);
	glClearColor(colf.red, colf.green, colf.blue, colf.alpha);
	mask = 0;
	if(mode & Camera::CLEARIMAGE)
		mask |= GL_COLOR_BUFFER_BIT;
	if(mode & Camera::CLEARZ)
		mask |= GL_DEPTH_BUFFER_BIT;
	if(mode & Camera::CLEARSTENCIL)
		mask |= GL_STENCIL_BUFFER_BIT;
	glDepthMask(GL_TRUE);
	static int lastVelClear = -1;
	if(gNeedVel && rsVelTex && currentFramebuffer == rsFbo && lastVelClear != gFrameIdx){
		// once per frame, at the first clear (the game may clear only depth, or clear again mid-frame):
		// motion vectors and reflection directions start at "not written" (alpha 0)
		lastVelClear = gFrameIdx;
		static const GLenum bufs[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
		static const float zero[4] = { 0, 0, 0, 0 };
		int n = rsReflTex ? 3 : 2;
		glDrawBuffers(n, bufs);
		glClearBufferfv(GL_COLOR, 1, zero);
		if(rsReflTex) glClearBufferfv(GL_COLOR, 2, zero);
		glDrawBuffers(1, bufs);
		gVelBoth = 1;
	}
	glClear(mask);
	glDepthMask(rwStateCache.zwrite);

	if(setScissor)
		glDisable(GL_SCISSOR_TEST);
}

extern "C" int regtaDumpRequest;

static void
showRaster(Raster *raster, uint32 flags)
{
	resolveRenderScale();
//	glViewport(raster->offsetX, raster->offsetY,
//		raster->width, raster->height);

#ifdef LIBRW_SDL2
	if(flags & Raster::FLIPWAITVSYNCH)
		SDL_GL_SetSwapInterval(1);
	else
		SDL_GL_SetSwapInterval(0);
	SDL_GL_SwapWindow(glGlobals.window);
#elif defined(LIBRW_SDL3)
	if(flags & Raster::FLIPWAITVSYNCH)
		SDL_GL_SetSwapInterval(1);
	else
		SDL_GL_SetSwapInterval(0);
	SDL_GL_SwapWindow(glGlobals.window);
#elif defined(LIBRW_GLFW)
	// debugging aid: other code can ask for a frame to be saved (regtaDumpRequest = frames to wait; 0 = none)
	{
		static int signalCount = 0;
		if(regtaDumpRequest > 0 && --regtaDumpRequest == 0){
			int fw, fh;
			glfwGetFramebufferSize(glGlobals.window, &fw, &fh);
			uint8 *px = (uint8*)malloc(fw*fh*3);
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glReadBuffer(GL_BACK);
			glReadPixels(0, 0, fw, fh, GL_RGB, GL_UNSIGNED_BYTE, px);
			char name[64];
			snprintf(name, sizeof(name), "regta_signal_%d.ppm", ++signalCount);
			if(FILE *f = fopen(name, "wb")){
				fprintf(f, "P6\n%d %d\n255\n", fw, fh);
				for(int y = fh-1; y >= 0; y--) fwrite(px + y*fw*3, 1, fw*3, f);
				fclose(f);
				printf("[dump] wrote %s (%dx%d)\n", name, fw, fh);
			}
			free(px);
		}
	}
	// debugging aid: REGTA_DUMP_FRAMES="n1,n2,.." writes those frames' back buffer to PPM files
	{
		static int frame = 0;
		static const char *spec = getenv("REGTA_DUMP_FRAMES");
		frame++;
		if(spec){
			char want[32];
			snprintf(want, sizeof(want), ",%d,", frame);
			char all[512];
			snprintf(all, sizeof(all), ",%s,", spec);
			if(strstr(all, want)){
				int fw, fh;
				glfwGetFramebufferSize(glGlobals.window, &fw, &fh);
				uint8 *px = (uint8*)malloc(fw*fh*3);
				glPixelStorei(GL_PACK_ALIGNMENT, 1);
				glReadBuffer(GL_BACK);
				glReadPixels(0, 0, fw, fh, GL_RGB, GL_UNSIGNED_BYTE, px);
				char name[64];
				snprintf(name, sizeof(name), "regta_frame_%d.ppm", frame);
				if(FILE *f = fopen(name, "wb")){
					fprintf(f, "P6\n%d %d\n255\n", fw, fh);
					for(int y = fh-1; y >= 0; y--)
						fwrite(px + y*fw*3, 1, fw*3, f);
					fclose(f);
					printf("[dump] wrote %s (%dx%d)\n", name, fw, fh);
				}
				free(px);
			}
		}
	}
	// only touch the swap interval when it changes: re-applying it every frame
	// reconfigures the display sync on macOS and shows up as flicker
	// macOS can silently reset it (window move/resize/focus), so also re-assert it every
	// 30 frames instead of trusting the cached value forever
	static int curInterval = -1;
	static int sinceSet = 0;
	int wantInterval = (flags & Raster::FLIPWAITVSYNCH) ? 1 : 0;
	if(wantInterval != curInterval || ++sinceSet >= 30){
		glfwSwapInterval(wantInterval);
		curInterval = wantInterval;
		sinceSet = 0;
	}
	glfwSwapBuffers(glGlobals.window);
	gScaledFrame = true;
	capTaken = false;
	capCount = 0;
	if(gJitterApplied) memcpy(gPrevVP, gCurVP, sizeof(gPrevVP));
	else gHistValid = false;
	gJitterApplied = false;
	gFrameIdx++;
#else
	not implemented
#endif
}

static bool32
rasterRenderFast(Raster *raster, int32 x, int32 y)
{
	Raster *src = raster;
	Raster *dst = Raster::getCurrentContext();
	Gl3Raster *natdst = PLUGINOFFSET(Gl3Raster, dst, nativeRasterOffset);
	Gl3Raster *natsrc = PLUGINOFFSET(Gl3Raster, src, nativeRasterOffset);

	switch(dst->type){
	case Raster::NORMAL:
	case Raster::TEXTURE:
	case Raster::CAMERATEXTURE:
		switch(src->type){
		case Raster::CAMERA:
			setActiveTexture(0);
			glBindTexture(GL_TEXTURE_2D, natdst->texid);
			{
				float sc = natdst->scale <= 0.0f ? 1.0f : natdst->scale;
				glCopyTexSubImage2D(GL_TEXTURE_2D, 0, (int32)(x*sc+.5f), (int32)(((dst->height-src->height)-y)*sc+.5f),
					0, 0, (int32)(src->width*sc+.5f), (int32)(src->height*sc+.5f));
			}
			glBindTexture(GL_TEXTURE_2D, boundTexture[0]);
			return 1;
		}
		break;
	}
	return 0;
}

#ifdef LIBRW_SDL2

static void
addVideoMode(int displayIndex, int modeIndex)
{
	int i;
	SDL_DisplayMode mode;

	SDL_GetDisplayMode(displayIndex, modeIndex, &mode);

	for(i = 1; i < glGlobals.numModes; i++){
		if(glGlobals.modes[i].mode.w == mode.w &&
		   glGlobals.modes[i].mode.h == mode.h &&
		   glGlobals.modes[i].mode.format == mode.format){
			// had this mode already, remember highest refresh rate
			if(mode.refresh_rate > glGlobals.modes[i].mode.refresh_rate)
				glGlobals.modes[i].mode.refresh_rate = mode.refresh_rate;
			return;
		}
	}

	// none found, add
	glGlobals.modes[glGlobals.numModes].mode = mode;
	glGlobals.modes[glGlobals.numModes].flags = VIDEOMODEEXCLUSIVE;
	glGlobals.numModes++;
}

static void
makeVideoModeList(int displayIndex)
{
	int i, num, depth;

	num = SDL_GetNumDisplayModes(displayIndex);
	rwFree(glGlobals.modes);
	glGlobals.modes = rwNewT(DisplayMode, num+1, ID_DRIVER | MEMDUR_EVENT);

	SDL_GetCurrentDisplayMode(displayIndex, &glGlobals.modes[0].mode);
	glGlobals.modes[0].flags = 0;
	glGlobals.numModes = 1;

	for(i = 0; i < num; i++)
		addVideoMode(displayIndex, i);

	for(i = 0; i < glGlobals.numModes; i++){
		depth = SDL_BITSPERPIXEL(glGlobals.modes[i].mode.format);
		// set depth to power of two
		for(glGlobals.modes[i].depth = 1; glGlobals.modes[i].depth < depth; glGlobals.modes[i].depth <<= 1);
	}
}

static int
openSDL2(EngineOpenParams *openparams)
{
	glGlobals.winWidth = openparams->width;
	glGlobals.winHeight = openparams->height;
	glGlobals.winTitle = openparams->windowtitle;
	glGlobals.pWindow = openparams->window;

	memset(&gl3Caps, 0, sizeof(gl3Caps));

	/* Init SDL */
	if(SDL_InitSubSystem(SDL_INIT_VIDEO)){
		RWERROR((ERR_GENERAL, SDL_GetError()));
		return 0;
	}

	glGlobals.currentDisplay = 0;
	glGlobals.numDisplays = SDL_GetNumVideoDisplays();

	makeVideoModeList(glGlobals.currentDisplay);

	return 1;
}

static int
closeSDL2(void)
{
	SDL_QuitSubSystem(SDL_INIT_VIDEO);
	return 1;
}

static struct {
	int gl;
	int major, minor;
} profiles[] = {
	{ SDL_GL_CONTEXT_PROFILE_CORE, 3, 3 },
	{ SDL_GL_CONTEXT_PROFILE_CORE, 2, 1 },
	{ SDL_GL_CONTEXT_PROFILE_ES, 3, 1 },
	{ SDL_GL_CONTEXT_PROFILE_ES, 2, 0 },
	{ 0, 0, 0 },
};

static int
startSDL2(void)
{
	SDL_Window *win;
	SDL_GLContext ctx;
	DisplayMode *mode;

	mode = &glGlobals.modes[glGlobals.currentMode];

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, glGlobals.numSamples);

	int i;
	for(i = 0; profiles[i].gl; i++){
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, profiles[i].gl);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, profiles[i].major);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, profiles[i].minor);

		if(mode->flags & VIDEOMODEEXCLUSIVE) {
			win = SDL_CreateWindow(glGlobals.winTitle, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, mode->mode.w, mode->mode.h, SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN);
			if (win)
				SDL_SetWindowDisplayMode(win, &mode->mode);
		} else {
			win = SDL_CreateWindow(glGlobals.winTitle, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, glGlobals.winWidth, glGlobals.winHeight, SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
			if (win)
				SDL_SetWindowDisplayMode(win, NULL);
		}
		if(win){
			gl3Caps.gles = profiles[i].gl == SDL_GL_CONTEXT_PROFILE_ES;
			gl3Caps.glversion = profiles[i].major*10 + profiles[i].minor;
			break;
		}
	}
	if(win == nil){
		RWERROR((ERR_GENERAL, SDL_GetError()));
		return 0;
	}
	ctx = SDL_GL_CreateContext(win);

	if (!((gl3Caps.gles ? gladLoadGLES2Loader : gladLoadGLLoader) ((GLADloadproc) SDL_GL_GetProcAddress, gl3Caps.glversion)) ) {
		RWERROR((ERR_GENERAL, "gladLoadGLLoader failed"));
		SDL_GL_DeleteContext(ctx);
		SDL_DestroyWindow(win);
		return 0;
	}

//	printf("OpenGL version: %s\n", glGetString(GL_VERSION));

	glGlobals.window = win;
	glGlobals.glcontext = ctx;
	*glGlobals.pWindow = win;
	glGlobals.presentWidth = 0;
	glGlobals.presentHeight = 0;
	glGlobals.presentOffX = 0;
	glGlobals.presentOffY = 0;
	return 1;
}

static int
stopSDL2(void)
{
	SDL_GL_DeleteContext(glGlobals.glcontext);
	SDL_DestroyWindow(glGlobals.window);
	return 1;
}

#elif defined(LIBRW_SDL3)

static void
addVideoMode(const SDL_DisplayMode *mode)
{
	int i;

	for(i = 1; i < glGlobals.numModes; i++){
		if(glGlobals.modes[i].mode.w == mode->w &&
		   glGlobals.modes[i].mode.h == mode->h &&
		   glGlobals.modes[i].mode.format == mode->format){
			// had this mode already, remember highest refresh rate
			if(mode->refresh_rate > glGlobals.modes[i].mode.refresh_rate)
				glGlobals.modes[i].mode.refresh_rate = mode->refresh_rate;
			return;
		}
	}

	// none found, add
	glGlobals.modes[glGlobals.numModes].mode = *mode;
	glGlobals.modes[glGlobals.numModes].flags = VIDEOMODEEXCLUSIVE;
	glGlobals.numModes++;
}

static void
makeVideoModeList(SDL_DisplayID displayIndex, SDL_DisplayID *displays)
{
	int i, num, depth;
	const SDL_DisplayMode *currentMode;
	SDL_DisplayMode **modes;

	currentMode = SDL_GetCurrentDisplayMode(displays[displayIndex]);
	modes = SDL_GetFullscreenDisplayModes(displays[displayIndex], &num);

	rwFree(glGlobals.modes);
	glGlobals.modes = rwNewT(DisplayMode, num+(currentMode != NULL ? 1 : 0), ID_DRIVER | MEMDUR_EVENT);

	if (currentMode) {
		glGlobals.modes[0].mode = *currentMode;
		glGlobals.modes[0].flags = 0;
		glGlobals.numModes = 1;
	}

	for(i = 0; i < num; i++)
		addVideoMode(modes[i]);

	for(i = 0; i < glGlobals.numModes; i++){
		depth = SDL_BITSPERPIXEL(glGlobals.modes[i].mode.format);
		// set depth to power of two
		for(glGlobals.modes[i].depth = 1; glGlobals.modes[i].depth < depth; glGlobals.modes[i].depth <<= 1);
	}

	SDL_free(modes);
}

static int
openSDL3(EngineOpenParams *openparams)
{
	glGlobals.winWidth = openparams->width;
	glGlobals.winHeight = openparams->height;
	glGlobals.winTitle = openparams->windowtitle;
	glGlobals.pWindow = openparams->window;

	memset(&gl3Caps, 0, sizeof(gl3Caps));

	/* Init SDL */
	if (!SDL_InitSubSystem(SDL_INIT_VIDEO)){
		RWERROR((ERR_GENERAL, SDL_GetError()));
		return 0;
	}

	glGlobals.currentDisplay = 0;
	SDL_DisplayID *displays = SDL_GetDisplays(&glGlobals.numDisplays);

	if (glGlobals.currentDisplay >= glGlobals.numDisplays) {
		RWERROR((ERR_GENERAL, SDL_GetError()));
		return 0;
	}

	makeVideoModeList(glGlobals.currentDisplay, displays);
	SDL_free(displays);

	return 1;
}

static int
closeSDL3(void)
{
	SDL_QuitSubSystem(SDL_INIT_VIDEO);
	return 1;
}

static struct {
	int gl;
	int major, minor;
} profiles[] = {
	{ SDL_GL_CONTEXT_PROFILE_CORE, 3, 3 },
	{ SDL_GL_CONTEXT_PROFILE_CORE, 2, 1 },
	{ SDL_GL_CONTEXT_PROFILE_ES, 3, 1 },
	{ SDL_GL_CONTEXT_PROFILE_ES, 2, 0 },
	{ 0, 0, 0 },
};

static int
startSDL3(void)
{
	SDL_Window *win;
	SDL_GLContext ctx;
	DisplayMode *mode;

	mode = &glGlobals.modes[glGlobals.currentMode];

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, glGlobals.numSamples);

	int i;
	for(i = 0; profiles[i].gl; i++){
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, profiles[i].gl);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, profiles[i].major);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, profiles[i].minor);

		if(mode->flags & VIDEOMODEEXCLUSIVE) {
			win = SDL_CreateWindow(glGlobals.winTitle, mode->mode.w, mode->mode.h, SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
			// This is the recommended way for SDL3.
			if (win) {
				SDL_SetWindowFullscreenMode(win, &mode->mode);
				SDL_SetWindowFullscreen(win, true);
			}
		} else {
			win = SDL_CreateWindow(glGlobals.winTitle, glGlobals.winWidth, glGlobals.winHeight, SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
			if (win)
				SDL_SetWindowFullscreenMode(win, NULL);
		}
		if(win){
			gl3Caps.gles = profiles[i].gl == SDL_GL_CONTEXT_PROFILE_ES;
			gl3Caps.glversion = profiles[i].major*10 + profiles[i].minor;
			break;
		}
	}
	if(win == nil){
		RWERROR((ERR_GENERAL, SDL_GetError()));
		return 0;
	}
	ctx = SDL_GL_CreateContext(win);

	if (!((gl3Caps.gles ? gladLoadGLES2Loader : gladLoadGLLoader) ((GLADloadproc) SDL_GL_GetProcAddress, gl3Caps.glversion)) ) {
		RWERROR((ERR_GENERAL, "gladLoadGLLoader failed"));
		SDL_GL_DestroyContext(ctx);
		SDL_DestroyWindow(win);
		return 0;
	}

//	printf("OpenGL version: %s\n", glGetString(GL_VERSION));

	glGlobals.window = win;
	glGlobals.glcontext = ctx;
	*glGlobals.pWindow = win;
	glGlobals.presentWidth = 0;
	glGlobals.presentHeight = 0;
	glGlobals.presentOffX = 0;
	glGlobals.presentOffY = 0;
	return 1;
}

static int
stopSDL3(void)
{
	SDL_GL_DestroyContext(glGlobals.glcontext);
	SDL_DestroyWindow(glGlobals.window);
	return 1;
}
#elif defined(LIBRW_GLFW)

static void
addVideoMode(const GLFWvidmode *mode)
{
	int i;

	for(i = 1; i < glGlobals.numModes; i++){
		if(glGlobals.modes[i].mode.width == mode->width &&
		   glGlobals.modes[i].mode.height == mode->height &&
		   glGlobals.modes[i].mode.redBits == mode->redBits &&
		   glGlobals.modes[i].mode.greenBits == mode->greenBits &&
		   glGlobals.modes[i].mode.blueBits == mode->blueBits){
			// had this mode already, remember highest refresh rate
			if(mode->refreshRate > glGlobals.modes[i].mode.refreshRate)
				glGlobals.modes[i].mode.refreshRate = mode->refreshRate;
			return;
		}
	}

	// none found, add
	glGlobals.modes[glGlobals.numModes].mode = *mode;
	glGlobals.modes[glGlobals.numModes].flags = VIDEOMODEEXCLUSIVE;
	glGlobals.numModes++;
}

static void
makeVideoModeList(GLFWmonitor *monitor)
{
	int i, num;
	const GLFWvidmode *modes;

	modes = glfwGetVideoModes(monitor, &num);
	rwFree(glGlobals.modes);
	glGlobals.modes = rwNewT(DisplayMode, num+1, ID_DRIVER | MEMDUR_EVENT);

	glGlobals.modes[0].mode = *glfwGetVideoMode(monitor);
	glGlobals.modes[0].flags = 0;
	glGlobals.numModes = 1;

	for(i = 0; i < num; i++)
		addVideoMode(&modes[i]);

	for(i = 0; i < glGlobals.numModes; i++){
		num = glGlobals.modes[i].mode.redBits +
			glGlobals.modes[i].mode.greenBits +
			glGlobals.modes[i].mode.blueBits;
		// set depth to power of two
		for(glGlobals.modes[i].depth = 1; glGlobals.modes[i].depth < num; glGlobals.modes[i].depth <<= 1);
	}
}

static int
openGLFW(EngineOpenParams *openparams)
{
	glGlobals.winWidth = openparams->width;
	glGlobals.winHeight = openparams->height;
	glGlobals.winTitle = openparams->windowtitle;
	glGlobals.pWindow = openparams->window;

	memset(&gl3Caps, 0, sizeof(gl3Caps));

	/* Init GLFW */
	if(!glfwInit()){
		RWERROR((ERR_GENERAL, "glfwInit() failed"));
		return 0;
	}

	glGlobals.monitor = glfwGetMonitors(&glGlobals.numMonitors)[0];

	makeVideoModeList(glGlobals.monitor);

	return 1;
}

static int
closeGLFW(void)
{
	glfwTerminate();
	return 1;
}

static void
glfwerr(int error, const char *desc)
{
	fprintf(stderr, "GLFW Error: %s\n", desc);
}

static struct {
	int gl;
	int major, minor;
} profiles[] = {
	{ GLFW_OPENGL_API, 3, 3 },
	{ GLFW_OPENGL_API, 2, 1 },
	{ GLFW_OPENGL_ES_API, 3, 1 },
	{ GLFW_OPENGL_ES_API, 2, 0 },
	{ 0, 0, 0 },
};

extern "C" { int regtaDumpRequest = 0; }

static int
startGLFW(void)
{
	GLFWwindow *win;
	DisplayMode *mode;

	mode = &glGlobals.modes[glGlobals.currentMode];

	glfwSetErrorCallback(glfwerr);
	{
		bool haveScaleIdx = false;
		if(FILE *ini = fopen("re3.ini", "r")){
			char line[256];
			bool inGraphics = false;
			while(fgets(line, sizeof(line), ini)){
				if(line[0] == '[') inGraphics = strncmp(line, "[Graphics]", 10) == 0;
				else if(inGraphics && strncmp(line, "RenderScale=", 12) == 0){
					float v = (float)atof(line + 12);
					if(v >= 0.4f && v <= 1.0f) gRenderScale = v;
				}else if(inGraphics && strncmp(line, "RenderSharpness=", 16) == 0)
					gRenderSharpness = (float)atof(line + 16);
				else if(inGraphics && strncmp(line, "RenderTemporal=", 15) == 0)
					gTemporal = atoi(line + 15) != 0;
				else if(inGraphics && strncmp(line, "VehicleSSR=", 11) == 0)
					gSSR = atoi(line + 11) != 0;
				else if(inGraphics && strncmp(line, "RenderRT=", 9) == 0)
					gRT = atoi(line + 9) != 0;
				else if(inGraphics && strncmp(line, "RenderRTStrength=", 17) == 0)
					gRTStrength = (float)atof(line + 17);
				else if(inGraphics && strncmp(line, "RenderRTHalfRes=", 16) == 0)
					gRTDiv = atoi(line + 16) ? 2 : 1;
				else if(inGraphics && strncmp(line, "RenderRTShadows=", 16) == 0)
					gRTShadows = atoi(line + 16) != 0;
				else if(inGraphics && strncmp(line, "RenderRTShadowStrength=", 23) == 0)
					gRTShadowStrength = (float)atof(line + 23);
				else if(inGraphics && strncmp(line, "RenderRTShadowDistance=", 23) == 0)
					gRTShadowDistance = (float)atof(line + 23);
				else if(inGraphics && strncmp(line, "RenderScaleIdx=", 15) == 0){
					int idx = atoi(line + 15);
					if(idx >= 0 && idx < 6){ gRenderScale = kScaleTable[idx]; regtaGfxScaleIdx = (int8_t)idx; haveScaleIdx = true; }
				}else if(inGraphics && strncmp(line, "RainPuddles=", 12) == 0)
					regtaGfxPuddles = atoi(line + 12) != 0;
				else if(inGraphics && strncmp(line, "RenderRTRoads=", 14) == 0)
					gRTRoadsOn = atoi(line + 14) != 0;
				else if(inGraphics && strncmp(line, "RenderRTRoadStrength=", 21) == 0)
					gRTRoadStrength = (float)atof(line + 21);
				else if(inGraphics && strncmp(line, "RenderRTDistance=", 17) == 0)
					gRTDistance = (float)atof(line + 17);
				else if(inGraphics && strncmp(line, "RenderMetalFX=", 14) == 0)
					gMetalFX = atoi(line + 14) != 0;
				else if(inGraphics && strncmp(line, "MetalFXJitterFlipX=", 19) == 0)
					gMfxJitterSignX = atoi(line + 19) ? -1.0f : 1.0f;
				else if(inGraphics && strncmp(line, "MetalFXJitterFlipY=", 19) == 0)
					gMfxJitterSignY = atoi(line + 19) ? -1.0f : 1.0f;
			}
			fclose(ini);
		}
		// everything the menu can switch on later needs the offscreen target and its attachments from the start
		gUseTarget = true;
		gNeedVel = true;
		regtaGfxTemporal = gTemporal; regtaGfxMetalFX = gMetalFX; regtaGfxSSR = gSSR;
		regtaGfxRT = gRT; regtaGfxRTShadows = gRTShadows; regtaGfxRTRoads = 1;
		if(!haveScaleIdx){
			int best = 0;
			for(int i = 0; i < 6; i++) if(fabsf(kScaleTable[i] - gRenderScale) < fabsf(kScaleTable[best] - gRenderScale)) best = i;
			regtaGfxScaleIdx = (int8_t)best;
		}
	}
#ifdef __APPLE__
	// The game sizes its framebuffer copies (post-processing, screen grabs) in window points.
	// A 2x Retina framebuffer makes those copy only the lower-left quarter and stretch it
	// (zoomed 3D view, ghosted menu), so keep one framebuffer pixel per window point.
	{
		// re3.ini [Graphics] Retina=1 draws at the screen's native pixels (screen-sized camera textures follow)
		bool retina = false;
		if(FILE *ini = fopen("re3.ini", "r")){
			char line[256];
			bool inGraphics = false;
			while(fgets(line, sizeof(line), ini)){
				if(line[0] == '[') inGraphics = strncmp(line, "[Graphics]", 10) == 0;
				else if(inGraphics && strncmp(line, "Retina=", 7) == 0) retina = atoi(line + 7) != 0;
			}
			fclose(ini);
		}
		glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, retina ? GLFW_TRUE : GLFW_FALSE);
	}
#endif
	glfwWindowHint(GLFW_RED_BITS, mode->mode.redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->mode.greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->mode.blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->mode.refreshRate);
	
	// GLX will round up to 2x or 4x if you ask for multisampling on with 1 sample
	// So only apply the SAMPLES hint if we actually want multisampling
	if (glGlobals.numSamples > 1)
		glfwWindowHint(GLFW_SAMPLES, glGlobals.numSamples);

	int i;
	for(i = 0; profiles[i].gl; i++){
		glfwWindowHint(GLFW_CLIENT_API, profiles[i].gl);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, profiles[i].major);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, profiles[i].minor);
#ifdef __APPLE__
		// macOS only provides 3.2+ contexts as forward-compatible core profiles;
		// without this it falls back to a legacy 2.1 context.
		if(profiles[i].gl == GLFW_OPENGL_API && profiles[i].major >= 3){
			glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
			glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		}
#endif

		if(mode->flags & VIDEOMODEEXCLUSIVE)
			win = glfwCreateWindow(mode->mode.width, mode->mode.height, glGlobals.winTitle, glGlobals.monitor, nil);
		else
			win = glfwCreateWindow(glGlobals.winWidth, glGlobals.winHeight, glGlobals.winTitle, nil, nil);
		if(win){
			gl3Caps.gles = profiles[i].gl == GLFW_OPENGL_ES_API;
			gl3Caps.glversion = profiles[i].major*10 + profiles[i].minor;
			break;
		}
	}
	if(win == nil){
		RWERROR((ERR_GENERAL, "glfwCreateWindow() failed"));
		return 0;
	}
	glfwMakeContextCurrent(win);

	/* Init GLAD */
	if (!((gl3Caps.gles ? gladLoadGLES2Loader : gladLoadGLLoader) ((GLADloadproc) glfwGetProcAddress, gl3Caps.glversion)) ) {
		RWERROR((ERR_GENERAL, "gladLoadGLLoader failed"));
		glfwDestroyWindow(win);
		return 0;
	}

	printf("OpenGL version: %s (%s)\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

	glGlobals.window = win;
	*glGlobals.pWindow = win;
	glGlobals.presentWidth = 0;
	glGlobals.presentHeight = 0;
	glGlobals.presentOffX = 0;
	glGlobals.presentOffY = 0;
	return 1;
}

static int
stopGLFW(void)
{
	glfwDestroyWindow(glGlobals.window);
	return 1;
}
#else
not implemented
#endif

static int
initOpenGL(void)
{
/*
	// this only works from 3.0 onward,
	// but luckily GLAD has already taken care of extensions for us
	int numExt;
	glGetIntegerv(GL_NUM_EXTENSIONS, &numExt);
	for(int i = 0; i < numExt; i++){
		const char *ext = (const char*)glGetStringi(GL_EXTENSIONS, i);
		if(ext == nil)
			continue;	// apparently that can happen...
		if(strcmp(ext, "GL_EXT_texture_compression_s3tc") == 0)
			gl3Caps.dxtSupported = true;
		else if(strcmp(ext, "GL_KHR_texture_compression_astc_ldr") == 0)
			gl3Caps.astcSupported = true;
//		printf("%d %s\n", i, ext);
	}
*/
	gl3Caps.dxtSupported = !!GLAD_GL_EXT_texture_compression_s3tc;
	gl3Caps.astcSupported = !!GLAD_GL_KHR_texture_compression_astc_ldr;
	printf("S3TC (DXT) supported: %d\n", (int)gl3Caps.dxtSupported);

	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &gl3Caps.maxAnisotropy);

	if(gl3Caps.gles){
		if(gl3Caps.glversion >= 30)
			shaderDecl = shaderDecl310es;
		else
			shaderDecl = shaderDecl100es;
	}else{
		if(gl3Caps.glversion >= 30)
			shaderDecl = shaderDecl330;
		else
			shaderDecl = shaderDecl120;
	}

#ifndef RW_GL_USE_UBOS
	u_alphaRef = registerUniform("u_alphaRef", UNIFORM_VEC4);
	u_fogData = registerUniform("u_fogData", UNIFORM_VEC4);
	u_fogColor = registerUniform("u_fogColor", UNIFORM_VEC4);
	u_proj = registerUniform("u_proj", UNIFORM_MAT4);
	u_view = registerUniform("u_view", UNIFORM_MAT4);
	u_world = registerUniform("u_world", UNIFORM_MAT4);
	u_ambLight = registerUniform("u_ambLight", UNIFORM_VEC4);
	u_lightParams = registerUniform("u_lightParams", UNIFORM_VEC4, MAX_LIGHTS);
	u_lightPosition = registerUniform("u_lightPosition", UNIFORM_VEC4, MAX_LIGHTS);
	u_lightDirection = registerUniform("u_lightDirection", UNIFORM_VEC4, MAX_LIGHTS);
	u_lightColor = registerUniform("u_lightColor", UNIFORM_VEC4, MAX_LIGHTS);
	lastShaderUploaded = nil;
#else
	registerBlock("Scene");
	registerBlock("Object");
	registerBlock("State");
#endif
	u_matColor = registerUniform("u_matColor", UNIFORM_VEC4);
	u_surfProps = registerUniform("u_surfProps", UNIFORM_VEC4);

	// for im2d
	registerUniform("u_xform", UNIFORM_VEC4);

	glClearColor(0.25, 0.25, 0.25, 1.0);

	byte whitepixel[4] = {0xFF, 0xFF, 0xFF, 0xFF};
	glGenTextures(1, &whitetex);
	glBindTexture(GL_TEXTURE_2D, whitetex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1,
	             0, GL_RGBA, GL_UNSIGNED_BYTE, &whitepixel);

	resetRenderState();

	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy);

	if(gl3Caps.glversion >= 30){
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}

#ifdef RW_GL_USE_UBOS
	glGenBuffers(1, &ubo_state);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_state);
	glBindBufferBase(GL_UNIFORM_BUFFER, gl3::findBlock("State"), ubo_state);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformState), &uniformState,
	             GL_STREAM_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &ubo_scene);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_scene);
	glBindBufferBase(GL_UNIFORM_BUFFER, gl3::findBlock("Scene"), ubo_scene);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformScene), &uniformScene,
	             GL_STREAM_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glGenBuffers(1, &ubo_object);
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_object);
	glBindBufferBase(GL_UNIFORM_BUFFER, gl3::findBlock("Object"), ubo_object);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformObject), &uniformObject,
	             GL_STREAM_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
#endif

#include "shaders/default_vs_gl.inc"
#include "shaders/simple_fs_gl.inc"
	const char *vs[] = { shaderDecl, "#define VEL_PIPE\n", "#define REFL_OUT\n", "#define WET_SURFACE\n", header_vert_src, default_vert_src, nil };
	const char *vs_fullLight[] = { shaderDecl, "#define VEL_PIPE\n", "#define REFL_OUT\n", "#define WET_SURFACE\n", "#define DIRECTIONALS\n#define POINTLIGHTS\n#define SPOTLIGHTS\n", header_vert_src, default_vert_src, nil };
	const char *fs[] = { shaderDecl, "#define VEL_PIPE\n", "#define REFL_OUT\n", "#define WET_SURFACE\n", header_frag_src, simple_frag_src, nil };
	const char *fs_noAT[] = { shaderDecl, "#define VEL_PIPE\n", "#define REFL_OUT\n", "#define WET_SURFACE\n", "#define NO_ALPHATEST\n", header_frag_src, simple_frag_src, nil };

	defaultShader = Shader::create(vs, fs);
	assert(defaultShader);
	defaultShader_noAT = Shader::create(vs, fs_noAT);
	assert(defaultShader_noAT);

	defaultShader_fullLight = Shader::create(vs_fullLight, fs);
	assert(defaultShader_fullLight);
	defaultShader_fullLight_noAT = Shader::create(vs_fullLight, fs_noAT);
	assert(defaultShader_fullLight_noAT);

	openIm2D();
	openIm3D();

	return 1;
}

static int
termOpenGL(void)
{
	closeIm3D();
	closeIm2D();

	defaultShader->destroy();
	defaultShader = nil;
	defaultShader_noAT->destroy();
	defaultShader_noAT = nil;
	defaultShader_fullLight->destroy();
	defaultShader_fullLight = nil;
	defaultShader_fullLight_noAT->destroy();
	defaultShader_fullLight_noAT = nil;

	glDeleteTextures(1, &whitetex);
	whitetex = 0;

	return 1;
}

static int
finalizeOpenGL(void)
{
	return 1;
}

#ifdef LIBRW_SDL2
static int
deviceSystemSDL2(DeviceReq req, void *arg, int32 n)
{
	VideoMode *rwmode;

	switch(req){
	case DEVICEOPEN:
		return openSDL2((EngineOpenParams*)arg);
	case DEVICECLOSE:
		return closeSDL2();

	case DEVICEINIT:
		return startSDL2() && initOpenGL();
	case DEVICETERM:
		return termOpenGL() && stopSDL2();

	case DEVICEFINALIZE:
		return finalizeOpenGL();


	case DEVICEGETNUMSUBSYSTEMS:
		return glGlobals.numDisplays;

	case DEVICEGETCURRENTSUBSYSTEM:
		return glGlobals.currentDisplay;

	case DEVICESETSUBSYSTEM:
		if (n >= SDL_GetNumVideoDisplays())
			return 0;
		glGlobals.currentDisplay = n;
		return 1;

	case DEVICEGETSUBSSYSTEMINFO: {
		const char *display_name = SDL_GetDisplayName(n);
		if (display_name == nil)
			return 0;
		strncpy(((SubSystemInfo*)arg)->name, display_name, sizeof(SubSystemInfo::name));
		return 1;
	}


	case DEVICEGETNUMVIDEOMODES:
		return glGlobals.numModes;

	case DEVICEGETCURRENTVIDEOMODE:
		return glGlobals.currentMode;

	case DEVICESETVIDEOMODE:
		if(n >= glGlobals.numModes)
			return 0;
		glGlobals.currentMode = n;
		return 1;

	case DEVICEGETVIDEOMODEINFO:
		rwmode = (VideoMode*)arg;
		rwmode->width = glGlobals.modes[n].mode.w;
		rwmode->height = glGlobals.modes[n].mode.h;
		rwmode->depth = glGlobals.modes[n].depth;
		rwmode->flags = glGlobals.modes[n].flags;
		return 1;

	case DEVICEGETMAXMULTISAMPLINGLEVELS:
		{
			GLint maxSamples;
			glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
			if(maxSamples == 0)
				return 1;
			return maxSamples;
		}
	case DEVICEGETMULTISAMPLINGLEVELS:
		if(glGlobals.numSamples == 0)
			return 1;
		return glGlobals.numSamples;
	case DEVICESETMULTISAMPLINGLEVELS:
		glGlobals.numSamples = (uint32)n;
		return 1;
	default:
		assert(0 && "not implemented");
		return 0;
	}
	return 1;
}

#elif defined(LIBRW_SDL3)

static int
deviceSystemSDL3(DeviceReq req, void *arg, int32 n)
{
	VideoMode *rwmode;

	switch(req){
	case DEVICEOPEN:
		return openSDL3((EngineOpenParams*)arg);
	case DEVICECLOSE:
		return closeSDL3();

	case DEVICEINIT:
		return startSDL3() && initOpenGL();
	case DEVICETERM:
		return termOpenGL() && stopSDL3();

	case DEVICEFINALIZE:
		return finalizeOpenGL();


	case DEVICEGETNUMSUBSYSTEMS:
		return glGlobals.numDisplays;

	case DEVICEGETCURRENTSUBSYSTEM:
		return glGlobals.currentDisplay;

	case DEVICESETSUBSYSTEM:
		if (n >= glGlobals.numDisplays)
			return 0;
		glGlobals.currentDisplay = n;
		return 1;

	case DEVICEGETSUBSSYSTEMINFO: {
		const char *display_name = SDL_GetDisplayName(n);
		if (display_name == nil)
			return 0;
		strncpy(((SubSystemInfo*)arg)->name, display_name, sizeof(SubSystemInfo::name));
		return 1;
	}


	case DEVICEGETNUMVIDEOMODES:
		return glGlobals.numModes;

	case DEVICEGETCURRENTVIDEOMODE:
		return glGlobals.currentMode;

	case DEVICESETVIDEOMODE:
		if(n >= glGlobals.numModes)
			return 0;
		glGlobals.currentMode = n;
		return 1;

	case DEVICEGETVIDEOMODEINFO:
		if (n < 0)
			return 0;
		rwmode = (VideoMode*)arg;
		rwmode->width = glGlobals.modes[n].mode.w;
		rwmode->height = glGlobals.modes[n].mode.h;
		rwmode->depth = glGlobals.modes[n].depth;
		rwmode->flags = glGlobals.modes[n].flags;
		return 1;

	case DEVICEGETMAXMULTISAMPLINGLEVELS:
		{
			GLint maxSamples;
			glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
			if(maxSamples == 0)
				return 1;
			return maxSamples;
		}
	case DEVICEGETMULTISAMPLINGLEVELS:
		if(glGlobals.numSamples == 0)
			return 1;
		return glGlobals.numSamples;
	case DEVICESETMULTISAMPLINGLEVELS:
		glGlobals.numSamples = (uint32)n;
		return 1;
	default:
		assert(0 && "not implemented");
		return 0;
	}
	return 1;
}

#elif defined(LIBRW_GLFW)

static int
deviceSystemGLFW(DeviceReq req, void *arg, int32 n)
{
	GLFWmonitor **monitors;
	VideoMode *rwmode;

	switch(req){
	case DEVICEOPEN:
		return openGLFW((EngineOpenParams*)arg);
	case DEVICECLOSE:
		return closeGLFW();

	case DEVICEINIT:
		return startGLFW() && initOpenGL();
	case DEVICETERM:
		return termOpenGL() && stopGLFW();

	case DEVICEFINALIZE:
		return finalizeOpenGL();


	case DEVICEGETNUMSUBSYSTEMS:
		return glGlobals.numMonitors;

	case DEVICEGETCURRENTSUBSYSTEM:
		return glGlobals.currentMonitor;

	case DEVICESETSUBSYSTEM:
		monitors = glfwGetMonitors(&glGlobals.numMonitors);
		if(n >= glGlobals.numMonitors)
			return 0;
		glGlobals.currentMonitor = n;
		glGlobals.monitor = monitors[glGlobals.currentMonitor];
		return 1;

	case DEVICEGETSUBSSYSTEMINFO:
		monitors = glfwGetMonitors(&glGlobals.numMonitors);
		if(n >= glGlobals.numMonitors)
			return 0;
		strncpy(((SubSystemInfo*)arg)->name, glfwGetMonitorName(monitors[n]), sizeof(SubSystemInfo::name));
		return 1;


	case DEVICEGETNUMVIDEOMODES:
		return glGlobals.numModes;

	case DEVICEGETCURRENTVIDEOMODE:
		return glGlobals.currentMode;

	case DEVICESETVIDEOMODE:
		if(n >= glGlobals.numModes)
			return 0;
		glGlobals.currentMode = n;
		return 1;

	case DEVICEGETVIDEOMODEINFO:
		rwmode = (VideoMode*)arg;
		rwmode->width = glGlobals.modes[n].mode.width;
		rwmode->height = glGlobals.modes[n].mode.height;
		rwmode->depth = glGlobals.modes[n].depth;
		rwmode->flags = glGlobals.modes[n].flags;
		return 1;

	case DEVICEGETMAXMULTISAMPLINGLEVELS:
		{
			GLint maxSamples;
			glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
			if(maxSamples == 0)
				return 1;
			return maxSamples;
		}
	case DEVICEGETMULTISAMPLINGLEVELS:
		if(glGlobals.numSamples == 0)
			return 1;
		return glGlobals.numSamples;
	case DEVICESETMULTISAMPLINGLEVELS:
		glGlobals.numSamples = (uint32)n;
		return 1;
	default:
		assert(0 && "not implemented");
		return 0;
	}
	return 1;
}

#else

not implemented

#endif

Device renderdevice = {
	-1.0f, 1.0f,
	gl3::beginUpdate,
	gl3::endUpdate,
	gl3::clearCamera,
	gl3::showRaster,
	gl3::rasterRenderFast,
	gl3::setRenderState,
	gl3::getRenderState,
	gl3::im2DRenderLine,
	gl3::im2DRenderTriangle,
	gl3::im2DRenderPrimitive,
	gl3::im2DRenderIndexedPrimitive,
	gl3::im3DTransform,
	gl3::im3DRenderPrimitive,
	gl3::im3DRenderIndexedPrimitive,
	gl3::im3DEnd,
#ifdef LIBRW_SDL2
	gl3::deviceSystemSDL2
#elif defined(LIBRW_SDL3)
	gl3::deviceSystemSDL3
#elif defined(LIBRW_GLFW)
	gl3::deviceSystemGLFW
#else
	not implemented
#endif
};

}
}

#else
// urgh, probably should get rid of that eventually
#include "rwgl3.h"
namespace rw {
namespace gl3 { 
Gl3Caps gl3Caps;
bool32 needToReadBackTextures;
}
}
#endif
