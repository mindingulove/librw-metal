#ifdef USE_UBOS
layout(std140) uniform State
{
	vec2 u_alphaRef;
	vec4  u_fogData;
	vec4  u_fogColor;
};
#else
uniform vec4 u_alphaRef;

uniform vec4  u_fogData;
uniform vec4  u_fogColor;
#endif

#define u_fogStart (u_fogData.x)
#define u_fogEnd (u_fogData.y)
#define u_fogRange (u_fogData.z)
#define u_fogDisable (u_fogData.w)

#ifndef GL2
out vec4 fragColor;
#endif

void DoAlphaTest(float a)
{
#ifndef NO_ALPHATEST
	if(a < u_alphaRef.x || a >= u_alphaRef.y)
		discard;
#endif
}

#ifdef WET_SURFACE
uniform vec4 u_glassRefl; // strength, owner, vehicle flag
#endif

#if defined(HAS_VEL_OUT) && defined(VEL_PIPE)
layout(location = 1) out vec4 velOut;
FSIN vec4 v_posCur;
FSIN vec4 v_posPrev;
uniform float u_velOn;
uniform float u_actorId;
float groundMark = 0.0;	// 1: level ground that applies the traced shadow itself (see DoWet)
void WriteVel(void)
{
	vec2 a = v_posCur.xy/v_posCur.w;
	vec2 b = v_posPrev.xy/v_posPrev.w;
#ifdef PED_MARK
	velOut = vec4((a - b)*0.5, u_velOn>0.5?u_actorId:groundMark, u_velOn*2.0);	// 2 = character
#else
	velOut = vec4((a - b)*0.5, u_velOn>0.5?u_actorId:groundMark, u_velOn);
#ifdef WET_SURFACE
	if(u_glassRefl.x>0.001 && u_glassRefl.z>0.5){
		velOut.b=-u_glassRefl.y;velOut.a=-max(gl_FragCoord.z,1.0e-8);
	}
#endif
#endif
}
#define WRITE_VEL() WriteVel()
#else
#define WRITE_VEL()
#endif

#ifdef WET_SURFACE
FSIN vec3 v_rrefl;
FSIN float v_rw;
FSIN float v_rfres;
FSIN vec3 v_rpos;
FSIN vec3 v_rrel;
FSIN float v_glossS;
uniform sampler2D u_glossTex;	// the neo road gloss texture of this material
uniform float u_glossMult;	// 0 when the material has none
uniform float u_roadPuddles;	// 1: rain puddles on, 0: only the faint wet film
uniform sampler2D u_rtPrev;	// last frame's ray traced reflections (screen space, already softened)
uniform vec2 u_rtPrevJitter; // previous traced image uses the jittered projection
uniform vec2 u_rtSize;
uniform vec2 u_rtParams;	// x: reflections available, y: strength
uniform sampler2D u_shadowPrev;	// last frame's traced sun shadows (softened, same size)
uniform vec2 u_shadowParams;	// x: shadows available, y: strength
vec3 glassDir = vec3(0.0);
float glassWeight = 0.0;
layout(location = 2) out vec4 reflOut;
float wetWeight = 0.0;

float puddleHash(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx)*0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y)*p3.z);
}

float puddleNoise(vec2 p)
{
	vec2 i = floor(p), f = fract(p);
	f = f*f*(3.0 - 2.0*f);
	return mix(mix(puddleHash(i), puddleHash(i + vec2(1.0, 0.0)), f.x),
	           mix(puddleHash(i + vec2(0.0, 1.0)), puddleHash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// puddles collect in blobs of a few metres once it rains; they are dark and mirror-like, the rest of the ground only gets a
// faint film that shows at grazing angles. Any grey, dark, upward facing surface counts as road or pavement.
void DoWet(inout vec4 color)
{
	wetWeight = 0.0;
	float mx = max(color.r, max(color.g, color.b)), mn = min(color.r, min(color.g, color.b));
	// face normal from the screen space derivatives of the world position (the map's models carry no normals)
	vec3 faceN = normalize(cross(dFdx(v_rrel), dFdy(v_rrel)));
	// only truly level ground: on bumpy ground the per-triangle normal changes from face to face and puddles would flicker
	float up = smoothstep(0.93, 0.985, abs(faceN.z));
	// The map has several overlapping ground layers (road, pavement, terrain underneath) that fight for the same pixels far
	// away. Every layer must decide the same way, so the test depends only on the position and level of the surface, never on
	// the colour of this layer's texture.
	float surface = up;
	float wetAmount = v_rw*surface;
	groundMark = up > 0.5 ? 1.0 : 0.0;
	// Sun shadows are composited from the current trace after geometry is complete. Sampling last frame here
	// turned pixels previously covered by a character/car into bright, unshadowed cutouts on the ground.
	bool coverOn = u_shadowParams.x > 1.5;
	glassWeight = 0.0;
	if(u_glassRefl.x > 0.001){
		// glass and windows: a mirror direction from the face normal; the traced colour of last frame is mixed in below
		vec3 viewDir = normalize(v_rrel);
		vec3 gn = dot(faceN, viewDir) > 0.0 ? -faceN : faceN;
		float cosT = clamp(-dot(viewDir, gn), 0.0, 1.0);
		float fres = 0.04 + 0.96*pow(1.0 - cosT, 5.0);
		glassDir = reflect(viewDir, gn);
		glassWeight = u_glassRefl.x*mix(0.3, 1.0, fres)*(u_glassRefl.z > 0.5 ? 1.0 : 1.0 - smoothstep(0.75, 0.95, abs(gn.z)));
		// The current-frame trace is composited after opaque and transparent geometry.

	}
	if(wetAmount > 0.001 && coverOn && v_posPrev.w > 0.0){
		// ground under a roof (traced last frame, the map does not move) stays dry
		vec2 cpu = (v_posPrev.xy/v_posPrev.w)*0.5 + 0.5 + u_rtPrevJitter;
		if(cpu.x > 0.0 && cpu.x < 1.0 && cpu.y > 0.0 && cpu.y < 1.0)
			wetAmount *= 1.0 - smoothstep(0.3, 0.8, texture(u_shadowPrev, cpu).b);
	}
	if(wetAmount > 0.001){
		float n = puddleNoise(v_rpos.xy*0.22)*0.65 + puddleNoise(v_rpos.xy*0.9)*0.35;
		float puddle = smoothstep(0.56, 0.72, n)*smoothstep(0.25, 0.6, clamp(wetAmount*1.4, 0.0, 1.0))*u_roadPuddles;
		wetWeight = wetAmount*mix(0.12, 0.9, v_rfres)*0.5 + puddle*0.95;
		color.rgb *= 1.0 - 0.15*puddle;
		// Current-frame compositing prevents old vehicle silhouettes entering wet-road colour.

	}
}
// The neo road gloss used to be a second additive pass over every road. Doing it in the same draw halves the road draw calls.
void DoGloss(inout vec4 color, vec2 uv, float fog)
{
	if(u_glossMult > 0.0){
		float s = v_glossS;
		float s2 = s*s; s2 *= s2; s2 *= s2;	// ^8
		color.rgb += texture(u_glossTex, vec2(uv.x, 1.0 - uv.y)).rgb*s2*u_glossMult*fog;
	}
}
#define WET_FS(color) DoWet(color)
#define GLOSS_FS(color) DoGloss(color, v_tex0, v_fog)
#define WRITE_REFL() (reflOut = glassWeight > wetWeight ? vec4(normalize(glassDir)*0.5 + 0.5, clamp(glassWeight, 0.0, 1.0)) : vec4(normalize(v_rrefl)*0.5 + 0.5, clamp(wetWeight, 0.0, 1.0)))	// half-float direction*0.5+0.5, weight
#elif defined(ZERO_REFL)
// shaders of things that are not wet ground (characters, other materials) clear the reflection weight under them
layout(location = 2) out vec4 reflOut;
#define WET_FS(color)
#define GLOSS_FS(color)
#define WRITE_REFL() (reflOut = vec4(0.0))
#else
#define WET_FS(color)
#define GLOSS_FS(color)
#define WRITE_REFL()
#endif
