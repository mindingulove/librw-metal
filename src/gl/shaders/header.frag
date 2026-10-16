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

#if defined(HAS_VEL_OUT) && defined(VEL_PIPE)
layout(location = 1) out vec4 velOut;
FSIN vec4 v_posCur;
FSIN vec4 v_posPrev;
uniform float u_velOn;
float groundMark = 0.0;	// 1: level ground that applies the traced shadow itself (see DoWet)
void WriteVel(void)
{
	vec2 a = v_posCur.xy/v_posCur.w;
	vec2 b = v_posPrev.xy/v_posPrev.w;
#ifdef PED_MARK
	velOut = vec4((a - b)*0.5, groundMark, u_velOn*2.0);	// 2 = character
#else
	velOut = vec4((a - b)*0.5, groundMark, u_velOn);
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
uniform vec2 u_rtSize;
uniform vec2 u_rtParams;	// x: reflections available, y: strength
uniform sampler2D u_shadowPrev;	// last frame's traced sun shadows (softened, same size)
uniform vec2 u_shadowParams;	// x: shadows available, y: strength
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
	// Level ground takes the traced sun shadow here, in its own shader, from last frame's image (exact for static ground).
	// Shadow and puddle then combine inside one shader instead of two screen passes fighting over the same pixels: the
	// shadow darkens the ground, the puddle mirror is mixed over it afterwards and is not darkened.
	bool shadowOn = mod(u_shadowParams.x, 2.0) > 0.5, coverOn = u_shadowParams.x > 1.5;
	if(groundMark > 0.5 && shadowOn && v_posPrev.w > 0.0){
		vec2 spu = (v_posPrev.xy/v_posPrev.w)*0.5 + 0.5;
		if(spu.x > 0.0 && spu.x < 1.0 && spu.y > 0.0 && spu.y < 1.0){
			float sh = texture(u_shadowPrev, spu).r;
			color.rgb *= 1.0 - clamp(sh*u_shadowParams.y, 0.0, 1.0);
		}
	}
	if(wetAmount > 0.001 && coverOn && v_posPrev.w > 0.0){
		// ground under a roof (traced last frame, the map does not move) stays dry
		vec2 cpu = (v_posPrev.xy/v_posPrev.w)*0.5 + 0.5;
		if(cpu.x > 0.0 && cpu.x < 1.0 && cpu.y > 0.0 && cpu.y < 1.0)
			wetAmount *= 1.0 - smoothstep(0.3, 0.8, texture(u_shadowPrev, cpu).b);
	}
	if(wetAmount > 0.001){
		float n = puddleNoise(v_rpos.xy*0.22)*0.65 + puddleNoise(v_rpos.xy*0.9)*0.35;
		float puddle = smoothstep(0.56, 0.72, n)*smoothstep(0.25, 0.6, clamp(wetAmount*1.4, 0.0, 1.0))*u_roadPuddles;
		wetWeight = wetAmount*mix(0.12, 0.9, v_rfres)*0.5 + puddle*0.95;
		color.rgb *= 1.0 - 0.15*puddle;
		// the reflection belongs to the ground itself: look it up where this point was last frame (the ground is static, so
		// that is exact) and mix it into the ground colour. Characters and cars drawn later simply cover it.
		if(u_rtParams.x > 0.5 && v_posPrev.w > 0.0){
			vec2 pu = (v_posPrev.xy/v_posPrev.w)*0.5 + 0.5;
			if(pu.x > 0.0 && pu.x < 1.0 && pu.y > 0.0 && pu.y < 1.0){
				vec4 r = texture(u_rtPrev, pu);
				// the filtered image holds colour*coverage in rgb and coverage in a: undo that so edges fade instead of flicker
				color.rgb = mix(color.rgb, r.rgb/max(r.a, 0.001), clamp(wetWeight*u_rtParams.y, 0.0, 1.0)*smoothstep(0.0, 0.6, r.a));
			}
		}
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
#define WRITE_REFL() (reflOut = vec4(normalize(v_rrefl)*0.5 + 0.5, clamp(wetWeight, 0.0, 1.0)))	// 8 bit: direction*0.5+0.5, weight
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
