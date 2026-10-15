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
void WriteVel(void)
{
	vec2 a = v_posCur.xy/v_posCur.w;
	vec2 b = v_posPrev.xy/v_posPrev.w;
	velOut = vec4((a - b)*0.5, 0.0, u_velOn);
}
#define WRITE_VEL() WriteVel()
#else
#define WRITE_VEL()
#endif
