uniform sampler2D tex0;

FSIN vec4 v_color;
FSIN vec2 v_tex0;
FSIN float v_fog;
#ifdef REFL_OUT
FSIN vec3 v_rrefl;
FSIN float v_rw;
FSIN float v_rfres;
FSIN vec2 v_rpos;
FSIN float v_rup;
uniform float u_roadPuddles;	// 1: rain puddles on, 0: only the faint wet film
layout(location = 2) out vec4 reflOut;

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
#endif

void
main(void)
{
	vec4 color = v_color*texture(tex0, vec2(v_tex0.x, 1.0-v_tex0.y));
#ifdef REFL_OUT
	// rain: puddles collect in blobs of a few metres once the road is wet; they are dark and mirror-like, the rest of the
	// road only gets a faint film that shows at grazing angles
	float wetN = clamp(v_rw*1.4, 0.0, 1.0);
	float puddle = 0.0;
	float weight = 0.0;
	// any grey, dark, upward facing surface counts as road or pavement (not every road of the map is flagged as one)
	float mx = max(color.r, max(color.g, color.b)), mn = min(color.r, min(color.g, color.b));
	float surface = v_rup*(1.0 - smoothstep(0.22, 0.42, (mx - mn)/max(mx, 0.01)))*(1.0 - smoothstep(0.55, 0.85, mx));
	float wetAmount = v_rw*surface;
	if(wetAmount > 0.001){
		float n = puddleNoise(v_rpos*0.22)*0.65 + puddleNoise(v_rpos*0.9)*0.35;
		puddle = smoothstep(0.50, 0.58, n)*smoothstep(0.25, 0.6, clamp(wetAmount*1.4, 0.0, 1.0))*u_roadPuddles;
		weight = wetAmount*mix(0.12, 0.9, v_rfres)*0.5 + puddle*1.4;
		color.rgb *= 1.0 - 0.35*puddle;
	}
#endif
	color.rgb = mix(u_fogColor.rgb, color.rgb, v_fog);
	DoAlphaTest(color.a);
	WRITE_VEL();
#ifdef REFL_OUT
	reflOut = vec4(normalize(v_rrefl), weight);
#endif
	FRAGCOLOR(color);
}
