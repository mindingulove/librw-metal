uniform sampler2D tex0;

FSIN vec4 v_color;
FSIN vec2 v_tex0;
FSIN float v_fog;
#ifdef REFL_OUT
FSIN vec3 v_rrefl;
FSIN float v_rw;
layout(location = 2) out vec4 reflOut;
#endif

void
main(void)
{
	vec4 color = v_color*texture(tex0, vec2(v_tex0.x, 1.0-v_tex0.y));
	color.rgb = mix(u_fogColor.rgb, color.rgb, v_fog);
	DoAlphaTest(color.a);
	WRITE_VEL();
#ifdef REFL_OUT
	reflOut = vec4(normalize(v_rrefl), v_rw);
#endif
	FRAGCOLOR(color);
}
