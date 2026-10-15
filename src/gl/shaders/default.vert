VSIN(ATTRIB_POS)	vec3 in_pos;

VSOUT vec4 v_color;
VSOUT vec2 v_tex0;
VSOUT float v_fog;
#ifdef REFL_OUT
// wet roads: world-space mirror direction and strength for the ray traced reflections (u_roadWet is 0 for everything but roads)
uniform float u_roadWet;
VSOUT vec3 v_rrefl;
VSOUT float v_rw;		// road wetness (0 for everything but roads)
VSOUT float v_rfres;	// fresnel term of the viewing angle
VSOUT vec2 v_rpos;	// world position, for the puddle pattern
#endif

void
main(void)
{
	vec4 Vertex = u_world * vec4(in_pos, 1.0);
	gl_Position = u_proj * u_view * Vertex;
	VELOCITY_VS(Vertex, in_pos);
#ifdef REFL_OUT
	{
		vec3 eye = -(transpose(mat3(u_view))*u_view[3].xyz);
		vec3 V = normalize(eye - Vertex.xyz);
		v_rrefl = vec3(-V.x, -V.y, V.z);	// mirror about the horizontal road surface
		v_rfres = pow(1.0 - clamp(V.z, 0.0, 1.0), 4.0);
		v_rw = u_roadWet;
		v_rpos = Vertex.xy;
	}
#endif
	vec3 Normal = mat3(u_world) * in_normal;

	v_tex0 = in_tex0;

	v_color = in_color;
	v_color.rgb += u_ambLight.rgb*surfAmbient;
	v_color.rgb += DoDynamicLight(Vertex.xyz, Normal)*surfDiffuse;
	v_color = clamp(v_color, 0.0, 1.0);
	v_color *= u_matColor;

	v_fog = DoFog(gl_Position.w);
}
