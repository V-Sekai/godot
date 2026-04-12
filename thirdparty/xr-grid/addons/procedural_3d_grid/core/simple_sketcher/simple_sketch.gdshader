shader_type spatial;

render_mode unshaded;
render_mode skip_vertex_transform;
render_mode cull_disabled;

mat4 get_line_billboard(vec3 a,vec3 b){
    mat4 billboard = mat4(1);
    billboard[0].xyz = (b-a)/2.;
    billboard[1].xyz = normalize( cross(b,a) );
    billboard[2].xyz = normalize( cross(billboard[0].xyz, billboard[1].xyz) );
    billboard[3].xyz = (b+a)/2.;
    return billboard;
}

void vertex(){
	VERTEX = ( MODELVIEW_MATRIX * vec4(VERTEX,1) ).xyz;
	vec3 tangent = mat3(MODELVIEW_MATRIX) * TANGENT;
	
	UV *= length(MODELVIEW_MATRIX[0].xyz);
	VERTEX += UV.x * get_line_billboard(VERTEX,VERTEX+tangent)[0].xyz + UV.y * get_line_billboard(VERTEX,VERTEX+tangent)[1].xyz;

	UV.y = normalize(UV.y);
}
void fragment(){
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
	
	// Add anti-aliasing.
	ALPHA *= clamp( (1.0-abs(UV.y))/fwidth(UV.y), 0.0, 1.0);
}