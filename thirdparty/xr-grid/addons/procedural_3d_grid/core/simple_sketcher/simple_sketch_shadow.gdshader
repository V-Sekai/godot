shader_type spatial;

render_mode unshaded;
render_mode skip_vertex_transform;
render_mode cull_disabled;
render_mode depth_draw_always;

mat4 getLineBillboard(vec3 a,vec3 b){
    mat4 billboard = mat4(1);
    billboard[0].xyz = (b-a)/2.;
    billboard[1].xyz = normalize( cross(b,a) );
    billboard[2].xyz = normalize( cross(billboard[0].xyz, billboard[1].xyz) );
    billboard[3].xyz = (b+a)/2.;
    return billboard;
}

varying vec3 model_pos;

void vertex(){
	vec3 tangent = TANGENT;
	
	float m = .7;
	VERTEX.x += VERTEX.y * m;
	tangent.x += tangent.y * m;
	
	UV *= 1.0;
	VERTEX += UV.x * tangent + UV.y * normalize(vec3(-tangent.z, 0, tangent.x));

	model_pos = VERTEX;

	VERTEX.y = 0.001;
	
	VERTEX = ( MODELVIEW_MATRIX * vec4(VERTEX,1) ).xyz;
}

void fragment(){
	ALBEDO = vec3(0.02);
	ALPHA = COLOR.a;
	
	if(model_pos.y < 0.0) discard;
}