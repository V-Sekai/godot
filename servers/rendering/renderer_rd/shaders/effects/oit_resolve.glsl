#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

void main() {
	vec2 vertex_base;
	if (gl_VertexIndex == 0) {
		vertex_base = vec2(-1.0, -1.0);
	} else if (gl_VertexIndex == 1) {
		vertex_base = vec2(-1.0, 3.0);
	} else {
		vertex_base = vec2(3.0, -1.0);
	}
	gl_Position = vec4(vertex_base, 0.0, 1.0);
	uv_interp = clamp(vertex_base, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0;
}

#[fragment]

#version 450

#VERSION_DEFINES

layout(location = 0) in vec2 uv_interp;

layout(set = 0, binding = 0) uniform sampler2D accum_texture;
layout(set = 0, binding = 1) uniform sampler2D revealage_texture;
layout(set = 0, binding = 2) uniform sampler2D background_texture;

layout(location = 0) out vec4 final_color;

void main() {
	vec4 accum = texture(accum_texture, uv_interp);
	float revealage = texture(revealage_texture, uv_interp).r;
	vec4 background = texture(background_texture, uv_interp);

	// Weighted blended OIT resolve
	// Final color = accum.rgb / max(accum.a, epsilon)
	vec3 transparent_color = accum.rgb / max(accum.a, 0.0001);

	// Blend with background: transparent * (1 - revealage) + background * revealage
	float alpha = 1.0 - revealage;
	final_color = vec4(transparent_color * alpha + background.rgb * (1.0 - alpha), 1.0);
}
