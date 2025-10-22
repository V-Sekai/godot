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

layout(set = 0, binding = 0) uniform sampler2D source_color;
layout(set = 0, binding = 1) uniform sampler2D source_depth;

layout(location = 0) out vec4 accum_color;
layout(location = 1) out float revealage;

void main() {
	vec4 color = texture(source_color, uv_interp);
	float depth = texture(source_depth, uv_interp).r;

	// Weighted blended OIT
	// Weight = pow(1 - depth, 4) - small epsilon to avoid division by zero
	float weight = pow(1.0 - depth, 4.0) + 0.0001;

	// Accumulate: color.rgb * color.a * weight
	accum_color = vec4(color.rgb * color.a * weight, color.a * weight);

	// Revealage: color.a * weight
	revealage = color.a * weight;
}
