#version 450

layout(location = 0) out vec2 frag_uv;

void main() {
	const vec2 positions[3] = vec2[](
			vec2(-1.0, -1.0),
			vec2(3.0, -1.0),
			vec2(-1.0, 3.0));
	const vec2 uvs[3] = vec2[](
			vec2(0.0, 0.0),
			vec2(2.0, 0.0),
			vec2(0.0, 2.0));
	gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
	frag_uv = uvs[gl_VertexIndex];
}

/* FRAGMENT SHADER START */
#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 o_color;

layout(binding = 0) uniform sampler2D opaque_texture;

struct TileData {
	uint32_t min_depth;
	uint32_t max_depth;
	uint32_t fragment_head;
	uint32_t fragment_count;
};
layout(std430, binding = 1) buffer OITTileBuffer {
	TileData tiles[];
};

struct FragmentData {
	float depth;
	uint32_t color_packed;
	uint32_t next_index;
};
layout(std430, binding = 2) buffer OITFragmentBuffer {
	FragmentData fragments[];
};

layout(std140, binding = 3) uniform OITParamsUBO {
	int multiview_enabled;
	int view_count;
	int viewport_width;
	int viewport_height;
	int max_frags;
	uint padding;
};

#define TILE_SIZE 16
#define DEPTH_BINS 4
#define MAX_FRAGS 16

vec4 unpack_rgba(uint packed) {
	vec4 c;
	c.a = float((packed >> 24u) & 0xFFu) / 255.0;
	c.r = float((packed >> 16u) & 0xFFu) / 255.0;
	c.g = float((packed >> 8u) & 0xFFu) / 255.0;
	c.b = float(packed & 0xFFu) / 255.0;
	return c;
}

uint get_tile_index(ivec2 tile, uint bin, uint view) {
	uint tiles_x = (uint(viewport_width) + TILE_SIZE - 1u) / TILE_SIZE;
	uint tiles_per_row = tiles_x;
	uint base = (uint(tile.y) * tiles_per_row + uint(tile.x)) * DEPTH_BINS;
	return base + bin * uint(view_count) + view;
}

void main() {
	vec2 uv = frag_uv * 0.5;
	uv.y = 1.0 - uv.y; // Flip Y
	uint view_idx = (multiview_enabled > 0) ? uint(gl_ViewIndex) : 0u;
	ivec2 coord = ivec2(uv * vec2(float(viewport_width), float(viewport_height)));
	ivec2 tile = coord / TILE_SIZE;
	vec3 accum = vec3(0.0);
	float trans = 1.0;
	for (uint b = 0u; b < uint(DEPTH_BINS); b++) {
		uint tile_idx = get_tile_index(tile, b, view_idx);
		TileData td = tiles[tile_idx];
		if (td.fragment_count == 0u) {
			continue;
		}
		uint idx = td.fragment_head;
		while (idx != 0xFFFFFFFFu) {
			if (idx >= uint(max_frags)) {
				break;
			}
			vec4 frag = unpack_rgba(fragments[idx].color_packed);
			float alpha = frag.a;
			accum += frag.rgb * alpha * trans;
			trans *= 1.0 - alpha;
			if (trans < 0.01) {
				break;
			}
			idx = fragments[idx].next_index;
		}
	}
	vec3 opaque_c = texture(opaque_texture, uv).rgb;
	accum += opaque_c * trans;
	o_color = vec4(accum, 1.0);
}
