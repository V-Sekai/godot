#[vertex]

#version 450

#VERSION_DEFINES

layout(location = 0) out vec2 uv_interp;

void main() {
	vec2 base_arr[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
	gl_Position = vec4(base_arr[gl_VertexIndex], 0.0, 1.0);
	uv_interp = clamp(gl_Position.xy, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0;
}

#[fragment]

#version 450

#VERSION_DEFINES

#if defined(USE_MULTIVIEW)
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#else
#define ViewIndex 0
#endif // USE_MULTIVIEW

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

// Define structs BEFORE using them in buffer declarations
struct TileData {
	uint fragment_head;
	uint min_depth;
	uint max_depth;
	uint fragment_count;
};

struct FragmentData {
	uint next;
	uint depth_packed;
	uint color_packed;
	uint alpha_packed;
};

// Now declare buffers using the structs
layout(set = 0, binding = 0) uniform sampler2D u_opaque_tex;
layout(set = 0, binding = 1) restrict readonly buffer TileBuffer {
	TileData tiles[];
};
layout(set = 0, binding = 2) restrict readonly buffer FragBuffer {
	FragmentData frags[];
};
layout(set = 0, binding = 3, std140) uniform UBO {
	int viewport_width;
	int viewport_height;
	int tile_count_x;
	int tile_count_y;
	int tile_size;
	int max_frags;
	int view_count;
	int multiview_enabled;
	// Padding to align to 48 bytes (3 vec4s for std140)
	uint padding[4];
}
ubo;

void main() {
	int view = (ubo.multiview_enabled > 0) ? int(ViewIndex) : 0;
	int tile_x = int(uv_interp.x * float(ubo.tile_count_x));
	int tile_y = int(uv_interp.y * float(ubo.tile_count_y));
	uint tile_idx = uint(tile_y * ubo.tile_count_x + tile_x);
	uint entry_idx = tile_idx * uint(ubo.view_count) + uint(view);

	TileData tile;
	if (entry_idx >= uint(ubo.tile_count_x * ubo.tile_count_y * ubo.view_count)) {
		frag_color = texture(u_opaque_tex, uv_interp);
		return;
	}

	tile = tiles[entry_idx];

	if (tile.fragment_head == ~0u || tile.fragment_count == 0u) {
		frag_color = texture(u_opaque_tex, uv_interp);
		return;
	}

	vec4 background = texture(u_opaque_tex, uv_interp);
	vec4 accum = background;

	// Collect fragments
	FragmentData collected[64];
	int num_frags = 0;
	uint current = tile.fragment_head;
	while (current != ~0u && num_frags < ubo.max_frags) {
		collected[num_frags] = frags[current];
		num_frags++;
		current = frags[current].next;
	}

	// Bubble sort ascending depth (near first: small depth_packed first)
	for (int i = 0; i < num_frags - 1; ++i) {
		for (int j = 0; j < num_frags - i - 1; ++j) {
			if (collected[j].depth_packed > collected[j + 1].depth_packed) {
				FragmentData temp = collected[j];
				collected[j] = collected[j + 1];
				collected[j + 1] = temp;
			}
		}
	}

	// Blend from far (array[num_frags-1]) to near (array[0])
	for (int i = num_frags - 1; i >= 0; --i) {
		FragmentData f = collected[i];
		uint col = f.color_packed;
		float r = float((col >> 24u) & 0xFFu) / 255.0;
		float g = float((col >> 16u) & 0xFFu) / 255.0;
		float b = float((col >> 8u) & 0xFFu) / 255.0;
		float a = float(col & 0xFFu) / 255.0;
		vec4 fcolor = vec4(r, g, b, a);
		accum.rgb = accum.rgb * (1.0 - fcolor.a) + fcolor.rgb * fcolor.a;
	}

	frag_color = vec4(accum.rgb, 1.0);
}
