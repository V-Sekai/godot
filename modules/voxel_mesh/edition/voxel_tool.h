/**************************************************************************/
/*  voxel_tool.h                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef VOXEL_TOOL_H
#define VOXEL_TOOL_H

#include "../util/math/box3i.h"
#include "funcs.h"
#include "voxel_raycast_result.h"

class VoxelBuffer;

// TODO Need to review VoxelTool to account for transformed volumes

// High-level generic voxel edition utility.
// Ease of use comes at cost.
// It's not a class to instantiate alone, get it from the voxel objects you want
// to work with
class VoxelTool : public RefCounted {
	GDCLASS(VoxelTool, RefCounted);

public:
	enum Mode { MODE_ADD,
		MODE_REMOVE,
		MODE_SET,
		MODE_TEXTURE_PAINT };

	VoxelTool();

	void set_value(uint64_t val);
	uint64_t get_value() const;

	void set_channel(int channel);
	int get_channel() const;

	void set_mode(Mode mode);
	Mode get_mode() const;

	void set_eraser_value(uint64_t value);
	uint64_t get_eraser_value() const;

	uint64_t get_voxel(VoxelVector3i pos) const;
	float get_voxel_f(VoxelVector3i pos) const;

	float get_sdf_scale() const;
	void set_sdf_scale(float s);

	void set_texture_index(int ti);
	int get_texture_index() const;

	void set_texture_opacity(float opacity);
	float get_texture_opacity() const;

	void set_texture_falloff(float falloff);
	float get_texture_falloff() const;

	// TODO Methods working on a whole area must use an implementation that
	// minimizes locking!

	// The following methods represent one edit each. Pick the correct one for the
	// job. For example, using `do_box` will be more efficient than calling
	// `do_point` many times.
	virtual void set_voxel(VoxelVector3i pos, uint64_t v);
	virtual void set_voxel_f(VoxelVector3i pos, float v);
	virtual void do_point(VoxelVector3i pos);
	virtual void do_line(VoxelVector3i begin, VoxelVector3i end);
	virtual void do_circle(VoxelVector3i pos, int radius,
			VoxelVector3i direction);
	virtual void do_sphere(Vector3 center, float radius);
	virtual void do_box(VoxelVector3i begin, VoxelVector3i end);

	void sdf_stamp_erase(Ref<VoxelBuffer> stamp, VoxelVector3i pos);

	virtual void copy(VoxelVector3i pos, Ref<VoxelBuffer> dst,
			uint8_t channels_mask);
	virtual void paste(VoxelVector3i pos, Ref<VoxelBuffer> p_voxels,
			uint8_t channels_mask, uint64_t mask_value);

	virtual Ref<VoxelRaycastResult> raycast(Vector3 pos, Vector3 dir,
			float max_distance,
			uint32_t collision_mask);

	// Checks if an edit affecting the given box can be applied, fully or
	// partially
	virtual bool is_area_editable(const Box3i &box) const;

	virtual void set_voxel_metadata(VoxelVector3i pos, Variant meta);
	virtual Variant get_voxel_metadata(VoxelVector3i pos);

protected:
	static void _bind_methods();

	// These methods never go alone, but may be used in others.
	// They don't represent an edit, they only abstract the lower-level API
	virtual uint64_t _get_voxel(VoxelVector3i pos) const;
	virtual float _get_voxel_f(VoxelVector3i pos) const;
	virtual void _set_voxel(VoxelVector3i pos, uint64_t v);
	virtual void _set_voxel_f(VoxelVector3i pos, float v);
	virtual void _post_edit(const Box3i &box);

private:
	// Bindings to convert to more specialized C++ types and handle virtuality,
	// cuz I don't know if it works by binding straight

	uint64_t _b_get_voxel(Vector3 pos) {
		return get_voxel(VoxelVector3i::from_floored(pos));
	}
	float _b_get_voxel_f(Vector3 pos) {
		return get_voxel_f(VoxelVector3i::from_floored(pos));
	}
	void _b_set_voxel(Vector3 pos, uint64_t v) {
		set_voxel(VoxelVector3i::from_floored(pos), v);
	}
	void _b_set_voxel_f(Vector3 pos, float v) {
		set_voxel_f(VoxelVector3i::from_floored(pos), v);
	}
	Ref<VoxelRaycastResult> _b_raycast(Vector3 pos, Vector3 dir,
			float max_distance,
			uint32_t collision_mask) {
		return raycast(pos, dir, max_distance, collision_mask);
	}
	void _b_do_point(Vector3 pos) { do_point(VoxelVector3i::from_floored(pos)); }
	void _b_do_line(Vector3 begin, Vector3 end) {
		do_line(VoxelVector3i::from_floored(begin),
				VoxelVector3i::from_floored(end));
	}
	void _b_do_circle(Vector3 pos, float radius, Vector3 direction) {
		do_circle(VoxelVector3i::from_floored(pos), radius,
				VoxelVector3i::from_floored(direction));
	}
	void _b_do_sphere(Vector3 pos, float radius) { do_sphere(pos, radius); }
	void _b_do_box(Vector3 begin, Vector3 end) {
		do_box(VoxelVector3i::from_floored(begin), VoxelVector3i(end));
	}
	void _b_copy(Vector3 pos, Ref<RefCounted> voxels, int channel_mask) {
		copy(VoxelVector3i::from_floored(pos), voxels, channel_mask);
	}
	void _b_paste(Vector3 pos, Ref<RefCounted> voxels, int channels_mask,
			int mask_value) {
		paste(VoxelVector3i::from_floored(pos), voxels, channels_mask, mask_value);
	}

	Variant _b_get_voxel_metadata(Vector3 pos) {
		return get_voxel_metadata(VoxelVector3i::from_floored(pos));
	}
	void _b_set_voxel_metadata(Vector3 pos, Variant meta) {
		return set_voxel_metadata(VoxelVector3i::from_floored(pos), meta);
	}

	bool _b_is_area_editable(AABB box) {
		return is_area_editable(Box3i(VoxelVector3i::from_floored(box.position),
				VoxelVector3i::from_floored(box.size)));
	}

protected:
	uint64_t _value = 0;
	uint64_t _eraser_value = 0; // air
	int _channel = 0;
	float _sdf_scale = 1.f;
	Mode _mode = MODE_ADD;

	struct TextureParams {
		float opacity = 1.f;
		float sharpness = 2.f;
		unsigned int index = 0;
	};

	struct TextureBlendSphereOp {
		Vector3 center;
		float radius;
		float radius_squared;
		TextureParams tp;

		TextureBlendSphereOp(Vector3 pCenter, float pRadius, TextureParams pTp) {
			center = pCenter;
			radius = pRadius;
			radius_squared = pRadius * pRadius;
			tp = pTp;
		}

		inline void operator()(VoxelVector3i pos, uint16_t &indices,
				uint16_t &weights) const {
			const float distance_squared = pos.to_vec3().distance_squared_to(center);
			if (distance_squared < radius_squared) {
				const float distance_from_radius =
						radius - Math::sqrt(distance_squared);
				const float target_weight =
						tp.opacity *
						clamp(tp.sharpness * (distance_from_radius / radius), 0.f, 1.f);
				blend_texture_packed_u16(tp.index, target_weight, indices, weights);
			}
		}
	};

	// Used on smooth terrain
	TextureParams _texture_params;
};

VARIANT_ENUM_CAST(VoxelTool::Mode)

#endif // VOXEL_TOOL_H
