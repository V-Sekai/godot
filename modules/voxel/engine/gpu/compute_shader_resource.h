/**************************************************************************/
/*  compute_shader_resource.h                                             */
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

#pragma once

#include "../../util/containers/span.h"
#include "../../util/godot/classes/ref_counted.h"
#include "../../util/godot/core/packed_byte_array.h"
#include "../../util/godot/core/rid.h"
#include "../../util/godot/core/transform_3d.h"
#include "../../util/godot/core/vector3i.h"
#include "../../util/godot/macros.h"
#include "../../util/macros.h"
#include <memory>

ZN_GODOT_FORWARD_DECLARE(class Image);
ZN_GODOT_FORWARD_DECLARE(class Curve);
ZN_GODOT_FORWARD_DECLARE(class RenderingDevice);

namespace zylann::voxel {

// This is our own thin wrapper for resources created with RenderingDevice. We can't use Godot's regular resources
// because they either don't exist, or assume that we use the main RenderingDevice (the one used for rendering).
// We use our own RenderingDevice because we want it to run heavy compute shaders that are not frame-based,
// and we use it in a dedicated thread because Godot doesn't have APIs to asynchronously upload/download data from
// it. RD functions are blocking (waiting for sync) so we don't want to block either the main thread or the rendering
// thread when doing our stuff.
//
// In addition to all that, Godot forces us to use a RenderingDevice on the same thread it was created.
// This means any resource originating from the main thread has to defer actions to our own thread. Despite resource
// creation actually being allowed to call from a different thread, we still had to defer it: because the thread
// creating the RenderingDevice might not have done so when the main thread starts creating resources...
//
// We had to pick a strategy to ensure consistency, so all resources are refcounted, and all methods are deferred.
// GPU tasks using these resources must hold a reference to them as long as they are in use, not just their RID.
// Note: a similar approach could have been to use the refcount of higher-level resources (generators?) to do this,
// instead of having these wrappers?

struct ComputeShaderResourceInternal {
	enum Type {
		TYPE_TEXTURE_2D,
		TYPE_TEXTURE_3D,
		TYPE_STORAGE_BUFFER,
		TYPE_UNINITIALIZED,
		TYPE_DEINITIALIZED //
	};

	// ComputeShaderResource();
	// ComputeShaderResource(ComputeShaderResource &&other);
	// ~ComputeShaderResource();

	void clear(RenderingDevice &rd);
	bool is_valid() const;
	Type get_type() const;
	void create_texture_2d(RenderingDevice &rd, const Image &image);
	void create_texture_2d(RenderingDevice &rd, const Curve &curve);
	void create_texture_3d_float32(RenderingDevice &rd, const PackedByteArray &data, const Vector3i size);
	void create_storage_buffer(RenderingDevice &rd, const PackedByteArray &data);
	void update_storage_buffer(RenderingDevice &rd, const PackedByteArray &data);

	// void operator=(ComputeShaderResource &&src);

	Type type = TYPE_UNINITIALIZED;
	RID rid;
};

class ComputeShaderResource;

// Normally I would make these functions static methods, but C++ annoyingly allows calling static methods on an instance
// of the object, which is a clear misuse. Also MSVC fails to report [[nodiscard]] misuses when called on a shared_ptr.
// So to workaround all that we have to move functions out...
struct ComputeShaderResourceFactory {
	ComputeShaderResourceFactory() = delete;

	[[nodiscard]]
	static std::shared_ptr<ComputeShaderResource> create_texture_2d(const Ref<Image> &image);

	[[nodiscard]]
	static std::shared_ptr<ComputeShaderResource> create_texture_2d(const Ref<Curve> &curve);

	[[nodiscard]]
	static std::shared_ptr<ComputeShaderResource> create_texture_3d_zxy(
			Span<const float> fdata_zxy,
			const Vector3i size
	);

	[[nodiscard]]
	static std::shared_ptr<ComputeShaderResource> create_storage_buffer(const PackedByteArray &data);
};

// Must be created with `ComputeShaderResourceFactory` and passed around with `shared_ptr`
class ComputeShaderResource {
public:
	friend struct ComputeShaderResourceFactory;

	static void update_storage_buffer(const std::shared_ptr<ComputeShaderResource> &res, const PackedByteArray &data);

	ComputeShaderResourceInternal::Type get_type() const;

	~ComputeShaderResource();

	// Only use on GPU task thread
	RID get_rid() const;

private:
	ComputeShaderResourceInternal::Type _type = ComputeShaderResourceInternal::TYPE_UNINITIALIZED;
	ComputeShaderResourceInternal _internal;
};

// Converts a 3D transform into a 4x4 matrix with a layout usable in GLSL.
void transform3d_to_mat4(const Transform3D &t, Span<float> dst);

} // namespace zylann::voxel
