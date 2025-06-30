/**************************************************************************/
/*  blocky_tint_sampler.h                                                 */
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

#include "../../util/math/color.h"
#include "../../util/math/vector3i.h"

namespace zylann::voxel {

class VoxelBuffer;

namespace blocky {

struct TintSampler {
	enum Mode {
		MODE_NONE,
		MODE_RAW,
		MODE_COUNT,
	};

	typedef Color (*Callback)(const TintSampler &, const Vector3i);

	const Callback f;
	const VoxelBuffer &voxels;

	static TintSampler create(const VoxelBuffer &p_voxels, const Mode mode);

	inline bool is_valid() const {
		return f != nullptr;
	}

	inline Color evaluate(const Vector3i pos) const {
		if (is_valid()) {
			return (*f)(*this, pos);
		} else {
			return Color(1, 1, 1, 1);
		}
	}
};

} // namespace blocky
} // namespace zylann::voxel
