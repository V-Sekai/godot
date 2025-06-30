/**************************************************************************/
/*  voxel_modifier_sdf.h                                                  */
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

#include "voxel_modifier.h"

namespace zylann::voxel {

class VoxelModifierSdf : public VoxelModifier {
public:
	enum Operation { //
		OP_ADD = 0,
		OP_SUBTRACT = 1
	};

	inline Operation get_operation() const {
		return _operation;
	}

	void set_operation(Operation op);

	inline float get_smoothness() const {
		return _smoothness;
	}

	void set_smoothness(float p_smoothness);

	bool is_sdf() const override {
		return true;
	}

protected:
#ifdef VOXEL_ENABLE_GPU
	void update_base_shader_data_no_lock();
#endif

private:
	Operation _operation = OP_ADD;
	float _smoothness = 0.f;
	// float _margin = 0.f;
};

} // namespace zylann::voxel
