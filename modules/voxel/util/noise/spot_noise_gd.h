/**************************************************************************/
/*  spot_noise_gd.h                                                       */
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

#include "../godot/classes/resource.h"

namespace zylann {

class ZN_SpotNoise : public Resource {
	GDCLASS(ZN_SpotNoise, Resource);

public:
	int get_seed() const;
	void set_seed(int seed);

	float get_cell_size() const;
	void set_cell_size(float cell_size);

	float get_spot_radius() const;
	void set_spot_radius(float r);

	float get_jitter() const;
	void set_jitter(float jitter);

	float get_noise_2d(real_t x, real_t y) const;
	float get_noise_3d(real_t x, real_t y, real_t z) const;

	float get_noise_2dv(Vector2 pos) const;
	float get_noise_3dv(Vector3 pos) const;

	PackedVector2Array get_spot_positions_in_area_2d(Rect2 rect) const;
	PackedVector3Array get_spot_positions_in_area_3d(AABB aabb) const;

private:
	static void _bind_methods();

	int _seed = 1337;
	float _cell_size = 32.f;
	float _spot_radius = 3.f;
	float _jitter = 0.9f;
};

}; // namespace zylann
