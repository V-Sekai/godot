/**************************************************************************/
/*  csg_sculpted_tube.h                                                   */
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

#include "csg_sculpted_primitive_base.h"

/**
 * Sculpted tube primitive with advanced sculpting parameters.
 * A hollow cylinder variant.
 */
class CSGSculptedTube3D : public CSGSculptedPrimitive3D {
	GDCLASS(CSGSculptedTube3D, CSGSculptedPrimitive3D);

	virtual CSGBrush *_build_brush() override;

	real_t inner_radius = 0.25;
	real_t outer_radius = 0.5;
	real_t height = 2.0;

protected:
	static void _bind_methods();

public:
	void set_inner_radius(const real_t p_inner_radius);
	real_t get_inner_radius() const;

	void set_outer_radius(const real_t p_outer_radius);
	real_t get_outer_radius() const;

	void set_height(const real_t p_height);
	real_t get_height() const;

	CSGSculptedTube3D();
};
