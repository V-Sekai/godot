/**************************************************************************/
/*  diff_cloth_integration.h                                              */
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

#include "core/object/object.h"
#include "core/variant/dictionary.h"
#include "lbfgsbpp.h"
#include "scene/resources/mesh.h"

class DiffClothIntegration : public LBFGSBSolver {
	GDCLASS(DiffClothIntegration, LBFGSBSolver);

public:
	Ref<ArrayMesh> character_collision_mesh;
	Ref<ArrayMesh> input_cloth_mesh;

	void set_character_collision_mesh(Ref<ArrayMesh> p_mesh);
	Ref<ArrayMesh> get_character_collision_mesh() const;

	void set_input_cloth_mesh(Ref<ArrayMesh> p_mesh);
	Ref<ArrayMesh> get_input_cloth_mesh() const;

	Dictionary optimize_cloth_drape();

protected:
	static void _bind_methods();

public:
	virtual double call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) override; // Added override
	DiffClothIntegration();
	~DiffClothIntegration();
};
