/**************************************************************************/
/*  test_diff_cloth_integration.h                                         */
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

#include "../diff_cloth_integration.h"
#include "../lbfgsb_capsule_fitter_solver.h"
#include "core/variant/dictionary.h"
#include "scene/resources/mesh.h"
#include "test_lbfgsb_capsule_fitter_solver.h"
#include "tests/test_macros.h"

// Assuming TestLBFGSBCapsuleFitterSolver namespace contains helper functions
namespace TestLBFGSBCapsuleFitterSolver {
// Forward declare or include necessary helper functions if they are not public in LBFGSBCapsuleFitterSolver tests
// For example: static Ref<ArrayMesh> create_cylinder_points_mesh(float r, float h, int num_points_circle, int num_height_steps);
}

namespace TestDiffClothIntegration {

TEST_CASE("[SceneTree][DiffClothIntegration] Optimize Cloth Draping via External Tool" * doctest::skip(true)) {
	DiffClothIntegration *integrator = memnew(DiffClothIntegration);

	Ref<ArrayMesh> dummy_char_cage_mesh = TestLBFGSBCapsuleFitterSolver::create_cylinder_points_mesh(0.5f, 1.5f, 10, 3);
	Ref<ArrayMesh> dummy_input_cloth_mesh = TestLBFGSBCapsuleFitterSolver::create_cylinder_points_mesh(0.8f, 2.0f, 10, 5);

	integrator->set_character_collision_mesh(dummy_char_cage_mesh);
	integrator->set_input_cloth_mesh(dummy_input_cloth_mesh);

	Dictionary result = integrator->call(SNAME("optimize_cloth_drape"));

	INFO("DiffCloth Integration result: ", result);
	CHECK_MESSAGE(!result.is_empty(), "DiffCloth Integration result should not be empty.");
	CHECK_MESSAGE(result.has("optimized_cloth_mesh"), "Result should contain 'optimized_cloth_mesh'.");

	Ref<ArrayMesh> optimized_cloth_mesh = result["optimized_cloth_mesh"];
	CHECK_MESSAGE(optimized_cloth_mesh.is_valid(), "Optimized cloth mesh should be valid.");
	CHECK_MESSAGE(optimized_cloth_mesh->get_surface_count() > 0, "Optimized cloth mesh should have surfaces.");

	memdelete(integrator);
}

} // namespace TestDiffClothIntegration
