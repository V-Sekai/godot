/**************************************************************************/
/*  test_csg.h                                                            */
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

#include "modules/csg/csg_shape.h"

#include "tests/test_macros.h"

namespace TestCSG {

TEST_CASE("[SceneTree][CSG] CSGBox3D") {
	SUBCASE("[SceneTree][CSG] CSGBox3D: Basic shape generation") {
		CSGBox3D *box = memnew(CSGBox3D);
		SceneTree::get_singleton()->get_root()->add_child(box);

		box->set_size(Vector3(2.0, 2.0, 2.0));
		Array faces = box->get_meshes();

		CHECK_MESSAGE(faces.size() > 0, "Box should generate meshes");

		SceneTree::get_singleton()->get_root()->remove_child(box);
		memdelete(box);
	}

	SUBCASE("[SceneTree][CSG] CSGBox3D: Property getters and setters") {
		CSGBox3D *box = memnew(CSGBox3D);
		box->set_size(Vector3(3.0, 4.0, 5.0));
		CHECK(box->get_size().is_equal_approx(Vector3(3.0, 4.0, 5.0)));

		memdelete(box);
	}
}

TEST_CASE("[SceneTree][CSG] CSGCylinder3D") {
	SUBCASE("[SceneTree][CSG] CSGCylinder3D: Basic shape generation") {
		CSGCylinder3D *cylinder = memnew(CSGCylinder3D);
		SceneTree::get_singleton()->get_root()->add_child(cylinder);

		cylinder->set_radius(1.0);
		cylinder->set_height(2.0);
		Array faces = cylinder->get_meshes();

		CHECK_MESSAGE(faces.size() > 0, "Cylinder should generate meshes");

		SceneTree::get_singleton()->get_root()->remove_child(cylinder);
		memdelete(cylinder);
	}

	SUBCASE("[SceneTree][CSG] CSGCylinder3D: Property getters and setters") {
		CSGCylinder3D *cylinder = memnew(CSGCylinder3D);
		cylinder->set_radius(1.5);
		cylinder->set_height(3.0);
		CHECK(cylinder->get_radius() == doctest::Approx(1.5));
		CHECK(cylinder->get_height() == doctest::Approx(3.0));

		memdelete(cylinder);
	}
}

TEST_CASE("[SceneTree][CSG] CSGSphere3D") {
	SUBCASE("[SceneTree][CSG] CSGSphere3D: Basic shape generation") {
		CSGSphere3D *sphere = memnew(CSGSphere3D);
		SceneTree::get_singleton()->get_root()->add_child(sphere);

		sphere->set_radius(1.0);
		Array faces = sphere->get_meshes();

		CHECK_MESSAGE(faces.size() > 0, "Sphere should generate meshes");

		SceneTree::get_singleton()->get_root()->remove_child(sphere);
		memdelete(sphere);
	}

	SUBCASE("[SceneTree][CSG] CSGSphere3D: Property getters and setters") {
		CSGSphere3D *sphere = memnew(CSGSphere3D);
		sphere->set_radius(2.5);
		CHECK(sphere->get_radius() == doctest::Approx(2.5));

		memdelete(sphere);
	}
}

TEST_CASE("[SceneTree][CSG] CSGTorus3D") {
	SUBCASE("[SceneTree][CSG] CSGTorus3D: Basic shape generation") {
		CSGTorus3D *torus = memnew(CSGTorus3D);
		SceneTree::get_singleton()->get_root()->add_child(torus);

		torus->set_inner_radius(0.25);
		torus->set_outer_radius(0.5);
		Array faces = torus->get_meshes();

		CHECK_MESSAGE(faces.size() > 0, "Torus should generate meshes");

		SceneTree::get_singleton()->get_root()->remove_child(torus);
		memdelete(torus);
	}

	SUBCASE("[SceneTree][CSG] CSGTorus3D: Property getters and setters") {
		CSGTorus3D *torus = memnew(CSGTorus3D);
		torus->set_inner_radius(0.3);
		torus->set_outer_radius(0.6);
		CHECK(torus->get_inner_radius() == doctest::Approx(0.3));
		CHECK(torus->get_outer_radius() == doctest::Approx(0.6));

		memdelete(torus);
	}
}

TEST_CASE("[SceneTree][CSG] CSGCombiner3D") {
	SUBCASE("[SceneTree][CSG] CSGCombiner3D: Basic functionality") {
		CSGCombiner3D *combiner = memnew(CSGCombiner3D);
		SceneTree::get_singleton()->get_root()->add_child(combiner);

		// Add a child shape
		CSGBox3D *box = memnew(CSGBox3D);
		combiner->add_child(box);

		Array faces = combiner->get_meshes();
		CHECK_MESSAGE(faces.size() > 0, "Combiner should generate meshes when it has children");

		SceneTree::get_singleton()->get_root()->remove_child(combiner);
		memdelete(combiner);
	}
}

TEST_CASE("[SceneTree][CSG] CSGMesh3D") {
	SUBCASE("[SceneTree][CSG] CSGMesh3D: Basic functionality") {
		CSGMesh3D *mesh = memnew(CSGMesh3D);
		SceneTree::get_singleton()->get_root()->add_child(mesh);

		// Create a simple mesh
		Ref<ArrayMesh> array_mesh;
		array_mesh.instantiate();
		mesh->set_mesh(array_mesh);

		Array faces = mesh->get_meshes();
		// CSGMesh3D may not generate faces if the mesh is empty
		// Just check that it doesn't crash

		SceneTree::get_singleton()->get_root()->remove_child(mesh);
		memdelete(mesh);
	}
}

} // namespace TestCSG
