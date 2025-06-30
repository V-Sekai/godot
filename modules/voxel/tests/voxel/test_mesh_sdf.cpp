/**************************************************************************/
/*  test_mesh_sdf.cpp                                                     */
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

#include "test_mesh_sdf.h"
#include "../../edition/voxel_mesh_sdf_gd.h"

namespace zylann::voxel::tests {

void test_voxel_mesh_sdf_issue463() {
	Ref<VoxelMeshSDF> msdf;
	msdf.instantiate();

	Dictionary d;
	d["roman"] = 22;
	d[22] = 25;
	// TODO The original report was creating a BoxShape3D, but for reasons beyond my understanding, Godot's
	// PhysicsServer3D is still not created after `MODULE_INITIALIZATION_LEVEL_SERVERS`. And not even SCENE or EDITOR
	// levels. It's impossible to use a level to do anything with physics.... Go figure.
	//
	// Ref<BoxShape3D> shape1;
	// shape1.instantiate();
	// Ref<BoxShape3D> shape2;
	// shape2.instantiate();
	// d[shape1] = shape2;
	Ref<Resource> res1;
	res1.instantiate();
	Ref<Resource> res2;
	res2.instantiate();
	d[res1] = res2;

	ZN_ASSERT(msdf->has_method("_set_data"));
	// Setting invalid data should cause an error but not crash or leak
	msdf->call("_set_data", d);
}

} // namespace zylann::voxel::tests
