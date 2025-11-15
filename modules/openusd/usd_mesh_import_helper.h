/**************************************************************************/
/*  usd_mesh_import_helper.h                                              */
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

#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/mesh.h"

// TODO: Update import helper to use TinyUSDZ (currently using GLTF-based import)
// TinyUSDZ headers
#include "tinyusdz.hh"
#include "prim-types.hh"
#include "usdGeom.hh"
#include "value-types.hh"

namespace tinyusdz {
	class Prim;
}

class UsdMeshImportHelper {
public:
	UsdMeshImportHelper();
	~UsdMeshImportHelper();

	// Import a USD mesh prim into a Godot mesh, delegates to the
	// appropriate import method based on the prim type
	// TODO: Update to use TinyUSDZ API (tinyusdz::Prim)
	Ref<Mesh> import_mesh_from_prim(const tinyusdz::Prim &p_prim);

	Ref<BoxMesh> import_cube(const tinyusdz::Prim &p_prim);
	Ref<SphereMesh> import_sphere(const tinyusdz::GeomSphere &p_sphere);
	Ref<CylinderMesh> import_cylinder(const tinyusdz::GeomCylinder &p_cylinder);
	Ref<CylinderMesh> import_cone(const tinyusdz::GeomCone &p_cone);
	Ref<CapsuleMesh> import_capsule(const tinyusdz::GeomCapsule &p_capsule);
	Ref<Mesh> import_geom_mesh(const tinyusdz::GeomMesh &p_mesh);

	// Helper method to handle non-uniform scaling
	void apply_non_uniform_scale(Ref<Mesh> p_mesh, const tinyusdz::value::float3 &p_scale);

	Ref<StandardMaterial3D> create_material(const tinyusdz::Prim &p_prim);
};
