/**************************************************************************/
/*  usd_mesh_import_helper.cpp                                            */
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

#include "usd_mesh_import_helper.h"

// TODO: Update import functionality to use TinyUSDZ API
// Old OpenUSD code removed - all methods stubbed out

UsdMeshImportHelper::UsdMeshImportHelper() {
	// Constructor
}

UsdMeshImportHelper::~UsdMeshImportHelper() {
	// Destructor
}

Ref<Mesh> UsdMeshImportHelper::import_mesh_from_prim(const tinyusdz::Prim &p_prim) {
	ERR_PRINT("USD Import: import_mesh_from_prim not yet implemented with TinyUSDZ");
	// TODO: Implement using TinyUSDZ API
	return Ref<Mesh>();
}

Ref<BoxMesh> UsdMeshImportHelper::import_cube(const tinyusdz::Prim &p_prim) {
	ERR_PRINT("USD Import: import_cube not yet implemented with TinyUSDZ");
	// TODO: Implement using TinyUSDZ API
	return Ref<BoxMesh>();
}

Ref<SphereMesh> UsdMeshImportHelper::import_sphere(const tinyusdz::GeomSphere &p_sphere) {
	ERR_PRINT("USD Import: import_sphere not yet implemented with TinyUSDZ");
	// TODO: Implement using TinyUSDZ API
	return Ref<SphereMesh>();
}

Ref<CylinderMesh> UsdMeshImportHelper::import_cylinder(const tinyusdz::GeomCylinder &p_cylinder) {
	ERR_PRINT("USD Import: import_cylinder not yet implemented with TinyUSDZ");
	// TODO: Implement using TinyUSDZ API
	return Ref<CylinderMesh>();
}

Ref<CylinderMesh> UsdMeshImportHelper::import_cone(const tinyusdz::GeomCone &p_cone) {
	ERR_PRINT("USD Import: import_cone not yet implemented with TinyUSDZ");
	// TODO: Implement using TinyUSDZ API
	return Ref<CylinderMesh>();
}

Ref<CapsuleMesh> UsdMeshImportHelper::import_capsule(const tinyusdz::GeomCapsule &p_capsule) {
	ERR_PRINT("USD Import: import_capsule not yet implemented with TinyUSDZ");
	// TODO: Implement using TinyUSDZ API
	return Ref<CapsuleMesh>();
}

Ref<Mesh> UsdMeshImportHelper::import_geom_mesh(const tinyusdz::GeomMesh &p_mesh) {
	ERR_PRINT("USD Import: import_geom_mesh not yet implemented with TinyUSDZ");
	// TODO: Implement using TinyUSDZ API
	return Ref<Mesh>();
}

void UsdMeshImportHelper::apply_non_uniform_scale(Ref<Mesh> p_mesh, const tinyusdz::value::float3 &p_scale) {
	ERR_PRINT("USD Import: apply_non_uniform_scale not yet implemented with TinyUSDZ");
	// TODO: Implement using TinyUSDZ API
}

Ref<StandardMaterial3D> UsdMeshImportHelper::create_material(const tinyusdz::Prim &p_prim) {
	ERR_PRINT("USD Import: create_material not yet implemented with TinyUSDZ");
	// TODO: Implement using TinyUSDZ API
	return Ref<StandardMaterial3D>();
}
