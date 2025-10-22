/**************************************************************************/
/*  usd_mesh_export_helper.h                                              */
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

// USD headers
#include <pxr/base/gf/vec3f.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/sphere.h>

PXR_NAMESPACE_USING_DIRECTIVE

class UsdMeshExportHelper {
public:
    UsdMeshExportHelper() {
    }
    ~UsdMeshExportHelper() {
    }

	// Export a Godot mesh to a USD prim
	pxr::UsdPrim export_mesh_to_prim(const Ref<Mesh> p_mesh, pxr::UsdStageRefPtr p_stage, const pxr::SdfPath &p_path);

private:
	// Helper methods for specific primitive types
	pxr::UsdGeomCube export_box(const Ref<BoxMesh> p_box, pxr::UsdStageRefPtr p_stage, const pxr::SdfPath &p_path);
	pxr::UsdGeomSphere export_sphere(const Ref<SphereMesh> p_sphere, pxr::UsdStageRefPtr p_stage, const pxr::SdfPath &p_path);
	pxr::UsdGeomCylinder export_cylinder(const Ref<CylinderMesh> p_cylinder, pxr::UsdStageRefPtr p_stage, const pxr::SdfPath &p_path);
	pxr::UsdGeomCone export_cone(const Ref<CylinderMesh> p_cone, pxr::UsdStageRefPtr p_stage, const pxr::SdfPath &p_path);
	pxr::UsdGeomCapsule export_capsule(const Ref<CapsuleMesh> p_capsule, pxr::UsdStageRefPtr p_stage, const pxr::SdfPath &p_path);
	pxr::UsdGeomMesh export_geom_mesh(const Ref<Mesh> p_mesh, pxr::UsdStageRefPtr p_stage, const pxr::SdfPath &p_path);

	// Helper method to handle non-uniform scaling
	void apply_non_uniform_scale(pxr::UsdGeomGprim &p_gprim, const pxr::GfVec3f &p_scale);
};
