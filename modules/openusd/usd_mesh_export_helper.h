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

// TODO: Update export functionality to use TinyUSDZ
// TinyUSDZ headers
// Workaround for Texture name conflict: rename TinyUSDZ's Texture before including
#define Texture TinyUSDZTexture
#include "tinyusdz.hh"
#undef Texture
#include "prim-types.hh"
#include "usdGeom.hh"
#include "value-types.hh"

namespace tinyusdz {
	class Stage;
	class Path;
	class Prim;
}

class UsdMeshExportHelper {
public:
    UsdMeshExportHelper() {
    }
    ~UsdMeshExportHelper() {
    }

	// Export a Godot mesh to a USD prim
	// TODO: Update to use TinyUSDZ API (tinyusdz::Prim, tinyusdz::Stage, tinyusdz::Path)
	tinyusdz::Prim export_mesh_to_prim(const Ref<Mesh> p_mesh, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path);
	
	// Export a GeomMesh (public for use by USDDocument)
	tinyusdz::Prim export_geom_mesh(const Ref<Mesh> p_mesh, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path);

private:
	// Helper methods for specific primitive types
	// TODO: Update to use TinyUSDZ API
	tinyusdz::Prim export_box(const Ref<BoxMesh> p_box, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path);
	tinyusdz::Prim export_sphere(const Ref<SphereMesh> p_sphere, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path);
	tinyusdz::Prim export_cylinder(const Ref<CylinderMesh> p_cylinder, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path);
	tinyusdz::Prim export_cone(const Ref<CylinderMesh> p_cone, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path);
	tinyusdz::Prim export_capsule(const Ref<CapsuleMesh> p_capsule, tinyusdz::Stage *p_stage, const tinyusdz::Path &p_path);

	// Helper method to handle non-uniform scaling
	// TODO: Update to use TinyUSDZ API
	void apply_non_uniform_scale(tinyusdz::Prim &p_prim, const tinyusdz::value::float3 &p_scale);
};
