/**************************************************************************/
/*  usd_document.cpp                                                      */
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

#include "usd_document.h"
#include "usd_mesh_export_helper.h"
#include "usd_mesh_import_helper.h"
#include "usd_state.h"

#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/mesh.h"

// USD headers
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/sdf/copyUtils.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdLux/sphereLight.h>

void UsdDocument::_bind_methods() {
	ClassDB::bind_method(D_METHOD("append_from_scene", "scene_root", "state", "flags"), &UsdDocument::append_from_scene, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("write_to_filesystem", "state", "path"), &UsdDocument::write_to_filesystem);
	ClassDB::bind_method(D_METHOD("get_file_extension_for_format", "binary"), &UsdDocument::get_file_extension_for_format);
	ClassDB::bind_method(D_METHOD("import_from_file", "path", "parent", "state"), &UsdDocument::import_from_file);
}

UsdDocument::UsdDocument() {
	// Initialize USD-specific members
	// This will be expanded as we implement the USD integration
}

Error UsdDocument::append_from_scene(Node *p_scene_root, Ref<UsdState> p_state, int32_t p_flags) {
	// This method will convert the Godot scene to USD

	if (!p_scene_root) {
		ERR_PRINT("USD Export: No scene root provided");
		return ERR_INVALID_PARAMETER;
	}

	if (p_state.is_null()) {
		ERR_PRINT("USD Export: No state provided");
		return ERR_INVALID_PARAMETER;
	}

	// Create a new USD stage in memory
	pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateInMemory();
	if (!stage) {
		ERR_PRINT("USD Export: Failed to create USD stage");
		return ERR_CANT_CREATE;
	}

	// Set up the stage
	stage->SetStartTimeCode(1);
	stage->SetEndTimeCode(1);
	stage->SetTimeCodesPerSecond(p_state->get_bake_fps());

	// Create a root prim for the scene
	pxr::UsdGeomXform root = pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Root"));
	if (!root.GetPrim().IsValid()) {
		ERR_PRINT("USD Export: Failed to define root prim");
		return ERR_CANT_CREATE;
	}
	stage->SetDefaultPrim(root.GetPrim());

	// Add metadata (using custom layer data for copyright)
	if (!p_state->get_copyright().is_empty()) {
		auto rootLayer = stage->GetRootLayer();
		if (!rootLayer) {
			ERR_PRINT("USD Export: Failed to get root layer");
			return ERR_CANT_CREATE;
		}
		auto customData = rootLayer->GetCustomLayerData();
		customData["copyright"] = pxr::VtValue(std::string(p_state->get_copyright().utf8().get_data()));
		rootLayer->SetCustomLayerData(customData);
	}

	// Store the stage in the state for later use in write_to_filesystem
	p_state->set_stage(stage);

	// Traverse the scene and convert nodes to USD prims
	Error err = _convert_node_to_prim(p_scene_root, stage, pxr::SdfPath("/Root"), p_state);
	if (err != OK) {
		// Error already printed in _convert_node_to_prim
		return err;
	}

	return OK;
}

Error UsdDocument::_convert_node_to_prim(Node *p_node, pxr::UsdStageRefPtr p_stage, const pxr::SdfPath &p_parent_path, Ref<UsdState> p_state) {
	// Check if the node is valid
	if (!p_node) {
		ERR_PRINT("USD Export: Invalid node");
		return ERR_INVALID_PARAMETER;
	}

	// Get the node name
	String node_name = p_node->get_name();

	// Create a valid USD path name (replace invalid characters)
	String valid_name = node_name.replace(":", "_").replace(".", "_");

	// Create a path for this node
	pxr::SdfPath node_path = p_parent_path.AppendChild(pxr::TfToken(valid_name.utf8().get_data()));

	// Create a USD prim based on the node type
	if (p_node->is_class("Node3D")) {
		// Create a transform prim
		pxr::UsdGeomXform xform = pxr::UsdGeomXform::Define(p_stage, node_path);
		if (!xform.GetPrim().IsValid()) {
			ERR_PRINT(vformat("USD Export: Failed to define Xform prim for node: %s", node_name));
			return ERR_CANT_CREATE;
		}

		// Set the transform
		Node3D *node_3d = Object::cast_to<Node3D>(p_node);
		if (node_3d) {
			// Get the transform
			Transform3D transform = node_3d->get_transform();

			// Extract basis (rotation and scale)
			Basis basis = transform.get_basis();

			// Extract translation
			Vector3 origin = transform.get_origin();

			// Create a USD matrix
			pxr::GfMatrix4d matrix;
			matrix.SetRow(0, pxr::GfVec4d(basis.get_column(0).x, basis.get_column(0).y, basis.get_column(0).z, 0.0));
			matrix.SetRow(1, pxr::GfVec4d(basis.get_column(1).x, basis.get_column(1).y, basis.get_column(1).z, 0.0));
			matrix.SetRow(2, pxr::GfVec4d(basis.get_column(2).x, basis.get_column(2).y, basis.get_column(2).z, 0.0));
			matrix.SetRow(3, pxr::GfVec4d(origin.x, origin.y, origin.z, 1.0));

			// Add a transform op
			pxr::UsdGeomXformOp transformOp = xform.AddTransformOp();
			if (!transformOp) {
				ERR_PRINT(vformat("USD Export: Failed to add transform op for node: %s", node_name));
				// Continue processing children even if transform fails? Or return error?
				// Let's return error for now.
				return ERR_CANT_CREATE;
			}
			transformOp.Set(matrix);

			//print_line("USD Export: Set transform for ", node_name);
		}

		// Check if it's a MeshInstance3D
		if (p_node->is_class("MeshInstance3D")) {
			MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(p_node);
			if (mesh_instance) {
				// Get the mesh
				Ref<Mesh> mesh = mesh_instance->get_mesh();
				if (mesh.is_valid()) {
					// Create a path for the mesh
					pxr::SdfPath mesh_path = node_path.AppendChild(pxr::TfToken("Mesh"));

					// Use the mesh export helper to convert the Godot mesh to a USD prim
					UsdMeshExportHelper mesh_helper;
					pxr::UsdPrim mesh_prim = mesh_helper.export_mesh_to_prim(mesh, p_stage, mesh_path);

					if (mesh_prim) {
						print_line("USD Export: Exported mesh for ", node_name);
					} else {
						// Error message already printed in export_mesh_to_prim if it failed
						// Consider returning an error here if mesh export failure is critical
						// return ERR_CANT_CREATE;
					}
				}
			}
		}

		// Process children
		for (int i = 0; i < p_node->get_child_count(); i++) {
			Node *child = p_node->get_child(i);
			Error err = _convert_node_to_prim(child, p_stage, node_path, p_state);
			if (err != OK) {
				return err; // Propagate error
			}
		}
	} else {
		print_line("USD Export: Skipping non-Node3D node: ", node_name);

		// Process children even for non-Node3D nodes
		for (int i = 0; i < p_node->get_child_count(); i++) {
			Node *child = p_node->get_child(i);
			// Pass the parent's path, not a new one based on the skipped node
			Error err = _convert_node_to_prim(child, p_stage, p_parent_path, p_state);
			if (err != OK) {
				return err; // Propagate error
			}
		}
	}
	return OK;
}

Error UsdDocument::write_to_filesystem(Ref<UsdState> p_state, const String &p_path) {
	// This method will write the USD document to a file using the USD API

	if (p_state.is_null()) {
		ERR_PRINT("USD Export: No state provided");
		return ERR_INVALID_PARAMETER;
	}

	if (p_path.is_empty()) {
		ERR_PRINT("USD Export: No file path provided");
		return ERR_INVALID_PARAMETER;
	}

	print_line("USD Export: Writing USD document to ", p_path);

	// Get the stage from the state
	pxr::UsdStageRefPtr stage = p_state->get_stage();
	if (!stage) {
		ERR_PRINT("USD Export: No stage found in state");
		return ERR_INVALID_PARAMETER;
	}

	// Export the stage to the file
	// Use a boolean return value check instead of try-catch
	bool success = stage->Export(p_path.utf8().get_data());
	if (!success) {
		ERR_PRINT(vformat("USD Export: Failed to export stage to %s", p_path));
		return ERR_CANT_CREATE;
	}

	print_line("USD Export: Successfully exported scene to ", p_path);
	return OK;
}

String UsdDocument::get_file_extension_for_format(bool p_binary) const {
	// Return the appropriate file extension based on the format
	return p_binary ? "usdc" : "usda";
}

Error UsdDocument::import_from_file(const String &p_path, Node *p_parent, Ref<UsdState> p_state) {
	// This method will import a USD file into a Godot scene

	if (!p_parent) {
		ERR_PRINT("USD Import: No parent node provided");
		return ERR_INVALID_PARAMETER;
	}

	if (p_state.is_null()) {
		ERR_PRINT("USD Import: No state provided");
		return ERR_INVALID_PARAMETER;
	}

	if (p_path.is_empty()) {
		ERR_PRINT("USD Import: No file path provided");
		return ERR_INVALID_PARAMETER;
	}

	print_line("USD Import: Importing USD file ", p_path);

	// Open the USD stage
	// Use RefPtr check instead of try-catch
	pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(p_path.utf8().get_data());
	if (!stage) {
		ERR_PRINT(vformat("USD Import: Failed to open USD stage at %s", p_path));
		return ERR_CANT_OPEN;
	}

	// Store the stage in the state
	p_state->set_stage(stage);

	// Get the default prim
	pxr::UsdPrim default_prim = stage->GetDefaultPrim();
	if (!default_prim) {
		// If there's no default prim, use the pseudo-root
		default_prim = stage->GetPseudoRoot();
		if (!default_prim) {
			// This should theoretically not happen if the stage opened successfully
			ERR_PRINT("USD Import: Failed to get default prim or pseudo-root");
			return ERR_CANT_OPEN;
		}
	}

	// Import the prim hierarchy
	return _import_prim_hierarchy(stage, default_prim.GetPath(), p_parent, p_state);
}

Error UsdDocument::_import_prim_hierarchy(const pxr::UsdStageRefPtr &p_stage, const pxr::SdfPath &p_prim_path, Node *p_parent, Ref<UsdState> p_state) {
	// This method will recursively import a USD prim hierarchy into a Godot scene

	// Get the prim
	pxr::UsdPrim prim = p_stage->GetPrimAtPath(p_prim_path);
	if (!prim) {
		// Use vformat for consistency
		ERR_PRINT(vformat("USD Import: Invalid prim path: %s", String(p_prim_path.GetString().c_str())));
		return ERR_INVALID_PARAMETER;
	}

	// Skip the pseudo-root
	if (prim.IsPseudoRoot()) {
		// Process children
		for (const pxr::UsdPrim &child : prim.GetChildren()) {
			Error err = _import_prim_hierarchy(p_stage, child.GetPath(), p_parent, p_state);
			if (err != OK) {
				return err; // Propagate error
			}
		}
		return OK;
	}

	// Create a new Node3D for this prim
	Node3D *node = memnew(Node3D);
	node->set_name(String(prim.GetName().GetString().c_str()));
	p_parent->add_child(node);
	node->set_owner(p_parent->get_owner() ? p_parent->get_owner() : p_parent);

	// Set the transform
	if (prim.IsA<pxr::UsdGeomXformable>()) {
		pxr::UsdGeomXformable xformable(prim);

		// Get the local transform matrix
		pxr::GfMatrix4d matrix;
		bool reset_xform_stack;
		// Check return value although it's unlikely to fail if prim is valid
		if (!xformable.GetLocalTransformation(&matrix, &reset_xform_stack)) {
			ERR_PRINT(vformat("USD Import: Failed to get local transformation for prim: %s", String(prim.GetPath().GetString().c_str())));
			// Decide how to handle this - skip transform or return error?
			// Let's skip setting the transform for now.
		} else {
			// Convert USD matrix to Godot transform
			Transform3D transform;

			// Extract basis (rotation and scale)
			Basis basis;
			basis.set_column(0, Vector3(matrix[0][0], matrix[0][1], matrix[0][2]));
			basis.set_column(1, Vector3(matrix[1][0], matrix[1][1], matrix[1][2]));
			basis.set_column(2, Vector3(matrix[2][0], matrix[2][1], matrix[2][2]));

			// Extract translation
			Vector3 origin(matrix[3][0], matrix[3][1], matrix[3][2]);

			// Set the transform
			transform.set_basis(basis);
			transform.set_origin(origin);
			node->set_transform(transform);
		}
	}

	// Handle specific prim types
	if (prim.IsA<pxr::UsdGeomGprim>()) {
		// Create a mesh instance for geometric primitives
		MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
		mesh_instance->set_name("MeshInstance3D");
		node->add_child(mesh_instance);
		mesh_instance->set_owner(p_parent->get_owner() ? p_parent->get_owner() : p_parent);

		// Use the mesh import helper to convert the USD prim to a Godot mesh
		UsdMeshImportHelper mesh_helper;
		Ref<Mesh> mesh = mesh_helper.import_mesh_from_prim(prim);

		if (mesh.is_valid()) {
			mesh_instance->set_mesh(mesh);
			//print_line("USD Import: Imported mesh for ", String(prim.GetName().GetString().c_str()));
		} else {
			//ERR_PRINT("USD Import: Failed to import mesh for ", String(prim.GetName().GetString().c_str()));
			// Error printed in helper, consider returning error if critical
			// return ERR_CANT_CREATE;
		}
	} else if (prim.IsA<pxr::UsdGeomCamera>()) {
		// Create a camera for camera prims
		Camera3D *camera = memnew(Camera3D);
		camera->set_name("Camera3D");
		node->add_child(camera);
		camera->set_owner(p_parent->get_owner() ? p_parent->get_owner() : p_parent);

		// Set camera properties
		pxr::UsdGeomCamera usd_camera(prim);

		// Get the focal length
		float focal_length = 50.0f; // Default focal length (in mm)
		// Add checks for attribute getting if necessary, though Get() often provides defaults
		usd_camera.GetFocalLengthAttr().Get(&focal_length);

		// Get the horizontal aperture
		float horizontal_aperture = 24.0f; // Default horizontal aperture (in mm)
		usd_camera.GetHorizontalApertureAttr().Get(&horizontal_aperture);

		// Convert focal length to FOV (approximation)
		// Use M_PI from <cmath> or Math_PI from Godot's core/math/math_defs.h
		// Assuming core/math/math_defs.h is included or accessible
		float fov = 2.0f * atan(horizontal_aperture / (2.0f * focal_length)) * (180.0f / Math::PI);

		// Set the FOV
		camera->set_fov(fov);

		//print_line("USD Import: Imported camera for ", String(prim.GetName().GetString().c_str()));
	} else if (prim.IsA<pxr::UsdLuxSphereLight>()) {
		// Create a light for light prims
		Light3D *light = memnew(Light3D);
		light->set_name("Light3D");
		node->add_child(light);
		light->set_owner(p_parent->get_owner() ? p_parent->get_owner() : p_parent);

		// Set light properties
		// This is a placeholder for now

		//print_line("USD Import: Imported light for ", String(prim.GetName().GetString().c_str()));
	}

	// Process children
	for (const pxr::UsdPrim &child : prim.GetChildren()) {
		Error err = _import_prim_hierarchy(p_stage, child.GetPath(), node, p_state);
		if (err != OK) {
			return err; // Propagate error
		}
	}

	return OK;
}
