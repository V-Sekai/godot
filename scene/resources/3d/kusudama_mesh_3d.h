/**************************************************************************/
/*  kusudama_mesh_3d.h                                                    */
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

#include "core/containers/local_vector.h"
#include "core/math/vector3.h"
#include "scene/resources/mesh.h"

// Forward declaration
class IKLimitCone3D;

/**
 * KusudamaMesh3D is a wireframe mesh primitive that visualizes kusudama multi-cone constraints.
 * It renders all cone parameters including control points, radii, and tangent circles for
 * inter-cone transitions. This mesh is used for gizmo visualization in the editor.
 */
class KusudamaMesh3D : public Mesh {
	GDCLASS(KusudamaMesh3D, Mesh);

	struct ConeData {
		Vector3 control_point = Vector3(0, 1, 0);
		real_t radius = 0.5;
		// Tangent circle parameters are precalculated/read-only
		// not included in serialization
	};

	LocalVector<ConeData> open_cones;
	real_t sphere_radius = 0.25;
	int segments = 16;

	mutable RID mesh_rid;
	mutable bool dirty = true;

	void _update_mesh() const;

protected:
	static void _bind_methods();

public:
	KusudamaMesh3D();
	virtual ~KusudamaMesh3D() override;

	// Cone data management via array inspector
	void set_cones(TypedArray<Dictionary> p_cones);
	TypedArray<Dictionary> get_cones() const;
	void add_cone(Dictionary p_cone_data);
	void remove_cone(int p_index);
	void clear_cones();
	int get_cone_count() const;

	// Mesh parameters
	void set_sphere_radius(real_t p_radius);
	real_t get_sphere_radius() const;

	void set_segments(int p_segments);
	int get_segments() const;

	// Mesh interface implementation
	virtual int get_surface_count() const override;
	virtual int surface_get_array_len(int p_surface) const override;
	virtual int surface_get_array_index_len(int p_surface) const override;
	virtual Array surface_get_arrays(int p_surface) const override;
	virtual Array surface_get_blend_shape_arrays(int p_surface) const override;
	virtual void surface_set_material(int p_surface, const Ref<Material> &p_material) override;
	virtual Ref<Material> surface_get_material(int p_surface) const override;
	virtual int surface_get_primitive_type(int p_surface) const override;
	virtual void surface_set_name(int p_surface, const String &p_name) override;
	virtual String surface_get_name(int p_surface) const override;
	virtual void clear_surfaces() override;
	virtual AABB get_aabb() const override;
	virtual RID get_rid() const override;
	virtual void rid_changed() override;
};
