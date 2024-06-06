/**************************************************************************/
/*  baked_subdiv_mesh.hpp                                                 */
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

#ifndef BAKED_SUBDIV_MESH_HPP
#define BAKED_SUBDIV_MESH_HPP

#include "core/io/resource.h"
#include "core/object/class_db.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"

class TopologyDataMesh;

//Uses a TopologyDataMesh in the background to function like a Mesh
class BakedSubdivMesh : public ArrayMesh {
	GDCLASS(BakedSubdivMesh, ArrayMesh);

protected:
	Ref<TopologyDataMesh> data_mesh;
	int subdiv_level = 0;
	void _update_subdiv();
	void _clear();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;

	static void _bind_methods();

public:
	void set_data_mesh(Ref<TopologyDataMesh> p_data_mesh);
	Ref<TopologyDataMesh> get_data_mesh() const;
	void set_subdiv_level(int p_level);
	int get_subdiv_level() const;
	BakedSubdivMesh();
	~BakedSubdivMesh();
};

#endif // BAKED_SUBDIV_MESH_HPP
