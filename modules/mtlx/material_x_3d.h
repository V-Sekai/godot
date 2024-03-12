/**************************************************************************/
/*  material_x_3d.h                                                       */
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

#ifndef MATERIAL_X_H
#define MATERIAL_X_H

#include "MaterialXCore/Generated.h"

#include "core/io/resource_importer.h"
#include "scene/resources/material.h"
#include "scene/resources/visual_shader.h"

#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/ShaderTranslator.h>

#include <MaterialXFormat/Environ.h>
#include <MaterialXFormat/Util.h>

#include <MaterialXCore/Util.h>

#include <MaterialXGenGlsl/EsslShaderGenerator.h>

#include <iostream>
#include <map>

using namespace godot;
namespace mx = MaterialX;
class MTLXLoader : public Resource {
	GDCLASS(MTLXLoader, Resource);
	mx::DocumentPtr _stdLib;
	void create_node(const mx::NodePtr &node, int depth, Ref<VisualShader> &shader, std::set<mx::NodePtr> &processed_nodes,
			int &id, std::map<mx::NodePtr, int> &node_ids) const;
	void connect_node(const mx::NodePtr &node, int depth, Ref<VisualShader> &shader, std::set<mx::NodePtr> &processed_nodes,
			int &id, const std::map<mx::NodePtr, int> &node_ids) const;
	int get_node_id(const mx::NodePtr &node, const std::map<mx::NodePtr, int> &node_ids) const;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("_load", "path", "original_path", "use_sub_threads", "cache_mode"), &MTLXLoader::_load);
	}

public:
	virtual Variant _load(const String &p_save_path, const String &p_original_path, bool p_use_sub_threads, int64_t p_cache_mode) const;
	MTLXLoader() {
	}
};

using MaterialPtr = std::shared_ptr<class Material>;

class DocumentModifiers {
public:
	mx::StringMap remapElements;
	mx::StringSet skipElements;
	std::string filePrefixTerminator;
};
#endif