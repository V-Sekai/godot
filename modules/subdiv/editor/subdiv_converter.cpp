/**************************************************************************/
/*  subdiv_converter.cpp                                                  */
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

#include "subdiv_converter.h"

void SubdivConverter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("convert_importer_mesh_instances_recursively", "node"), &SubdivConverter::convert_importer_mesh_instances_recursively);
}

void SubdivConverter::convert_importer_mesh_instances_recursively(Node *node) {
	for (int i = 0; i < node->get_child_count(); ++i) {
		Node *child = node->get_child(i);
		convert_importer_mesh_instances_recursively(child);

		if (Object::cast_to<ImporterMeshInstance3D>(child)) {
			importer->convert_importer_meshinstance_to_subdiv(Object::cast_to<ImporterMeshInstance3D>(child), TopologyDataImporter::ImportMode(import_mode), subdiv_level);
		}
	}
}

SubdivConverter::SubdivConverter() {
}
