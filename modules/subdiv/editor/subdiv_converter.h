/**************************************************************************/
/*  subdiv_converter.h                                                    */
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

#ifndef SUBDIV_CONVERTER_H
#define SUBDIV_CONVERTER_H

#include "core/object/ref_counted.h"

#include "../src/import/topology_data_importer.hpp"
#include "scene/3d/importer_mesh_instance_3d.h"

class SubdivConverter : public RefCounted {
	GDCLASS(SubdivConverter, RefCounted);

	Ref<TopologyDataImporter> importer;
	TopologyDataImporter::ImportMode import_mode;
	int subdiv_level = 0;

protected:
	static void _bind_methods();

public:
	SubdivConverter(const TopologyDataImporter::ImportMode p_import_mode, int p_subdiv_level) {
		importer.instantiate();
		import_mode = p_import_mode;
		subdiv_level = p_subdiv_level;
	}
	void convert_importer_mesh_instances_recursively(Node *node);

	SubdivConverter();
};

#endif // SUBDIV_CONVERTER_H
