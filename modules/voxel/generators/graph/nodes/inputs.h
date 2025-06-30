/**************************************************************************/
/*  inputs.h                                                              */
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

#include "../node_type_db.h"

namespace zylann::voxel::pg {

void register_input_nodes(Span<NodeType> types) {
	{
		NodeType &t = types[VoxelGraphFunction::NODE_INPUT_X];
		t.name = "InputX";
		t.category = CATEGORY_INPUT;
		t.outputs.push_back(NodeType::Port("x"));
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_INPUT_Y];
		t.name = "InputY";
		t.category = CATEGORY_INPUT;
		t.outputs.push_back(NodeType::Port("y"));
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_INPUT_Z];
		t.name = "InputZ";
		t.category = CATEGORY_INPUT;
		t.outputs.push_back(NodeType::Port("z"));
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_INPUT_SDF];
		t.name = "InputSDF";
		t.category = CATEGORY_INPUT;
		t.outputs.push_back(NodeType::Port("sdf"));
	}
	{
		NodeType &t = types[VoxelGraphFunction::NODE_CUSTOM_INPUT];
		t.name = "CustomInput";
		// t.params.push_back(NodeType::Param("binding", Variant::INT, 0));
		t.category = CATEGORY_INPUT;
		t.outputs.push_back(NodeType::Port("value"));
	}
}

} // namespace zylann::voxel::pg
