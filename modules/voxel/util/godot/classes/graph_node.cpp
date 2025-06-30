/**************************************************************************/
/*  graph_node.cpp                                                        */
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

#include "graph_node.h"
#include "../core/version.h"

namespace zylann::godot {

// There were changes to GraphNode in Godot PR #79311 2167694965ca2f4f16cfc1362d32a2fa01e817a2

// For some reason these getters cannot be const...

Vector2 get_graph_node_input_port_position(GraphNode &node, int port_index) {
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 1
	// Can't directly use inputs and output positions... Godot pre-scales them, which makes them unusable
	// inside NOTIFICATION_DRAW because the node is already scaled
	const Vector2 scale = node.get_global_transform().get_scale();
	return node.get_connection_input_position(port_index) / scale;
#else
	return node.get_input_port_position(port_index);
#endif
}

Vector2 get_graph_node_output_port_position(GraphNode &node, int port_index) {
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 1
	const Vector2 scale = node.get_global_transform().get_scale();
	return node.get_connection_output_position(port_index) / scale;
#else
	return node.get_output_port_position(port_index);
#endif
}

Color get_graph_node_input_port_color(GraphNode &node, int port_index) {
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 1
	return node.get_connection_input_color(port_index);
#else
	return node.get_input_port_color(port_index);
#endif
}

} // namespace zylann::godot
