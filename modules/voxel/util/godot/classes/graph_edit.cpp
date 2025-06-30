/**************************************************************************/
/*  graph_edit.cpp                                                        */
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

#include "graph_edit.h"

namespace zylann::godot {

void get_graph_edit_connections(const GraphEdit &self, StdVector<GraphEditConnection> &out_connections) {
#if defined(ZN_GODOT)

#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 2
	List<GraphEdit::Connection> connections_list;
	self.get_connection_list(&connections_list);

	for (const GraphEdit::Connection &src_con : connections_list) {
		GraphEditConnection dst_con;

#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 1
		dst_con.from = src_con.from;
		dst_con.to = src_con.to;
#else
		dst_con.from = src_con.from_node;
		dst_con.to = src_con.to_node;
#endif
		dst_con.from_port = src_con.from_port;
		dst_con.to_port = src_con.to_port;

		out_connections.push_back(dst_con);
	}

#else // Godot 4.3+

#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 3
	const List<Ref<GraphEdit::Connection>> &connections_list = self.get_connection_list();
#else
	const Vector<Ref<GraphEdit::Connection>> &connections_list = self.get_connections();
#endif

	for (const Ref<GraphEdit::Connection> &src_con : connections_list) {
		GraphEditConnection dst_con;

		dst_con.from = src_con->from_node;
		dst_con.to = src_con->to_node;
		dst_con.from_port = src_con->from_port;
		dst_con.to_port = src_con->to_port;

		out_connections.push_back(dst_con);
	}
#endif

#elif defined(ZN_GODOT_EXTENSION)
	Array list = self.get_connection_list();
	const int count = list.size();

#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 1
	const String from_key = "from";
	const String to_key = "to";
#else
	const String from_key = "from_node";
	const String to_key = "to_node";
#endif
	const String from_port_key = "from_port";
	const String to_port_key = "to_port";

	for (int i = 0; i < count; ++i) {
		Dictionary d = list[i];
		GraphEditConnection con;
		con.from = d[from_key];
		con.from_port = d[from_port_key];
		con.to = d[to_key];
		con.to_port = d[to_port_key];
		out_connections.push_back(con);
	}
#endif
}

Vector2 get_graph_edit_scroll_offset(const GraphEdit &self) {
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 1
	return self.get_scroll_ofs();
#else
	return self.get_scroll_offset();
#endif
}

bool is_graph_edit_using_snapping(const GraphEdit &self) {
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 1
	return self.is_using_snap();
#else
	return self.is_snapping_enabled();
#endif
}

int get_graph_edit_snapping_distance(const GraphEdit &self) {
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 1
	return self.get_snap();
#else
	return self.get_snapping_distance();
#endif
}

} // namespace zylann::godot
