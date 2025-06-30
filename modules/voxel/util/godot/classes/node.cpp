/**************************************************************************/
/*  node.cpp                                                              */
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

#include "node.h"

namespace zylann::godot {

template <typename F>
void for_each_node_depth_first(Node *parent, F f) {
	ERR_FAIL_COND(parent == nullptr);
	f(parent);
	for (int i = 0; i < parent->get_child_count(); ++i) {
		for_each_node_depth_first(parent->get_child(i), f);
	}
}

void set_nodes_owner(Node *root, Node *owner) {
	for_each_node_depth_first(root, [owner](Node *node) { //
		node->set_owner(owner);
	});
}

void set_nodes_owner_except_root(Node *root, Node *owner) {
	ERR_FAIL_COND(root == nullptr);
	for (int i = 0; i < root->get_child_count(); ++i) {
		set_nodes_owner(root->get_child(i), owner);
	}
}

void get_node_groups(const Node &node, StdVector<StringName> &out_groups) {
#if defined(ZN_GODOT)
	List<Node::GroupInfo> gi;
	node.get_groups(&gi);
	for (const Node::GroupInfo &g : gi) {
		out_groups.push_back(g.name);
	}

#elif defined(ZN_GODOT_EXTENSION)
	TypedArray<StringName> groups = node.get_groups();
	for (int i = 0; i < groups.size(); ++i) {
		out_groups.push_back(groups[i]);
	}
#endif
}

} // namespace zylann::godot
