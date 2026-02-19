/**************************************************************************/
/*  test_graph_edit_arranger.cpp                                         */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_graph_edit_arranger)

#ifndef ADVANCED_GUI_DISABLED

#include "core/io/resource_loader.h"
#include "core/string/ustring.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/graph_edit_arranger.h"
#include "scene/gui/graph_node.h"
#include "scene/main/window.h"
#include "scene/resources/visual_shader.h"
#include "tests/test_utils.h"

namespace TestGraphEditArranger {

// Build a GraphEdit in the scene tree with the given nodes and edges, then run the arranger.
// Each node gets one input port and one output port (slot 0). Edges are (from, to) node names.
static GraphEdit *build_graph_and_arrange(const Vector<StringName> &p_node_names, const Vector<Pair<StringName, StringName>> &p_edges) {
	Window *root = SceneTree::get_singleton()->get_root();
	GraphEdit *graph_edit = memnew(GraphEdit);
	graph_edit->set_name("GraphEdit");
	root->add_child(graph_edit);

	for (const StringName &name : p_node_names) {
		GraphNode *node = memnew(GraphNode);
		node->set_name(name);
		node->set_slot(0, true, 0, Color(), true, 0, Color());
		node->set_custom_minimum_size(Size2(80, 40));
		graph_edit->add_child(node);
	}

	for (const Pair<StringName, StringName> &e : p_edges) {
		Error err = graph_edit->connect_node(e.first, 0, e.second, 0);
		CHECK_MESSAGE(err == OK, "connect_node must succeed");
	}

	Ref<GraphEditArranger> arranger = memnew(GraphEditArranger(graph_edit));
	arranger->arrange_nodes();

	return graph_edit;
}

// Get position (x, y) of a node by name. Caller must keep graph_edit and nodes alive.
static Vector2 get_node_position(GraphEdit *p_graph_edit, const StringName &p_name) {
	for (int i = 0; i < p_graph_edit->get_child_count(); i++) {
		GraphNode *gn = Object::cast_to<GraphNode>(p_graph_edit->get_child(i));
		if (gn && gn->get_name() == p_name) {
			return gn->get_position_offset();
		}
	}
	return Vector2(FLT_MAX, FLT_MAX);
}

TEST_CASE("[GraphEditArranger] Jellyfish layout: sources left, sinks right, long stingers") {
	SUBCASE("Linear chain A->B->C->D forms long stinger (strictly increasing x)") {
		Vector<StringName> nodes;
		nodes.push_back("A");
		nodes.push_back("B");
		nodes.push_back("C");
		nodes.push_back("D");
		Vector<Pair<StringName, StringName>> edges;
		edges.push_back(Pair<StringName, StringName>("A", "B"));
		edges.push_back(Pair<StringName, StringName>("B", "C"));
		edges.push_back(Pair<StringName, StringName>("C", "D"));

		GraphEdit *graph_edit = build_graph_and_arrange(nodes, edges);
		REQUIRE(graph_edit != nullptr);

		Vector2 pos_a = get_node_position(graph_edit, "A");
		Vector2 pos_b = get_node_position(graph_edit, "B");
		Vector2 pos_c = get_node_position(graph_edit, "C");
		Vector2 pos_d = get_node_position(graph_edit, "D");

		CHECK_MESSAGE(pos_a.x < pos_b.x, "Jellyfish: source A should be left of B (long stinger)");
		CHECK_MESSAGE(pos_b.x < pos_c.x, "Jellyfish: B should be left of C (long stinger)");
		CHECK_MESSAGE(pos_c.x < pos_d.x, "Jellyfish: C should be left of D (long stinger)");

		graph_edit->get_parent()->remove_child(graph_edit);
		memdelete(graph_edit);
	}

	SUBCASE("Single source fan-out: one node in first column, no large cap") {
		Vector<StringName> nodes;
		nodes.push_back("A");
		nodes.push_back("B");
		nodes.push_back("C");
		nodes.push_back("D");
		Vector<Pair<StringName, StringName>> edges;
		edges.push_back(Pair<StringName, StringName>("A", "B"));
		edges.push_back(Pair<StringName, StringName>("A", "C"));
		edges.push_back(Pair<StringName, StringName>("A", "D"));

		GraphEdit *graph_edit = build_graph_and_arrange(nodes, edges);
		REQUIRE(graph_edit != nullptr);

		Vector2 pos_a = get_node_position(graph_edit, "A");
		Vector2 pos_b = get_node_position(graph_edit, "B");
		Vector2 pos_c = get_node_position(graph_edit, "C");
		Vector2 pos_d = get_node_position(graph_edit, "D");

		CHECK_MESSAGE(pos_a.x < pos_b.x, "Jellyfish: source A should be left of B");
		CHECK_MESSAGE(pos_a.x < pos_c.x, "Jellyfish: source A should be left of C");
		CHECK_MESSAGE(pos_a.x < pos_d.x, "Jellyfish: source A should be left of D");

		graph_edit->get_parent()->remove_child(graph_edit);
		memdelete(graph_edit);
	}

	SUBCASE("Diamond: source leftmost, sink rightmost") {
		Vector<StringName> nodes;
		nodes.push_back("A");
		nodes.push_back("B");
		nodes.push_back("C");
		nodes.push_back("D");
		Vector<Pair<StringName, StringName>> edges;
		edges.push_back(Pair<StringName, StringName>("A", "B"));
		edges.push_back(Pair<StringName, StringName>("A", "C"));
		edges.push_back(Pair<StringName, StringName>("B", "D"));
		edges.push_back(Pair<StringName, StringName>("C", "D"));

		GraphEdit *graph_edit = build_graph_and_arrange(nodes, edges);
		REQUIRE(graph_edit != nullptr);

		Vector2 pos_a = get_node_position(graph_edit, "A");
		Vector2 pos_d = get_node_position(graph_edit, "D");

		CHECK_MESSAGE(pos_a.x < pos_d.x, "Jellyfish: source A should be left of sink D");

		graph_edit->get_parent()->remove_child(graph_edit);
		memdelete(graph_edit);
	}

	SUBCASE("Two parallel chains: both form long stingers") {
		Vector<StringName> nodes;
		nodes.push_back("A1");
		nodes.push_back("B1");
		nodes.push_back("C1");
		nodes.push_back("A2");
		nodes.push_back("B2");
		nodes.push_back("C2");
		Vector<Pair<StringName, StringName>> edges;
		edges.push_back(Pair<StringName, StringName>("A1", "B1"));
		edges.push_back(Pair<StringName, StringName>("B1", "C1"));
		edges.push_back(Pair<StringName, StringName>("A2", "B2"));
		edges.push_back(Pair<StringName, StringName>("B2", "C2"));

		GraphEdit *graph_edit = build_graph_and_arrange(nodes, edges);
		REQUIRE(graph_edit != nullptr);

		Vector2 pos_a1 = get_node_position(graph_edit, "A1");
		Vector2 pos_b1 = get_node_position(graph_edit, "B1");
		Vector2 pos_c1 = get_node_position(graph_edit, "C1");
		Vector2 pos_a2 = get_node_position(graph_edit, "A2");
		Vector2 pos_b2 = get_node_position(graph_edit, "B2");
		Vector2 pos_c2 = get_node_position(graph_edit, "C2");

		CHECK_MESSAGE(pos_a1.x < pos_b1.x, "Jellyfish: chain A1->B1->C1 should have increasing x (A1 left of B1)");
		CHECK_MESSAGE(pos_b1.x < pos_c1.x, "Jellyfish: chain A1->B1->C1 should have increasing x (B1 left of C1)");
		CHECK_MESSAGE(pos_a2.x < pos_b2.x, "Jellyfish: chain A2->B2->C2 should have increasing x (A2 left of B2)");
		CHECK_MESSAGE(pos_b2.x < pos_c2.x, "Jellyfish: chain A2->B2->C2 should have increasing x (B2 left of C2)");

		graph_edit->get_parent()->remove_child(graph_edit);
		memdelete(graph_edit);
	}
}

// Load input and expected arrangement .tres (VisualShader), verify both load and have same graph structure.
// Data: tests/data/arrangements/input/*.tres (disliked layout), tests/data/arrangements/expected/*.tres (arranged).
static void test_arrangement_tres_pair(const String &p_name) {
	const String input_path = TestUtils::get_data_path("arrangements/input/" + p_name + ".tres");
	const String expected_path = TestUtils::get_data_path("arrangements/expected/" + p_name + ".tres");

	Error err;
	Ref<Resource> res_input = ResourceLoader::load(input_path, "", ResourceFormatLoader::CACHE_MODE_IGNORE, &err);
	CHECK_MESSAGE(err == OK, vformat("Load input %s: err=%d", input_path, err));
	REQUIRE(res_input.is_valid());
	Ref<VisualShader> vs_input = Ref<VisualShader>(Object::cast_to<VisualShader>(res_input.ptr()));
	CHECK_MESSAGE(vs_input.is_valid(), "Input should be a VisualShader");

	Ref<Resource> res_expected = ResourceLoader::load(expected_path, "", ResourceFormatLoader::CACHE_MODE_IGNORE, &err);
	CHECK_MESSAGE(err == OK, vformat("Load expected %s: err=%d", expected_path, err));
	REQUIRE(res_expected.is_valid());
	Ref<VisualShader> vs_expected = Ref<VisualShader>(Object::cast_to<VisualShader>(res_expected.ptr()));
	CHECK_MESSAGE(vs_expected.is_valid(), "Expected should be a VisualShader");

	if (!vs_input.is_valid() || !vs_expected.is_valid()) {
		return;
	}

	Vector<int> nodes_input = vs_input->get_node_list(VisualShader::TYPE_VERTEX);
	Vector<int> nodes_expected = vs_expected->get_node_list(VisualShader::TYPE_VERTEX);
	CHECK_MESSAGE(nodes_input.size() == nodes_expected.size(),
			vformat("Input and expected should have same node count: %d vs %d", nodes_input.size(), nodes_expected.size()));

	List<VisualShader::Connection> conn_input;
	List<VisualShader::Connection> conn_expected;
	vs_input->get_node_connections(VisualShader::TYPE_VERTEX, &conn_input);
	vs_expected->get_node_connections(VisualShader::TYPE_VERTEX, &conn_expected);
	CHECK_MESSAGE(conn_input.size() == conn_expected.size(),
			vformat("Input and expected should have same connection count: %d vs %d", (int)conn_input.size(), (int)conn_expected.size()));
}

TEST_CASE("[GraphEditArranger] Arrangement data: input and expected VisualShader tres") {
	SUBCASE("MyShader: load input and expected, same graph structure") {
		test_arrangement_tres_pair("MyShader");
	}
	SUBCASE("MyShader2: load input and expected, same graph structure") {
		test_arrangement_tres_pair("MyShader2");
	}
	SUBCASE("MyShader3: load input and expected, same graph structure") {
		test_arrangement_tres_pair("MyShader3");
	}
}

} // namespace TestGraphEditArranger

#endif // ADVANCED_GUI_DISABLED
