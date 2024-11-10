/**************************************************************************/
/*  bimdf.h                                                               */
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

#ifndef BIMDF_H
#define BIMDF_H

#include "bimdf.h"
#include <string>

#include <libTimekeeper/StopWatchPrinting.hh>
#include <libsatsuma/Extra/Highlevel.hh>
#include <libsatsuma/Problems/BiMDF.hh>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/object/ref_counted.h"

class MinimumDeviationFlow : public RefCounted {
	GDCLASS(MinimumDeviationFlow, RefCounted);

	 HashMap<int, String> node_names;
	 HashMap<String, Pair<int, int>> edge_nodes;
	 HashMap<String, double> edge_targets;
	 HashMap<String, double> edge_weights;
	 HashMap<String, int> edge_lowers;
	 HashMap<String, int> edge_uppers;
	 HashMap<String, bool> edge_u_heads;
	 HashMap<String, bool> edge_v_heads;

	std::map<std::string, Satsuma::BiMDF::Edge> edges;
	std::vector<Satsuma::BiMDF::Node> nodes;
	Satsuma::BiMDF bimdf;
	const std::map<std::string, Satsuma::BiMDF::Edge> &get_edges() const;
	const std::vector<Satsuma::BiMDF::Node> &get_nodes() const;
	
	const  HashMap<int, String> &get_node_names() const;
	const  HashMap<String, Pair<int, int>> &get_edge_nodes() const;
	const  HashMap<String, double> &get_edge_targets() const;
	const  HashMap<String, double> &get_edge_weights() const;
	const  HashMap<String, int> &get_edge_lowers() const;
	const  HashMap<String, int> &get_edge_uppers() const;
	const  HashMap<String, bool> &get_edge_u_heads() const;
	const  HashMap<String, bool> &get_edge_v_heads() const;
	void print_solution(const Satsuma::BiMDFFullResult &result, const std::map<std::string, Satsuma::BiMDF::Edge> &edges);
	std::string format_solution(const Satsuma::BiMDFFullResult &result, const std::map<std::string, Satsuma::BiMDF::Edge> &edges);

protected:
	static void _bind_methods();

public:
	void clear();
	int add_node(const String &name);
	void add_edge_abs(int p_u, int p_v, double p_target, double p_weight, int p_lower, int p_upper, bool p_u_head, bool p_v_head);
	void add_edge_quad(int p_u, int p_v, double p_target, double p_weight, int p_lower, int p_upper, bool p_u_head, bool p_v_head);
	void add_edge_zero(int p_u, int p_v, int p_lower, int p_upper, bool p_u_head, bool p_v_head);
	void solve();
	MinimumDeviationFlow() {}
};

#endif