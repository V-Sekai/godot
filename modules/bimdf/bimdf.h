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

#include <libTimekeeper/StopWatchPrinting.hh>
#include <libsatsuma/Extra/Highlevel.hh>
#include <libsatsuma/Problems/BiMDF.hh>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/object/ref_counted.h"

class MinimumDeviationFlowState : public RefCounted {
	GDCLASS(MinimumDeviationFlowState, RefCounted);

public:
	Satsuma::BiMDF bimdf;
	std::map<std::string, Satsuma::BiMDF::Edge> edges;
	std::vector<Satsuma::BiMDF::Node> nodes;
};

class MinimumDeviationFlow : public RefCounted {
	GDCLASS(MinimumDeviationFlow, RefCounted);
	static void print_solution(const Satsuma::BiMDFFullResult &result, const std::map<std::string, Satsuma::BiMDF::Edge> &edges);
	static std::string format_solution(const Satsuma::BiMDFFullResult &result, const std::map<std::string, Satsuma::BiMDF::Edge> &edges);

protected:
	static void _bind_methods();

public:
	static int add_node(Ref<MinimumDeviationFlowState> state);
	static void add_edge_abs(Ref<MinimumDeviationFlowState> state, int u, int v, double target, double weight, int lower, int upper, bool u_head, bool v_head);
	static void add_edge_quad(Ref<MinimumDeviationFlowState> state, int u, int v, double target, double weight, int lower, int upper, bool u_head, bool v_head);
	static void add_edge_zero(Ref<MinimumDeviationFlowState> state, int u, int v, int lower, int upper, bool u_head, bool v_head);
	static void solve(Ref<MinimumDeviationFlowState> state);
	MinimumDeviationFlow();
};

#endif // BIMDF_H
