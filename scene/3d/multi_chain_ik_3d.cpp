/**************************************************************************/
/*  multi_chain_ik_3d.cpp                                                 */
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

#include "multi_chain_ik_3d.h"

MultiChainIK3D::MultiChainIK3D() {
}

MultiChainIK3D::~MultiChainIK3D() {
	chains.clear();
	solvers.clear();
}

void MultiChainIK3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			skeleton = Object::cast_to<Skeleton3D>(get_parent());
			_update_solvers();
		} break;
		case NOTIFICATION_PROCESS: {
			if (!skeleton) {
				return;
			}
			for (uint32_t i = 0; i < solvers.size(); i++) {
				if (solvers[i]) {
					solvers[i]->set_max_iterations(max_iterations);
					solvers[i]->set_min_distance(min_distance);
					solvers[i]->set_angular_delta_limit(angular_delta_limit);
				}
			}
		} break;
	}
}

void MultiChainIK3D::add_chain(const StringName &p_root_bone, const StringName &p_tip_bone, const NodePath &p_target) {
	ChainInfo chain;
	chain.root_bone = p_root_bone;
	chain.tip_bone = p_tip_bone;
	chain.target_node = p_target;
	chains.push_back(chain);
	_update_solvers();
}

void MultiChainIK3D::remove_chain(int p_index) {
	if (p_index < 0 || p_index >= (int)chains.size()) {
		return;
	}
	chains.remove_at(p_index);
	_update_solvers();
}

int MultiChainIK3D::get_chain_count() const {
	return chains.size();
}

void MultiChainIK3D::_update_solvers() {
	solvers.clear();
	if (!skeleton) {
		return;
	}

	for (uint32_t i = 0; i < chains.size(); i++) {
		IterateIK3D *solver = memnew(IterateIK3D);
		add_child(solver);
		solver->set_max_iterations(max_iterations);
		solver->set_min_distance(min_distance);
		solver->set_angular_delta_limit(angular_delta_limit);
		solvers.push_back(solver);
	}
}

void MultiChainIK3D::set_max_iterations(int p_max_iterations) {
	max_iterations = p_max_iterations;
	for (uint32_t i = 0; i < solvers.size(); i++) {
		if (solvers[i]) {
			solvers[i]->set_max_iterations(max_iterations);
		}
	}
}

int MultiChainIK3D::get_max_iterations() const {
	return max_iterations;
}

void MultiChainIK3D::set_min_distance(double p_min_distance) {
	min_distance = p_min_distance;
	for (uint32_t i = 0; i < solvers.size(); i++) {
		if (solvers[i]) {
			solvers[i]->set_min_distance(min_distance);
		}
	}
}

double MultiChainIK3D::get_min_distance() const {
	return min_distance;
}

void MultiChainIK3D::set_angular_delta_limit(double p_angular_delta_limit) {
	angular_delta_limit = p_angular_delta_limit;
	for (uint32_t i = 0; i < solvers.size(); i++) {
		if (solvers[i]) {
			solvers[i]->set_angular_delta_limit(angular_delta_limit);
		}
	}
}

double MultiChainIK3D::get_angular_delta_limit() const {
	return angular_delta_limit;
}

StringName MultiChainIK3D::get_chain_root_bone(int p_index) const {
	if (p_index < 0 || p_index >= (int)chains.size()) {
		return StringName();
	}
	return chains[p_index].root_bone;
}

StringName MultiChainIK3D::get_chain_tip_bone(int p_index) const {
	if (p_index < 0 || p_index >= (int)chains.size()) {
		return StringName();
	}
	return chains[p_index].tip_bone;
}

NodePath MultiChainIK3D::get_chain_target(int p_index) const {
	if (p_index < 0 || p_index >= (int)chains.size()) {
		return NodePath();
	}
	return chains[p_index].target_node;
}

void MultiChainIK3D::set_chain_target(int p_index, const NodePath &p_target) {
	if (p_index < 0 || p_index >= (int)chains.size()) {
		return;
	}
	chains[p_index].target_node = p_target;
}

void MultiChainIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::INT, "max_iterations", PROPERTY_HINT_RANGE, "1,64,1"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "min_distance", PROPERTY_HINT_RANGE, "0.0001,1.0,0.0001"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "angular_delta_limit", PROPERTY_HINT_RANGE, "0.001,0.5,0.001"));

	for (uint32_t i = 0; i < chains.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::NIL, "chains/" + itos(i), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_GROUP));
		p_list->push_back(PropertyInfo(Variant::STRING_NAME, "chains/" + itos(i) + "/root_bone"));
		p_list->push_back(PropertyInfo(Variant::STRING_NAME, "chains/" + itos(i) + "/tip_bone"));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, "chains/" + itos(i) + "/target_node"));
	}
}

bool MultiChainIK3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path == "max_iterations") {
		r_ret = max_iterations;
		return true;
	} else if (path == "min_distance") {
		r_ret = min_distance;
		return true;
	} else if (path == "angular_delta_limit") {
		r_ret = angular_delta_limit;
		return true;
	}

	if (path.begins_with("chains/")) {
		Vector<String> parts = path.split("/");
		if (parts.size() < 3) {
			return false;
		}

		int chain_idx = parts[1].to_int();
		if (chain_idx < 0 || chain_idx >= (int)chains.size()) {
			return false;
		}

		String property = parts[2];
		if (property == "root_bone") {
			r_ret = chains[chain_idx].root_bone;
			return true;
		} else if (property == "tip_bone") {
			r_ret = chains[chain_idx].tip_bone;
			return true;
		} else if (property == "target_node") {
			r_ret = chains[chain_idx].target_node;
			return true;
		}
	}

	return false;
}

bool MultiChainIK3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path == "max_iterations") {
		set_max_iterations(p_value);
		return true;
	} else if (path == "min_distance") {
		set_min_distance(p_value);
		return true;
	} else if (path == "angular_delta_limit") {
		set_angular_delta_limit(p_value);
		return true;
	}

	if (path.begins_with("chains/")) {
		Vector<String> parts = path.split("/");
		if (parts.size() < 3) {
			return false;
		}

		int chain_idx = parts[1].to_int();
		if (chain_idx < 0 || chain_idx >= (int)chains.size()) {
			return false;
		}

		String property = parts[2];
		if (property == "root_bone") {
			chains[chain_idx].root_bone = p_value;
			return true;
		} else if (property == "tip_bone") {
			chains[chain_idx].tip_bone = p_value;
			return true;
		} else if (property == "target_node") {
			chains[chain_idx].target_node = p_value;
			return true;
		}
	}

	return false;
}

void MultiChainIK3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_chain", "root_bone", "tip_bone", "target_node"), &MultiChainIK3D::add_chain);
	ClassDB::bind_method(D_METHOD("remove_chain", "index"), &MultiChainIK3D::remove_chain);
	ClassDB::bind_method(D_METHOD("get_chain_count"), &MultiChainIK3D::get_chain_count);

	ClassDB::bind_method(D_METHOD("set_max_iterations", "max_iterations"), &MultiChainIK3D::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations"), &MultiChainIK3D::get_max_iterations);
	ClassDB::bind_method(D_METHOD("set_min_distance", "min_distance"), &MultiChainIK3D::set_min_distance);
	ClassDB::bind_method(D_METHOD("get_min_distance"), &MultiChainIK3D::get_min_distance);
	ClassDB::bind_method(D_METHOD("set_angular_delta_limit", "angular_delta_limit"), &MultiChainIK3D::set_angular_delta_limit);
	ClassDB::bind_method(D_METHOD("get_angular_delta_limit"), &MultiChainIK3D::get_angular_delta_limit);

	ClassDB::bind_method(D_METHOD("get_chain_root_bone", "index"), &MultiChainIK3D::get_chain_root_bone);
	ClassDB::bind_method(D_METHOD("get_chain_tip_bone", "index"), &MultiChainIK3D::get_chain_tip_bone);
	ClassDB::bind_method(D_METHOD("get_chain_target", "index"), &MultiChainIK3D::get_chain_target);
	ClassDB::bind_method(D_METHOD("set_chain_target", "index", "target"), &MultiChainIK3D::set_chain_target);
}
