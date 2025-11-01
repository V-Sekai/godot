/**************************************************************************/
/*  multi_chain_ik_3d.h                                                   */
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

#ifndef MULTI_CHAIN_IK_3D_H
#define MULTI_CHAIN_IK_3D_H

#include "scene/3d/iterate_ik_3d.h"
#include "scene/3d/skeleton_3d.h"

class MultiChainIK3D : public Node3D {
	GDCLASS(MultiChainIK3D, Node3D);

public:
	struct ChainInfo {
		StringName root_bone;
		StringName tip_bone;
		NodePath target_node;
	};

protected:
	Skeleton3D *skeleton = nullptr;
	LocalVector<ChainInfo> chains;
	LocalVector<IterateIK3D *> solvers;

	// Shared configuration for all chains
	int max_iterations = 4;
	double min_distance = 0.001;
	double angular_delta_limit = Math::deg_to_rad(2.0);

	void _update_solvers();
	void _get_property_list(List<PropertyInfo> *p_list) const;
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	static void _bind_methods();

public:
	MultiChainIK3D();
	~MultiChainIK3D();

	// Chain management
	void add_chain(const StringName &p_root_bone, const StringName &p_tip_bone, const NodePath &p_target);
	void remove_chain(int p_index);
	int get_chain_count() const;

	// Configuration (merged across all chains)
	void set_max_iterations(int p_max_iterations);
	int get_max_iterations() const;
	void set_min_distance(double p_min_distance);
	double get_min_distance() const;
	void set_angular_delta_limit(double p_angular_delta_limit);
	double get_angular_delta_limit() const;

	// Per-chain access
	StringName get_chain_root_bone(int p_index) const;
	StringName get_chain_tip_bone(int p_index) const;
	NodePath get_chain_target(int p_index) const;
	void set_chain_target(int p_index, const NodePath &p_target);

	virtual void _notification(int p_what);
};

#endif // MULTI_CHAIN_IK_3D_H
