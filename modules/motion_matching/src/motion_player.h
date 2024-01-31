/**************************************************************************/
/*  motion_player.h                                                       */
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

#ifndef MOTION_PLAYER_H
#define MOTION_PLAYER_H

#include "core/string/node_path.h"
#include "core/variant/variant.h"
#include "scene/main/node.h"

#include "scene/resources/animation.h"
#include "scene/resources/animation_library.h"

#include "scene/3d/skeleton_3d.h"

#include <bitset>

#include "../thirdparty/kdtree.hpp"
#include "motion_features.h"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/skewness.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <cstdint>

class MotionPlayer : public Node {
	GDCLASS(MotionPlayer, Node)

	static constexpr float interval = 0.1;
	static constexpr float time_delta = 1.f / 30.f;

public:
	MotionPlayer() = default;
	~MotionPlayer() = default;

	PackedFloat32Array MotionData;

	PackedFloat32Array get_MotionData() const {
		return MotionData;
	}

	void set_MotionData(const PackedFloat32Array &value) {
		MotionData = value;
	}

	Dictionary blackboard;

	Dictionary get_blackboard() const {
		return blackboard;
	}

	void set_blackboard(const Dictionary &value) {
		blackboard = value;
	}
	TypedArray<String> category_track_names;

	TypedArray<String> get_category_track_names() const {
		return category_track_names;
	}

	void set_category_track_names(const TypedArray<String> &value) {
		category_track_names = value;
	}

	NodePath skeleton_path;
	void set_skeleton(NodePath path);
	NodePath get_skeleton();

	// The KdTree.
	Kdtree::KdTree *kdt = nullptr;

	NodePath main_node;

	NodePath get_main_node() const {
		return main_node;
	}

	void set_main_node(const NodePath &value) {
		main_node = value;
	}

	// Animation Library. Each one will be analysed
	Ref<AnimationLibrary> animation_library;

	Ref<AnimationLibrary> get_animation_library() const {
		return animation_library;
	}

	void set_animation_library(const Ref<AnimationLibrary> &value) {
		animation_library = value;
	}

	// Array of the motion features.
	Array motion_features;

	Array get_motion_features() const {
		return motion_features;
	}

	void set_motion_features(const Array &value) {
		motion_features = value;
	}

	// Dimensional Stats.
	PackedFloat32Array weights;

	PackedFloat32Array get_weights() const {
		return weights;
	}

	void set_weights(const PackedFloat32Array &value) {
		weights = value;
	}
	PackedFloat32Array means;

	PackedFloat32Array get_means() const {
		return means;
	}

	void set_means(const PackedFloat32Array &value) {
		means = value;
	}
	PackedFloat32Array variances;

	PackedFloat32Array get_variances() const {
		return variances;
	}

	void set_variances(const PackedFloat32Array &value) {
		variances = value;
	}

	Array densities;

	Array get_densities() const {
		return densities;
	}

	void set_densities(const Array &value) {
		densities = value;
	}

	// Database. A pose is just the index of a row in the kdtree.
	// Usage : db_anim_*[result.index] =
	PackedInt32Array db_anim_index;

	PackedInt32Array get_db_anim_index() const {
		return db_anim_index;
	}

	void set_db_anim_index(const PackedInt32Array &value) {
		db_anim_index = value;
	}
	PackedFloat32Array db_anim_timestamp;

	PackedFloat32Array get_db_anim_timestamp() const {
		return db_anim_timestamp;
	}

	void set_db_anim_timestamp(const PackedFloat32Array &value) {
		db_anim_timestamp = value;
	}

	PackedInt32Array db_anim_category;

	PackedInt32Array get_db_anim_category() const {
		return db_anim_category;
	}

	void set_db_anim_category(const PackedInt32Array &value) {
		db_anim_category = value;
	}

	// Simple helper. Might be removed from this class
	int category_value = INT_MAX;

	int get_category_value() const {
		return category_value;
	}

	void set_category_value(int value) {
		category_value = value;
	}

	// How the kdtree calculate the distance.
	// 0 (L0) : Maximum of each difference in all dimensions.
	// 1 (L1) : Manhattan distance (default)
	// 2 (L2) : Distance squared.
	int distance_type = 1;
	int get_distance_type();
	void set_distance_type(int value);

protected:
	void _notification(int p_what);

public:
	// Useful while baking data and in editor.
	void set_skeleton_to_pose(Ref<Animation> animation, double time);

	// Reset the skeleton poses.
	void reset_skeleton_poses();

	// Bake the data into the KdTree.
	// Goes through all the animations and construct the kdtree with each features at the interval.
	virtual void baking_data();

	// A predicate for searching animation categories.
	// included_category_bitfield : The animations must at least contains those categories
	// excluded_category_bitfield : Exclude every animations with any of those categories
	struct Category_Pred : Kdtree::KdNodePredicate {
		const std::bitset<64> m_desired_category;
		const std::bitset<64> m_exclude_category;
		Category_Pred(int64_t included_category_bitfield, int64_t excluded_category_bitfield = 0) :
				m_desired_category{ static_cast<uint64_t>(included_category_bitfield) }, m_exclude_category{ static_cast<uint64_t>(excluded_category_bitfield) } {}

		virtual bool operator()(const Kdtree::KdNode &node) const {
			static constexpr std::bitset<64> zero = {};
			const std::bitset<64> node_category = *((int32_t *)node.data);
			const bool include = (m_desired_category & node_category) == node_category;
			const bool exclude = (m_exclude_category & node_category) == zero;
			return include && exclude;
		}
	};
	// Calculate the weights using the features get_weights() functions.
	// Take into consideration the number of dimensions.
	// The calculation might be reconsidered, but it's the best I found.
	void recalculate_weights();

	// query the kdtree.
	// Can include or exclude categories.
	TypedArray<Dictionary> query_pose(int64_t included_category = std::numeric_limits<int64_t>::max(), int64_t exclude = 0);

	// Bypass the feature query, and ask directly which poses is the most similar.
	// The query must be of the correct dimension.
	Array check_query_results(PackedFloat32Array query, int64_t nb_result = 1);

protected:
	// Binding.
	static void _bind_methods();
};

#endif // MOTION_PLAYER_H