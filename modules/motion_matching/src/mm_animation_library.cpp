/**************************************************************************/
/*  mm_animation_library.cpp                                              */
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

#include "mm_animation_library.h"

#include "common.h"
#include "core/math/math_funcs.h"
#include "features/mm_feature.h"
#include "math/hash.h"
#include "math/stats.hpp"
#include "mm_character.h"

void MMAnimationLibrary::bake_data(const MMCharacter *p_character, const AnimationMixer *p_player, const Skeleton3D *p_skeleton) {
	motion_data.clear();
	db_anim_index.clear();
	db_time_index.clear();
	db_pose_offset.clear();

	int32_t dim_count = 0;
	for (auto i = 0; i < features.size(); ++i) {
		MMFeature *f = Object::cast_to<MMFeature>(features[i]);
		dim_count += f->get_dimension_count();
		f->setup_skeleton(p_character, p_player, p_skeleton);
	}

	List<StringName> animation_list;
	get_animation_list(&animation_list);

	// Normalization data
	std::vector<std::vector<StatsAccumulator>> stats(features.size());

	PackedFloat32Array data;
	int32_t current_pose_offset = 0;
	// For every animation
	int64_t animation_index = 0;
	for (const StringName &anim_name : animation_list) {
		Ref<Animation> animation = get_animation(anim_name);
		db_pose_offset.push_back(current_pose_offset);

		// Initialize features
		for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
			MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);
			stats[feature_index].resize(feature->get_dimension_count());
			feature->setup_for_animation(animation);
		}

		const double animation_length = animation->get_length();
		const double time_step = 1.0f / get_sampling_rate();
		// Every time step
		for (double time = 0; time < animation_length; time += time_step) {
			PackedFloat32Array pose_data;
			// For every feature
			for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
				const MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);
				const PackedFloat32Array feature_data = feature->bake_animation_pose(animation, time);

				ERR_FAIL_COND(feature_data.size() != feature->get_dimension_count());

				// Update stats
				for (int64_t feature_element_index = 0; feature_element_index < feature_data.size(); feature_element_index++) {
					stats[feature_index][feature_element_index].add_sample(feature_data[feature_element_index]);
				}

				pose_data.append_array(feature_data);
				current_pose_offset += feature->get_dimension_count();
			}

			ERR_FAIL_COND(pose_data.size() != dim_count);

			// Update dataset
			data.append_array(pose_data);
			db_anim_index.push_back(animation_index);
			db_time_index.push_back(time);
		}
		animation_index++;
	}

	// Compute mean and standard deviation
	for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
		MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);

		PackedFloat32Array feature_means;
		feature_means.resize(feature->get_dimension_count());
		PackedFloat32Array feature_std_devs;
		feature_std_devs.resize(feature->get_dimension_count());
		PackedFloat32Array feature_mins;
		feature_mins.resize(feature->get_dimension_count());
		PackedFloat32Array feature_maxes;
		feature_maxes.resize(feature->get_dimension_count());

		for (int64_t feature_element_index = 0; feature_element_index < feature->get_dimension_count(); feature_element_index++) {
			feature_means.set(feature_element_index, stats[feature_index][feature_element_index].get_mean());
			feature_std_devs.set(feature_element_index, stats[feature_index][feature_element_index].get_standard_deviation());
			feature_mins.set(feature_element_index, stats[feature_index][feature_element_index].get_min());
			feature_maxes.set(feature_element_index, stats[feature_index][feature_element_index].get_max());
		}

		feature->set_means(feature_means);
		feature->set_std_devs(feature_std_devs);
		feature->set_mins(feature_mins);
		feature->set_maxes(feature_maxes);
	}

	_normalize_data(data, dim_count);

	motion_data = data.duplicate();

	schema_hash = compute_features_hash();

	_kd_tree = std::make_unique<KDTree>(motion_data.ptr(), dim_count, ((int32_t)motion_data.size()) / dim_count);
	node_indices = PackedInt32Array(_kd_tree->get_node_indices());
}

MMQueryOutput MMAnimationLibrary::query(const MMQueryInput &p_query_input) {
	PackedFloat32Array query_vector;
	for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
		const MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);
		PackedFloat32Array feature_data = feature->evaluate_runtime_data(p_query_input);
		feature->normalize(feature_data.ptrw());
		query_vector.append_array(feature_data);
	}

	if (node_indices.is_empty()) {
		return _search_naive(query_vector);
	}
	return _search_kd_tree(query_vector);
}

int64_t MMAnimationLibrary::get_dim_count() const {
	int64_t dim_count = 0;
	for (auto i = 0; i < features.size(); ++i) {
		MMFeature *f = Object::cast_to<MMFeature>(features[i]);
		dim_count += f->get_dimension_count();
	}

	return dim_count;
}

int64_t MMAnimationLibrary::get_animation_pose_count(String p_animation_name) const {
	Ref<Animation> animation = get_animation(p_animation_name);
	if (animation.is_null()) {
		return 0;
	}
	const double animation_length = animation->get_length();
	const double time_step = 1.0f / get_sampling_rate();
	return static_cast<int32_t>(Math::floor(animation_length / time_step));
}

#ifdef TOOLS_ENABLED
void MMAnimationLibrary::display_data(const Ref<EditorNode3DGizmo> &p_gizmo, const Transform3D &p_transform, String p_animation_name, int32_t p_pose_index) const {
	List<StringName> anim_list;
	get_animation_list(&anim_list);
	int32_t anim_index = -1;
	int idx = 0;
	for (const StringName &anim_name : anim_list) {
		if (anim_name == p_animation_name) {
			anim_index = idx;
			break;
		}
		idx++;
	}
	ERR_FAIL_COND(anim_index < 0);

	const int32_t dim_count = get_dim_count();
	int32_t start_frame_index = db_pose_offset[anim_index];
	int32_t frame_index = start_frame_index + p_pose_index * dim_count;

	for (int feature_index = 0; feature_index < (int)features.size(); feature_index++) {
		const MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);
		const float *frame_motion_data = motion_data.ptr() + frame_index;
		feature->display_data(p_gizmo, p_transform, frame_motion_data);
		frame_index += feature->get_dimension_count();
	}
}
#endif

int64_t MMAnimationLibrary::compute_features_hash() const {
	int64_t hash = 0;
	for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
		MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);
		List<PropertyInfo> plist;
		feature->get_property_list(&plist);
		for (const PropertyInfo &E : plist) {
			hash = hash_combine(hash, (int64_t)E.name.hash());
			hash = hash_combine(hash, (int64_t)E.type);
		}
	}
	return hash;
}

bool MMAnimationLibrary::needs_baking() const {
	return motion_data.is_empty() || db_anim_index.is_empty() || db_time_index.is_empty();
}

float MMAnimationLibrary::_compute_feature_costs(int p_pose_index, const PackedFloat32Array &p_query, Dictionary *p_feature_costs) const {
	float pose_cost = 0.f;
	int32_t dim_count = p_query.size();
	int32_t start_frame_index = p_pose_index * dim_count;
	int32_t start_feature_index = 0;

	for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
		const MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);
		if (!feature) {
			continue;
		}

		float feature_cost = 0.f;
		for (int64_t feature_dim_index = 0; feature_dim_index < feature->get_dimension_count(); feature_dim_index++) {
			feature_cost += distance_squared((motion_data.ptr() + start_frame_index + start_feature_index + feature_dim_index),
									(p_query.ptr() + start_feature_index + feature_dim_index),
									1) *
					feature->calculate_normalized_weight(feature_dim_index);
		}

		if (p_feature_costs) {
			p_feature_costs->get_or_add(feature->get_class(), feature_cost);
		}

		pose_cost += feature_cost;
		start_feature_index += feature->get_dimension_count();
	}
	return pose_cost;
}

MMQueryOutput MMAnimationLibrary::_search_naive(const PackedFloat32Array &p_query) const {
	float cost = FLT_MAX;
	MMQueryOutput result;
	List<StringName> animation_list;
	get_animation_list(&animation_list);
	int32_t dim_count = p_query.size();
	int64_t best_pose_index = -1;
	for (int64_t start_frame_index = 0; start_frame_index < motion_data.size(); start_frame_index += dim_count) {
		int start_feature_index = start_frame_index;
		float pose_cost = 0.f;
		Dictionary feature_costs;
		for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
			const MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);
			if (!feature) {
				continue;
			}

			float feature_cost = 0.f;
			for (int64_t feature_dim_index = 0; feature_dim_index < feature->get_dimension_count(); feature_dim_index++) {
				feature_cost += distance_squared((motion_data.ptr() + start_frame_index + start_feature_index + feature_dim_index),
										(p_query.ptr() + start_feature_index + feature_dim_index),
										1) *
						feature->calculate_normalized_weight(feature_dim_index);
			}

			feature_costs.get_or_add(feature->get_class(), feature_cost);
			pose_cost += feature_cost;
			start_feature_index += feature->get_dimension_count();
		}

		if (pose_cost < cost) {
			cost = pose_cost;
			best_pose_index = start_frame_index / dim_count;
		}
	}

	String library_name = get_path().get_file().get_basename() + "/";
	if (library_name.is_empty()) {
		library_name = get_name() + "/";
	}
	result.animation_match = library_name + String(animation_list.get(db_anim_index[best_pose_index]));
	result.time_match = db_time_index[best_pose_index];

	if (include_cost_results) {
		result.matched_frame_data = motion_data.slice(
				best_pose_index * dim_count,
				(best_pose_index + 1) * dim_count);

		result.cost = _compute_feature_costs(
				best_pose_index,
				p_query,
				&result.feature_costs);
	}

	return result;
}

MMQueryOutput MMAnimationLibrary::_search_kd_tree(const PackedFloat32Array &p_query) {
	if (!_kd_tree) {
		int32_t dim_count = p_query.size();
		_kd_tree = std::make_unique<KDTree>(dim_count);
		_kd_tree->rebuild_tree(((int32_t)motion_data.size()) / dim_count, node_indices);
	}

	std::vector<float> dimension_weights;
	for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
		const MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);
		if (!feature) {
			continue;
		}
		for (int64_t feature_dim_index = 0; feature_dim_index < feature->get_dimension_count(); feature_dim_index++) {
			dimension_weights.push_back(feature->calculate_normalized_weight(feature_dim_index));
		}
	}

	int best_pose_index = _kd_tree->search_nn(
			motion_data.ptr(),
			p_query.ptr(),
			dimension_weights);

	MMQueryOutput result;
	List<StringName> animation_list;
	get_animation_list(&animation_list);
	String library_name = get_path().get_file().get_basename() + "/";
	if (library_name.is_empty()) {
		library_name = get_name() + "/";
	}

	result.animation_match = library_name + String(animation_list.get(db_anim_index[best_pose_index]));
	result.time_match = db_time_index[best_pose_index];

	if (include_cost_results) {
		result.matched_frame_data = motion_data.slice(
				best_pose_index * p_query.size(),
				(best_pose_index + 1) * p_query.size());

		result.cost =
				_compute_feature_costs(
						best_pose_index,
						p_query,
						&result.feature_costs);
	}

	return result;
}

void MMAnimationLibrary::_bind_methods() {
	// Use "Resource" so C# glue resolves (MMFeature is abstract); avoids extra instantiable classes that trigger LeakSanitizer in regression test.
	BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::ARRAY, features, PROPERTY_HINT_ARRAY_TYPE, "Resource");
	BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::FLOAT, sampling_rate);
	BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::BOOL, include_cost_results);
	BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_FLOAT32_ARRAY, motion_data, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
	BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT32_ARRAY, db_anim_index, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
	BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_FLOAT32_ARRAY, db_time_index, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
	BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT32_ARRAY, db_pose_offset, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
	BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::INT, schema_hash, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
	BINDER_PROPERTY_PARAMS(MMAnimationLibrary, Variant::PACKED_INT32_ARRAY, node_indices, PROPERTY_HINT_NONE, "", DEBUG_PROPERTY_STORAGE_FLAG);
}

void MMAnimationLibrary::_normalize_data(PackedFloat32Array &p_data, size_t p_dim_count) const {
	ERR_FAIL_COND(p_data.size() % p_dim_count != 0);

	for (int64_t frame_index = 0; frame_index < p_data.size(); frame_index += p_dim_count) {
		int dim_index = 0;
		for (int64_t feature_index = 0; feature_index < features.size(); feature_index++) {
			const MMFeature *feature = Object::cast_to<MMFeature>(features[feature_index]);
			feature->normalize(p_data.ptrw() + frame_index + dim_index);
			dim_index += feature->get_dimension_count();
		}
	}
}
