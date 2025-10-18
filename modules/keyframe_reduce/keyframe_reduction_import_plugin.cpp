/**************************************************************************/
/*  keyframe_reduction_import_plugin.cpp                                  */
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

#include "keyframe_reduction_import_plugin.h"

#include "scene/resources/animation.h"
#include <cstdint>

void KeyframeReductionImportPlugin::get_import_options(const String &p_path, List<ResourceImporter::ImportOption> *r_options) {
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "keyframe_reduction/enable", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "keyframe_reduction/max_error", PROPERTY_HINT_RANGE, "0.001,1.0,0.001", PROPERTY_USAGE_DEFAULT), 0.05f));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "keyframe_reduction/step_size", PROPERTY_HINT_RANGE, "0.01,1.0,0.01", PROPERTY_USAGE_DEFAULT), 0.1f));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "keyframe_reduction/tangent_split_angle", PROPERTY_HINT_RANGE, "1,180,1", PROPERTY_USAGE_DEFAULT), 45.0f));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "keyframe_reduction/use_one_euro_filter", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), true));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "keyframe_reduction/one_euro_min_cutoff", PROPERTY_HINT_RANGE, "0.1,10.0,0.1", PROPERTY_USAGE_DEFAULT), 1.0f));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "keyframe_reduction/one_euro_beta", PROPERTY_HINT_RANGE, "0.0,1.0,0.01", PROPERTY_USAGE_DEFAULT), 0.1f));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "keyframe_reduction/one_euro_dcutoff", PROPERTY_HINT_RANGE, "0.1,10.0,0.1", PROPERTY_USAGE_DEFAULT), 1.0f));
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "keyframe_reduction/conversion_sample_rate", PROPERTY_HINT_RANGE, "0.01,0.5,0.01", PROPERTY_USAGE_DEFAULT), 0.05f));
}

Variant KeyframeReductionImportPlugin::get_option_visibility(const String &p_path, const String &p_scene_import_type, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	if (p_option.begins_with("keyframe_reduction/") && p_option != "keyframe_reduction/enable") {
		if (p_options.has("keyframe_reduction/enable") && !bool(p_options["keyframe_reduction/enable"])) {
			return false;
		}
	}
	return true;
}

void KeyframeReductionImportPlugin::internal_process(EditorScenePostImportPlugin::InternalImportCategory p_category, Node *p_base_scene, Node *p_node, Ref<Resource> p_resource, const Dictionary &p_options) {
	if (p_category != INTERNAL_IMPORT_CATEGORY_ANIMATION) {
		return;
	}

	Ref<Animation> animation = p_resource;
	if (animation.is_null()) {
		return;
	}

	// Extract options
	ImportOptions opts;
	if (p_options.has("keyframe_reduction/enable")) {
		opts.enable_reduction = p_options["keyframe_reduction/enable"];
	}
	if (p_options.has("keyframe_reduction/max_error")) {
		opts.max_error = p_options["keyframe_reduction/max_error"];
	}
	if (p_options.has("keyframe_reduction/step_size")) {
		opts.step_size = p_options["keyframe_reduction/step_size"];
	}
	if (p_options.has("keyframe_reduction/tangent_split_angle")) {
		opts.tangent_split_angle_threshold = p_options["keyframe_reduction/tangent_split_angle"];
	}
	if (p_options.has("keyframe_reduction/use_one_euro_filter")) {
		opts.use_one_euro_filter = p_options["keyframe_reduction/use_one_euro_filter"];
	}
	if (p_options.has("keyframe_reduction/one_euro_min_cutoff")) {
		opts.one_euro_min_cutoff = p_options["keyframe_reduction/one_euro_min_cutoff"];
	}
	if (p_options.has("keyframe_reduction/one_euro_beta")) {
		opts.one_euro_beta = p_options["keyframe_reduction/one_euro_beta"];
	}
	if (p_options.has("keyframe_reduction/one_euro_dcutoff")) {
		opts.one_euro_dcutoff = p_options["keyframe_reduction/one_euro_dcutoff"];
	}
	if (p_options.has("keyframe_reduction/conversion_sample_rate")) {
		opts.conversion_sample_rate = p_options["keyframe_reduction/conversion_sample_rate"];
	}

	if (!opts.enable_reduction) {
		return;
	}

	// Process each track in the animation
	int track_count = animation->get_track_count();
	for (int i = 0; i < track_count; i++) {
		Animation::TrackType track_type = animation->track_get_type(i);

		// Convert track to Bezier if it's not already
		if (track_type != Animation::TYPE_BEZIER) {
			_convert_track_to_bezier(animation, i, opts);
		}

		// Apply keyframe reduction to Bezier tracks
		if (animation->track_get_type(i) == Animation::TYPE_BEZIER) {
			_reduce_bezier_track(animation, i, opts);
		}
	}
}

void KeyframeReductionImportPlugin::_convert_track_to_bezier(Ref<Animation> p_animation, int p_track_idx, const ImportOptions &p_options) {
	Animation::TrackType track_type = p_animation->track_get_type(p_track_idx);

	if (track_type == Animation::TYPE_BEZIER) {
		return; // Already a Bezier track
	}

	NodePath base_path = p_animation->track_get_path(p_track_idx);
	real_t animation_length = p_animation->get_length();
	real_t sample_rate = p_options.conversion_sample_rate;

	// Handle multi-component tracks by creating multiple Bezier tracks
	switch (track_type) {
		case Animation::TYPE_POSITION_3D:
		case Animation::TYPE_SCALE_3D: {
			// Create 3 separate Bezier tracks for x, y, z components
			const char *components[3] = { ":x", ":y", ":z" };

			for (int comp = 0; comp < 3; comp++) {
				LocalVector<BezierKeyframeReduce::Bezier> bezier_points;

				for (real_t t = 0.0f; t <= animation_length; t += sample_rate) {
					real_t value = 0.0f;

					if (track_type == Animation::TYPE_POSITION_3D) {
						Vector3 pos = p_animation->position_track_interpolate(p_track_idx, t);
						value = pos[comp];
					} else { // TYPE_SCALE_3D
						Vector3 scale = p_animation->scale_track_interpolate(p_track_idx, t);
						value = scale[comp];
					}

					// Create Bezier point
					BezierKeyframeReduce::Vector2Bezier time_value(t, value);
					BezierKeyframeReduce::Vector2Bezier in_handle(t - sample_rate * 0.3f, value);
					BezierKeyframeReduce::Vector2Bezier out_handle(t + sample_rate * 0.3f, value);

					BezierKeyframeReduce::Bezier bezier_point(time_value, in_handle, out_handle);
					bezier_points.push_back(bezier_point);
				}

				// Add new Bezier track for this component
				String comp_suffix = components[comp];
				NodePath comp_path = NodePath(String(base_path.get_concatenated_names()) + comp_suffix);
				int new_track_idx = p_animation->add_track(Animation::TYPE_BEZIER);
				p_animation->track_set_path(new_track_idx, comp_path);

				// Add Bezier keyframes
				for (int i = 0; i < bezier_points.size(); i++) {
					const BezierKeyframeReduce::Bezier &bezier = bezier_points[i];
					p_animation->bezier_track_insert_key(new_track_idx, bezier.time_value.x,
							bezier.time_value.y, bezier.in_handle, bezier.out_handle);
				}
			}
		} break;

		case Animation::TYPE_ROTATION_3D: {
			// Convert quaternion track to 6D representation, apply reduction, then convert back to quaternion
			LocalVector<Quaternion> original_rotations;

			// Sample original quaternion track
			for (real_t t = 0.0f; t <= animation_length; t += sample_rate) {
				Quaternion rot = p_animation->rotation_track_interpolate(p_track_idx, t);
				original_rotations.push_back(rot);
			}

			// Convert to 6D Bezier tracks, reduce, then reconstruct
			LocalVector<LocalVector<BezierKeyframeReduce::Bezier>> component_bezier_tracks;
			component_bezier_tracks.resize(6);

			for (int comp = 0; comp < 6; comp++) {
				LocalVector<BezierKeyframeReduce::Bezier> bezier_points;

				for (int i = 0; i < original_rotations.size(); i++) {
					real_t t = i * sample_rate;
					Basis basis = Basis(original_rotations[i]);

					real_t value = 0.0f;
					if (comp < 3) {
						// First column (x, y, z)
						value = basis.get_column(0)[comp];
					} else {
						// Second column (x, y, z)
						value = basis.get_column(1)[comp - 3];
					}

					// Create Bezier point
					BezierKeyframeReduce::Vector2Bezier time_value(t, value);
					BezierKeyframeReduce::Vector2Bezier in_handle(t - sample_rate * 0.3f, value);
					BezierKeyframeReduce::Vector2Bezier out_handle(t + sample_rate * 0.3f, value);

					BezierKeyframeReduce::Bezier bezier_point(time_value, in_handle, out_handle);
					bezier_points.push_back(bezier_point);
				}

				component_bezier_tracks[comp] = bezier_points;
			}

			// Apply keyframe reduction to each 6D component
			LocalVector<LocalVector<BezierKeyframeReduce::Bezier>> reduced_components;
			reduced_components.resize(6);
			for (int comp = 0; comp < 6; comp++) {
				if (component_bezier_tracks[comp].size() < 3) {
					reduced_components[comp] = component_bezier_tracks[comp];
					continue;
				}

				Ref<BezierKeyframeReduce> reducer;
				reducer.instantiate();

				BezierKeyframeReduce::KeyframeReductionSetting settings;
				settings.max_error = p_options.max_error;
				settings.step_size = p_options.step_size;
				settings.tangent_split_angle_threshold_in_degrees = p_options.tangent_split_angle_threshold;
				settings.use_one_euro_filter = p_options.use_one_euro_filter;
				settings.one_euro_min_cutoff = p_options.one_euro_min_cutoff;
				settings.one_euro_beta = p_options.one_euro_beta;
				settings.one_euro_dcutoff = p_options.one_euro_dcutoff;

				LocalVector<BezierKeyframeReduce::Bezier> reduced_keyframes;
				reducer->reduce(component_bezier_tracks[comp], reduced_keyframes, settings);
				reduced_components[comp] = reduced_keyframes;
			}

			// Reconstruct quaternion track from reduced 6D components
			p_animation->remove_track(p_track_idx);
			int new_track_idx = p_animation->add_track(Animation::TYPE_ROTATION_3D, p_track_idx);
			p_animation->track_set_path(new_track_idx, base_path);

			// For each time point in the reduced keyframes, reconstruct quaternion
			// Use the first component to determine the time points
			const LocalVector<BezierKeyframeReduce::Bezier> &first_component = reduced_components[0];
			for (int i = 0; i < first_component.size(); i++) {
				real_t time = first_component[i].time_value.x;

				// Sample all 6 components at this time point
				Vector3 col0, col1;
				for (int comp = 0; comp < 6; comp++) {
					const LocalVector<BezierKeyframeReduce::Bezier> &component_track = reduced_components[comp];

					// Find the keyframe at this time (or interpolate)
					real_t value = 0.0f;
					for (int j = 0; j < component_track.size(); j++) {
						if (Math::is_equal_approx(component_track[j].time_value.x, time)) {
							value = component_track[j].time_value.y;
							break;
						}
					}

					if (comp < 3) {
						col0[comp] = value;
					} else {
						col1[comp - 3] = value;
					}
				}

				// Reconstruct third column: cross product of first two columns
				Vector3 col2 = col0.cross(col1);

				// Create rotation matrix and convert to quaternion
				Basis reconstructed_basis(col0, col1, col2);
				Quaternion reconstructed_rot = reconstructed_basis.get_rotation_quaternion();

				// Add keyframe to rotation track
				p_animation->rotation_track_insert_key(new_track_idx, time, reconstructed_rot);
			}
		} break;

		case Animation::TYPE_BLEND_SHAPE:
		case Animation::TYPE_VALUE: {
			// Single component tracks - convert directly to one Bezier track
			LocalVector<BezierKeyframeReduce::Bezier> bezier_points;

			for (real_t t = 0.0f; t <= animation_length; t += sample_rate) {
				real_t value = 0.0f;

				if (track_type == Animation::TYPE_BLEND_SHAPE) {
					value = p_animation->blend_shape_track_interpolate(p_track_idx, t);
				} else { // TYPE_VALUE
					Variant v = p_animation->value_track_interpolate(p_track_idx, t);
					if (v.get_type() == Variant::FLOAT || v.get_type() == Variant::INT) {
						value = v;
					}
				}

				// Create Bezier point
				BezierKeyframeReduce::Vector2Bezier time_value(t, value);
				BezierKeyframeReduce::Vector2Bezier in_handle(t - sample_rate * 0.3f, value);
				BezierKeyframeReduce::Vector2Bezier out_handle(t + sample_rate * 0.3f, value);

				BezierKeyframeReduce::Bezier bezier_point(time_value, in_handle, out_handle);
				bezier_points.push_back(bezier_point);
			}

			// Replace the old track with new Bezier track
			p_animation->remove_track(p_track_idx);

			int new_track_idx = p_animation->add_track(Animation::TYPE_BEZIER, p_track_idx);
			p_animation->track_set_path(new_track_idx, base_path);

			// Add Bezier keyframes
			for (int i = 0; i < bezier_points.size(); i++) {
				const BezierKeyframeReduce::Bezier &bezier = bezier_points[i];
				p_animation->bezier_track_insert_key(new_track_idx, bezier.time_value.x,
						bezier.time_value.y, bezier.in_handle, bezier.out_handle);
			}
		} break;

		default:
			return; // Skip unsupported track types
	}

	// Remove the original track (already done for single-component tracks above)
	if (track_type == Animation::TYPE_POSITION_3D || track_type == Animation::TYPE_ROTATION_3D || track_type == Animation::TYPE_SCALE_3D) {
		p_animation->remove_track(p_track_idx);
	}
}

void KeyframeReductionImportPlugin::_reduce_bezier_track(Ref<Animation> p_animation, int p_track_idx, const ImportOptions &p_options) {
	if (p_animation->track_get_type(p_track_idx) != Animation::TYPE_BEZIER) {
		return;
	}

	// Extract Bezier keyframes from the animation track
	LocalVector<BezierKeyframeReduce::Bezier> bezier_points;

	int key_count = p_animation->track_get_key_count(p_track_idx);
	for (int i = 0; i < key_count; i++) {
		real_t time = p_animation->track_get_key_time(p_track_idx, i);
		real_t value = p_animation->bezier_track_get_key_value(p_track_idx, i);
		Vector2 in_handle = p_animation->bezier_track_get_key_in_handle(p_track_idx, i);
		Vector2 out_handle = p_animation->bezier_track_get_key_out_handle(p_track_idx, i);

		BezierKeyframeReduce::Vector2Bezier time_value(time, value);
		BezierKeyframeReduce::Vector2Bezier in_h(in_handle.x, in_handle.y);
		BezierKeyframeReduce::Vector2Bezier out_h(out_handle.x, out_handle.y);

		BezierKeyframeReduce::Bezier bezier(time_value, in_h, out_h);
		bezier_points.push_back(bezier);
	}

	if (bezier_points.size() < 3) {
		return; // Not enough points to reduce
	}

	// Apply keyframe reduction
	Ref<BezierKeyframeReduce> reducer;
	reducer.instantiate();

	BezierKeyframeReduce::KeyframeReductionSetting settings;
	settings.max_error = p_options.max_error;
	settings.step_size = p_options.step_size;
	settings.tangent_split_angle_threshold_in_degrees = p_options.tangent_split_angle_threshold;
	settings.use_one_euro_filter = p_options.use_one_euro_filter;
	settings.one_euro_min_cutoff = p_options.one_euro_min_cutoff;
	settings.one_euro_beta = p_options.one_euro_beta;
	settings.one_euro_dcutoff = p_options.one_euro_dcutoff;

	LocalVector<BezierKeyframeReduce::Bezier> reduced_keyframes;
	reducer->reduce(bezier_points, reduced_keyframes, settings);

	// Replace the track with reduced keyframes
	p_animation->remove_track(p_track_idx);
	int new_track_idx = p_animation->add_track(Animation::TYPE_BEZIER, p_track_idx);
	p_animation->track_set_path(new_track_idx, p_animation->track_get_path(p_track_idx));

	// Add reduced Bezier keyframes
	for (uint32_t i = 0; i < reduced_keyframes.size(); i++) {
		const BezierKeyframeReduce::Bezier &bezier = reduced_keyframes[i];
		p_animation->bezier_track_insert_key(new_track_idx, bezier.time_value.x,
				bezier.time_value.y,
				Vector2(bezier.in_handle.x, bezier.in_handle.y),
				Vector2(bezier.out_handle.x, bezier.out_handle.y));
	}
}

void KeyframeReductionImportPlugin::_bind_methods() {
}

KeyframeReductionImportPlugin::KeyframeReductionImportPlugin() {
}
