/**************************************************************************/
/*  motion_features.h                                                     */
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

#ifndef MOTION_FEATURES_H
#define MOTION_FEATURES_H

#include "core/io/resource.h"
#include "core/math/math_defs.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/os/time.h"
#include "core/string/node_path.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_plugin.h"
#include "editor/plugins/node_3d_editor_gizmos.h"
#endif

#include "scene/3d/skeleton_3d.h"
#include "scene/main/node.h"
#include "scene/resources/animation.h"
#include "scene/resources/primitive_meshes.h"

class MotionFeature : public Resource {
	GDCLASS(MotionFeature, Resource)
public:
	static constexpr float delta = 0.016f;

	MotionFeature();

	virtual void physics_update(double p_delta) {}

	virtual int get_dimension() { return 0; }

	virtual PackedFloat32Array get_weights() { return {}; }

	virtual void setup_nodes(Variant character) {}

	virtual void setup_for_animation(Ref<Animation> animation) {}
	virtual PackedFloat32Array bake_animation_pose(Ref<Animation> animation, float time) { return {}; }

	virtual PackedFloat32Array broadphase_query_pose(Dictionary p_blackboard, float p_delta) { return {}; }

	virtual float narrowphase_evaluate_cost(PackedFloat32Array to_convert) { return 0.0; }

	virtual void debug_pose_gizmo(Ref<RefCounted> gizmo, const PackedFloat32Array data, Transform3D tr = Transform3D{}) { return; }

protected:
	static void _bind_methods();
};

#include "scene/3d/physics_body_3d.h"

class RootVelocityMotionFeature : public MotionFeature {
	GDCLASS(RootVelocityMotionFeature, MotionFeature)
public:
	CharacterBody3D *body;
	CharacterBody3D *get_body() { return body; }
	void set_body(CharacterBody3D *value) { body = value; }
	int root_track_pos = -1, root_track_quat = -1; //, root_track_scale = -1;

	String root_bone_name = "%GeneralSkeleton:Root";
	void set_root_bone_name(String value) {
		root_bone_name = value;
	}
	String get_root_bone_name() { return root_bone_name; }

	virtual int get_dimension() override {
		return 3;
	}

	float weight{ 1.0f }; // Default value initialization with 1.0f
	float get_weight() {
		return weight;
	}
	void set_weight(float value) {
		weight = value;
	}
	virtual PackedFloat32Array get_weights() override {
		PackedFloat32Array weights{ weight, weight, weight };
		return weights;
	}

	virtual void setup_nodes(Variant character) override {
		// Node::get_node();
		body = Object::cast_to<CharacterBody3D>(character);
	}
	virtual void setup_for_animation(Ref<Animation> animation) override {
		root_track_pos = animation->find_track(NodePath(root_bone_name), Animation::TrackType::TYPE_POSITION_3D);
		root_track_quat = animation->find_track(NodePath(root_bone_name), Animation::TrackType::TYPE_ROTATION_3D);
		// root_track_scale = animation->find_track(NodePath(root_bone_name),Animation::TrackType::TYPE_SCALE_3D);
		// u::prints("Root Tracks for",root_track_pos,root_track_quat);
	}

	virtual PackedFloat32Array bake_animation_pose(Ref<Animation> animation, float time) override {
		auto pos = animation->position_track_interpolate(root_track_pos, time);
		auto prev_pos = animation->position_track_interpolate(root_track_pos, time - 0.1);
		Quaternion rotation = animation->rotation_track_interpolate(root_track_quat, time).normalized();

		Vector3 vel = rotation.xform_inv(pos - prev_pos) / 0.1;

		PackedFloat32Array result{};
		result.push_back(vel.x);
		result.push_back(vel.y);
		result.push_back(vel.z);
		return result;
	}

	virtual PackedFloat32Array broadphase_query_pose(Dictionary p_blackboard, float p_delta) override {
		auto vel = body->get_quaternion().xform_inv(body->get_velocity());
		PackedFloat32Array result{};
		result.push_back(vel.x);
		result.push_back(vel.y);
		result.push_back(vel.z);
		return result;
	}

	virtual float narrowphase_evaluate_cost(PackedFloat32Array to_convert) override {
		Vector3 data_vel = { to_convert[0], to_convert[1], to_convert[2] };
		return (body->get_velocity() - data_vel).length_squared();
	}

protected:
	static void _bind_methods();

	virtual void debug_pose_gizmo(Ref<RefCounted> p_gizmo, const PackedFloat32Array data, Transform3D tr = Transform3D{}) override {
#ifdef TOOLS_ENABLED
		Ref<EditorNode3DGizmo> gizmo = p_gizmo;
		if (data.size() == get_dimension()) {
			Vector3 vel = tr.xform(Vector3(data[0], data[1], data[2]));
			auto mat = gizmo->get_plugin()->get_material("white", gizmo);
			PackedVector3Array lines{ tr.origin, tr.origin + vel };
			gizmo->add_lines(lines, mat);
		}
#endif
	}
};

class BonePositionVelocityMotionFeature : public MotionFeature {
	GDCLASS(BonePositionVelocityMotionFeature, MotionFeature)
	Skeleton3D *skeleton = nullptr;
	PackedStringArray bone_names{};
	PackedInt32Array bones_id{};
	CharacterBody3D *the_char = nullptr;
	HashMap<uint32_t, PackedInt32Array> bone_tracks{};
	float last_time_queried = 0.0f;
	float weight_bone_pos{ 1.0f };
	float weight_bone_vel{ 1.0f };
public:
	NodePath to_skeleton{};
	void set_to_skeleton(NodePath path);
	NodePath get_to_skeleton();
	void set_bone_names(PackedStringArray value);
	PackedStringArray get_bone_names();
	virtual int get_dimension() override;
	virtual void setup_nodes(Variant character) override;
	virtual void setup_for_animation(Ref<Animation> animation) override;
	void set_skeleton_to_animation_timestamp(Ref<Animation> anim, float time);
	virtual PackedFloat32Array bake_animation_pose(Ref<Animation> animation, float time) override;
	PackedVector3Array last_known_positions{};
	PackedVector3Array last_known_velocities{};
	PackedFloat32Array last_known_result{};
	virtual PackedFloat32Array broadphase_query_pose(Dictionary blackboard, float delta) override;
	virtual float narrowphase_evaluate_cost(PackedFloat32Array to_convert) override;
	float get_weight_bone_pos() const {
		return weight_bone_pos;
	}
	void set_weight_bone_pos(float value) {
		weight_bone_pos = value;
	}
	float get_weight_bone_vel() const {
		return weight_bone_vel;
	}
	void set_weight_bone_vel(float value) {
		weight_bone_vel = value;
	}
	virtual PackedFloat32Array get_weights() override;

protected:
	static void _bind_methods();
	virtual void debug_pose_gizmo(Ref<RefCounted> gizmo, const PackedFloat32Array data, Transform3D tr = Transform3D{}) override;
};

class PredictionMotionFeature : public MotionFeature {
	GDCLASS(PredictionMotionFeature, MotionFeature)
public:
	Skeleton3D *skeleton{ nullptr };

	Skeleton3D *get_skeleton() {
		return skeleton;
	}

	void set_skeleton(Skeleton3D *value) {
		skeleton = value;
	}
	String root_bone_name{ "%GeneralSkeleton:Root" };

	String get_root_bone_name() const {
		return root_bone_name;
	}

	void set_root_bone_name(const String &value) {
		root_bone_name = value;
	}
	NodePath character_path;

	NodePath get_character_path() const {
		return character_path;
	}

	void set_character_path(const NodePath &value) {
		character_path = value;
	}
	float halflife_velocity{ 0.2f };

	float get_halflife_velocity() const {
		return halflife_velocity;
	}

	void set_halflife_velocity(float value) {
		halflife_velocity = value;
	}
	float halflife_angular_velocity{ 0.13f };

	float get_halflife_angular_velocity() const {
		return halflife_angular_velocity;
	}

	void set_halflife_angular_velocity(float value) {
		halflife_angular_velocity = value;
	}
	PackedFloat32Array past_time_dt;

	PackedFloat32Array get_past_time_dt() {
		return past_time_dt;
	}

	void set_past_time_dt(const PackedFloat32Array &value) {
		past_time_dt = value;
	}
	PackedFloat32Array future_time_dt;

	PackedFloat32Array get_future_time_dt() {
		return future_time_dt;
	}

	void set_future_time_dt(const PackedFloat32Array &value) {
		future_time_dt = value;
	}
	int past_count{ 4 };

	int get_past_count() const {
		return past_count;
	}

	void set_past_count(int value) {
		past_count = value;
	}
	float past_delta{ 0.7f / past_count };

	float get_past_delta() const {
		return past_delta;
	}

	void set_past_delta(float value) {
		past_delta = value;
	}
	int future_count{ 6 };

	int get_future_count() const {
		return future_count;
	}

	void set_future_count(int value) {
		future_count = value;
	}
	float future_delta{ 1.2f / future_count };

	float get_future_delta() const {
		return future_delta;
	}

	void set_future_delta(float value) {
		future_delta = value;
	}
	float weight_history_pos{ 1.0f };

	float get_weight_history_pos() const {
		return weight_history_pos;
	}

	void set_weight_history_pos(float value) {
		weight_history_pos = value;
	}
	float weight_prediction_pos{ 1.0f };

	float get_weight_prediction_pos() const {
		return weight_prediction_pos;
	}

	void set_weight_prediction_pos(float value) {
		weight_prediction_pos = value;
	}
	float weight_prediction_angle{ 1.0f };

	float get_weight_prediction_angle() const {
		return weight_prediction_angle;
	}

	void set_weight_prediction_angle(float value) {
		weight_prediction_angle = value;
	}

	virtual PackedFloat32Array get_weights() override;
	PredictionMotionFeature();

private:
	void create_default_dt();

public:
	virtual int get_dimension() override;
	CharacterBody3D *body = nullptr;
	virtual void setup_nodes(Variant character) override;

	int root_tracks[3] = { 0, 0, 0 };
	Vector3 start_pos, start_vel, end_pos, end_vel;
	Quaternion start_rot, end_rot, end_ang_vel;
	float start_time = 0.0f, end_time = 0.0f;
	virtual void setup_for_animation(Ref<Animation> animation) override;
	virtual PackedFloat32Array bake_animation_pose(Ref<Animation> animation, float time) override;
	virtual PackedFloat32Array broadphase_query_pose(Dictionary blackboard, float delta) override;
	virtual float narrowphase_evaluate_cost(PackedFloat32Array to_convert) override { return 0.0; }

protected:
	static void _bind_methods();
	virtual void debug_pose_gizmo(Ref<RefCounted> gizmo, const PackedFloat32Array data, Transform3D tr = Transform3D{}) override;
};

#endif // MOTION_FEATURES_H