/**************************************************************************/
/*  qbo_document.cpp                                                      */
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

#include "qbo_document.h"

#include "core/io/file_access.h"
#include "core/io/file_access_memory.h"
#include "modules/gltf/skin_tool.h"
#include "modules/gltf/structures/gltf_animation.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/mesh.h"
#include "scene/resources/surface_tool.h"

#define BVH_X_POSITION 1
#define BVH_Y_POSITION 2
#define BVH_Z_POSITION 3
#define BVH_X_ROTATION 4
#define BVH_Y_ROTATION 5
#define BVH_Z_ROTATION 6
#define BVH_W_ROTATION 7

// FIXME: Hardcoded to avoid editor dependency.
#define QBO_IMPORT_GENERATE_TANGENT_ARRAYS 8
#define QBO_IMPORT_USE_NAMED_SKIN_BINDS 16
#define QBO_IMPORT_DISCARD_MESHES_AND_MATERIALS 32
#define QBO_IMPORT_FORCE_DISABLE_MESH_COMPRESSION 64


Error QBODocument::_parse_motion(Ref<FileAccess> f, List<Skeleton3D *> &r_skeletons, AnimationPlayer **r_animation) {
	bool motion = false;
	int frame_count = -1;
	double frame_time = 0.03333333;
	HashMap<int, int> tracks;
	HashMap<int, int> parents;
	HashMap<int, int> children;
	HashMap<String, int> bone_names;
	Vector<String> frames;
	Vector<int> bones;
	Vector<Quaternion> orientations;
	Vector<Vector3> offsets;
	Vector<Vector<int>> channels;
	Ref<Animation> animation;
	Ref<AnimationLibrary> animation_library;
	String animation_library_name = f->get_path().get_basename().strip_edges();
	if (animation_library_name.contains(".")) {
		animation_library_name = animation_library_name.substr(0, animation_library_name.find("."));
	}
	animation_library_name = AnimationLibrary::validate_library_name(animation_library_name);
	if (r_animation != nullptr) {
		if (*r_animation == nullptr) {
			animation_library.instantiate();
			animation_library->set_name(animation_library_name);
			*r_animation = memnew(AnimationPlayer);
			(*r_animation)->add_animation_library(animation_library_name, animation_library);
		} else {
			List<StringName> libraries;
			(*r_animation)->get_animation_library_list(&libraries);
			for (int i = 0; i < libraries.size(); i++) {
				String library_name = libraries.get(i);
				if (library_name.is_empty()) {
					continue;
				}
				animation_library = (*r_animation)->get_animation_library(animation_library_name);
				if (animation_library.is_valid()) {
					animation_library_name = animation_library->get_name();
					break;
				}
			}
		}
	}

	int loops = 0;
	int blanks = 0;
	while (true) {
		String l = f->get_line().strip_edges();
		//print_verbose(l);
		if (++loops % 100 == 0 && OS::get_singleton()->has_feature("debug")) {
			print_verbose(String::num_int64(loops) + " BVH loops");
		}
		if (l.is_empty() && f->eof_reached()) {
			break;
		}
		if (l.is_empty()) {
			if (++blanks > 1) {
				break;
			}
			continue;
		}
		if (l.begins_with("#")) {
			continue;
		}

		if (motion) {
		if (l.begins_with("HIERARCHY")) {
				motion = false;
			} else if (l.begins_with("Frame Time: ")) {
				Vector<String> s = l.split(":");
				l = s[1].strip_edges();
				frame_time = l.to_float();
			} else if (l.begins_with("Frames: ")) {
				Vector<String> s = l.split(":");
				l = s[1].strip_edges();
				frame_count = l.to_int();
			} else {
				frames.append(l);
				if (frames.size() == frame_count) {
					motion = false;
				}
			}
			continue;
		}

		if (l.begins_with("HIERARCHY")) {
			l = l.substr(10);
			if (!l.is_empty()) {
				animation_library_name = l;
			}
			continue;
		} else if (l.begins_with("ROOT")) {
			String bone_name = "";
			int bone = -1;
			bones.clear();
			offsets.clear();
			orientations.clear();
			channels.clear();
			r_skeletons.push_back(memnew(Skeleton3D));
			l = l.substr(5);
			if (!l.is_empty()) {
				bone_name += l;
			} else {
				bone_name += "ROOT";
			}
			if (!bone_name.is_empty()) {
				if (bone_names.has(bone_name)) {
					bone_names[bone_name] += 1;
					bone_name += String::num_int64(bone_names[bone_name]);
				} else {
					bone_names[bone_name] = 1;
				}
				r_skeletons.back()->get()->set_name(animation_library_name);
				bone = r_skeletons.back()->get()->add_bone(bone_name);
			}
			if (bone >= 0) {
				bones.append(bone);
				orientations.append(Quaternion());
				offsets.append(Vector3());
				channels.append(Vector<int>({ bones[bones.size() - 1] }));
			}
		} else if (l.begins_with("MOTION")) {
			motion = true;
			if (animation_library.is_valid() && animation.is_valid() && r_animation != nullptr && frames.size() == frame_count) {
				if (!channels.is_empty() && !r_skeletons.is_empty()) {
					tracks.clear();
					for (int i = 0; i < channels.size(); i++) {
						if (channels[i].size() < 2) {
							continue;
						}
						tracks[channels[i][0]] = animation->add_track(Animation::TrackType::TYPE_POSITION_3D);
						tracks[r_skeletons.back()->get()->get_bone_count() + channels[i][0]] = animation->add_track(Animation::TrackType::TYPE_ROTATION_3D);
					}
					for (int i = 0; i < frame_count; i++) {
						int bone_index = 0;
						int channel_index = 0;
						String frame = frames[i];
						Vector<String> s;
						if (frame.contains(" ")) {
							s = frame.split(" ");
						} else {
							s = frame.split("\t");
						}
						for (int j = 0; j < s.size(); j++) {
							channel_index++;
							if (channel_index >= channels[bone_index].size() || channels[bone_index].size() < 2) {
								do {
									bone_index++;
									if (bone_index >= channels.size()) {
							break;
						}
								} while (channels[bone_index].size() < 2);
								channel_index = 1;
								if (bone_index >= channels.size()) {
									break;
								}
							}
							if (bone_index < 0 || bone_index >= channels.size()) {
								break;
							}
							Vector3 position;
							Quaternion rotation;
							String bone_name = r_skeletons.back()->get()->get_bone_name(channels[bone_index][0]);
							int position_track = tracks[channels[bone_index][0]];
							int rotation_track = tracks[r_skeletons.back()->get()->get_bone_count() + channels[bone_index][0]];
							int insertion = -1;
							switch (channels[bone_index][channel_index]) {
								case BVH_X_POSITION:
								case BVH_Y_POSITION:
								case BVH_Z_POSITION:
									if (channel_index + 2 < channels[bone_index].size()) {
										position = Vector3();
										for (int k = j; k < j + 3 && k < s.size(); k++) {
											switch (channels[bone_index][channel_index + (k - j)]) {
												case BVH_X_POSITION:
													position.x = s[k].strip_edges().to_float();
													break;
												case BVH_Y_POSITION:
													position.y = s[k].strip_edges().to_float();
													break;
												case BVH_Z_POSITION:
													position.z = s[k].strip_edges().to_float();
													break;
											}
										}
										animation->track_set_imported(position_track, true);
										animation->track_set_path(position_track, "" + r_skeletons.back()->get()->get_name() + ":" + bone_name);
										//insertion = animation->position_track_insert_key(position_track, frame_time * static_cast<double>(i), position);
										j += 2;
										channel_index += 2;
										//print_verbose(position);
									} else {
										print_verbose("poselse" + String::num_int64(channel_index) + " " + String::num_int64(channels[bone_index].size()));
									}
									break;
								case BVH_X_ROTATION:
								case BVH_Y_ROTATION:
								case BVH_Z_ROTATION:
								case BVH_W_ROTATION:
									if (channel_index + 3 < channels[bone_index].size()) {
										rotation = Quaternion();
										for (int k = j; k < j + 4 && k < s.size(); k++) {
											switch (channels[bone_index][channel_index + (k - j)]) {
												case BVH_X_ROTATION:
													rotation.x = s[k].strip_edges().to_float();
													break;
												case BVH_Y_ROTATION:
													rotation.y = s[k].strip_edges().to_float();
													break;
												case BVH_Z_ROTATION:
													rotation.z = s[k].strip_edges().to_float();
													break;
												case BVH_W_ROTATION:
													rotation.w = s[k].strip_edges().to_float();
													break;
											}
										}
										animation->track_set_imported(rotation_track, true);
										animation->track_set_path(rotation_track, "" + r_skeletons.back()->get()->get_name() + ":" + bone_name);
										insertion = animation->rotation_track_insert_key(rotation_track, frame_time * static_cast<double>(i), rotation);
										j += 3;
										channel_index += 3;
										//print_verbose(rotation);
									} else {
										print_verbose("rotelse" + String::num_int64(channel_index) + " " + String::num_int64(channels[bone_index].size()));
									}
									break;
								default:
									print_verbose(String::num_int64(position_track) + " " + String::num_int64(rotation_track) + bone_name + " @ " + String::num_int64(channel_index));
									break;
							}
							if (insertion < 0 && OS::get_singleton()->has_feature("debug")) {
								//print_verbose(String::num_int64(insertion) + " BVH track insertion");
							}
						}
					}
				}
				if (!animation->get_name().is_empty()) {
					animation_library_name = animation->get_name();
				}
				animation->set_step(frame_time);
				animation_library->add_animation(animation_library_name, animation);
				if (r_animation != nullptr) {
					(*r_animation)->set_assigned_animation(animation_library->get_name() + "/" + animation->get_name());
				}
			}
			animation.instantiate();
			frame_count = -1;
			frames.clear();
			l = l.substr(7);
			if (!l.is_empty()) {
				animation->set_name(l.strip_edges());
			} else {
				animation->set_name("MOTION");
			}
		} else if (l.begins_with("End ")) {
			ERR_FAIL_COND_V(r_skeletons.is_empty(), ERR_FILE_CORRUPT);
			l = l.substr(4);
			if (!l.is_empty()) {
				if (bone_names.has(l)) {
					bone_names[l] += 1;
					l += String::num_int64(bone_names[l]);
				} else {
					bone_names[l] = 1;
				}
				bones.append(r_skeletons.back()->get()->add_bone(l));
				orientations.append(Quaternion());
				offsets.append(Vector3());
				channels.append(Vector<int>({ bones[bones.size() - 1] }));
				if (bones.size() > 1) {
					r_skeletons.back()->get()->set_bone_parent(bones[bones.size() - 1], bones[bones.size() - 2]);
					parents[bones[bones.size() - 1]] = bones[bones.size() - 2];
				}
			}
		} else {
			Vector<String> s;
			if (l.contains(" ")) {
				s = l.split(" ");
			} else {
				s = l.split("\t");
			}
			if (s.size() > 1) {
				if (s[0].casecmp_to("OFFSET") == 0) {
					ERR_FAIL_COND_V(s.size() < 4, ERR_FILE_CORRUPT);
					Vector3 offset;
					offset.x = s[1].to_float();
					offset.y = s[2].to_float();
					offset.z = s[3].to_float();
					// Convert OFFSET for axis convention (QBO/BVH to Godot)
					// BVH uses Y-up, Z-forward, X-right
					// Godot uses Y-up, -Z-forward, X-right
					// Swap X and Z components for coordinate system conversion
					offset = Vector3(offset.z, offset.y, offset.x);
					if (offsets.is_empty()) {
						offsets.append(offset);
					} else {
						offsets.set(offsets.size() - 1, offset);
					}
				} else if (s[0].casecmp_to("ORIENT") == 0) {
					ERR_FAIL_COND_V(s.size() < 5, ERR_FILE_CORRUPT);
					Quaternion orientation;
					orientation.x = s[1].to_float();
					orientation.y = s[2].to_float();
					orientation.z = s[3].to_float();
					orientation.w = s[4].to_float();
					if (!orientation.is_normalized()) {
						print_verbose("UNNORMALIZED ORIENTATION!!!")
					}
					if (orientations.is_empty()) {
						orientations.append(orientation);
					} else {
						orientations.set(orientations.size() - 1, orientation);
					}
				} else if (s[0].casecmp_to("CHANNELS") == 0) {
					int channel_count = s[1].to_int();
					ERR_FAIL_COND_V(s.size() < channel_count + 2 || bones.is_empty(), ERR_FILE_CORRUPT);
					Vector<int> channel;
					channel.append(bones[bones.size() - 1]);
					for (int i = 0; i < channel_count; i++) {
						String channel_name = s[i + 2].strip_edges();
						//print_verbose(channel_name);
						if (channel_name.casecmp_to("Xposition") == 0) {
							channel.append(BVH_X_POSITION);
						} else if (channel_name.casecmp_to("Yposition") == 0) {
							channel.append(BVH_Y_POSITION);
						} else if (channel_name.casecmp_to("Zposition") == 0) {
							channel.append(BVH_Z_POSITION);
						} else if (channel_name.casecmp_to("Xrotation") == 0) {
							channel.append(BVH_X_ROTATION);
						} else if (channel_name.casecmp_to("Yrotation") == 0) {
							channel.append(BVH_Y_ROTATION);
						} else if (channel_name.casecmp_to("Zrotation") == 0) {
							channel.append(BVH_Z_ROTATION);
						} else if (channel_name.casecmp_to("Wrotation") == 0) {
							channel.append(BVH_W_ROTATION);
						} else {
							channel_name.clear();
						}
						ERR_FAIL_COND_V(channel_name.is_empty(), ERR_FILE_CORRUPT);
					}
					ERR_FAIL_COND_V(channel.size() < 2, ERR_FILE_CORRUPT);
					if (channels.is_empty()) {
						channels.append(channel);
					} else if (channels[channels.size() - 1].size() < 2) {
						channels.remove_at(channels.size() - 1);
						channels.append(channel);
					}
				} else if (s[0].casecmp_to("JOINT") == 0) {
					ERR_FAIL_COND_V(r_skeletons.is_empty() || bones.is_empty(), ERR_FILE_CORRUPT);
					int parent = bones[bones.size() - 1];
					String bone_name = s[1];
					if (bone_names.has(bone_name)) {
						bone_names[bone_name] += 1;
						if (bone_name.ends_with("_")) {
							bone_name += "_";
						}
						bone_name += String::num_int64(bone_names[bone_name]);
					} else {
						bone_names[bone_name] = 1;
					}
					bones.append(r_skeletons.back()->get()->add_bone(bone_name));
					orientations.append(Quaternion());
					offsets.append(Vector3());
					channels.append(Vector<int>({ bones[bones.size() - 1] }));
					r_skeletons.back()->get()->set_bone_parent(bones[bones.size() - 1], parent);
					parents[bones[bones.size() - 1]] = parent;
				}
			} else {
				if (l.casecmp_to("{") == 0) {
					int child = 0;
					if (!bones.is_empty() && parents.has(bones[bones.size() - 1])) {
						if (children.has(parents[bones[bones.size() - 1]])) {
							children[parents[bones[bones.size() - 1]]] += 1;
						} else {
							children[parents[bones[bones.size() - 1]]] = 1;
						}
						child += children[parents[bones[bones.size() - 1]]];
					}
					print_verbose(r_skeletons.back()->get()->get_bone_name(parents[bones[bones.size() - 1]]) + " -> " + String::num_int64(child));
				} else if (l.casecmp_to("}") == 0) {
					ERR_FAIL_COND_V(r_skeletons.is_empty() || bones.is_empty() || offsets.is_empty() || orientations.is_empty() || channels.is_empty(), ERR_FILE_CORRUPT);
					int bone = bones[bones.size() - 1];
					Transform3D rest;
					Vector3 scale = Vector3(1.0, 1.0, 1.0);
					Vector3 offset = offsets[offsets.size() - 1];
					Quaternion orientation = orientations[orientations.size() - 1];
					bones.remove_at(bones.size() - 1);
					offsets.remove_at(offsets.size() - 1);
					orientations.remove_at(orientations.size() - 1);
					rest.basis.set_quaternion_scale(orientation, scale);

					// For root bone (no parent), the OFFSET is the skeleton's position
					// Apply it to the skeleton's transform and set bone rest origin to zero
					int bone_parent = r_skeletons.back()->get()->get_bone_parent(bone);
					if (bone_parent < 0) {
						// This is a root bone (no parent)
						r_skeletons.back()->get()->set_transform(Transform3D(Basis(), offset));
						rest.origin = Vector3();
					} else {
						rest.origin = offset;
					}

					if (bone < r_skeletons.back()->get()->get_bone_count()) {
						print_verbose(r_skeletons.back()->get()->get_bone_name(bone) + " @ " + String::num_int64(bone) + " = " + String(offset));
						r_skeletons.back()->get()->set_bone_rest(bone, rest);
						r_skeletons.back()->get()->set_bone_pose_rotation(bone, orientation);
						r_skeletons.back()->get()->set_bone_pose_scale(bone, scale);
					} else {
						print_verbose(r_skeletons.back()->get()->get_bone_name(bone) + " @ " + String::num_int64(bone));
					}
					print_verbose(String::num_int64(bones.size()));
				}
			}
		}
	}

	//print_verbose(String::num_int64(frames.size())+" "+String::num_int64(frame_count));
	if (animation_library.is_valid() && animation.is_valid() && r_animation != nullptr && frames.size() == frame_count) {
		if (!channels.is_empty() && !r_skeletons.is_empty()) {
			tracks.clear();
			for (int i = 0; i < channels.size(); i++) {
				//print_verbose(channels[i]);
				if (channels[i].size() < 2) {
				continue;
				}
				tracks[channels[i][0]] = animation->add_track(Animation::TrackType::TYPE_POSITION_3D);
				tracks[r_skeletons.back()->get()->get_bone_count() + channels[i][0]] = animation->add_track(Animation::TrackType::TYPE_ROTATION_3D);
			}
			for (int i = 0; i < frame_count; i++) {
				int bone_index = 0;
				int channel_index = 0;
				String frame = frames[i];
				Vector<String> s;
				if (frame.contains(" ")) {
					s = frame.split(" ");
				} else {
					s = frame.split("\t");
				}
				for (int j = 0; j < s.size(); j++) {
					channel_index++;
					if (channel_index >= channels[bone_index].size() || channels[bone_index].size() < 2) {
						do {
							bone_index++;
							if (bone_index >= channels.size()) {
								break;
							}
						} while (channels[bone_index].size() < 2);
						channel_index = 1;
						if (bone_index >= channels.size()) {
							break;
						}
					}
					if (bone_index < 0 || bone_index >= channels.size()) {
						break;
					}
					Vector3 position;
					Quaternion rotation;
					String bone_name = r_skeletons.back()->get()->get_bone_name(channels[bone_index][0]);
					int position_track = tracks[channels[bone_index][0]];
					int rotation_track = tracks[r_skeletons.back()->get()->get_bone_count() + channels[bone_index][0]];
					int insertion = -1;
					switch (channels[bone_index][channel_index + 1]) {
						case BVH_X_POSITION:
						case BVH_Y_POSITION:
						case BVH_Z_POSITION:
							if (channel_index + 2 < channels[bone_index].size()) {
								position = Vector3();
								for (int k = j; k < j + 3 && k < s.size(); k++) {
									switch (channels[bone_index][channel_index + (k - j)]) {
										case BVH_X_POSITION:
											position.x = s[k].strip_edges().to_float();
											break;
										case BVH_Y_POSITION:
											position.y = s[k].strip_edges().to_float();
											break;
										case BVH_Z_POSITION:
											position.z = s[k].strip_edges().to_float();
											break;
									}
								}
								animation->track_set_imported(position_track, true);
								animation->track_set_path(position_track, "" + r_skeletons.back()->get()->get_name() + ":" + bone_name);
								//insertion = animation->position_track_insert_key(position_track, frame_time * static_cast<double>(i), position);
								j += 2;
								channel_index += 2;
								//print_verbose(position);
			} else {
								print_verbose("poselse" + String::num_int64(channel_index) + " " + String::num_int64(channels[bone_index].size()));
							}
							break;
						case BVH_X_ROTATION:
						case BVH_Y_ROTATION:
						case BVH_Z_ROTATION:
						case BVH_W_ROTATION:
							if (channel_index + 3 < channels[bone_index].size()) {
								rotation = Quaternion();
								for (int k = j; k < j + 4 && k < s.size(); k++) {
									switch (channels[bone_index][channel_index + (k - j)]) {
										case BVH_X_ROTATION:
											rotation.x = s[k].strip_edges().to_float();
											break;
										case BVH_Y_ROTATION:
											rotation.y = s[k].strip_edges().to_float();
											break;
										case BVH_Z_ROTATION:
											rotation.z = s[k].strip_edges().to_float();
											break;
										case BVH_W_ROTATION:
											rotation.w = s[k].strip_edges().to_float();
											break;
									}
								}
								animation->track_set_imported(rotation_track, true);
								animation->track_set_path(rotation_track, "" + r_skeletons.back()->get()->get_name() + ":" + bone_name);
								insertion = animation->rotation_track_insert_key(rotation_track, frame_time * static_cast<double>(i), rotation);
								j += 3;
								channel_index += 3;
								//print_verbose(rotation);
				} else {
								print_verbose("rotelse" + String::num_int64(channel_index) + " " + String::num_int64(channels[bone_index].size()));
							}
							break;
						default:
							print_verbose(String::num_int64(position_track) + " " + String::num_int64(rotation_track) + bone_name + " @ " + String::num_int64(channel_index));
							break;
					}
					if (insertion < 0 && OS::get_singleton()->has_feature("debug")) {
						//rint_verbose(String::num_int64(insertion) + " BVH track insertion");
					}
				}
			}
		}
		if (!animation->get_name().is_empty()) {
			animation_library_name = animation->get_name();
		}
		animation->set_step(frame_time);
		animation_library->add_animation(animation_library_name, animation);
		(*r_animation)->set_assigned_animation(animation_library->get_name() + "/" + animation->get_name());
		List<StringName> animations;
		animation_library->get_animation_list(&animations);
		for (int i = 0; i < animations.size(); i++) {
			Vector<int> duds;
			animation = animation_library->get_animation(animations.get(i));
			for (int j = 0; j < animation->get_track_count(); j++) {
				if (animation->track_get_path(j).is_empty()) {
					duds.append(j);
				}
			}
			for (int j = 0; j < duds.size(); j++) {
				for (int k = j + 1; k < duds.size(); k++) {
					duds.set(k, duds[k] - 1);
				}
				animation->remove_track(duds[j]);
			}
		}
	}

	return OK;
}

Error QBODocument::_parse_material_library(const String &p_path, HashMap<String, Ref<StandardMaterial3D>> &material_map, List<String> *r_missing_deps) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, vformat("Couldn't open MTL file '%s', it may not exist or not be readable.", p_path));

	Ref<StandardMaterial3D> current;
	String current_name;
	String base_path = p_path.get_base_dir();
	while (true) {
		String l = f->get_line().strip_edges();

		if (l.begins_with("newmtl ")) {
			//vertex

			current_name = l.replace("newmtl", "").strip_edges();
			current.instantiate();
			current->set_name(current_name);
			material_map[current_name] = current;
		} else if (l.begins_with("Ka ")) {
			//uv
			WARN_PRINT("OBJ: Ambient light for material '" + current_name + "' is ignored in PBR");

		} else if (l.begins_with("Kd ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_INVALID_DATA);
			Color c = current->get_albedo();
			c.r = v[1].to_float();
			c.g = v[2].to_float();
			c.b = v[3].to_float();
			current->set_albedo(c);
		} else if (l.begins_with("Ks ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_INVALID_DATA);
			float r = v[1].to_float();
			float g = v[2].to_float();
			float b = v[3].to_float();
			float metalness = MAX(r, MAX(g, b));
			current->set_metallic(metalness);
		} else if (l.begins_with("Ns ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() != 2, ERR_INVALID_DATA);
			float s = v[1].to_float();
			current->set_metallic((1000.0 - s) / 1000.0);
		} else if (l.begins_with("d ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() != 2, ERR_INVALID_DATA);
			float d = v[1].to_float();
			Color c = current->get_albedo();
			c.a = d;
			current->set_albedo(c);
			if (c.a < 0.99) {
				current->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
			}
		} else if (l.begins_with("Tr ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() != 2, ERR_INVALID_DATA);
			float d = v[1].to_float();
			Color c = current->get_albedo();
			c.a = 1.0 - d;
			current->set_albedo(c);
			if (c.a < 0.99) {
				current->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
			}

		} else if (l.begins_with("map_Ka ")) {
			//uv
			WARN_PRINT("OBJ: Ambient light texture for material '" + current_name + "' is ignored in PBR");

		} else if (l.begins_with("map_Kd ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);

			String p = l.replace("map_Kd", "").replace_char('\\', '/').strip_edges();
			String path;
			if (p.is_absolute_path()) {
				path = p;
		} else {
				path = base_path.path_join(p);
			}

			Ref<Texture2D> texture = ResourceLoader::load(path);

			if (texture.is_valid()) {
				current->set_texture(StandardMaterial3D::TEXTURE_ALBEDO, texture);
			} else if (r_missing_deps) {
				r_missing_deps->push_back(path);
			}

		} else if (l.begins_with("map_Ks ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);

			String p = l.replace("map_Ks", "").replace_char('\\', '/').strip_edges();
			String path;
			if (p.is_absolute_path()) {
				path = p;
					} else {
				path = base_path.path_join(p);
			}

			Ref<Texture2D> texture = ResourceLoader::load(path);

			if (texture.is_valid()) {
				current->set_texture(StandardMaterial3D::TEXTURE_METALLIC, texture);
			} else if (r_missing_deps) {
				r_missing_deps->push_back(path);
			}

		} else if (l.begins_with("map_Ns ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);

			String p = l.replace("map_Ns", "").replace_char('\\', '/').strip_edges();
			String path;
			if (p.is_absolute_path()) {
				path = p;
			} else {
				path = base_path.path_join(p);
			}

			Ref<Texture2D> texture = ResourceLoader::load(path);

			if (texture.is_valid()) {
				current->set_texture(StandardMaterial3D::TEXTURE_ROUGHNESS, texture);
			} else if (r_missing_deps) {
				r_missing_deps->push_back(path);
			}
		} else if (l.begins_with("map_bump ")) {
			//normal
			ERR_FAIL_COND_V(current.is_null(), ERR_FILE_CORRUPT);

			String p = l.replace("map_bump", "").replace_char('\\', '/').strip_edges();
			String path = base_path.path_join(p);

			Ref<Texture2D> texture = ResourceLoader::load(path);

			if (texture.is_valid()) {
				current->set_feature(StandardMaterial3D::FEATURE_NORMAL_MAPPING, true);
				current->set_texture(StandardMaterial3D::TEXTURE_NORMAL, texture);
			} else if (r_missing_deps) {
				r_missing_deps->push_back(path);
			}
		} else if (f->eof_reached()) {
						break;
		}
	}

	return OK;
}

Error QBODocument::_parse_hierarchy_to_gltf(Ref<FileAccess> f, Ref<GLTFState> p_state, HashMap<String, GLTFNodeIndex> &r_bone_name_to_node, Vector<BoneData> &r_bone_data, Vector<GLTFNodeIndex> &r_root_nodes) {
	bool motion = false;
	int frame_count = -1;
	double frame_time = 0.03333333;
	HashMap<String, int> bone_names;
	Vector<String> frames;
	Vector<int> bone_indices; // Index into r_bone_data
	Vector<Quaternion> orientations;
	Vector<Vector3> offsets;
	Vector<Vector<int>> channels;
	Ref<GLTFAnimation> gltf_animation;
	String animation_name;

	int loops = 0;
	int blanks = 0;
	while (true) {
		String l = f->get_line().strip_edges();
		if (++loops % 100 == 0 && OS::get_singleton()->has_feature("debug")) {
			print_verbose(String::num_int64(loops) + " QBO loops");
		}
		if (l.is_empty() && f->eof_reached()) {
			break;
		}
		if (l.is_empty()) {
			if (++blanks > 1) {
				break;
			}
			continue;
		}
		if (l.begins_with("#")) {
			continue;
		}

		if (l.begins_with("HIERARCHY")) {
			// HIERARCHY section - continue parsing
			continue;
		} else if (l.begins_with("ROOT")) {
			String bone_name = "";
			bone_indices.clear();
			offsets.clear();
			orientations.clear();
			channels.clear();
			l = l.substr(5);
			if (!l.is_empty()) {
				bone_name += l;
		} else {
				bone_name += "ROOT";
			}
			if (!bone_name.is_empty()) {
				if (bone_names.has(bone_name)) {
					bone_names[bone_name] += 1;
					bone_name += String::num_int64(bone_names[bone_name]);
				} else {
					bone_names[bone_name] = 1;
				}
				// Create GLTFNode for root bone
				Ref<GLTFNode> node;
				node.instantiate();
				node->set_original_name(bone_name);
				node->set_name(_gen_unique_name_static(p_state->unique_names, bone_name));
				node->joint = true;
				node->transform = Transform3D(); // Will be set when we parse OFFSET/ORIENT
				
				GLTFNodeIndex node_index = p_state->append_gltf_node(node, nullptr, -1);
				r_root_nodes.push_back(node_index);
				
				BoneData bone_data;
				bone_data.name = bone_name;
				bone_data.gltf_node_index = node_index;
				bone_data.parent_bone_index = -1;
				bone_data.offset = Vector3();
				bone_data.orientation = Quaternion();
				bone_data.rest_transform = Transform3D();
				r_bone_data.push_back(bone_data);
				r_bone_name_to_node[bone_name] = node_index;
				bone_indices.push_back(r_bone_data.size() - 1);
				orientations.append(Quaternion());
				offsets.append(Vector3());
				channels.append(Vector<int>({ bone_indices[bone_indices.size() - 1] }));
			}
		} else if (l.begins_with("MOTION")) {
			motion = true;
			// Create GLTFAnimation directly (not AnimationPlayer)
			gltf_animation = Ref<GLTFAnimation>();
			gltf_animation.instantiate();
			frame_count = -1;
			frames.clear();
			print_line(vformat("QBO: Found MOTION section, creating GLTFAnimation"));
			l = l.substr(7);
			if (!l.is_empty()) {
				animation_name = l.strip_edges();
			} else {
				animation_name = "MOTION";
			}
			gltf_animation->set_original_name(animation_name);
			// Generate unique animation name
			String u_name = animation_name;
			int index = 1;
			while (p_state->unique_animation_names.has(u_name)) {
				u_name = animation_name + itos(index);
				index++;
			}
			p_state->unique_animation_names.insert(u_name);
			gltf_animation->set_name(u_name);
		} else if (motion && l.begins_with("Frame Time: ")) {
			Vector<String> s = l.split(":");
			if (s.size() >= 2) {
				frame_time = s[1].strip_edges().to_float();
				print_line(vformat("QBO: Frame Time: %f", frame_time));
			}
		} else if (motion && l.begins_with("Frames: ")) {
			Vector<String> s = l.split(":");
			if (s.size() >= 2) {
				frame_count = s[1].strip_edges().to_int();
				print_line(vformat("QBO: Frames: %d", frame_count));
			}
		} else if (motion && frame_count > 0 && !l.is_empty() && !l.begins_with("#")) {
			// Parse frame data
			frames.append(l);
			if (frames.size() == frame_count && gltf_animation.is_valid()) {
				print_line(vformat("QBO: Processing animation '%s' with %d frames, %d channels (r_bone_data has %d entries).", gltf_animation->get_name(), frame_count, channels.size(), r_bone_data.size()));
				// Create GLTFAnimation NodeTracks for each bone
				// Map from bone_data_idx to GLTFNodeIndex for tracking
				HashMap<int, GLTFNodeIndex> bone_data_to_node;
				HashMap<GLTFNodeIndex, GLTFAnimation::NodeTrack> node_tracks;
				
				// First pass: identify which bones have animation channels
				// Iterate over channels (not bone_indices) since channels contains the bone_data_idx
				for (int channel_idx = 0; channel_idx < channels.size(); channel_idx++) {
					if (channels[channel_idx].size() < 2) {
						continue;
					}
					int bone_data_idx = channels[channel_idx][0];
					if (bone_data_idx < 0 || bone_data_idx >= r_bone_data.size()) {
						continue;
					}
					GLTFNodeIndex node_index = r_bone_data[bone_data_idx].gltf_node_index;
					bone_data_to_node[bone_data_idx] = node_index;
					
					// Initialize NodeTrack for this node
					if (!node_tracks.has(node_index)) {
						node_tracks[node_index] = GLTFAnimation::NodeTrack();
					}
				}
				
				// Second pass: parse frames and populate tracks
				for (int frame_i = 0; frame_i < frame_count; frame_i++) {
					String frame = frames[frame_i];
					Vector<String> s;
					if (frame.contains(" ")) {
						s = frame.split(" ");
					} else {
						s = frame.split("\t");
					}
					
					int channel_idx = 0;
					int data_idx = 0;
					
					while (data_idx < s.size() && channel_idx < channels.size()) {
						if (channels[channel_idx].size() < 2) {
							channel_idx++;
							continue;
						}
						
						int bone_data_idx = channels[channel_idx][0];
						if (bone_data_idx < 0 || bone_data_idx >= r_bone_data.size() || !bone_data_to_node.has(bone_data_idx)) {
							channel_idx++;
							continue;
						}
						
						GLTFNodeIndex node_index = bone_data_to_node[bone_data_idx];
						GLTFAnimation::NodeTrack &track = node_tracks[node_index];
						
						Vector3 position;
						Quaternion rotation;
						bool position_set = false;
						bool rotation_set = false;
						
						// Parse channels for this bone
						for (int ch = 1; ch < channels[channel_idx].size() && data_idx < s.size(); ch++) {
							int channel_type = channels[channel_idx][ch];
							float value = s[data_idx].strip_edges().to_float();
							
							switch (channel_type) {
								case BVH_X_POSITION:
									position.x = value;
									position_set = true;
									break;
								case BVH_Y_POSITION:
									position.y = value;
									position_set = true;
									break;
								case BVH_Z_POSITION:
									position.z = value;
									position_set = true;
									break;
								case BVH_X_ROTATION:
									rotation.x = value;
									rotation_set = true;
									break;
								case BVH_Y_ROTATION:
									rotation.y = value;
									rotation_set = true;
									break;
								case BVH_Z_ROTATION:
									rotation.z = value;
									rotation_set = true;
									break;
								case BVH_W_ROTATION:
									rotation.w = value;
									rotation_set = true;
									break;
							}
							data_idx++;
						}
						
						// Convert position for axis convention (QBO/BVH to Godot)
						// BVH uses Y-up, Z-forward, X-right
						// Godot uses Y-up, -Z-forward, X-right
						// Swap X and Z components for coordinate system conversion
						if (position_set) {
							position = Vector3(position.z, position.y, position.x);
						}
						
						// Convert rotation for axis convention (same as ORIENT in skeleton)
						// Convert quaternion for axis convention using Euler order conversion
						// BVH/QBO may use YZX Euler order, Godot uses YXZ
						// Convert to Euler with YZX order, swap X and Z components, then convert back with YXZ
						if (rotation_set) {
							if (!rotation.is_normalized()) {
								rotation.normalize();
							}
							Basis basis = Basis(rotation);
							Vector3 euler_yzx = basis.get_euler(EulerOrder::YZX);
							// Swap X and Z Euler angles to match coordinate system conversion
							Vector3 euler_swapped = Vector3(euler_yzx.z, euler_yzx.y, euler_yzx.x);
							// Convert back to quaternion using Godot's YXZ order
							Basis converted_basis = Basis::from_euler(euler_swapped, EulerOrder::YXZ);
							rotation = converted_basis.get_rotation_quaternion();
						}
						
						// Add to GLTFAnimation tracks
						double time = frame_time * static_cast<double>(frame_i);
						if (position_set) {
							track.position_track.times.push_back(time);
							track.position_track.values.push_back(position);
						}
						if (rotation_set) {
							track.rotation_track.times.push_back(time);
							track.rotation_track.values.push_back(rotation);
						}
						
						channel_idx++;
					}
				}
				
				// Set interpolation and add tracks to GLTFAnimation
				int tracks_added = 0;
				for (KeyValue<GLTFNodeIndex, GLTFAnimation::NodeTrack> &kv : node_tracks) {
					GLTFAnimation::NodeTrack &track = kv.value;
					if (!track.position_track.times.is_empty()) {
						track.position_track.interpolation = GLTFAnimation::INTERP_LINEAR;
					}
					if (!track.rotation_track.times.is_empty()) {
						track.rotation_track.interpolation = GLTFAnimation::INTERP_LINEAR;
					}
					// Only add track if it has data
					if (!track.position_track.times.is_empty() || !track.rotation_track.times.is_empty() || !track.scale_track.times.is_empty()) {
						gltf_animation->get_node_tracks()[kv.key] = track;
						tracks_added++;
					}
				}
				
				// Add GLTFAnimation to GLTFState
				print_line(vformat("QBO: Animation '%s' has %d tracks added, is_empty_of_tracks=%s", gltf_animation->get_name(), tracks_added, gltf_animation->is_empty_of_tracks() ? "true" : "false"));
				if (!gltf_animation->is_empty_of_tracks()) {
					p_state->animations.push_back(gltf_animation);
					print_line(vformat("QBO: Added animation '%s' with %d tracks to GLTFState (total animations: %d).", gltf_animation->get_name(), tracks_added, p_state->animations.size()));
			} else {
					WARN_PRINT(vformat("QBO: Animation '%s' has no tracks (node_tracks had %d entries, but none had data) and will not be imported.", gltf_animation->get_name(), node_tracks.size()));
				}
				
				// Animation parsing complete
				motion = false;
			}
		} else if (l.begins_with("End ")) {
			ERR_FAIL_COND_V(bone_indices.is_empty(), ERR_FILE_CORRUPT);
			l = l.substr(4);
			if (!l.is_empty()) {
				String bone_name = l;
				if (bone_names.has(bone_name)) {
					bone_names[bone_name] += 1;
					bone_name += String::num_int64(bone_names[bone_name]);
				} else {
					bone_names[bone_name] = 1;
				}
				// Create GLTFNode for end site
				Ref<GLTFNode> node;
				node.instantiate();
				node->set_original_name(bone_name);
				node->set_name(_gen_unique_name_static(p_state->unique_names, bone_name));
				node->joint = true;
				
				int parent_bone_idx = bone_indices[bone_indices.size() - 1];
				GLTFNodeIndex parent_node_index = r_bone_data[parent_bone_idx].gltf_node_index;
				GLTFNodeIndex node_index = p_state->append_gltf_node(node, nullptr, parent_node_index);
				
				BoneData bone_data;
				bone_data.name = bone_name;
				bone_data.gltf_node_index = node_index;
				bone_data.parent_bone_index = parent_bone_idx;
				bone_data.offset = Vector3();
				bone_data.orientation = Quaternion();
				bone_data.rest_transform = Transform3D();
				r_bone_data.push_back(bone_data);
				r_bone_name_to_node[bone_name] = node_index;
				bone_indices.push_back(r_bone_data.size() - 1);
				orientations.append(Quaternion());
				offsets.append(Vector3());
				channels.append(Vector<int>({ bone_indices[bone_indices.size() - 1] }));
			}
		} else {
			Vector<String> s;
			if (l.contains(" ")) {
				s = l.split(" ");
			} else {
				s = l.split("\t");
			}
			if (s.size() > 1) {
				if (s[0].casecmp_to("OFFSET") == 0) {
					ERR_FAIL_COND_V(s.size() < 4, ERR_FILE_CORRUPT);
					Vector3 offset;
					offset.x = s[1].to_float();
					offset.y = s[2].to_float();
					offset.z = s[3].to_float();
					// Convert OFFSET for axis convention (QBO/BVH to Godot)
					offset = Vector3(offset.z, offset.y, offset.x);
					if (offsets.is_empty()) {
						offsets.append(offset);
					} else {
						offsets.set(offsets.size() - 1, offset);
					}
				} else if (s[0].casecmp_to("ORIENT") == 0) {
					ERR_FAIL_COND_V(s.size() < 5, ERR_FILE_CORRUPT);
					Quaternion orientation;
					orientation.x = s[1].to_float();
					orientation.y = s[2].to_float();
					orientation.z = s[3].to_float();
					orientation.w = s[4].to_float();
					if (!orientation.is_normalized()) {
						print_verbose("UNNORMALIZED ORIENTATION!!!")
					}
					// Convert quaternion for axis convention using Euler order conversion
					// BVH/QBO may use YZX Euler order, Godot uses YXZ
					// Convert to Euler with YZX order, swap X and Z components, then convert back with YXZ
					Basis basis = Basis(orientation);
					Vector3 euler_yzx = basis.get_euler(EulerOrder::YZX);
					// Swap X and Z Euler angles to match coordinate system conversion
					Vector3 euler_swapped = Vector3(euler_yzx.z, euler_yzx.y, euler_yzx.x);
					// Convert back to quaternion using Godot's YXZ order
					Basis converted_basis = Basis::from_euler(euler_swapped, EulerOrder::YXZ);
					orientation = converted_basis.get_rotation_quaternion();
					if (orientations.is_empty()) {
						orientations.append(orientation);
		} else {
						orientations.set(orientations.size() - 1, orientation);
					}
				} else if (s[0].casecmp_to("CHANNELS") == 0) {
					int channel_count = s[1].to_int();
					ERR_FAIL_COND_V(s.size() < channel_count + 2 || bone_indices.is_empty(), ERR_FILE_CORRUPT);
					Vector<int> channel;
					channel.append(bone_indices[bone_indices.size() - 1]);
					for (int i = 0; i < channel_count; i++) {
						String channel_name = s[i + 2].strip_edges();
						if (channel_name.casecmp_to("Xposition") == 0) {
							channel.append(BVH_X_POSITION);
						} else if (channel_name.casecmp_to("Yposition") == 0) {
							channel.append(BVH_Y_POSITION);
						} else if (channel_name.casecmp_to("Zposition") == 0) {
							channel.append(BVH_Z_POSITION);
						} else if (channel_name.casecmp_to("Xrotation") == 0) {
							channel.append(BVH_X_ROTATION);
						} else if (channel_name.casecmp_to("Yrotation") == 0) {
							channel.append(BVH_Y_ROTATION);
						} else if (channel_name.casecmp_to("Zrotation") == 0) {
							channel.append(BVH_Z_ROTATION);
						} else if (channel_name.casecmp_to("Wrotation") == 0) {
							channel.append(BVH_W_ROTATION);
						} else {
							channel_name.clear();
						}
						ERR_FAIL_COND_V(channel_name.is_empty(), ERR_FILE_CORRUPT);
					}
					ERR_FAIL_COND_V(channel.size() < 2, ERR_FILE_CORRUPT);
					if (channels.is_empty()) {
						channels.append(channel);
					} else if (channels[channels.size() - 1].size() < 2) {
						channels.remove_at(channels.size() - 1);
						channels.append(channel);
					}
				} else if (s[0].casecmp_to("JOINT") == 0) {
					ERR_FAIL_COND_V(bone_indices.is_empty(), ERR_FILE_CORRUPT);
					int parent_bone_idx = bone_indices[bone_indices.size() - 1];
					String bone_name = s[1];
					if (bone_names.has(bone_name)) {
						bone_names[bone_name] += 1;
						if (bone_name.ends_with("_")) {
							bone_name += "_";
						}
						bone_name += String::num_int64(bone_names[bone_name]);
					} else {
						bone_names[bone_name] = 1;
					}
					// Create GLTFNode for joint
					Ref<GLTFNode> node;
					node.instantiate();
					node->set_original_name(bone_name);
					node->set_name(_gen_unique_name_static(p_state->unique_names, bone_name));
					node->joint = true;
					node->transform = Transform3D(); // Will be set when we parse OFFSET/ORIENT
					
					GLTFNodeIndex parent_node_index = r_bone_data[parent_bone_idx].gltf_node_index;
					GLTFNodeIndex node_index = p_state->append_gltf_node(node, nullptr, parent_node_index);
					
					BoneData bone_data;
					bone_data.name = bone_name;
					bone_data.gltf_node_index = node_index;
					bone_data.parent_bone_index = parent_bone_idx;
					bone_data.offset = Vector3();
					bone_data.orientation = Quaternion();
					bone_data.rest_transform = Transform3D();
					r_bone_data.push_back(bone_data);
					r_bone_name_to_node[bone_name] = node_index;
					bone_indices.push_back(r_bone_data.size() - 1);
					orientations.append(Quaternion());
					offsets.append(Vector3());
					channels.append(Vector<int>({ bone_indices[bone_indices.size() - 1] }));
				}
			} else {
				if (l.casecmp_to("{") == 0) {
					// Opening brace, continue
				} else if (l.casecmp_to("}") == 0) {
					ERR_FAIL_COND_V(bone_indices.is_empty() || offsets.is_empty() || orientations.is_empty() || channels.is_empty(), ERR_FILE_CORRUPT);
					int bone_idx = bone_indices[bone_indices.size() - 1];
					Vector3 offset = offsets[offsets.size() - 1];
					Quaternion orientation = orientations[orientations.size() - 1];
					bone_indices.remove_at(bone_indices.size() - 1);
					offsets.remove_at(offsets.size() - 1);
					orientations.remove_at(orientations.size() - 1);
					
					// Store global orientation first (will convert to local in second pass)
					// ORIENT in QBO/BVH is typically in global space
					r_bone_data.write[bone_idx].offset = offset;
					r_bone_data.write[bone_idx].orientation = orientation; // Store global orientation for now
					
					// For now, use global orientation for rest transform
					// We'll convert to local space in a second pass after all bones are parsed
					Transform3D rest;
					Vector3 scale = Vector3(1.0, 1.0, 1.0);
					rest.basis.set_quaternion_scale(orientation, scale);
					rest.origin = offset;
					r_bone_data.write[bone_idx].rest_transform = rest;
					
					// Update GLTFNode transform (will be updated in second pass with local orientation)
					GLTFNodeIndex node_index = r_bone_data[bone_idx].gltf_node_index;
					Ref<GLTFNode> node = p_state->nodes[node_index];
					node->transform = rest;
					node->set_additional_data("GODOT_rest_transform", rest);
				}
			}
		}
	}

	return OK;
}

Error QBODocument::_parse_obj_to_gltf(Ref<FileAccess> f, const String &p_base_path, Ref<GLTFState> p_state, const HashMap<String, GLTFNodeIndex> &p_bone_name_to_node, const Vector<BoneData> &p_bone_data, const HashMap<String, int> &p_bone_name_to_skeleton_bone_index, bool p_generate_tangents, bool p_disable_compression, List<String> *r_missing_deps) {
	// Parse OBJ section and create GLTFMesh and GLTFSkin directly in GLTFState
	// This reuses the OBJ parsing logic but converts to GLTFMesh instead of ImporterMesh
	// and creates GLTFSkin from vertex weights using bone name to GLTFNode mapping
	
	// First pass: Scan for weights to determine if we need 8 weights per vertex
	uint64_t file_start_pos = f->get_position();
	int max_weights_per_vertex = 0;
	bool in_hierarchy_scan = false;
	bool in_motion_scan = false;
	
	while (true) {
		String l = f->get_line().strip_edges();
		if (f->eof_reached()) {
						break;
					}

		// Skip HIERARCHY and MOTION sections
		if (l.begins_with("HIERARCHY")) {
			in_hierarchy_scan = true;
			continue;
		}
		if (l.begins_with("MOTION")) {
			in_hierarchy_scan = false;
			in_motion_scan = true;
			continue;
		}
		if (in_hierarchy_scan || in_motion_scan) {
			continue;
		}
		
		// Count weights per vertex
		if (l.begins_with("vw ") && !p_bone_name_to_node.is_empty()) {
			Vector<String> v = l.split(" ", false);
			if (v.size() >= 4 && (v.size() - 2) % 2 == 0) {
				// Count only valid bone names
				int valid_weight_count = 0;
				for (int i = 2; i < v.size() - 1; i += 2) {
					String b = v[i];
					if (p_bone_name_to_node.has(b)) {
						valid_weight_count++;
					}
				}
				if (valid_weight_count > max_weights_per_vertex) {
					max_weights_per_vertex = valid_weight_count;
				}
			}
		}
	}
	
	// Reset file position for actual parsing
	f->seek(file_start_pos);
	
	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();
	
	bool generate_tangents = p_generate_tangents;
	Vector3 scale_mesh = Vector3(1.0, 1.0, 1.0);
	Vector3 offset_mesh = Vector3(0.0, 0.0, 0.0);
	
	Vector<HashMap<String, float>> weights; // Vertex weights: bone name -> weight
	Vector<Vector3> vertices;
	Vector<Vector3> normals;
	Vector<Vector2> uvs;
	Vector<Color> colors;
	const String default_name = "QBO";
	String name = default_name;
	
		// Track bone names used in weights to determine skin joint order
		HashSet<String> used_bone_names;
		// Mapping from bone name to joint index in skin (created when we first process faces)
		// This is still needed for skin creation (skins use joint indices)
		HashMap<String, int> bone_name_to_joint_index;
		bool bone_mapping_created = false;
	
	HashMap<String, HashMap<String, Ref<StandardMaterial3D>>> material_map;
	
	Ref<SurfaceTool> surf_tool = memnew(SurfaceTool);
	// Set skin weight count based on scan results
	SurfaceTool::SkinWeightCount skin_weight_count = (max_weights_per_vertex > 4) ? SurfaceTool::SKIN_8_WEIGHTS : SurfaceTool::SKIN_4_WEIGHTS;
	if (max_weights_per_vertex > 4) {
		surf_tool->set_skin_weight_count(SurfaceTool::SkinWeightCount::SKIN_8_WEIGHTS);
	}
	surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);

	String current_material_library;
	String current_material;
	String current_group;
	uint32_t smooth_group = 0;
	bool smoothing = true;
	const uint32_t no_smoothing_smooth_group = (uint32_t)-1;
	
	// Skip HIERARCHY and MOTION sections (already parsed)
	bool in_hierarchy = false;
	bool in_motion = false;
	
	while (true) {
		String l = f->get_line().strip_edges();
		while (l.length() && l[l.length() - 1] == '\\') {
			String add = f->get_line().strip_edges();
			l += add;
			if (add.is_empty()) {
						break;
			}
		}
		
		// Check for EOF - if EOF reached, process it before breaking
		bool is_eof = f->eof_reached();
		
		if (l.is_empty() && !is_eof) {
							continue;
						}
		if (l.begins_with("#")) {
			continue;
		}

		// Skip HIERARCHY and MOTION sections
		if (l.begins_with("HIERARCHY")) {
			in_hierarchy = true;
			continue;
		}
		if (l.begins_with("MOTION")) {
			in_motion = true;
			in_hierarchy = false;
			continue;
		}
		if (in_hierarchy || in_motion) {
			// Skip until we reach OBJ data
			if (l.begins_with("v ") || l.begins_with("vt ") || l.begins_with("vn ") || l.begins_with("f ")) {
				in_hierarchy = false;
				in_motion = false;
			} else {
				continue;
			}
		}
		
		if (l.begins_with("v ")) {
			//vertex
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_FILE_CORRUPT);
			Vector3 vtx;
			vtx.x = v[1].to_float() * scale_mesh.x + offset_mesh.x;
			vtx.y = v[2].to_float() * scale_mesh.y + offset_mesh.y;
			vtx.z = v[3].to_float() * scale_mesh.z + offset_mesh.z;
			vertices.push_back(vtx);
			//vertex color
			if (v.size() >= 7) {
				while (colors.size() < vertices.size() - 1) {
					colors.push_back(Color(1.0, 1.0, 1.0));
				}
				Color c;
				c.r = v[4].to_float();
				c.g = v[5].to_float();
				c.b = v[6].to_float();
				colors.push_back(c);
			} else if (!colors.is_empty()) {
				colors.push_back(Color(1.0, 1.0, 1.0));
				}
			} else if (l.begins_with("vt ")) {
			//uv
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 3, ERR_FILE_CORRUPT);
			Vector2 uv;
			uv.x = v[1].to_float();
			uv.y = 1.0 - v[2].to_float();
				uvs.push_back(uv);
			} else if (l.begins_with("vn ")) {
			//normal
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_FILE_CORRUPT);
			Vector3 nrm;
			nrm.x = v[1].to_float();
			nrm.y = v[2].to_float();
			nrm.z = v[3].to_float();
			normals.push_back(nrm);
			} else if (l.begins_with("vw ")) {
			//weight - store for later GLTFSkin creation
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(p_bone_name_to_node.is_empty() || v.size() < 4 || (v.size() - 2) % 2 != 0, ERR_FILE_CORRUPT);
			int vertex_idx = v[1].to_int() - 1; // Convert 1-based to 0-based
			if (vertex_idx < 0) {
				vertex_idx += vertices.size() + 1;
			}
			// Ensure weights list is large enough
			while (weights.size() <= vertex_idx) {
				weights.push_back(HashMap<String, float>());
			}
			HashMap<String, float> weight;
			for (int i = 2; i < v.size() - 1; i += 2) {
				String b = v[i];
				float w = v[i + 1].to_float();
				// Validate bone name exists in bone mapping
					if (!p_bone_name_to_node.has(b)) {
					continue; // Skip invalid bone names
				}
				weight[b] = w;
				// Track used bone names for joint index mapping
				used_bone_names.insert(b);
				// Note: set_skin_weight_count must be called before begin(), but we don't know
				// if we need 8 weights until we parse vw lines. SurfaceTool's add_vertex will
				// automatically handle capping to 4 weights if we set more, but if we need 8,
				// we'd need to restart. For now, we'll let SurfaceTool handle it in add_vertex
				// which will cap and normalize weights > 4 to 4 weights.
			}
			weights.set(vertex_idx, weight);
			} else if (l.begins_with("f ")) {
			// Create mapping from bone name to joint index in skin when we first process a face
			// This ensures we've seen all weights before creating the mapping
			// The mapping must match the skin joint order (based on p_bone_data order)
			if (!bone_mapping_created && !used_bone_names.is_empty()) {
				int joint_idx = 0;
				for (int i = 0; i < p_bone_data.size(); i++) {
					if (used_bone_names.has(p_bone_data[i].name)) {
						bone_name_to_joint_index[p_bone_data[i].name] = joint_idx;
						joint_idx++;
					}
				}
				bone_mapping_created = true;
			}
			
			//face - reuse existing face parsing logic
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_FILE_CORRUPT);
			
			Vector<String> face[3];
			face[0] = v[1].split("/");
			face[1] = v[2].split("/");
			ERR_FAIL_COND_V(face[0].is_empty(), ERR_FILE_CORRUPT);
			ERR_FAIL_COND_V(face[0].size() != face[1].size(), ERR_FILE_CORRUPT);
			
			for (int i = 2; i < v.size() - 1; i++) {
				face[2] = v[i + 1].split("/");
				ERR_FAIL_COND_V(face[0].size() != face[2].size(), ERR_FILE_CORRUPT);
				
				for (int j = 0; j < 3; j++) {
					int idx = j;
					if (idx < 2) {
						idx = 1 ^ idx;
					}
					
					if (face[idx].size() >= 3) {
						int norm = face[idx][2].to_int() - 1;
						if (norm < 0) {
							norm += normals.size() + 1;
						}
						ERR_FAIL_INDEX_V(norm, normals.size(), ERR_FILE_CORRUPT);
						surf_tool->set_normal(normals[norm]);
						if (generate_tangents && uvs.is_empty()) {
							Vector3 tan = Vector3(normals[norm].z, -normals[norm].x, normals[norm].y).cross(normals[norm].normalized()).normalized();
							surf_tool->set_tangent(Plane(tan.x, tan.y, tan.z, 1.0));
						}
			} else {
						if (generate_tangents && uvs.is_empty()) {
							surf_tool->set_tangent(Plane(1.0, 0.0, 0.0, 1.0));
						}
					}
					
					if (face[idx].size() >= 2 && !face[idx][1].is_empty()) {
						int uv = face[idx][1].to_int() - 1;
						if (uv < 0) {
							uv += uvs.size() + 1;
						}
						ERR_FAIL_INDEX_V(uv, uvs.size(), ERR_FILE_CORRUPT);
						surf_tool->set_uv(uvs[uv]);
					}
					
					int vtx = face[idx][0].to_int() - 1;
					if (vtx < 0) {
						vtx += vertices.size() + 1;
					}
					ERR_FAIL_INDEX_V(vtx, vertices.size(), ERR_FILE_CORRUPT);
					
					Vector3 vertex = vertices[vtx];
					if (!colors.is_empty()) {
						surf_tool->set_color(colors[vtx]);
					}
					
					// Set bones and weights for this vertex
					// If we have any weights in the mesh, we must set bones/weights on ALL vertices
					// to ensure the format flag is set and the mesh is detected as rigged
					// Use skeleton bone indices directly (from p_bone_name_to_skeleton_bone_index)
					if (!weights.is_empty() && (!p_bone_name_to_skeleton_bone_index.is_empty() || bone_mapping_created)) {
						if (vtx < weights.size() && !weights.get(vtx).is_empty()) {
							// Collect bone indices and weights
							struct WeightPair {
								int bone_idx;
								float weight;
								bool operator<(const WeightPair &p_right) const {
									return weight > p_right.weight; // Sort descending
								}
							};
							Vector<WeightPair> bone_weight_pairs;
							for (HashMap<String, float>::ConstIterator itr = weights.get(vtx).begin(); itr; ++itr) {
								if (itr->key.is_empty()) {
							continue;
						}
								// Map bone name to skeleton bone index (for mesh vertices)
								// Use skeleton bone index directly instead of joint index
								if (p_bone_name_to_skeleton_bone_index.has(itr->key)) {
									int skeleton_bone_idx = p_bone_name_to_skeleton_bone_index[itr->key];
									WeightPair wp = {};
									wp.bone_idx = skeleton_bone_idx;
									wp.weight = itr->value;
									bone_weight_pairs.append(wp);
								} else if (bone_name_to_joint_index.has(itr->key)) {
									// Fallback: if skeleton bone index not found, use joint index
									// (This shouldn't happen if skeletons were created properly)
									int joint_idx = bone_name_to_joint_index[itr->key];
									WeightPair wp = {};
									wp.bone_idx = joint_idx;
									wp.weight = itr->value;
									bone_weight_pairs.append(wp);
								}
							}
							
							if (!bone_weight_pairs.is_empty()) {
								// Sort by weight (descending) to keep highest weights
								bone_weight_pairs.sort();
								
								// Limit to 4 or 8 weights based on skin weight count
								int max_weights = (surf_tool->get_skin_weight_count() == SurfaceTool::SKIN_8_WEIGHTS) ? 8 : 4;
								if (bone_weight_pairs.size() > max_weights) {
									bone_weight_pairs.resize(max_weights);
								}
								
								// Normalize weights
								float total_weight = 0.0f;
								for (int i = 0; i < bone_weight_pairs.size(); i++) {
									total_weight += bone_weight_pairs[i].weight;
								}
								
								Vector<int> bone_indices_vec;
								Vector<float> weight_vec;
								if (total_weight > 0.0f) {
									for (int i = 0; i < bone_weight_pairs.size(); i++) {
										bone_indices_vec.append(bone_weight_pairs[i].bone_idx);
										weight_vec.append(bone_weight_pairs[i].weight / total_weight);
									}
								}
								
								// Pad to required size (4 or 8)
								int required_size = max_weights;
								while (bone_indices_vec.size() < required_size) {
									bone_indices_vec.append(0);
									weight_vec.append(0.0f);
								}
								
								surf_tool->set_bones(bone_indices_vec);
								surf_tool->set_weights(weight_vec);
			} else {
								// Vertex has no weights, but mesh is rigged - pad with zeros
								int max_weights = (surf_tool->get_skin_weight_count() == SurfaceTool::SKIN_8_WEIGHTS) ? 8 : 4;
								Vector<int> bone_indices_vec;
								Vector<float> weight_vec;
								bone_indices_vec.resize(max_weights);
								weight_vec.resize(max_weights);
								for (int i = 0; i < max_weights; i++) {
									bone_indices_vec.write[i] = 0;
									weight_vec.write[i] = 0.0f;
								}
								surf_tool->set_bones(bone_indices_vec);
								surf_tool->set_weights(weight_vec);
			}
		} else {
							// Vertex index out of range or no weights for this vertex, but mesh is rigged - pad with zeros
							int max_weights = (surf_tool->get_skin_weight_count() == SurfaceTool::SKIN_8_WEIGHTS) ? 8 : 4;
							Vector<int> bone_indices_vec;
							Vector<float> weight_vec;
							bone_indices_vec.resize(max_weights);
							weight_vec.resize(max_weights);
							for (int i = 0; i < max_weights; i++) {
								bone_indices_vec.write[i] = 0;
								weight_vec.write[i] = 0.0f;
							}
							surf_tool->set_bones(bone_indices_vec);
							surf_tool->set_weights(weight_vec);
						}
					} else {
						// No weights in mesh - don't set bones/weights
						// (This allows unrigged meshes to work correctly)
					}
					surf_tool->set_smooth_group(smoothing ? smooth_group : no_smoothing_smooth_group);
					surf_tool->add_vertex(vertex);
				}
				face[1] = face[2];
			}
		} else if (l.begins_with("s ")) {
			//smoothing
			String what = l.substr(2, l.length()).strip_edges();
			bool do_smooth;
			if (what == "off") {
				do_smooth = false;
		} else {
				do_smooth = true;
			}
			if (do_smooth) {
				smooth_group++;
			} else {
				smooth_group = no_smoothing_smooth_group;
			}
			smoothing = do_smooth;
		} else if (l.begins_with("usemtl ")) {
			//commit group to mesh
			if (surf_tool->get_vertex_array().size() > 0) {
				Ref<StandardMaterial3D> material;
				if (!current_material.is_empty() && material_map.has(current_material_library) && material_map[current_material_library].has(current_material)) {
					material = material_map[current_material_library][current_material];
				}
				
				// Generate normals if they don't exist
				if (normals.is_empty()) {
					surf_tool->generate_normals();
				}
				
				// Generate tangents if needed
				if (generate_tangents && !uvs.is_empty()) {
					surf_tool->generate_tangents();
				}
				
				// Index the mesh for better performance
				surf_tool->index();
				
				uint32_t mesh_flags = 0;
				if (!p_disable_compression) {
					mesh_flags |= RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
				}
				// Note: Octahedral compression is handled automatically by Godot
				
				Array array = surf_tool->commit_to_arrays();
				
				if (mesh_flags & RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES && generate_tangents) {
					Vector<Vector3> norms = array[Mesh::ARRAY_NORMAL];
					Vector<float> tangents = array[Mesh::ARRAY_TANGENT];
					for (int vert = 0; vert < norms.size(); vert++) {
						Vector3 tan = Vector3(tangents[vert * 4 + 0], tangents[vert * 4 + 1], tangents[vert * 4 + 2]);
						if (abs(tan.dot(norms[vert])) > 0.0001) {
							mesh_flags &= ~RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
						}
					}
				}
				
				importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, array, TypedArray<Array>(), Dictionary(), material, name, mesh_flags);
				
				if (!current_material.is_empty()) {
					if (importer_mesh->get_surface_count() >= 1) {
						importer_mesh->set_surface_name(importer_mesh->get_surface_count() - 1, current_material.get_basename());
					}
				} else if (!current_group.is_empty()) {
					if (importer_mesh->get_surface_count() >= 1) {
						importer_mesh->set_surface_name(importer_mesh->get_surface_count() - 1, current_group);
					}
				}
				
				surf_tool->clear();
				// Restore skin weight count after clear (clear resets it to SKIN_4_WEIGHTS)
				if (skin_weight_count == SurfaceTool::SKIN_8_WEIGHTS) {
					surf_tool->set_skin_weight_count(SurfaceTool::SkinWeightCount::SKIN_8_WEIGHTS);
				}
				surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
			}
		} else if (l.begins_with("o ") || is_eof) {
			// When we see 'o' or EOF, finish the current mesh and create mesh node
			// First, commit any pending surface from surf_tool
			if (surf_tool->get_vertex_array().size() > 0) {
				Ref<StandardMaterial3D> material;
				if (!current_material.is_empty() && material_map.has(current_material_library) && material_map[current_material_library].has(current_material)) {
					material = material_map[current_material_library][current_material];
				}
				
				// Generate normals if they don't exist
				if (normals.is_empty()) {
					surf_tool->generate_normals();
				}
				
				// Generate tangents if needed
				if (generate_tangents && !uvs.is_empty()) {
					surf_tool->generate_tangents();
				}
				
				// Index the mesh for better performance
		surf_tool->index();

				uint32_t mesh_flags = 0;
				if (!p_disable_compression) {
					mesh_flags |= RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
				}
				// Note: Octahedral compression is handled automatically by Godot
				
				Array array = surf_tool->commit_to_arrays();
				
				if (mesh_flags & RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES && generate_tangents) {
					Vector<Vector3> norms = array[Mesh::ARRAY_NORMAL];
					Vector<float> tangents = array[Mesh::ARRAY_TANGENT];
					for (int vert = 0; vert < norms.size(); vert++) {
						Vector3 tan = Vector3(tangents[vert * 4 + 0], tangents[vert * 4 + 1], tangents[vert * 4 + 2]);
						if (abs(tan.dot(norms[vert])) > 0.0001) {
							mesh_flags &= ~RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
						}
					}
				}
				
				importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, array, TypedArray<Array>(), Dictionary(), material, name, mesh_flags);
				
				if (!current_material.is_empty()) {
					if (importer_mesh->get_surface_count() >= 1) {
						importer_mesh->set_surface_name(importer_mesh->get_surface_count() - 1, current_material.get_basename());
					}
				} else if (!current_group.is_empty()) {
					if (importer_mesh->get_surface_count() >= 1) {
						importer_mesh->set_surface_name(importer_mesh->get_surface_count() - 1, current_group);
					}
				}
				
				surf_tool->clear();
				// Restore skin weight count after clear (clear resets it to SKIN_4_WEIGHTS)
				if (skin_weight_count == SurfaceTool::SKIN_8_WEIGHTS) {
					surf_tool->set_skin_weight_count(SurfaceTool::SkinWeightCount::SKIN_8_WEIGHTS);
				}
				surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
			}
			
			// Create mesh node if we have surfaces (when we see 'o', finish previous mesh)
			// Check both importer_mesh (previous object) and surf_tool (current object)
			// to ensure we create a mesh even if surfaces were just committed
			bool has_surfaces = importer_mesh->get_surface_count() > 0;
			bool has_pending_vertices = surf_tool->get_vertex_array().size() > 0;
			
			if (has_surfaces || has_pending_vertices) {
				// If we have pending vertices but no surfaces yet, commit them first
				if (has_pending_vertices && !has_surfaces) {
					Ref<StandardMaterial3D> material;
					if (!current_material.is_empty() && material_map.has(current_material_library) && material_map[current_material_library].has(current_material)) {
						material = material_map[current_material_library][current_material];
					}
					
					// Generate normals if they don't exist
		if (normals.is_empty()) {
			surf_tool->generate_normals();
		}
					
					// Generate tangents if needed
					if (generate_tangents && !uvs.is_empty()) {
			surf_tool->generate_tangents();
		}

					// Index the mesh for better performance
					surf_tool->index();
					
					uint32_t mesh_flags = 0;
					if (!p_disable_compression) {
						mesh_flags |= RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
					}
					
					Array array = surf_tool->commit_to_arrays();
					
					if (mesh_flags & RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES && generate_tangents) {
						Vector<Vector3> norms = array[Mesh::ARRAY_NORMAL];
						Vector<float> tangents = array[Mesh::ARRAY_TANGENT];
						for (int vert = 0; vert < norms.size(); vert++) {
							Vector3 tan = Vector3(tangents[vert * 4 + 0], tangents[vert * 4 + 1], tangents[vert * 4 + 2]);
							if (abs(tan.dot(norms[vert])) > 0.0001) {
								mesh_flags &= ~RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
							}
						}
					}
					
					importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, array, TypedArray<Array>(), Dictionary(), material, name, mesh_flags);
					
					if (!current_material.is_empty()) {
						if (importer_mesh->get_surface_count() >= 1) {
							importer_mesh->set_surface_name(importer_mesh->get_surface_count() - 1, current_material.get_basename());
						}
					} else if (!current_group.is_empty()) {
						if (importer_mesh->get_surface_count() >= 1) {
							importer_mesh->set_surface_name(importer_mesh->get_surface_count() - 1, current_group);
						}
					}
					
					surf_tool->clear();
					surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
				}
				
				// Now create mesh node if we have surfaces
				if (importer_mesh->get_surface_count() > 0) {
					importer_mesh->set_name(name);
				// Convert ImporterMesh to GLTFMesh
				Ref<GLTFMesh> gltf_mesh;
				gltf_mesh.instantiate();
				if (!importer_mesh->get_name().is_empty()) {
					gltf_mesh->set_original_name(importer_mesh->get_name());
					gltf_mesh->set_name(_gen_unique_name_static(p_state->unique_names, importer_mesh->get_name()));
				}
				gltf_mesh->set_mesh(importer_mesh);
				GLTFMeshIndex mesh_index = p_state->meshes.size();
		p_state->meshes.push_back(gltf_mesh);
				
				// Create GLTFNode for mesh
				Ref<GLTFNode> mesh_node;
				mesh_node.instantiate();
				mesh_node->set_original_name(importer_mesh->get_name());
				mesh_node->set_name(_gen_unique_name_static(p_state->unique_names, importer_mesh->get_name()));
				mesh_node->mesh = mesh_index;
				mesh_node->transform = Transform3D();
				
				// Create GLTFSkin if we have bones/joints (always create for meshes with joints)
				GLTFNodeIndex skeleton_root_node = -1;
				if (!p_bone_name_to_node.is_empty()) {
					Ref<GLTFSkin> gltf_skin;
					gltf_skin.instantiate();
					gltf_skin->set_name(_gen_unique_name_static(p_state->unique_names, "qboSkin"));
					
					// Use the function-level used_bone_names that was populated during vw parsing
					// No need to re-collect - it's already been tracked
					
					// Create joints and inverse binds from bone data
					HashMap<String, GLTFNodeIndex> bone_name_to_gltf_node;
					for (int i = 0; i < p_bone_data.size(); i++) {
						bone_name_to_gltf_node[p_bone_data[i].name] = p_bone_data[i].gltf_node_index;
					}
					
					// Determine which nodes to include in the skin
					HashSet<GLTFNodeIndex> nodes_to_include;
					if (!used_bone_names.is_empty()) {
						// We have weights - include only bones with weights and their ancestors
						for (int i = 0; i < p_bone_data.size(); i++) {
							if (used_bone_names.has(p_bone_data[i].name)) {
								// Add this bone and all its ancestors
								int bone_idx = i;
								while (bone_idx >= 0) {
									GLTFNodeIndex node_index = p_bone_data[bone_idx].gltf_node_index;
									nodes_to_include.insert(node_index);
									bone_idx = p_bone_data[bone_idx].parent_bone_index;
								}
							}
						}
				} else {
						// No weights - include all bones in the hierarchy
						for (int i = 0; i < p_bone_data.size(); i++) {
							GLTFNodeIndex node_index = p_bone_data[i].gltf_node_index;
							nodes_to_include.insert(node_index);
						}
					}
					
					// Add joints to skin (in order of bone_data to maintain hierarchy)
					for (int i = 0; i < p_bone_data.size(); i++) {
						if (nodes_to_include.has(p_bone_data[i].gltf_node_index)) {
							GLTFNodeIndex node_index = p_bone_data[i].gltf_node_index;
							// Add to joints_original if it has weights (for inverse binds), or if no weights at all (include all)
							if (used_bone_names.is_empty() || used_bone_names.has(p_bone_data[i].name)) {
								gltf_skin->joints_original.push_back(node_index);
								// Calculate inverse bind matrix from rest transform
								Transform3D rest = p_bone_data[i].rest_transform;
								Transform3D inverse_bind = rest.affine_inverse();
								gltf_skin->inverse_binds.push_back(inverse_bind);
							}
							// Add to joints if it's a joint node, otherwise it will be added to non_joints by _expand_skin
							if (p_state->nodes[node_index]->joint) {
								if (!gltf_skin->joints.has(node_index)) {
									gltf_skin->joints.push_back(node_index);
								}
							}
							
							// Mark as root if no parent and remember first root bone
							if (p_bone_data[i].parent_bone_index < 0) {
								gltf_skin->roots.push_back(node_index);
								if (skeleton_root_node < 0) {
									skeleton_root_node = node_index;
								}
							}
						}
					}
					
					// Set skin_root to first root bone
					if (!gltf_skin->roots.is_empty()) {
						gltf_skin->skin_root = gltf_skin->roots[0];
					}
					
					// Add skin to state
					GLTFSkinIndex skin_index = p_state->skins.size();
					p_state->skins.push_back(gltf_skin);
					
					// Expand and verify the skin (like FBXDocument does)
					// This will include all nodes in the subtree, not just bones with weights
					// Save the ancestors we added to joints before _expand_skin
					HashSet<GLTFNodeIndex> ancestors_in_joints;
					for (GLTFNodeIndex node_index : gltf_skin->joints) {
						if (nodes_to_include.has(node_index) && !used_bone_names.has(p_state->nodes[node_index]->get_original_name())) {
							// This is an ancestor (in nodes_to_include but not in used_bone_names)
							ancestors_in_joints.insert(node_index);
						}
					}
					
					Error skin_err = SkinTool::_expand_skin(p_state->nodes, gltf_skin);
					ERR_FAIL_COND_V(skin_err != OK, skin_err);
					
					// Ensure ancestors are still in joints after _expand_skin
					for (GLTFNodeIndex ancestor_node : ancestors_in_joints) {
						if (p_state->nodes[ancestor_node]->joint && !gltf_skin->joints.has(ancestor_node)) {
							gltf_skin->joints.push_back(ancestor_node);
						}
					}
					
					skin_err = SkinTool::_verify_skin(p_state->nodes, gltf_skin);
					ERR_FAIL_COND_V(skin_err != OK, skin_err);
					
					// Set skin on mesh node
					mesh_node->skin = skin_index;
					// Skeleton will be set after SkinTool determines it
				}
				
				// Add mesh node to scene as root node
				// append_gltf_node with parent=-1 already adds to root_nodes, so no need to push_back again
				GLTFNodeIndex mesh_node_index = p_state->append_gltf_node(mesh_node, nullptr, -1);
				
					// Reset importer_mesh for next mesh group
					importer_mesh.instantiate();
				}
			}
			
			// Handle object name and reset state for new object
			if (l.begins_with("o ")) {
				name = l.substr(2, l.length()).strip_edges();
				current_group = "";
				current_material = "";
			}
			
			if (is_eof) {
				break;
			}
		} else if (l.begins_with("usemtl ")) {
			current_material = l.replace("usemtl", "").strip_edges();
		} else if (l.begins_with("g ")) {
			// Groups are standard OBJ but are NOT mesh boundaries in Godot's OBJ importer
			// They're only used for surface naming. Only 'o' (object) creates new meshes.
			current_group = l.substr(2, l.length()).strip_edges();
		} else if (l.begins_with("mtllib ")) {
			//parse material
			current_material_library = l.replace("mtllib", "").strip_edges();
			if (!material_map.has(current_material_library)) {
				HashMap<String, Ref<StandardMaterial3D>> lib;
				String lib_path = current_material_library;
				if (lib_path.is_relative_path()) {
					lib_path = p_base_path.get_base_dir().path_join(current_material_library);
				}
				Error err = _parse_material_library(lib_path, lib, r_missing_deps);
				if (err == OK) {
					material_map[current_material_library] = lib;
				}
			}
		}
	}
	
	// Handle final mesh if any
	if (importer_mesh->get_surface_count() > 0) {
		importer_mesh->set_name(name);
		Ref<GLTFMesh> gltf_mesh;
		gltf_mesh.instantiate();
		if (!importer_mesh->get_name().is_empty()) {
			gltf_mesh->set_original_name(importer_mesh->get_name());
			gltf_mesh->set_name(_gen_unique_name_static(p_state->unique_names, importer_mesh->get_name()));
		}
		gltf_mesh->set_mesh(importer_mesh);
		GLTFMeshIndex mesh_index = p_state->meshes.size();
		p_state->meshes.push_back(gltf_mesh);

			Ref<GLTFNode> mesh_node;
			mesh_node.instantiate();
		mesh_node->set_original_name(importer_mesh->get_name());
		mesh_node->set_name(_gen_unique_name_static(p_state->unique_names, importer_mesh->get_name()));
		mesh_node->mesh = mesh_index;
		mesh_node->transform = Transform3D();
		
		// Create GLTFSkin if we have bones/joints (always create for meshes with joints)
		GLTFNodeIndex skeleton_root_node = -1;
		if (!p_bone_name_to_node.is_empty()) {
			Ref<GLTFSkin> gltf_skin;
			gltf_skin.instantiate();
			gltf_skin->set_name(_gen_unique_name_static(p_state->unique_names, "qboSkin"));
			
			// Determine which nodes to include in the skin
			HashSet<GLTFNodeIndex> nodes_to_include;
			if (!used_bone_names.is_empty()) {
				// We have weights - include only bones with weights and their ancestors
				for (int i = 0; i < p_bone_data.size(); i++) {
					if (used_bone_names.has(p_bone_data[i].name)) {
						// Add this bone and all its ancestors
						int bone_idx = i;
						while (bone_idx >= 0) {
							GLTFNodeIndex node_index = p_bone_data[bone_idx].gltf_node_index;
							nodes_to_include.insert(node_index);
							bone_idx = p_bone_data[bone_idx].parent_bone_index;
						}
					}
				}
			} else {
				// No weights - include all bones in the hierarchy
				for (int i = 0; i < p_bone_data.size(); i++) {
					GLTFNodeIndex node_index = p_bone_data[i].gltf_node_index;
					nodes_to_include.insert(node_index);
				}
			}
			
			// Add joints to skin (in order of bone_data to maintain hierarchy)
			for (int i = 0; i < p_bone_data.size(); i++) {
				if (nodes_to_include.has(p_bone_data[i].gltf_node_index)) {
					GLTFNodeIndex node_index = p_bone_data[i].gltf_node_index;
					// Add to joints_original if it has weights (for inverse binds), or if no weights at all (include all)
					if (used_bone_names.is_empty() || used_bone_names.has(p_bone_data[i].name)) {
						gltf_skin->joints_original.push_back(node_index);
						// Calculate inverse bind matrix from rest transform
						Transform3D rest = p_bone_data[i].rest_transform;
						Transform3D inverse_bind = rest.affine_inverse();
						gltf_skin->inverse_binds.push_back(inverse_bind);
					}
					// Add to joints if it's a joint node, otherwise it will be added to non_joints by _expand_skin
					if (p_state->nodes[node_index]->joint) {
						if (!gltf_skin->joints.has(node_index)) {
							gltf_skin->joints.push_back(node_index);
						}
					}
					
					if (p_bone_data[i].parent_bone_index < 0) {
						gltf_skin->roots.push_back(node_index);
						if (skeleton_root_node < 0) {
							skeleton_root_node = node_index;
						}
					}
				}
			}
			
			if (!gltf_skin->roots.is_empty()) {
				gltf_skin->skin_root = gltf_skin->roots[0];
			}
			
			GLTFSkinIndex skin_index = p_state->skins.size();
			p_state->skins.push_back(gltf_skin);
			
			// Expand and verify the skin (like FBXDocument does)
			// This will include all nodes in the subtree, not just bones with weights
			Error skin_err = SkinTool::_expand_skin(p_state->nodes, gltf_skin);
			ERR_FAIL_COND_V(skin_err != OK, skin_err);
			skin_err = SkinTool::_verify_skin(p_state->nodes, gltf_skin);
			ERR_FAIL_COND_V(skin_err != OK, skin_err);
			
			mesh_node->skin = skin_index;
		}
		
		// Add mesh node to scene as root node
		// append_gltf_node with parent=-1 already adds to root_nodes, so no need to push_back again
		GLTFNodeIndex mesh_node_index = p_state->append_gltf_node(mesh_node, nullptr, -1);
	}
	
	return OK;
}

Error QBODocument::_parse_obj(Ref<FileAccess> f, const String &p_base_path, List<Ref<ImporterMesh>> &r_meshes, bool p_single_mesh, bool p_generate_tangents, bool p_optimize, Vector3 p_scale_mesh, Vector3 p_offset_mesh, bool p_disable_compression, List<String> *r_missing_deps, List<Skeleton3D *> &r_skeletons, AnimationPlayer **r_animation) {
	Ref<ImporterMesh> mesh;
	mesh.instantiate();

	bool generate_tangents = p_generate_tangents;
	Vector3 scale_mesh = p_scale_mesh;
	Vector3 offset_mesh = p_offset_mesh;

	Vector<HashMap<String, float>> weights;
	Vector<Vector3> vertices;
	Vector<Vector3> normals;
	Vector<Vector2> uvs;
	Vector<Color> colors;
	const String default_name = "QBO";
	String name = default_name;

	HashMap<String, HashMap<String, Ref<StandardMaterial3D>>> material_map;

	Ref<SurfaceTool> surf_tool = memnew(SurfaceTool);
	surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);

	String current_material_library;
	String current_material;
	String current_group;
	uint32_t smooth_group = 0;
	bool smoothing = true;
	const uint32_t no_smoothing_smooth_group = (uint32_t)-1;

	Error err = _parse_motion(f, r_skeletons, r_animation);
	ERR_FAIL_COND_V_MSG(err != OK, err, vformat("Couldn't parse QBO file, it may be corrupt."));
	if (r_animation != nullptr) {
		List<StringName> animations;
		(*r_animation)->get_animation_list(&animations);
		for (int i = 0; i < animations.size(); ++i) {
			print_verbose(animations.get(i));
		}
	}

	while (true) {
		String l = f->get_line().strip_edges();
		while (l.length() && l[l.length() - 1] == '\\') {
			String add = f->get_line().strip_edges();
			l += add;
			if (add.is_empty()) {
				break;
			}
		}

		if (l.begins_with("v ")) {
			//vertex
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_FILE_CORRUPT);
			Vector3 vtx;
			vtx.x = v[1].to_float() * scale_mesh.x + offset_mesh.x;
			vtx.y = v[2].to_float() * scale_mesh.y + offset_mesh.y;
			vtx.z = v[3].to_float() * scale_mesh.z + offset_mesh.z;
			vertices.push_back(vtx);
			//vertex color
			if (v.size() >= 7) {
				while (colors.size() < vertices.size() - 1) {
					colors.push_back(Color(1.0, 1.0, 1.0));
				}
				Color c;
				c.r = v[4].to_float();
				c.g = v[5].to_float();
				c.b = v[6].to_float();
				colors.push_back(c);
			} else if (!colors.is_empty()) {
				colors.push_back(Color(1.0, 1.0, 1.0));
			}
		} else if (l.begins_with("vt ")) {
			//uv
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 3, ERR_FILE_CORRUPT);
			Vector2 uv;
			uv.x = v[1].to_float();
			uv.y = 1.0 - v[2].to_float();
			uvs.push_back(uv);
		} else if (l.begins_with("vn ")) {
			//normal
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_FILE_CORRUPT);
			Vector3 nrm;
			nrm.x = v[1].to_float();
			nrm.y = v[2].to_float();
			nrm.z = v[3].to_float();
			normals.push_back(nrm);
		} else if (l.begins_with("vw ")) {
			//weight ( https://github.com/tinyobjloader/tinyobjloader/blob/v2.0.0rc13/tiny_obj_loader.h#L2696 )
			// Format: vw <vertex_index> <bone_name> <weight> [<bone_name> <weight> ...]
			// vertex_index is 1-based, convert to 0-based for storage
			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(r_skeletons.is_empty() || v.size() < 4 || (v.size() - 2) % 2 != 0, ERR_FILE_CORRUPT);
			int vertex_idx = v[1].to_int() - 1; // Convert 1-based to 0-based
			if (vertex_idx < 0) {
				vertex_idx += vertices.size() + 1;
			}
			// Ensure weights list is large enough
			while (weights.size() <= vertex_idx) {
				weights.push_back(HashMap<String, float>());
			}
			HashMap<String, float> weight;
			for (int i = 2; i < v.size() - 1; i += 2) {
				String b = v[i];
				float w = v[i + 1].to_float();
				// Validate bone name exists in skeleton before storing
				bool bone_found = false;
				for (int j = 0; j < r_skeletons.size(); j++) {
					if (r_skeletons.get(j)->find_bone(b) > -1) {
						bone_found = true;
					break;
				}
				}
				ERR_FAIL_COND_V(!bone_found, ERR_FILE_CORRUPT);
				// Only store weight if bone is valid
				weight[b] = w;
				// Note: set_skin_weight_count must be called before begin(), but we don't know
				// if we need 8 weights until we parse vw lines. SurfaceTool's add_vertex will
				// automatically handle capping to 4 weights if we set more, but if we need 8,
				// we'd need to restart. For now, we'll let SurfaceTool handle it in add_vertex
				// which will cap and normalize weights > 4 to 4 weights.
			}
			weights.set(vertex_idx, weight);
		} else if (l.begins_with("f ")) {
			//vertex

			Vector<String> v = l.split(" ", false);
			ERR_FAIL_COND_V(v.size() < 4, ERR_FILE_CORRUPT);

			//not very fast, could be sped up

			Vector<String> face[3];
			face[0] = v[1].split("/");
			face[1] = v[2].split("/");
			ERR_FAIL_COND_V(face[0].is_empty(), ERR_FILE_CORRUPT);

			ERR_FAIL_COND_V(face[0].size() != face[1].size(), ERR_FILE_CORRUPT);
			for (int i = 2; i < v.size() - 1; i++) {
				face[2] = v[i + 1].split("/");

				ERR_FAIL_COND_V(face[0].size() != face[2].size(), ERR_FILE_CORRUPT);
				for (int j = 0; j < 3; j++) {
					int idx = j;

					if (idx < 2) {
						idx = 1 ^ idx;
					}

					if (face[idx].size() >= 3) {
						int norm = face[idx][2].to_int() - 1;
						if (norm < 0) {
							norm += normals.size() + 1;
						}
						ERR_FAIL_INDEX_V(norm, normals.size(), ERR_FILE_CORRUPT);
						surf_tool->set_normal(normals[norm]);
						if (generate_tangents && uvs.is_empty()) {
							// We can't generate tangents without UVs, so create dummy tangents.
							Vector3 tan = Vector3(normals[norm].z, -normals[norm].x, normals[norm].y).cross(normals[norm].normalized()).normalized();
							surf_tool->set_tangent(Plane(tan.x, tan.y, tan.z, 1.0));
			}
		} else {
						// No normals, use a dummy tangent since normals and tangents will be generated.
						if (generate_tangents && uvs.is_empty()) {
							// We can't generate tangents without UVs, so create dummy tangents.
							surf_tool->set_tangent(Plane(1.0, 0.0, 0.0, 1.0));
						}
					}

					if (face[idx].size() >= 2 && !face[idx][1].is_empty()) {
						int uv = face[idx][1].to_int() - 1;
						if (uv < 0) {
							uv += uvs.size() + 1;
						}
						ERR_FAIL_INDEX_V(uv, uvs.size(), ERR_FILE_CORRUPT);
						surf_tool->set_uv(uvs[uv]);
					}

					int vtx = face[idx][0].to_int() - 1;
					if (vtx < 0) {
						vtx += vertices.size() + 1;
					}
					ERR_FAIL_INDEX_V(vtx, vertices.size(), ERR_FILE_CORRUPT);

					Vector3 vertex = vertices[vtx];
					if (!colors.is_empty()) {
						surf_tool->set_color(colors[vtx]);
					}
					if (!weights.is_empty() && vtx < weights.size() && !weights.get(vtx).is_empty()) {
						Vector<int> bones;
						Vector<float> weight;
						for (HashMap<String, float>::ConstIterator itr = weights.get(vtx).begin(); itr; ++itr) {
							if (itr->key.is_empty()) {
				continue;
			}
							if (itr->key.is_numeric()) {
								bones.append(itr->key.to_int());
							} else if (!r_skeletons.is_empty()) {
								int bone_idx = r_skeletons.back()->get()->find_bone(itr->key);
								if (bone_idx < 0) {
									continue;
								}
								bones.append(bone_idx);
			} else {
								continue;
							}
							if (bones.is_empty() || bones[bones.size() - 1] < 0) {
								if (!bones.is_empty()) {
									bones.remove_at(bones.size() - 1);
								}
								continue;
							}
							weight.append(itr->value);
						}
						if (!bones.is_empty()) {
							surf_tool->set_bones(bones);
							surf_tool->set_weights(weight);
			} else {
							surf_tool->set_bones(Vector<int>());
							surf_tool->set_weights(Vector<float>());
						}
					} else {
						surf_tool->set_bones(Vector<int>());
						surf_tool->set_weights(Vector<float>());
					}
					surf_tool->set_smooth_group(smoothing ? smooth_group : no_smoothing_smooth_group);
					surf_tool->add_vertex(vertex);
				}

				face[1] = face[2];
			}
		} else if (l.begins_with("s ")) { //smoothing
			String what = l.substr(2, l.length()).strip_edges();
			bool do_smooth;
			if (what == "off") {
				do_smooth = false;
			} else {
				do_smooth = true;
			}
			if (do_smooth != smoothing) {
				smoothing = do_smooth;
				if (smoothing) {
					smooth_group++;
				}
			}
		} else if (/*l.begins_with("g ") ||*/ l.begins_with("usemtl ") || (l.begins_with("o ") || f->eof_reached())) { //commit group to mesh
			uint64_t mesh_flags = RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;

			if (p_disable_compression) {
				mesh_flags = 0;
			} else {
				bool is_mesh_2d = true;

				// Disable compression if all z equals 0 (the mesh is 2D).
				for (int i = 0; i < vertices.size(); i++) {
					if (!Math::is_zero_approx(vertices[i].z)) {
						is_mesh_2d = false;
						break;
					}
				}

				if (is_mesh_2d) {
					mesh_flags = 0;
				}
			}

			//groups are too annoying
			if (surf_tool->get_vertex_array().size()) {
				//another group going on, commit it
				if (normals.size() == 0) {
			surf_tool->generate_normals();
		}

				if (generate_tangents && uvs.size()) {
			surf_tool->generate_tangents();
		}

				surf_tool->index();

				print_verbose("OBJ: Current material library " + current_material_library + " has " + itos(material_map.has(current_material_library)));
				print_verbose("OBJ: Current material " + current_material + " has " + itos(material_map.has(current_material_library) && material_map[current_material_library].has(current_material)));
				Ref<StandardMaterial3D> material;
				if (material_map.has(current_material_library) && material_map[current_material_library].has(current_material)) {
					material = material_map[current_material_library][current_material];
					if (!colors.is_empty()) {
						material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
					}
					surf_tool->set_material(material);
				}

				Array array = surf_tool->commit_to_arrays();

				if (mesh_flags & RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES && generate_tangents) {
					// Compression is enabled, so let's validate that the normals and tangents are correct.
					Vector<Vector3> norms = array[Mesh::ARRAY_NORMAL];
					Vector<float> tangents = array[Mesh::ARRAY_TANGENT];
					for (int vert = 0; vert < norms.size(); vert++) {
						Vector3 tan = Vector3(tangents[vert * 4 + 0], tangents[vert * 4 + 1], tangents[vert * 4 + 2]);
						if (abs(tan.dot(norms[vert])) > 0.0001) {
							// Tangent is not perpendicular to the normal, so we can't use compression.
							mesh_flags &= ~RS::ARRAY_FLAG_COMPRESS_ATTRIBUTES;
						}
					}
				}

				mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, array, TypedArray<Array>(), Dictionary(), material, name, mesh_flags);
				print_verbose("OBJ: Added surface :" + mesh->get_surface_name(mesh->get_surface_count() - 1));

				if (!current_material.is_empty()) {
					if (mesh->get_surface_count() >= 1) {
						mesh->set_surface_name(mesh->get_surface_count() - 1, current_material.get_basename());
					}
				} else if (!current_group.is_empty()) {
					if (mesh->get_surface_count() >= 1) {
						mesh->set_surface_name(mesh->get_surface_count() - 1, current_group);
					}
				}

				surf_tool->clear();
				surf_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
			}

			if (l.begins_with("o ") || f->eof_reached()) {
				if (!p_single_mesh) {
					if (mesh->get_surface_count() > 0) {
						mesh->set_name(name);
						r_meshes.push_back(mesh);
						mesh.instantiate();
					}
					name = default_name;
					current_group = "";
					current_material = "";
				}
			}

			if (f->eof_reached()) {
				break;
			}

			if (l.begins_with("o ")) {
				name = l.substr(2, l.length()).strip_edges();
			}

			if (l.begins_with("usemtl ")) {
				current_material = l.replace("usemtl", "").strip_edges();
			}

			if (l.begins_with("g ")) {
				current_group = l.substr(2, l.length()).strip_edges();
			}

		} else if (l.begins_with("mtllib ")) { //parse material

			current_material_library = l.replace("mtllib", "").strip_edges();
			if (!material_map.has(current_material_library)) {
				HashMap<String, Ref<StandardMaterial3D>> lib;
				String lib_path = current_material_library;
				if (lib_path.is_relative_path()) {
					lib_path = p_base_path.get_base_dir().path_join(current_material_library);
				}
				err = _parse_material_library(lib_path, lib, r_missing_deps);
				if (err == OK) {
					material_map[current_material_library] = lib;
				}
			}
		}
	}

	if (p_single_mesh && mesh->get_surface_count() > 0) {
		r_meshes.push_back(mesh);
	}

	return OK;
}

Error QBODocument::parse_qbo_data(Ref<FileAccess> f, Ref<GLTFState> p_state, uint32_t p_flags, String p_base_path, String p_path) {
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Cannot open QBO file.");
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);

	if (p_base_path.is_empty()) {
		p_base_path += p_path.get_base_dir();
	}

	// Parse QBO directly into GLTFState (proper architecture)
	// 1. Parse HIERARCHY section → Create GLTFNode objects for bones
	// 2. Parse OBJ section → Create GLTFMesh and GLTFSkin
	// 3. Use SkinTool to determine skeletons
	
	HashMap<String, GLTFNodeIndex> bone_name_to_node;
	Vector<BoneData> bone_data;
	Vector<GLTFNodeIndex> hierarchy_root_nodes;
	
	// Reset file to beginning
	f->seek(0);
	
	// Parse HIERARCHY section to create bone GLTFNodes (for skeletons, but not skinning)
	// Always import animations by default (unless explicitly disabled via GLTFState::create_animations)
	Error err = _parse_hierarchy_to_gltf(f, p_state, bone_name_to_node, bone_data, hierarchy_root_nodes);
	if (err != OK) {
		return err;
	}
	
	// Compute node heights (required for _find_highest_node to work correctly)
	// This sets height=0 for root nodes, height=1 for their children, etc.
	_compute_node_heights(p_state);
	
	// Determine skeletons from hierarchy (before parsing OBJ so we can use skeleton bone indices directly)
	HashMap<String, int> bone_name_to_skeleton_bone_index;
	if (!hierarchy_root_nodes.is_empty()) {
		// Determine skeletons from hierarchy root nodes
		err = SkinTool::_determine_skeletons(p_state->skins, p_state->nodes, p_state->skeletons, hierarchy_root_nodes, false);
		if (err != OK) {
			return err;
		}
		
		// Create skeletons to get bone order and joint_i_to_bone_i mapping
		HashMap<ObjectID, SkinSkeletonIndex> skeleton_map;
		HashMap<GLTFNodeIndex, Node *> scene_nodes;
		err = SkinTool::_create_skeletons(p_state->unique_names, p_state->skins, p_state->nodes,
				skeleton_map, p_state->skeletons, scene_nodes, get_naming_version());
		if (err != OK) {
			return err;
		}
		
		// Create mapping from bone name to skeleton bone index
		// This allows us to set skeleton bone indices directly during OBJ parsing
		for (GLTFSkeletonIndex skel_i = 0; skel_i < p_state->skeletons.size(); ++skel_i) {
			Ref<GLTFSkeleton> skeleton = p_state->skeletons[skel_i];
			if (skeleton.is_valid() && skeleton->godot_skeleton != nullptr) {
				Skeleton3D *godot_skeleton = skeleton->godot_skeleton;
				for (int bone_i = 0; bone_i < godot_skeleton->get_bone_count(); ++bone_i) {
					String bone_name = godot_skeleton->get_bone_name(bone_i);
					bone_name_to_skeleton_bone_index[bone_name] = bone_i;
				}
			}
		}
		
		// Clear scene_nodes and godot_skeleton pointers so they can be recreated later
		// But keep joint_i_to_bone_i mapping as it's still valid
		p_state->scene_nodes.clear();
		for (GLTFSkeletonIndex skel_i = 0; skel_i < p_state->skeletons.size(); ++skel_i) {
			Ref<GLTFSkeleton> skeleton = p_state->skeletons[skel_i];
			if (skeleton.is_valid()) {
				skeleton->godot_skeleton = nullptr;
			}
		}
	}
	
	// Reset file to beginning for OBJ parsing
	f->seek(0);
	
	// Parse OBJ section to create GLTFMesh and GLTFSkin
	// Pass bone_name_to_skeleton_bone_index so we can set skeleton bone indices directly
	List<String> missing_deps;
	err = _parse_obj_to_gltf(f, p_base_path, p_state, bone_name_to_node, bone_data, bone_name_to_skeleton_bone_index, p_flags & QBO_IMPORT_GENERATE_TANGENT_ARRAYS, p_flags & QBO_IMPORT_FORCE_DISABLE_MESH_COMPRESSION, &missing_deps);
	if (err != OK) {
		return err;
	}
	
	// Assign skins to existing skeletons (if skeletons were already determined from hierarchy)
	// Or determine skeletons from skins if no skeletons exist yet
	if (!p_state->skins.is_empty() && !p_state->skeletons.is_empty()) {
		// Skeletons already determined from hierarchy - assign skins to them
		// Match skins to skeletons by checking if skin roots/joints are in skeleton
		for (GLTFSkinIndex skin_i = 0; skin_i < p_state->skins.size(); ++skin_i) {
			Ref<GLTFSkin> skin = p_state->skins[skin_i];
			if (skin.is_null() || skin->get_skeleton() >= 0) {
				continue; // Already assigned or invalid
			}
			
			// Find which skeleton contains this skin's nodes
			for (GLTFSkeletonIndex skel_i = 0; skel_i < p_state->skeletons.size(); ++skel_i) {
				Ref<GLTFSkeleton> skeleton = p_state->skeletons[skel_i];
				if (skeleton.is_null()) {
					continue;
				}
				
				// Check if any of the skin's roots or joints are in this skeleton
				bool matches = false;
				for (GLTFNodeIndex root : skin->roots) {
					if (skeleton->roots.has(root) || skeleton->joints.has(root)) {
						matches = true;
						break;
					}
				}
				if (!matches) {
					for (GLTFNodeIndex joint : skin->joints) {
						if (skeleton->joints.has(joint)) {
							matches = true;
							break;
						}
					}
				}
				
				if (matches) {
					skin->skeleton = skel_i;
					break;
				}
			}
		}
		
		// Set skeleton index on mesh nodes that have skins
		for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); ++node_i) {
			Ref<GLTFNode> node = p_state->nodes[node_i];
			if (node->mesh >= 0 && node->skin >= 0) {
				Ref<GLTFSkin> skin = p_state->skins[node->skin];
				GLTFSkeletonIndex skeleton_index = skin->get_skeleton();
				if (skeleton_index >= 0 && skeleton_index < p_state->skeletons.size()) {
					node->skeleton = skeleton_index;
				}
			}
		}
		
		// Add skeleton root nodes to root_nodes (if not already added)
		for (GLTFSkeletonIndex skel_i = 0; skel_i < p_state->skeletons.size(); ++skel_i) {
			Ref<GLTFSkeleton> skeleton = p_state->skeletons[skel_i];
			for (GLTFNodeIndex root_node : skeleton->roots) {
				if (!p_state->root_nodes.has(root_node)) {
					p_state->root_nodes.push_back(root_node);
				}
			}
		}
	} else if (!p_state->skins.is_empty() && p_state->skeletons.is_empty()) {
		// Skins exist but no skeletons yet - use skins to determine skeletons
		err = SkinTool::_determine_skeletons(p_state->skins, p_state->nodes, p_state->skeletons, Vector<GLTFNodeIndex>(), false);
		if (err != OK) {
			return err;
		}
		
		// Set skeleton index on mesh nodes that have skins
		for (GLTFNodeIndex node_i = 0; node_i < p_state->nodes.size(); ++node_i) {
			Ref<GLTFNode> node = p_state->nodes[node_i];
			if (node->mesh >= 0 && node->skin >= 0) {
				Ref<GLTFSkin> skin = p_state->skins[node->skin];
				GLTFSkeletonIndex skeleton_index = skin->get_skeleton();
				if (skeleton_index >= 0) {
					node->skeleton = skeleton_index;
				}
			}
		}
		
		// Add skeleton root nodes to root_nodes (after skeleton creation)
		for (GLTFSkeletonIndex skel_i = 0; skel_i < p_state->skeletons.size(); ++skel_i) {
			Ref<GLTFSkeleton> skeleton = p_state->skeletons[skel_i];
			for (GLTFNodeIndex root_node : skeleton->roots) {
				if (!p_state->root_nodes.has(root_node)) {
					p_state->root_nodes.push_back(root_node);
				}
			}
		}
	} else if (!hierarchy_root_nodes.is_empty()) {
		// No skins - skeleton root nodes were already added above
		for (GLTFSkeletonIndex skel_i = 0; skel_i < p_state->skeletons.size(); ++skel_i) {
			Ref<GLTFSkeleton> skeleton = p_state->skeletons[skel_i];
			for (GLTFNodeIndex root_node : skeleton->roots) {
				if (!p_state->root_nodes.has(root_node)) {
					p_state->root_nodes.push_back(root_node);
				}
			}
		}
	}

	return OK;
}

Error QBODocument::append_from_file(const String &p_path, Ref<GLTFState> p_state, uint32_t p_flags, const String &p_base_path) {
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_FILE_NOT_FOUND);
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	p_state->set_filename(p_path.get_file().get_basename());
	p_state->use_named_skin_binds = p_flags & QBO_IMPORT_USE_NAMED_SKIN_BINDS;
	p_state->discard_meshes_and_materials = p_flags & QBO_IMPORT_DISCARD_MESHES_AND_MATERIALS;
	p_state->force_generate_tangents = p_flags & QBO_IMPORT_GENERATE_TANGENT_ARRAYS;
	p_state->force_disable_compression = p_flags & QBO_IMPORT_FORCE_DISABLE_MESH_COMPRESSION;
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, "Cannot open QBO file.");
	return parse_qbo_data(f, p_state, p_flags, p_base_path, p_path);
}

Error QBODocument::append_from_buffer(const PackedByteArray &p_bytes, const String &p_base_path, Ref<GLTFState> p_state, uint32_t p_flags) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_bytes.is_empty(), ERR_INVALID_PARAMETER);
	p_state->use_named_skin_binds = p_flags & QBO_IMPORT_USE_NAMED_SKIN_BINDS;
	p_state->discard_meshes_and_materials = p_flags & QBO_IMPORT_DISCARD_MESHES_AND_MATERIALS;
	p_state->force_generate_tangents = p_flags & QBO_IMPORT_GENERATE_TANGENT_ARRAYS;
	p_state->force_disable_compression = p_flags & QBO_IMPORT_FORCE_DISABLE_MESH_COMPRESSION;
	Ref<FileAccessMemory> memfile;
	memfile.instantiate();
	const Error open_error = memfile->open_custom(p_bytes.ptr(), p_bytes.size());
	ERR_FAIL_COND_V_MSG(open_error != OK, open_error, "Could not create memory file for QBO buffer.");
	ERR_FAIL_COND_V_MSG(memfile.is_null(), ERR_CANT_OPEN, "Cannot open QBO file.");
	return parse_qbo_data(memfile, p_state, p_flags, p_base_path, String());
}

Error QBODocument::append_from_scene(Node *p_node, Ref<GLTFState> p_state, uint32_t p_flags) {
	// QBO is an import-only format, cannot export from scene
	return ERR_UNAVAILABLE;
}

Node *QBODocument::generate_scene(Ref<GLTFState> p_state, float p_bake_fps, bool p_trimming, bool p_remove_immutable_tracks) {
	// Since QBODocument uses GLTFState (not a custom state), delegate to parent implementation
	return GLTFDocument::generate_scene(p_state, p_bake_fps, p_trimming, p_remove_immutable_tracks);
}

PackedByteArray QBODocument::generate_buffer(Ref<GLTFState> p_state) {
	// QBO is an import-only format, cannot export to buffer
	return PackedByteArray();
}

Error QBODocument::write_to_filesystem(Ref<GLTFState> p_state, const String &p_path) {
	// QBO is an import-only format, cannot export to filesystem
	return ERR_UNAVAILABLE;
}
