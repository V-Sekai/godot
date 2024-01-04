/**************************************************************************/
/*  fbx_light.cpp                                                         */
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

#include "fbx_light.h"

#include "scene/3d/light_3d.h"

#include "ufbx.h"

Ref<FBXLight> FBXLight::from_node(const Light3D *p_light) {
	Ref<FBXLight> light;
	light.instantiate();
	light->set_name(p_light->get_name());

	if (!p_light) {
		ERR_FAIL_V_MSG(light, "Provided Light3D node is null.");
	}

	light->set_color(p_light->get_color());
	light->set_intensity(p_light->get_param(Light3D::PARAM_ENERGY) / 100.0f);
	light->set_cast_shadows(p_light->has_shadow());

	Transform3D light_local_transform = p_light->get_transform();

	if (cast_to<DirectionalLight3D>(p_light)) {
		light->set_type(int(UFBX_LIGHT_DIRECTIONAL));
		Vector3 direction = -light_local_transform.basis.get_column(Vector3::AXIS_Z);
		light->set_local_direction(direction.normalized());
	} else if (cast_to<OmniLight3D>(p_light)) {
		light->set_type(int(UFBX_LIGHT_POINT));
	} else if (cast_to<SpotLight3D>(p_light)) {
		const SpotLight3D *spot_light = cast_to<SpotLight3D>(p_light);
		light->set_type(int(UFBX_LIGHT_SPOT));
		Vector3 direction = -light_local_transform.basis.get_column(Vector3::AXIS_Z);
		light->set_local_direction(direction.normalized());
		light->set_inner_angle(spot_light->get_param(SpotLight3D::PARAM_SPOT_ANGLE));
		light->set_outer_angle(spot_light->get_param(SpotLight3D::PARAM_SPOT_ATTENUATION));
	} else {
		ERR_FAIL_V_MSG(light, "Unsupported Light3D type for FBXLight conversion.");
	}
	light->set_decay(UFBX_LIGHT_DECAY_QUADRATIC);

	return light;
}

Light3D *FBXLight::to_node() const {
	Light3D *light = nullptr;

	switch (get_type()) {
		case int(UFBX_LIGHT_POINT):
			light = memnew(OmniLight3D);
			break;

		case int(UFBX_LIGHT_DIRECTIONAL):
			light = memnew(DirectionalLight3D);
			break;

		case int(UFBX_LIGHT_SPOT):
			light = memnew(SpotLight3D);
			break;

		default:
			ERR_FAIL_COND_V(!light, nullptr);
	}

	if (light) {
		light->set_name(get_name());
		light->set_color(color);
		light->set_param(Light3D::PARAM_ENERGY, intensity * 100.0f);

		light->set_shadow(cast_shadows);

		Transform3D transform;
		Vector3 up_vector = Vector3(0, 1, 0);
		DirectionalLight3D *dir_light = Object::cast_to<DirectionalLight3D>(light);
		SpotLight3D *spot_light = Object::cast_to<SpotLight3D>(light);
		OmniLight3D *omni_light = Object::cast_to<OmniLight3D>(light);
		if (dir_light) {
			transform.set_look_at(Vector3(), -local_direction.normalized(), up_vector);
			dir_light->set_transform(transform);
		} else if (spot_light) {
			transform.set_look_at(Vector3(), -local_direction.normalized(), up_vector);
			spot_light->set_transform(transform);
			spot_light->set_param(SpotLight3D::PARAM_SPOT_ANGLE, inner_angle);
		}
		if (omni_light || spot_light) {
			switch (decay) {
				case UFBX_LIGHT_DECAY_NONE:
					light->set_param(Light3D::PARAM_ATTENUATION, 0.0f);
					break;
				case UFBX_LIGHT_DECAY_LINEAR:
					light->set_param(Light3D::PARAM_ATTENUATION, 1.0f);
					break;
				case UFBX_LIGHT_DECAY_QUADRATIC:
					light->set_param(Light3D::PARAM_ATTENUATION, 2.0f);
					break;
				case UFBX_LIGHT_DECAY_CUBIC:
					light->set_param(Light3D::PARAM_ATTENUATION, 3.0f);
					break;
			}
		}
	}

	return light;
}

Ref<FBXLight> FBXLight::from_dictionary(const Dictionary p_dictionary) {
	ERR_FAIL_COND_V_MSG(!p_dictionary.has("type"), Ref<FBXLight>(), "Failed to parse GLTF camera, missing required field 'type'.");
	Ref<FBXLight> light;
	light.instantiate();
	return light;
}

Dictionary FBXLight::to_dictionary() const {
	Dictionary d;
	return d;
}

void FBXLight::_bind_methods() {
	ClassDB::bind_static_method("FBXLight", D_METHOD("from_node", "camera_node"), &FBXLight::from_node);
	ClassDB::bind_static_method("FBXLight", D_METHOD("from_dictionary", "dictionary"), &FBXLight::from_dictionary);

	ClassDB::bind_method(D_METHOD("to_node"), &FBXLight::to_node);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &FBXLight::to_dictionary);

	ClassDB::bind_method(D_METHOD("set_color", "color"), &FBXLight::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &FBXLight::get_color);

	ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &FBXLight::set_intensity);
	ClassDB::bind_method(D_METHOD("get_intensity"), &FBXLight::get_intensity);

	ClassDB::bind_method(D_METHOD("set_local_direction", "local_direction"), &FBXLight::set_local_direction);
	ClassDB::bind_method(D_METHOD("get_local_direction"), &FBXLight::get_local_direction);

	ClassDB::bind_method(D_METHOD("set_type", "type"), &FBXLight::set_type);
	ClassDB::bind_method(D_METHOD("get_type"), &FBXLight::get_type);

	ClassDB::bind_method(D_METHOD("set_decay", "decay"), &FBXLight::set_decay);
	ClassDB::bind_method(D_METHOD("get_decay"), &FBXLight::get_decay);

	ClassDB::bind_method(D_METHOD("set_area_shape", "area_shape"), &FBXLight::set_area_shape);
	ClassDB::bind_method(D_METHOD("get_area_shape"), &FBXLight::get_area_shape);

	ClassDB::bind_method(D_METHOD("set_inner_angle", "inner_angle"), &FBXLight::set_inner_angle);
	ClassDB::bind_method(D_METHOD("get_inner_angle"), &FBXLight::get_inner_angle);

	ClassDB::bind_method(D_METHOD("set_outer_angle", "outer_angle"), &FBXLight::set_outer_angle);
	ClassDB::bind_method(D_METHOD("get_outer_angle"), &FBXLight::get_outer_angle);

	ClassDB::bind_method(D_METHOD("set_cast_light", "cast_light"), &FBXLight::set_cast_light);
	ClassDB::bind_method(D_METHOD("is_casting_light"), &FBXLight::is_casting_light);

	ClassDB::bind_method(D_METHOD("set_cast_shadows", "cast_shadows"), &FBXLight::set_cast_shadows);
	ClassDB::bind_method(D_METHOD("is_casting_shadows"), &FBXLight::is_casting_shadows);

	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "intensity"), "set_intensity", "get_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "local_direction"), "set_local_direction", "get_local_direction");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type"), "set_type", "get_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "decay"), "set_decay", "get_decay");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "area_shape"), "set_area_shape", "get_area_shape");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_angle"), "set_inner_angle", "get_inner_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_angle"), "set_outer_angle", "get_outer_angle");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cast_light"), "set_cast_light", "is_casting_light");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cast_shadows"), "set_cast_shadows", "is_casting_shadows");
}
