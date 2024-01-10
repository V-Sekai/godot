/**************************************************************************/
/*  fbx_light.h                                                           */
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

#ifndef FBX_LIGHT_H
#define FBX_LIGHT_H

#include "core/io/resource.h"
#include "modules/gltf/structures/gltf_camera.h"
#include "scene/3d/light_3d.h"

class Light3D;

class FBXLight : public GLTFCamera {
	GDCLASS(FBXLight, GLTFCamera);

private:
	Color color;
	float intensity = 0.0f;
	Vector3 local_direction = Vector3(0, 1, 0);
	int type = -1;
	int decay = 0;
	int area_shape = -1;
	float inner_angle = 0.0f;
	float outer_angle = 0.0f;
	bool cast_light = true;
	bool cast_shadows = true;

protected:
	static void _bind_methods();

public:
	static Ref<FBXLight> from_node(const Light3D *p_camera);
	Light3D *to_node() const;

	static Ref<FBXLight> from_dictionary(const Dictionary p_dictionary);
	Dictionary to_dictionary() const;

	void set_color(Color p_color) { color = p_color; }
	// Color and intensity of the light, usually you want to use `color * intensity`
	// NOTE: `intensity` is 0.01x of the property `"Intensity"` as that matches
	// matches values in DCC programs before exporting.
	void set_intensity(float p_intensity) { intensity = p_intensity; }
	void set_local_direction(Vector3 p_local_direction) { local_direction = p_local_direction; }
	void set_type(int p_type) { type = p_type; }
	void set_decay(int p_decay) { decay = p_decay; }
	void set_area_shape(int p_area_shape) { area_shape = p_area_shape; }
	void set_inner_angle(float p_inner_angle) { inner_angle = p_inner_angle; }
	void set_outer_angle(float p_outer_angle) { outer_angle = p_outer_angle; }
	void set_cast_light(bool p_cast_light) { cast_light = p_cast_light; }
	void set_cast_shadows(bool p_cast_shadows) { cast_shadows = p_cast_shadows; }
	Color get_color() const { return color; }
	float get_intensity() const { return intensity; }
	Vector3 get_local_direction() const { return local_direction; }
	int get_type() const { return type; }
	int get_decay() const { return decay; }
	int get_area_shape() const { return area_shape; }
	float get_inner_angle() const { return inner_angle; }
	float get_outer_angle() const { return outer_angle; }
	bool is_casting_light() const { return cast_light; }
	bool is_casting_shadows() const { return cast_shadows; }
};

#endif // FBX_LIGHT_H