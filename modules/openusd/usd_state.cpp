/**************************************************************************/
/*  usd_state.cpp                                                         */
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

#include "usd_state.h"

void UsdState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_copyright", "copyright"), &UsdState::set_copyright);
	ClassDB::bind_method(D_METHOD("get_copyright"), &UsdState::get_copyright);

	ClassDB::bind_method(D_METHOD("set_bake_fps", "fps"), &UsdState::set_bake_fps);
	ClassDB::bind_method(D_METHOD("get_bake_fps"), &UsdState::get_bake_fps);

	// Add properties to the inspector
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "copyright", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT), "set_copyright", "get_copyright");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bake_fps", PROPERTY_HINT_RANGE, "1,120,0.1"), "set_bake_fps", "get_bake_fps");
}

UsdState::UsdState() {
	// Set default values
	_copyright = "";
	_bake_fps = 30.0f;
	_stage = nullptr;
}

void UsdState::set_copyright(const String &p_copyright) {
	_copyright = p_copyright;
}

String UsdState::get_copyright() const {
	return _copyright;
}

void UsdState::set_bake_fps(float p_fps) {
	_bake_fps = p_fps;
}

float UsdState::get_bake_fps() const {
	return _bake_fps;
}

void UsdState::set_stage(tinyusdz::Stage *p_stage) {
	// TODO: Update to use TinyUSDZ API
	_stage = p_stage;
}

tinyusdz::Stage *UsdState::get_stage() const {
	// TODO: Update to use TinyUSDZ API
	return _stage;
}
