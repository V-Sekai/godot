/**************************************************************************/
/*  animation_tree_handler_plugin.cpp                                     */
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

#include "animation_tree_handler_plugin.h"

AnimationTreeHandlerPlugin *AnimationTreeHandlerPlugin::_singleton = nullptr;

AnimationTreeHandlerPlugin::AnimationTreeHandlerPlugin() {
	_singleton = this;
	_animation_tree = nullptr;
}

bool AnimationTreeHandlerPlugin::handles(Object *p_object) const {
	return Object::cast_to<AnimationTree>(p_object) != nullptr;
}

void AnimationTreeHandlerPlugin::edit(Object *p_object) {
	_animation_tree = Object::cast_to<AnimationTree>(p_object);
}

AnimationTreeHandlerPlugin *AnimationTreeHandlerPlugin::get_singleton() {
	return _singleton;
}

void AnimationTreeHandlerPlugin::_bind_methods() {
}
