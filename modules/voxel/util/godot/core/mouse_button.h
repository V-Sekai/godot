/**************************************************************************/
/*  mouse_button.h                                                        */
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

#pragma once

#if defined(ZN_GODOT)
#include <core/input/input_enums.h>

#define ZN_GODOT_MouseButton_NONE MouseButton::NONE
#define ZN_GODOT_MouseButton_WHEEL_UP MouseButton::WHEEL_UP
#define ZN_GODOT_MouseButton_WHEEL_DOWN MouseButton::WHEEL_DOWN
#define ZN_GODOT_MouseButtonMask_MIDDLE MouseButtonMask::MIDDLE

#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/global_constants.hpp>
using namespace godot;

#define ZN_GODOT_MouseButton_NONE MouseButton::MOUSE_BUTTON_NONE
#define ZN_GODOT_MouseButton_WHEEL_UP MouseButton::MOUSE_BUTTON_WHEEL_UP
#define ZN_GODOT_MouseButton_WHEEL_DOWN MouseButton::MOUSE_BUTTON_WHEEL_DOWN
#define ZN_GODOT_MouseButtonMask_MIDDLE MouseButtonMask::MOUSE_BUTTON_MASK_MIDDLE

#endif
