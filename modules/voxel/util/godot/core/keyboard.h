/**************************************************************************/
/*  keyboard.h                                                            */
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

// Key enums are not defined the same way between Godot and GDExtension.
// This defines aliases so using them is the same in both module and extension builds.

#if defined(ZN_GODOT)
#include <core/os/keyboard.h>

// Expose as in GodotCpp

namespace godot {
static const KeyModifierMask KEY_CODE_MASK = KeyModifierMask::CODE_MASK;
static const KeyModifierMask KEY_MODIFIER_MASK = KeyModifierMask::MODIFIER_MASK;
static const KeyModifierMask KEY_MASK_CMD_OR_CTRL = KeyModifierMask::CMD_OR_CTRL;
static const KeyModifierMask KEY_MASK_SHIFT = KeyModifierMask::SHIFT;
static const KeyModifierMask KEY_MASK_ALT = KeyModifierMask::ALT;
static const KeyModifierMask KEY_MASK_META = KeyModifierMask::META;
static const KeyModifierMask KEY_MASK_CTRL = KeyModifierMask::CTRL;
static const KeyModifierMask KEY_MASK_KPAD = KeyModifierMask::KPAD;
static const KeyModifierMask KEY_MASK_GROUP_SWITCH = KeyModifierMask::GROUP_SWITCH;

static const Key KEY_NONE = Key::NONE;
static const Key KEY_R = Key::R;
static const Key KEY_UP = Key::UP;
static const Key KEY_DOWN = Key::DOWN;
static const Key KEY_ENTER = Key::ENTER;
}; // namespace godot

#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/global_constants.hpp>

// TODO GDX: The operator `Key & KeyModifierMask` is defined in core, but not in GDExtension...
constexpr godot::Key operator&(godot::Key a, godot::KeyModifierMask b) {
	return (godot::Key)((int)a & (int)b);
}

// TODO GDX: The operator `Key | KeyModifierMask` is defined in core, but not in GDExtension...
constexpr godot::Key operator|(godot::KeyModifierMask a, godot::Key b) {
	return (godot::Key)((int)a | (int)b);
}

// TODO GDX: The operator `KeyModifierMask | KeyModifierMask` is defined in core, but not in GDExtension...
constexpr godot::KeyModifierMask operator|(godot::KeyModifierMask a, godot::KeyModifierMask b) {
	return (godot::KeyModifierMask)((int)a | (int)b);
}

#endif
