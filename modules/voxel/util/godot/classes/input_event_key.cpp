/**************************************************************************/
/*  input_event_key.cpp                                                   */
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

#include "input_event_key.h"
#include "../core/keyboard.h"

namespace zylann::godot {

Ref<InputEventKey> create_input_event_from_key(Key p_keycode_with_modifier_masks, bool p_physical) {
#if defined(ZN_GODOT)
	return InputEventKey::create_reference(p_keycode_with_modifier_masks, p_physical);

#elif defined(ZN_GODOT_EXTENSION)
	Key p_keycode = p_keycode_with_modifier_masks;

	// Ported from core `InputEventKey::create_reference`

	Ref<InputEventKey> ie;
	ie.instantiate();
	if (p_physical) {
		ie->set_physical_keycode(p_keycode & ::godot::KEY_CODE_MASK);
	} else {
		ie->set_keycode(p_keycode & ::godot::KEY_CODE_MASK);
	}

	ie->set_unicode(char32_t(p_keycode & ::godot::KEY_CODE_MASK));

	if ((p_keycode & ::godot::KEY_MASK_SHIFT) != ::godot::KEY_NONE) {
		ie->set_shift_pressed(true);
	}
	if ((p_keycode & ::godot::KEY_MASK_ALT) != ::godot::KEY_NONE) {
		ie->set_alt_pressed(true);
	}
	if ((p_keycode & ::godot::KEY_MASK_CMD_OR_CTRL) != ::godot::KEY_NONE) {
		ie->set_command_or_control_autoremap(true);
		if ((p_keycode & ::godot::KEY_MASK_CTRL) != ::godot::KEY_NONE ||
			(p_keycode & ::godot::KEY_MASK_META) != ::godot::KEY_NONE) {
			WARN_PRINT(
					"Invalid Key Modifiers: Command or Control autoremapping is enabled, Meta and Control values "
					"are ignored!"
			);
		}
	} else {
		if ((p_keycode & ::godot::KEY_MASK_CTRL) != ::godot::KEY_NONE) {
			ie->set_ctrl_pressed(true);
		}
		if ((p_keycode & ::godot::KEY_MASK_META) != ::godot::KEY_NONE) {
			ie->set_meta_pressed(true);
		}
	}

	return ie;
#endif
}

} // namespace zylann::godot
