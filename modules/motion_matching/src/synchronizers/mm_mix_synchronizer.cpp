/**************************************************************************/
/*  mm_mix_synchronizer.cpp                                               */
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

#include "mm_mix_synchronizer.h"

#include "mm_character.h"

void MMMixSynchronizer::sync(MMCharacter *p_controller, Node3D *p_character, float p_delta_time) {
	Vector3 position_delta = p_character->get_global_position() - p_controller->get_global_position();

	Vector3 controller_position = p_controller->get_global_position() + position_delta * root_motion_amount;

	Vector3 character_position = p_character->get_global_position() - position_delta * (1.0 - root_motion_amount);

	controller_position.y = p_controller->get_global_position().y;
	character_position.y = p_controller->get_global_position().y;

	p_controller->set_global_position(controller_position);
	p_character->set_global_position(character_position);

	// p_character->set_global_rotation(character_rotation);
}

void MMMixSynchronizer::_bind_methods() {
	BINDER_PROPERTY_PARAMS(MMMixSynchronizer, Variant::FLOAT, root_motion_amount, PROPERTY_HINT_RANGE, "0.0,1.0,0.01");
}
