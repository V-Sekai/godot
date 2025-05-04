/**************************************************************************/
/*  input_event_codec.h                                                   */
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

#include "core/input/input_event.h"

/// @brief Encodes an InputEvent from the specified byte array.
/// @param p_event The event to encode.
/// @param r_data The byte array to encode the event into.
/// @return True if the event was successfully encoded, false otherwise.
bool encode_input_event(const Ref<InputEvent> &p_event, PackedByteArray &r_data);

/// @brief Decodes an InputEvent from the specified byte array.
///
/// @note If the event type is not recognized, the event will remain invalid.
///
/// @param p_data The byte array to decode the event from.
/// @param r_event The event to decode into.
void decode_input_event(const PackedByteArray &p_data, Ref<InputEvent> &r_event);

/// @brief Encodes an InputKeyEvent to the specified byte array.
/// @param p_event The event to encode.
/// @param r_data The byte array to encode the event into.
void encode_input_event_key(const Ref<InputEventKey> &p_event, PackedByteArray &r_data);

/// @brief Encodes an InputMouseButtonEvent to the specified byte array.
/// @param p_event The event to encode.
/// @param r_data The byte array to encode the event into.
void encode_input_event_mouse_button(const Ref<InputEventMouseButton> &p_event, PackedByteArray &r_data);

/// @brief Encodes an InputMouseButtonMotion to the specified byte array.
/// @param p_event The event to encode.
/// @param r_data The byte array to encode the event into.
void encode_input_event_mouse_motion(const Ref<InputEventMouseMotion> &p_event, PackedByteArray &r_data);

/// @brief Encodes an InputEventJoypadButten to the specified byte array.
/// @param p_event The event to encode.
/// @param r_data The byte array to encode the event into.
void encode_input_event_joypad_button(const Ref<InputEventJoypadButton> &p_event, PackedByteArray &r_data);

/// @brief Encodes an InputEventJoypadMotion to the specified byte array.
/// @param p_event The event to encode.
/// @param r_data The byte array to encode the event into.
void encode_input_event_joypad_motion(const Ref<InputEventJoypadMotion> &p_event, PackedByteArray &r_data);

/// @brief Encodes an InputEventPanGesture to the specified byte array.
/// @param p_event The event to encode.
/// @param r_data The byte array to encode the event into.
void encode_input_event_gesture_pan(const Ref<InputEventPanGesture> &p_event, PackedByteArray &r_data);

/// @brief Encodes an InputEventMagnifyGesture to the specified byte array.
/// @param p_event The event to encode.
/// @param r_data The byte array to encode the event into.
void encode_input_event_gesture_magnify(const Ref<InputEventMagnifyGesture> &p_event, PackedByteArray &r_data);
