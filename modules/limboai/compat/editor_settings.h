/**************************************************************************/
/*  editor_settings.h                                                     */
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

/**
 * editor_settings.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef COMPAT_EDITOR_SETTINGS_H
#define COMPAT_EDITOR_SETTINGS_H

#ifdef LIMBOAI_MODULE

#ifdef TOOLS_ENABLED
#include "editor/editor_interface.h"
#include "editor/settings/editor_settings.h"
#endif // TOOLS_ENABLED

#define EDITOR_SETTINGS() (EditorSettings::get_singleton())

#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION

#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/editor_settings.hpp>
using namespace godot;

#define EDITOR_SETTINGS() (EditorInterface::get_singleton()->get_editor_settings())

#define EDITOR_GET(m_var) _EDITOR_GET(m_var)
Variant _EDITOR_GET(const String &p_setting);

#define EDITOR_DEF(m_setting, m_value)                                                                         \
	do { /* do-while(0) ideom to avoid any potential semicolon errors. */                                      \
		EditorInterface::get_singleton()->get_editor_settings()->set_initial_value(m_setting, m_value, false); \
		if (!EDITOR_SETTINGS()->has_setting(m_setting)) {                                                      \
			EDITOR_SETTINGS()->set_setting(m_setting, m_value);                                                \
		}                                                                                                      \
	} while (0)

#endif // LIMBOAI_GDEXTENSION

#endif // COMPAT_EDITOR_SETTINGS_H
