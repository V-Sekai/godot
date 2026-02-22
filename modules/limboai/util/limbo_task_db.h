/**************************************************************************/
/*  limbo_task_db.h                                                       */
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
 * limbo_task_db.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBO_TASK_DB_H
#define LIMBO_TASK_DB_H

#ifdef LIMBOAI_MODULE
#include "core/object/class_db.h"
#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/templates/hash_map.hpp>
#include <godot_cpp/templates/list.hpp>
#include <godot_cpp/variant/string.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

class LimboTaskDB {
private:
	static HashMap<String, List<String>> core_tasks;
	static HashMap<String, List<String>> tasks_cache;

	struct ComparatorByTaskName {
		bool operator()(const String &p_left, const String &p_right) const {
			return get_task_name(p_left) < get_task_name(p_right);
		}
	};

public:
	template <class T>
	static void register_task() {
		GDREGISTER_CLASS(T);
		HashMap<String, List<String>>::Iterator E = core_tasks.find(T::get_task_category());
		if (E) {
			E->value.push_back(T::get_class_static());
		} else {
			List<String> tasks;
			tasks.push_back(T::get_class_static());
			core_tasks.insert(T::get_task_category(), tasks);
		}
	}

	static void scan_user_tasks();
	static _FORCE_INLINE_ String get_misc_category() { return "Misc"; }
	static List<String> get_categories();
	static List<String> get_tasks_in_category(const String &p_category);
	static _FORCE_INLINE_ String get_task_name(String p_class_or_script_path) {
		if (p_class_or_script_path.begins_with("res:")) {
			return p_class_or_script_path.get_file().get_basename().trim_prefix("BT").to_pascal_case();
		} else {
			return p_class_or_script_path.trim_prefix("BT");
		}
	}
};

#ifdef LIMBOAI_MODULE
#define LIMBO_REGISTER_TASK(m_class)              \
	if constexpr (GD_IS_CLASS_ENABLED(m_class)) { \
		::LimboTaskDB::register_task<m_class>();  \
	}
#elif LIMBOAI_GDEXTENSION
#define LIMBO_REGISTER_TASK(m_class) LimboTaskDB::register_task<m_class>();
#endif

#define TASK_CATEGORY(m_cat)                           \
public:                                                \
	static _FORCE_INLINE_ String get_task_category() { \
		return String(#m_cat);                         \
	}                                                  \
                                                       \
private:

#endif // LIMBO_TASK_DB_H
