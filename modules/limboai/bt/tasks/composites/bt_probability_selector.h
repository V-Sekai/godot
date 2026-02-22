/**************************************************************************/
/*  bt_probability_selector.h                                             */
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
 * bt_probability_selector.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_PROBABILITY_SELECTOR_H
#define BT_PROBABILITY_SELECTOR_H

#include "../../../util/limbo_string_names.h"
#include "../bt_composite.h"

#ifdef LIMBOAI_MODULE
#include "core/templates/hash_set.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/templates/hash_set.hpp>
#endif // LIMBOAI_GDEXTENSION

class BTProbabilitySelector : public BTComposite {
	GDCLASS(BTProbabilitySelector, BTComposite);
	TASK_CATEGORY(Composites);

private:
	HashSet<Ref<BTTask>> failed_tasks;
	Ref<BTTask> selected_task;
	bool abort_on_failure = false;

	void _select_task();
	_FORCE_INLINE_ double _get_weight(int p_index) const { return get_child(p_index)->get_meta(LW_NAME(_weight_), 1.0); }
	_FORCE_INLINE_ double _get_weight(Ref<BTTask> p_task) const { return p_task->get_meta(LW_NAME(_weight_), 1.0); }
	_FORCE_INLINE_ void _set_weight(int p_index, double p_weight) {
		get_child(p_index)->set_meta(LW_NAME(_weight_), Variant(p_weight));
		get_child(p_index)->emit_signal(LW_NAME(changed));
	}
	_FORCE_INLINE_ double _get_total_weight() const {
		double total = 0.0;
		for (int i = 0; i < get_child_count(); i++) {
			if (get_child(i)->is_enabled()) {
				total += _get_weight(i);
			}
		}
		return total;
	}

protected:
	static void _bind_methods();

	virtual void _enter() override;
	virtual void _exit() override;
	virtual Status _tick(double p_delta) override;

public:
	double get_weight(int p_index) const;
	void set_weight(int p_index, double p_weight);
	double get_total_weight() const { return _get_total_weight(); };

	double get_probability(int p_index) const;
	void set_probability(int p_index, double p_probability);

	bool has_probability(int p_index) const;

	void set_abort_on_failure(bool p_abort_on_failure);
	bool get_abort_on_failure() const;
};

#endif // BT_PROBABILITY_SELECTOR_H
