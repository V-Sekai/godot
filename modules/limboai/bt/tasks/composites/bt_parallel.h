/**************************************************************************/
/*  bt_parallel.h                                                         */
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
 * bt_parallel.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef BT_PARALLEL_H
#define BT_PARALLEL_H

#include "../bt_composite.h"

class BTParallel : public BTComposite {
	GDCLASS(BTParallel, BTComposite);
	TASK_CATEGORY(Composites);

private:
	int num_successes_required = 1;
	int num_failures_required = 1;
	bool repeat = false;

protected:
	static void _bind_methods();

	virtual void _enter() override;
	virtual Status _tick(double p_delta) override;

public:
	int get_num_successes_required() const { return num_successes_required; }
	void set_num_successes_required(int p_value) {
		num_successes_required = p_value;
		emit_changed();
	}
	int get_num_failures_required() const { return num_failures_required; }
	void set_num_failures_required(int p_value) {
		num_failures_required = p_value;
		emit_changed();
	}
	bool get_repeat() const { return repeat; }
	void set_repeat(bool p_value) {
		repeat = p_value;
		emit_changed();
	}
};

#endif // BT_PARALLEL_H
