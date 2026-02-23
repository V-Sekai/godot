/**************************************************************************/
/*  test_probability.h                                                    */
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
 * test_probability.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_PROBABILITY_H
#define TEST_PROBABILITY_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_probability.h"

#include "core/math/math_funcs.h"

namespace TestProbability {

TEST_CASE("[Modules][LimboAI] BTProbability") {
	Ref<BTProbability> prob = memnew(BTProbability);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(prob->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction);
	prob->add_child(task);

	Math::randomize();

	SUBCASE("Check if probability meets expectation") {
		task->ret_status = BTTask::SUCCESS;
		prob->set_run_chance(0.5);

		for (int i = 0; i < 1000; i++) {
			prob->execute(0.01666);
		}

		CHECK(task->num_ticks > 450);
		CHECK(task->num_ticks < 550);
	}

	SUBCASE("When probability is 0") {
		task->ret_status = BTTask::SUCCESS;
		prob->set_run_chance(0.0);

		for (int i = 0; i < 1000; i++) {
			prob->execute(0.01666);
		}

		CHECK(task->num_ticks == 0);
	}

	SUBCASE("When probability is 1") {
		task->ret_status = BTTask::SUCCESS;
		prob->set_run_chance(1.0);

		for (int i = 0; i < 1000; i++) {
			prob->execute(0.01666);
		}

		CHECK(task->num_ticks == 1000);
	}

	SUBCASE("Test return status") {
		prob->set_run_chance(1.0);

		task->ret_status = BTTask::SUCCESS;
		CHECK(prob->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 1, 1);

		task->ret_status = BTTask::RUNNING;
		CHECK(prob->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 2, 2, 1);

		task->ret_status = BTTask::FAILURE;
		CHECK(prob->execute(0.01666) == BTTask::FAILURE);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 2, 3, 2);
	}
}

} //namespace TestProbability

#endif // TEST_PROBABILITY_H
