/**************************************************************************/
/*  test_delay.h                                                          */
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
 * test_delay.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_DELAY_H
#define TEST_DELAY_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_delay.h"

namespace TestDelay {

TEST_CASE("[Modules][LimboAI] BTDelay") {
	Ref<BTDelay> del = memnew(BTDelay);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(del->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction(BTTask::SUCCESS));
	del->add_child(task);

	SUBCASE("Check if delay is observed correctly") {
		del->set_seconds(1.0);
		CHECK(del->execute(0.35) == BTTask::RUNNING); // * first execution: elapsed 0.0
		CHECK_ENTRIES_TICKS_EXITS(task, 0, 0, 0);
		CHECK(del->execute(0.35) == BTTask::RUNNING); // * second execution: elapsed 0.35
		CHECK_ENTRIES_TICKS_EXITS(task, 0, 0, 0);
		CHECK(del->execute(0.35) == BTTask::RUNNING); // * third execution: elapsed 0.7
		CHECK_ENTRIES_TICKS_EXITS(task, 0, 0, 0);

		SUBCASE("When child task returns SUCCESS") {
			task->ret_status = BTTask::SUCCESS;
			CHECK(del->execute(0.35) == BTTask::SUCCESS); // * fourth execution: elapsed 1.05
			CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
		}
		SUBCASE("When child task returns FAILURE") {
			task->ret_status = BTTask::FAILURE;
			CHECK(del->execute(0.35) == BTTask::FAILURE); // * fourth execution: elapsed 1.05
			CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
		}
		SUBCASE("When child task returns RUNNING") {
			task->ret_status = BTTask::RUNNING;
			CHECK(del->execute(0.35) == BTTask::RUNNING); // * fourth execution: elapsed 1.05
			CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 0);

			task->ret_status = BTTask::SUCCESS;
			CHECK(del->execute(0.35) == BTTask::SUCCESS); // * fifth execution: elapsed 1.4
			CHECK_ENTRIES_TICKS_EXITS(task, 1, 2, 1);
		}
	}
}

} //namespace TestDelay

#endif // TEST_DELAY_H
