/**************************************************************************/
/*  test_run_limit.h                                                      */
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
 * test_run_limit.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_RUN_LIMIT_H
#define TEST_RUN_LIMIT_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_run_limit.h"

namespace TestRunLimit {

TEST_CASE("[Modules][LimboAI] BTRunLimit") {
	Ref<BTRunLimit> lim = memnew(BTRunLimit);

	SUBCASE("When empty") {
		ERR_PRINT_OFF;
		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	Ref<BTTestAction> task = memnew(BTTestAction);
	lim->add_child(task);

	SUBCASE("With run limit set to 2") {
		lim->set_run_limit(2);

		task->ret_status = BTTask::SUCCESS;
		CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 1, 1); // * task executed

		SUBCASE("When the child task succeeds") {
			task->ret_status = BTTask::SUCCESS;

			lim->set_count_policy(BTRunLimit::COUNT_FAILED);
			CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 2, 2, 2); // * task executed

			lim->set_count_policy(BTRunLimit::COUNT_SUCCESSFUL);
			CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 3, 3, 3); // * task executed
		}
		SUBCASE("When the child task fails") {
			task->ret_status = BTTask::FAILURE;

			lim->set_count_policy(BTRunLimit::COUNT_SUCCESSFUL);
			CHECK(lim->execute(0.01666) == BTTask::FAILURE);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 2, 2, 2); // * task executed

			lim->set_count_policy(BTRunLimit::COUNT_FAILED);
			CHECK(lim->execute(0.01666) == BTTask::FAILURE);
			CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::FAILURE, 3, 3, 3); // * task executed
		}

		task->ret_status = BTTask::SUCCESS;
		lim->set_count_policy(BTRunLimit::COUNT_SUCCESSFUL);

		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 3, 3, 3); // * task not executed

		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 3, 3, 3); // * task not executed
	}

	SUBCASE("When the child task takes more than one tick to finish") {
		lim->set_run_limit(2);
		lim->set_count_policy(BTRunLimit::COUNT_ALL);

		task->ret_status = BTTask::RUNNING;
		CHECK(lim->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 1, 1, 0);

		task->ret_status = BTTask::SUCCESS;
		CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 1, 2, 1);

		task->ret_status = BTTask::RUNNING;
		CHECK(lim->execute(0.01666) == BTTask::RUNNING);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::RUNNING, 2, 3, 1);

		task->ret_status = BTTask::SUCCESS;
		CHECK(lim->execute(0.01666) == BTTask::SUCCESS);
		CHECK_STATUS_ENTRIES_TICKS_EXITS(task, BTTask::SUCCESS, 2, 4, 2);

		CHECK(lim->execute(0.01666) == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 2, 4, 2);
	}
}

} //namespace TestRunLimit

#endif // TEST_RUN_LIMIT_H
