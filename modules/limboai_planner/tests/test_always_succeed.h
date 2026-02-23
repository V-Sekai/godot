/**************************************************************************/
/*  test_always_succeed.h                                                 */
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
 * test_always_succeed.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_ALWAYS_SUCCEED_H
#define TEST_ALWAYS_SUCCEED_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/decorators/bt_always_succeed.h"

namespace TestAlwaysSucceed {

TEST_CASE("[Modules][LimboAI] BTAlwaysSucceed") {
	Ref<BTAlwaysSucceed> as = memnew(BTAlwaysSucceed);

	SUBCASE("When empty") {
		CHECK(as->execute(0.01666) == BTTask::SUCCESS);
	}

	Ref<BTTestAction> task = memnew(BTTestAction);

	as->add_child(task);

	SUBCASE("When child returns FAILURE") {
		task->ret_status = BTTask::FAILURE;

		CHECK(as->execute(0.01666) == BTTask::SUCCESS);

		CHECK(task->get_status() == BTTask::FAILURE);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
	}

	SUBCASE("When child returns SUCCESS") {
		task->ret_status = BTTask::SUCCESS;

		CHECK(as->execute(0.01666) == BTTask::SUCCESS);

		CHECK(task->get_status() == BTTask::SUCCESS);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 1);
	}

	SUBCASE("When child returns RUNNING") {
		task->ret_status = BTTask::RUNNING;

		CHECK(as->execute(0.01666) == BTTask::RUNNING);

		CHECK(task->get_status() == BTTask::RUNNING);
		CHECK_ENTRIES_TICKS_EXITS(task, 1, 1, 0);
	}
}

} //namespace TestAlwaysSucceed

#endif // TEST_ALWAYS_SUCCEED_H
