/**************************************************************************/
/*  test_check_trigger.h                                                  */
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
 * test_check_trigger.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_CHECK_TRIGGER_H
#define TEST_CHECK_TRIGGER_H

#include "limbo_test.h"

#include "modules/limboai/bt/tasks/blackboard/bt_check_trigger.h"
#include "modules/limboai/bt/tasks/bt_task.h"

namespace TestCheckTrigger {

TEST_CASE("[Modules][LimboAI] BTCheckTrigger") {
	Ref<BTCheckTrigger> ct = memnew(BTCheckTrigger);
	Node *dummy = memnew(Node);
	Ref<Blackboard> bb = memnew(Blackboard);

	ct->initialize(dummy, bb, dummy);

	SUBCASE("Empty") {
		ERR_PRINT_OFF;
		ct->set_variable("");
		CHECK(ct->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	ct->set_variable("trigger");

	SUBCASE("When variable is not found") {
		ERR_PRINT_OFF;
		CHECK(ct->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}
	SUBCASE("When variable set to false") {
		bb->set_var("trigger", false);
		CHECK(ct->execute(0.01666) == BTTask::FAILURE);
		CHECK(bb->get_var("trigger", false) == Variant(false));
	}
	SUBCASE("When variable set to true") {
		bb->set_var("trigger", true);
		CHECK(bb->get_var("trigger", false) == Variant(true));
		CHECK(ct->execute(0.01666) == BTTask::SUCCESS);
		CHECK(bb->get_var("trigger", false) == Variant(false));
	}
	SUBCASE("When variable set to non-bool") {
		bb->set_var("trigger", "Some text");
		CHECK(ct->execute(0.01666) == BTTask::FAILURE);
		CHECK(bb->get_var("trigger", Variant()) == "Some text");
	}

	memdelete(dummy);
}

} //namespace TestCheckTrigger

#endif // TEST_CHECK_TRIGGER_H
