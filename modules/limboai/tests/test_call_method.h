/**************************************************************************/
/*  test_call_method.h                                                    */
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
 * test_call_method.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_CALL_METHOD_H
#define TEST_CALL_METHOD_H

#include "limbo_test.h"

#include "modules/limboai/blackboard/bb_param/bb_node.h"
#include "modules/limboai/blackboard/blackboard.h"
#include "modules/limboai/bt/tasks/bt_task.h"
#include "modules/limboai/bt/tasks/utility/bt_call_method.h"

#include "core/os/memory.h"
#include "core/variant/array.h"

namespace TestCallMethod {

TEST_CASE("[Modules][LimboAI] BTCallMethod") {
	Ref<BTCallMethod> cm = memnew(BTCallMethod);

	SUBCASE("When node parameter is null") {
		cm->set_node_param(nullptr);
		cm->set_method("test");
		ERR_PRINT_OFF;
		CHECK(cm->execute(0.01666) == BTTask::FAILURE);
		ERR_PRINT_ON;
	}

	SUBCASE("With object on the blackboard") {
		Node *dummy = memnew(Node);
		Ref<Blackboard> bb = memnew(Blackboard);

		Ref<BBNode> node_param = memnew(BBNode);
		cm->set_node_param(node_param);
		Ref<CallbackCounter> callback_counter = memnew(CallbackCounter);
		bb->set_var("object", callback_counter);
		node_param->set_value_source(BBParam::BLACKBOARD_VAR);
		node_param->set_variable("object");
		cm->set_method("callback");

		cm->initialize(dummy, bb, dummy);

		SUBCASE("When method is empty") {
			cm->set_method("");
			ERR_PRINT_OFF;
			CHECK(cm->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
		}
		SUBCASE("When method doesn't exist") {
			cm->set_method("not_found");
			ERR_PRINT_OFF;
			CHECK(cm->execute(0.01666) == BTTask::FAILURE);
			ERR_PRINT_ON;
		}
		SUBCASE("When method exists") {
			CHECK(cm->execute(0.01666) == BTTask::SUCCESS);
			CHECK(callback_counter->num_callbacks == 1);
		}
		SUBCASE("With arguments") {
			cm->set_method("callback_delta");

			SUBCASE("Should fail with 0 arguments") {
				cm->set_include_delta(false);
				cm->set_args(TypedArray<BBVariant>());
				ERR_PRINT_OFF;
				CHECK(cm->execute(0.01666) == BTTask::FAILURE);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 0);
			}
			SUBCASE("Should fail with too many arguments") {
				cm->set_include_delta(true);
				TypedArray<BBVariant> args;
				args.push_back(memnew(BBVariant(0.2)));
				cm->set_args(args);
				ERR_PRINT_OFF;
				CHECK(cm->execute(0.01666) == BTTask::FAILURE);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 0);
			}
			SUBCASE("Should fail with a wrong type arg") {
				cm->set_include_delta(false);
				TypedArray<BBVariant> args;
				args.push_back(memnew(BBVariant("wrong data type")));
				cm->set_args(args);
				ERR_PRINT_OFF;
				CHECK(cm->execute(0.01666) == BTTask::FAILURE);
				ERR_PRINT_ON;
				CHECK(callback_counter->num_callbacks == 1);
			}
			SUBCASE("Should succeed with delta included") {
				cm->set_include_delta(true);
				CHECK(cm->execute(0.01666) == BTTask::SUCCESS);
				CHECK(callback_counter->num_callbacks == 1);
			}
			SUBCASE("Should succeed with one float arg") {
				cm->set_include_delta(false);
				TypedArray<BBVariant> args;
				args.push_back(memnew(BBVariant(0.2)));
				cm->set_args(args);
				CHECK(cm->execute(0.01666) == BTTask::SUCCESS);
				CHECK(callback_counter->num_callbacks == 1);
			}
		}

		memdelete(dummy);
	}
}

} //namespace TestCallMethod

#endif // TEST_CALL_METHOD_H
