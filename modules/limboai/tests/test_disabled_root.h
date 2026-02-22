/**************************************************************************/
/*  test_disabled_root.h                                                  */
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
 * test_disabled_root.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef TEST_DISABLED_ROOT_H
#define TEST_DISABLED_ROOT_H

#include "limbo_test.h"

#include "modules/limboai/blackboard/blackboard.h"
#include "modules/limboai/bt/behavior_tree.h"
#include "modules/limboai/bt/bt_instance.h"

namespace TestDisabledRoot {

TEST_CASE("[SceneTree][LimboAI] BTInstance when root task disabled FAILURE") {
	Ref<BehaviorTree> bt = memnew(BehaviorTree);
	Ref<BTTestAction> task = memnew(BTTestAction(BTTask::SUCCESS));
	Node *dummy = memnew(Node);

	SceneTree::get_singleton()->get_root()->add_child(dummy);
	dummy->set_owner(SceneTree::get_singleton()->get_root());
	task->set_enabled(false);
	bt->set_root_task(task);
	Ref<Blackboard> blackboard = memnew(Blackboard);
	Ref<BTInstance> bti = bt->instantiate(dummy, blackboard, dummy);

	CHECK(bti->update(0.1666) == BTTask::FAILURE);
	CHECK(task->get_status() == BTTask::FRESH);

	SceneTree::get_singleton()->get_root()->remove_child(dummy);
	memdelete(dummy);
}

} //namespace TestDisabledRoot

#endif // TEST_DISABLED_ROOT_H
