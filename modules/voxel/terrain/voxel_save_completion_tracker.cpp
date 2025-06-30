/**************************************************************************/
/*  voxel_save_completion_tracker.cpp                                     */
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

#include "voxel_save_completion_tracker.h"

namespace zylann::voxel {

Ref<VoxelSaveCompletionTracker> VoxelSaveCompletionTracker::create(std::shared_ptr<AsyncDependencyTracker> tracker) {
	Ref<VoxelSaveCompletionTracker> self;
	self.instantiate();
	self->_tracker = tracker;
	self->_total_tasks = tracker->get_remaining_count();
	return self;
}

bool VoxelSaveCompletionTracker::is_complete() const {
	ZN_ASSERT_RETURN_V(_tracker != nullptr, false);
	return _tracker->is_complete();
}

bool VoxelSaveCompletionTracker::is_aborted() const {
	ZN_ASSERT_RETURN_V(_tracker != nullptr, false);
	return _tracker->is_aborted();
}

int VoxelSaveCompletionTracker::get_total_tasks() const {
	return _total_tasks;
}

int VoxelSaveCompletionTracker::get_remaining_tasks() const {
	ZN_ASSERT_RETURN_V(_tracker != nullptr, 0);
	return _tracker->get_remaining_count();
}

void VoxelSaveCompletionTracker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_complete"), &VoxelSaveCompletionTracker::is_complete);
	ClassDB::bind_method(D_METHOD("is_aborted"), &VoxelSaveCompletionTracker::is_aborted);
	ClassDB::bind_method(D_METHOD("get_total_tasks"), &VoxelSaveCompletionTracker::get_total_tasks);
	ClassDB::bind_method(D_METHOD("get_remaining_tasks"), &VoxelSaveCompletionTracker::get_remaining_tasks);
}

} // namespace zylann::voxel
