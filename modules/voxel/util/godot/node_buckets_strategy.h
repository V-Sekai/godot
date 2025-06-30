/**************************************************************************/
/*  node_buckets_strategy.h                                               */
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

#include "../containers/std_vector.h"
#include "../errors.h"
#include "macros.h"

ZN_GODOT_FORWARD_DECLARE(class Node)

namespace zylann::godot {

// A workaround for the fact Godot is very slow at removing nodes from the scene tree if they have many siblings...
// See https://github.com/godotengine/godot/issues/61929
template <typename TBucket>
class NodeBucketsStrategy {
public:
	static const unsigned int BUCKET_SIZE = 20;

	NodeBucketsStrategy(Node &parent) : _parent(parent) {}

	void add_child(Node *node) {
		ZN_ASSERT(node != nullptr);
		if (_free_buckets.size() == 0) {
			TBucket *bucket = memnew(TBucket);
			_used_buckets.push_back(bucket);
			bucket->add_child(node);
			_parent.add_child(bucket);
		} else {
			TBucket *bucket = _free_buckets.back();
			_free_buckets.pop_back();
			_used_buckets.push_back(bucket);
			bucket->add_child(node);
		}
	}

	void remove_empty_buckets() {
		for (unsigned int i = 0; i < _used_buckets.size();) {
			TBucket *bucket = _used_buckets[i];
			if (bucket->get_child_count() == 0) {
				bucket->queue_free();
				_used_buckets[i] = _used_buckets.back();
				_used_buckets.pop_back();
			} else {
				++i;
			}
		}
	}

private:
	Node &_parent;
	StdVector<TBucket *> _free_buckets;
	StdVector<TBucket *> _used_buckets;
};

} // namespace zylann::godot
