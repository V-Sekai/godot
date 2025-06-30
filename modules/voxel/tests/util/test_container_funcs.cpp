/**************************************************************************/
/*  test_container_funcs.cpp                                              */
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

#include "test_container_funcs.h"
#include "../../util/containers/container_funcs.h"
#include "../../util/containers/std_vector.h"
#include "../../util/testing/test_macros.h"

namespace zylann::tests {

void test_unordered_remove_if() {
	struct L {
		static unsigned int count(const StdVector<int> &vec, int v) {
			unsigned int n = 0;
			for (size_t i = 0; i < vec.size(); ++i) {
				if (vec[i] == v) {
					++n;
				}
			}
			return n;
		}
	};
	// Remove one at beginning
	{
		StdVector<int> vec;
		vec.push_back(0);
		vec.push_back(1);
		vec.push_back(2);
		vec.push_back(3);

		unordered_remove_if(vec, [](int v) { return v == 0; });

		ZN_TEST_ASSERT(vec.size() == 3);
		ZN_TEST_ASSERT(
				L::count(vec, 0) == 0 && L::count(vec, 1) == 1 && L::count(vec, 2) == 1 && L::count(vec, 3) == 1
		);
	}
	// Remove one in middle
	{
		StdVector<int> vec;
		vec.push_back(0);
		vec.push_back(1);
		vec.push_back(2);
		vec.push_back(3);

		unordered_remove_if(vec, [](int v) { return v == 2; });

		ZN_TEST_ASSERT(vec.size() == 3);
		ZN_TEST_ASSERT(
				L::count(vec, 0) == 1 && L::count(vec, 1) == 1 && L::count(vec, 2) == 0 && L::count(vec, 3) == 1
		);
	}
	// Remove one at end
	{
		StdVector<int> vec;
		vec.push_back(0);
		vec.push_back(1);
		vec.push_back(2);
		vec.push_back(3);

		unordered_remove_if(vec, [](int v) { return v == 3; });

		ZN_TEST_ASSERT(vec.size() == 3);
		ZN_TEST_ASSERT(
				L::count(vec, 0) == 1 && L::count(vec, 1) == 1 && L::count(vec, 2) == 1 && L::count(vec, 3) == 0
		);
	}
	// Remove multiple
	{
		StdVector<int> vec;
		vec.push_back(0);
		vec.push_back(1);
		vec.push_back(2);
		vec.push_back(3);

		unordered_remove_if(vec, [](int v) { return v == 1 || v == 2; });

		ZN_TEST_ASSERT(vec.size() == 2);
		ZN_TEST_ASSERT(
				L::count(vec, 0) == 1 && L::count(vec, 1) == 0 && L::count(vec, 2) == 0 && L::count(vec, 3) == 1
		);
	}
	// Remove last
	{
		StdVector<int> vec;
		vec.push_back(0);

		unordered_remove_if(vec, [](int v) { return v == 0; });

		ZN_TEST_ASSERT(vec.size() == 0);
	}
}

} // namespace zylann::tests
