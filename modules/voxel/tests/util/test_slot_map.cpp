/**************************************************************************/
/*  test_slot_map.cpp                                                     */
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

#include "test_slot_map.h"
#include "../../util/containers/slot_map.h"
#include "../../util/testing/test_macros.h"

namespace zylann::tests {

void test_slot_map() {
	SlotMap<int> map;

	const SlotMap<int>::Key key1 = map.add(1);
	const SlotMap<int>::Key key2 = map.add(2);
	const SlotMap<int>::Key key3 = map.add(3);

	ZN_TEST_ASSERT(key1 != key2 && key2 != key3);
	ZN_TEST_ASSERT(map.exists(key1));
	ZN_TEST_ASSERT(map.exists(key2));
	ZN_TEST_ASSERT(map.exists(key3));
	ZN_TEST_ASSERT(map.count() == 3);

	map.remove(key2);
	ZN_TEST_ASSERT(map.exists(key1));
	ZN_TEST_ASSERT(!map.exists(key2));
	ZN_TEST_ASSERT(map.exists(key3));
	ZN_TEST_ASSERT(map.count() == 2);

	const SlotMap<int>::Key key4 = map.add(4);
	ZN_TEST_ASSERT(key4 != key2);
	ZN_TEST_ASSERT(map.count() == 3);

	const int v1 = map.get(key1);
	const int v4 = map.get(key4);
	const int v3 = map.get(key3);
	ZN_TEST_ASSERT(v1 == 1);
	ZN_TEST_ASSERT(v4 == 4);
	ZN_TEST_ASSERT(v3 == 3);

	map.clear();
	ZN_TEST_ASSERT(!map.exists(key1));
	ZN_TEST_ASSERT(!map.exists(key4));
	ZN_TEST_ASSERT(!map.exists(key3));
	ZN_TEST_ASSERT(map.count() == 0);
}

} // namespace zylann::tests
