/**************************************************************************/
/*  newlib_open.hpp                                                       */
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

const auto g_path = machine.sysarg(0);
const int flags = machine.template sysarg<int>(1);
const int mode = machine.template sysarg<int>(2);
// This is a custom syscall for the newlib mini guest
std::string path = machine.memory.memstring(g_path);
if (machine.has_file_descriptors() && machine.fds().permit_filesystem) {
	if (machine.fds().filter_open != nullptr) {
		// filter_open() can modify the path
		if (!machine.fds().filter_open(machine.template get_userdata<void>(), path)) {
			machine.set_result(-EPERM);
			return;
		}

#if !defined(_MSC_VER)
		int res = open(path.c_str(), flags, mode);
		if (res > 0)
			res = machine.fds().assign_file(res);
		machine.set_result_or_error(res);
		return;
#endif
	}
}
machine.set_result(-EPERM);
return;
