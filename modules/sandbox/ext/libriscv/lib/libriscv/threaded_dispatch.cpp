/**************************************************************************/
/*  threaded_dispatch.cpp                                                 */
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

#include "common.hpp"
#include "internal_common.hpp"
#define DISPATCH_MODE_THREADED
#define DISPATCH_ATTR RISCV_HOT_PATH()
#define DISPATCH_FUNC simulate_threaded

#define EXECUTE_INSTR()                                         \
	if constexpr (FUZZING) {                                    \
		if (UNLIKELY(decoder->get_bytecode() >= BYTECODES_MAX)) \
			abort();                                            \
	}                                                           \
	goto *computed_opcode[decoder->get_bytecode()];
#define UNUSED_FUNCTION() \
	RISCV_UNREACHABLE();

#include "cpu_dispatch.cpp"

#include "cpu_inaccurate_dispatch.cpp"

namespace riscv {
INSTANTIATE_32_IF_ENABLED(CPU);
INSTANTIATE_64_IF_ENABLED(CPU);
INSTANTIATE_128_IF_ENABLED(CPU);
} //namespace riscv
