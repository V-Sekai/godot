/**************************************************************************/
/*  tr_api.hpp                                                            */
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

#ifndef TR_API_HPP
#define TR_API_HPP
#include <cstdint>

namespace riscv {
template <int W>
using syscall_t = void (*)(Machine<W> &);

template <int W>
struct CallbackTable {
	address_type<W> (*mem_read)(CPU<W> &, address_type<W> addr, unsigned size);
	void (*mem_write)(CPU<W> &, address_type<W> addr, address_type<W> value, unsigned size);
	void (*vec_load)(CPU<W> &, int vd, address_type<W> addr);
	void (*vec_store)(CPU<W> &, address_type<W> addr, int vd);
	syscall_t<W> *syscalls;
	int (*system_call)(CPU<W> &, int);
	void (*unknown_syscall)(CPU<W> &, address_type<W>);
	void (*system)(CPU<W> &, uint32_t);
	unsigned (*execute)(CPU<W> &, uint32_t);
	unsigned (*execute_handler)(CPU<W> &, unsigned, uint32_t);
	void (**handlers)(CPU<W> &, uint32_t);
	void (*trigger_exception)(CPU<W> &, address_type<W>, int);
	void (*trace)(CPU<W> &, const char *, address_type<W>, uint32_t);
	float (*sqrtf32)(float);
	double (*sqrtf64)(double);
	int (*clz)(uint32_t);
	int (*clzl)(uint64_t);
	int (*ctz)(uint32_t);
	int (*ctzl)(uint64_t);
	int (*cpop)(uint32_t);
	int (*cpopl)(uint64_t);
};
} //namespace riscv

#endif // TR_API_HPP
