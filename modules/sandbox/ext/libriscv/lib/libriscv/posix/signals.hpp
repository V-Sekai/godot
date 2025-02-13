/**************************************************************************/
/*  signals.hpp                                                           */
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

#ifndef SIGNALS_HPP
#define SIGNALS_HPP
#include "../types.hpp"
#include <map>
#include <set>

namespace riscv {
template <int W>
struct Machine;
template <int W>
struct Registers;

template <int W>
struct SignalStack {
	address_type<W> ss_sp = 0x0;
	int ss_flags = 0x0;
	address_type<W> ss_size = 0;
};

template <int W>
struct SignalAction {
	static constexpr address_type<W> SIG_UNSET = ~(address_type<W>)0x0;
	bool is_unset() const noexcept {
		return handler == 0x0 || handler == SIG_UNSET;
	}
	address_type<W> handler = SIG_UNSET;
	bool altstack = false;
	unsigned mask = 0x0;
};

template <int W>
struct SignalReturn {
	Registers<W> regs;
};

template <int W>
struct SignalPerThread {
	SignalStack<W> stack;
	SignalReturn<W> sigret;
};

template <int W>
struct Signals {
	SignalAction<W> &get(int sig);
	void enter(Machine<W> &, int sig);

	// TODO: Lock this in the future, for multiproessing
	auto &per_thread(int tid) { return m_per_thread[tid]; }

	Signals();
	~Signals();

private:
	std::array<SignalAction<W>, 64> signals{};
	std::map<int, SignalPerThread<W>> m_per_thread;
};

} //namespace riscv

#endif // SIGNALS_HPP
