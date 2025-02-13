/**************************************************************************/
/*  signals.cpp                                                           */
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

#include "signals.hpp"

#include "../internal_common.hpp"
#include "../machine.hpp"
#include "../threads.hpp"

namespace riscv {

template <int W>
Signals<W>::Signals() {}
template <int W>
Signals<W>::~Signals() {}

template <int W>
SignalAction<W> &Signals<W>::get(int sig) {
	if (sig > 0)
		return signals.at(sig - 1);
	throw MachineException(ILLEGAL_OPERATION, "Signal 0 invoked");
}

template <int W>
void Signals<W>::enter(Machine<W> &machine, int sig) {
	if (sig == 0)
		return;

	auto &sigact = signals.at(sig);
	if (sigact.altstack) {
		auto *thread = machine.threads().get_thread();
		// Change to alternate per-thread stack
		auto &stack = per_thread(thread->tid).stack;
		machine.cpu.reg(REG_SP) = stack.ss_sp + stack.ss_size;
	}
	// We have to jump to handler-4 because we are mid-instruction
	// WARNING: Assumption.
	machine.cpu.jump(sigact.handler - 4);
}

INSTANTIATE_32_IF_ENABLED(Signals);
INSTANTIATE_64_IF_ENABLED(Signals);
INSTANTIATE_128_IF_ENABLED(Signals);
} //namespace riscv
