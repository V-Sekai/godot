/**************************************************************************/
/*  microthread.cpp                                                       */
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

#include "microthread.hpp"

extern "C" void microthread_set_tp(void *);

namespace microthread {
static Thread main_thread{ nullptr };

void trampoline(Thread *thread) {
	thread->startfunc();
}
void oneshot_exit() {
	auto *thread = self();
	// after this point stack unusable
	free((char *)thread + sizeof(Thread) - Thread::STACK_SIZE);
	syscall(THREAD_SYSCALLS_BASE + 1, 0);
	__builtin_unreachable();
}

/* glibc sets up its own main thread, *required* by C++ exceptions */
#ifndef __GLIBC__
__attribute__((constructor, used)) static void init_threads() {
	microthread_set_tp(&main_thread);
}
#endif
} //namespace microthread

asm(".section .text\n"
	".global microthread_set_tp\n"
	".type microthread_set_tp, @function\n"
	"microthread_set_tp:\n"
	"  mv tp, a0\n"
	"  ret\n");

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

// This function never returns (so no ret)
asm(".global threadcall_destructor\n"
	".type threadcall_destructor, @function\n"
	"threadcall_destructor:\n"
	"	li a7, " STRINGIFY(THREAD_SYSCALLS_BASE + 9) "\n"
													 "	ecall\n");
