/**************************************************************************/
/*  start.cpp                                                             */
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

#include <stdlib.h>
#include <tinyprintf.h>
#include <cstdint>
#include <include/syscall.hpp>
static struct _reent reent;
struct _reent *_impure_ptr = &reent;
__FILE __sf[3];

extern "C" {
__attribute__((noreturn)) void _exit(int exitval);
void __print_putchr(void *file, char c);
}

static void
init_stdlib() {
	_REENT_INIT_PTR_ZEROED(_impure_ptr);

	// 2. Enable printf facilities
	init_printf(NULL, __print_putchr);

	// 3. Call global C/C++ constructors
	extern void (*__init_array_start[])();
	extern void (*__init_array_end[])();
	const int count = __init_array_end - __init_array_start;
	for (int i = 0; i < count; i++) {
		__init_array_start[i]();
	}
}

extern "C" __attribute__((visibility("hidden"), used)) void libc_start(int argc, char **argv, char **envp) {
	init_stdlib();

	// Call main() :)
	extern int main(int, char **, char **);
	_exit(main(argc, argv, envp));
}

// 1. wrangle with argc and argc
// 2. initialize the global pointer to __global_pointer
// NOTE: have to disable relaxing first
asm("  .global _start				\t\n\
_start:                         \t\n\
    lw   a0, 0(sp) 				\t\n"
#if __riscv_xlen == 32
	"addi a1, sp, 4		 		\t\n"
#else
	"addi a1, sp, 8		 		\t\n"
#endif
	"andi sp, sp, -16 /* not needed */\t\n\
.option push 					\t\n\
.option norelax 				\t\n\
	1:auipc gp, %pcrel_hi(__global_pointer$) \t\n\
	addi  gp, gp, %pcrel_lo(1b) \t\n\
.option pop						\t\n\
	call libc_start				\t\n\
");
