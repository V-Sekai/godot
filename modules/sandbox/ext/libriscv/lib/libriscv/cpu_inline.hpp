/**************************************************************************/
/*  cpu_inline.hpp                                                        */
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

// Use a trick to access the Machine directly on g++/clang, Linux-only for now
#if (defined(__GNUG__) || defined(__clang__)) && defined(__linux__)
template <int W>
RISCV_ALWAYS_INLINE inline Machine<W> &CPU<W>::machine() noexcept { return *reinterpret_cast<Machine<W> *>(this); }
template <int W>
RISCV_ALWAYS_INLINE inline const Machine<W> &CPU<W>::machine() const noexcept { return *reinterpret_cast<const Machine<W> *>(this); }
#else
template <int W>
RISCV_ALWAYS_INLINE inline Machine<W> &CPU<W>::machine() noexcept { return this->m_machine; }
template <int W>
RISCV_ALWAYS_INLINE inline const Machine<W> &CPU<W>::machine() const noexcept { return this->m_machine; }
#endif

template <int W>
RISCV_ALWAYS_INLINE inline Memory<W> &CPU<W>::memory() noexcept { return machine().memory; }
template <int W>
RISCV_ALWAYS_INLINE inline const Memory<W> &CPU<W>::memory() const noexcept { return machine().memory; }

template <int W>
inline CPU<W>::CPU(Machine<W> &machine) :
		m_machine{ machine }, m_exec(empty_execute_segment().get()) {
}
template <int W>
inline void CPU<W>::reset_stack_pointer() noexcept {
	// initial stack location
	this->reg(2) = machine().memory.stack_initial();
}

template <int W>
inline void CPU<W>::jump(const address_t dst) {
	// it's possible to jump to a misaligned address
	if constexpr (!compressed_enabled) {
		if (UNLIKELY(dst & 0x3)) {
			trigger_exception(MISALIGNED_INSTRUCTION, dst);
		}
	} else {
		if (UNLIKELY(dst & 0x1)) {
			trigger_exception(MISALIGNED_INSTRUCTION, dst);
		}
	}
	this->registers().pc = dst;
}

template <int W>
inline void CPU<W>::aligned_jump(const address_t dst) noexcept {
	this->registers().pc = dst;
}

template <int W>
inline void CPU<W>::increment_pc(int delta) noexcept {
	registers().pc += delta;
}
