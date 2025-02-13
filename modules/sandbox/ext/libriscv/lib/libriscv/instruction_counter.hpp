/**************************************************************************/
/*  instruction_counter.hpp                                               */
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

#ifndef INSTRUCTION_COUNTER_HPP
#define INSTRUCTION_COUNTER_HPP
#include <cstdint>

namespace riscv {
template <int W>
struct Machine;

// In fastsim mode the instruction counter becomes a register
// the function, and we only update m_counter in Machine on exit
// When binary translation is enabled we cannot do this optimization.
struct InstrCounter {
	InstrCounter(uint64_t icounter, uint64_t maxcounter) :
			m_counter(icounter),
			m_max(maxcounter) {}
	~InstrCounter() = default;

	template <int W>
	void apply(Machine<W> &machine) {
		machine.set_instruction_counter(m_counter);
		machine.set_max_instructions(m_max);
	}
	template <int W>
	void apply_counter(Machine<W> &machine) {
		machine.set_instruction_counter(m_counter);
	}
	// Used by binary translator to compensate for its own function already being counted
	// TODO: Account for this inside the binary translator instead. Very minor impact.
	template <int W>
	void apply_counter_minus_1(Machine<W> &machine) {
		machine.set_instruction_counter(m_counter - 1);
		machine.set_max_instructions(m_max);
	}
	template <int W>
	void retrieve_max_counter(Machine<W> &machine) {
		m_max = machine.max_instructions();
	}
	template <int W>
	void retrieve_counters(Machine<W> &machine) {
		m_counter = machine.instruction_counter();
		m_max = machine.max_instructions();
	}

	uint64_t value() const noexcept {
		return m_counter;
	}
	uint64_t max() const noexcept {
		return m_max;
	}
	void stop() noexcept {
		m_max = 0; // This stops the machine
	}
	void set_counters(uint64_t value, uint64_t max) {
		m_counter = value;
		m_max = max;
	}
	void increment_counter(uint64_t cnt) {
		m_counter += cnt;
	}
	bool overflowed() const noexcept {
		return m_counter >= m_max;
	}

private:
	uint64_t m_counter;
	uint64_t m_max;
};
} //namespace riscv

#endif // INSTRUCTION_COUNTER_HPP
