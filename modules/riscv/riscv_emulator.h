/**************************************************************************/
/*  riscv_emulator.h                                                      */
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

#ifndef RISCV_EMULATOR_H
#define RISCV_EMULATOR_H

#include "core/io/resource.h"
#include "core/variant/variant.h"

#include <libriscv/cpu.hpp>
#include <libriscv/machine.hpp>
#include <libriscv/threaded_bytecodes.hpp>

using gaddr_t = riscv::address_type<riscv::RISCV64>;
using machine_t = riscv::Machine<riscv::RISCV64>;

class RiscvEmulator : public Resource {
	GDCLASS(RiscvEmulator, Resource);

protected:
	static void _bind_methods();

	virtual String to_string() override;

public:
	static constexpr uint64_t MAX_INSTRUCTIONS = 16'000'000'000ULL;

	RiscvEmulator();
	~RiscvEmulator();

	auto &machine() { return *m_machine; }
	const auto &machine() const { return *m_machine; }

	// Functions.
	void load(const PackedByteArray p_buffer, const PackedStringArray p_arguments);
	void exec();
	void fork_exec();
	int64_t invoke_function(String p_function);
	gaddr_t _address_of(std::string_view p_name) const;

	PackedByteArray get_buffer() const { return buffer; }
	void set_buffer(const PackedByteArray &p_new_buffer) { buffer = p_new_buffer; }

	PackedStringArray get_arguments() const { return arguments; }
	void set_arguments(const PackedStringArray &p_new_arguments) { arguments = p_new_arguments; }

private:
	void _handle_exception(gaddr_t);
	void _handle_timeout(gaddr_t);

	machine_t *m_machine = nullptr;
	std::vector<uint8_t> m_binary;

	bool m_last_newline = false;
	unsigned m_budget_overruns = 0;

	PackedByteArray buffer;
	PackedStringArray arguments;
};

#endif // RISCV_EMULATOR_H
