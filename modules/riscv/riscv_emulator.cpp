/**************************************************************************/
/*  riscv_emulator.cpp                                                    */
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

#include "riscv_emulator.h"
#include <cstdint>

using namespace riscv;

String RiscvEmulator::to_string() {
	return "[ GDExtension::RiscvEmulator <--> Instance ID:" + uitos(get_instance_id()) + " ]";
}

void RiscvEmulator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load", "buffer", "arguments"), &RiscvEmulator::load);
	ClassDB::bind_method(D_METHOD("exec"), &RiscvEmulator::exec);
	ClassDB::bind_method(D_METHOD("fork_exec"), &RiscvEmulator::fork_exec);
	ClassDB::bind_method(D_METHOD("invoke_function", "function"), &RiscvEmulator::invoke_function);

	ClassDB::bind_method(D_METHOD("get_buffer"), &RiscvEmulator::get_buffer);
	ClassDB::bind_method(D_METHOD("set_buffer", "new_buffer"), &RiscvEmulator::set_buffer);
	ClassDB::bind_method(D_METHOD("get_arguments"), &RiscvEmulator::get_arguments);
	ClassDB::bind_method(D_METHOD("set_arguments", "new_arguments"), &RiscvEmulator::set_arguments);
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "buffer"), "set_buffer", "get_buffer");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "arguments"), "set_arguments", "get_arguments");
}

RiscvEmulator::RiscvEmulator() {
	// In order to reduce checks we guarantee that this
	// class is well-formed at all times.
	this->m_machine = new machine_t{};
}

RiscvEmulator::~RiscvEmulator() {
	delete this->m_machine;
}

void RiscvEmulator::load(const PackedByteArray p_buffer, const PackedStringArray p_arguments) {
	print_line("Loading file from buffer");

	m_binary = std::vector<uint8_t>{ buffer.ptr(), buffer.ptr() + buffer.size() };

	delete this->m_machine;
	this->m_machine = new machine_t{ this->m_binary };
	machine_t &m = machine();

	m.setup_minimal_syscalls();
	m.setup_argv({ "program" });
}

void RiscvEmulator::exec() {
	machine_t &m = machine();
	print_line("Simulating...");
	m.simulate(MAX_INSTRUCTIONS);
	print_line("Done, instructions: ", m.instruction_counter(),
			" result: ", m.return_value<int64_t>());
}

void RiscvEmulator::fork_exec() {
}

int64_t RiscvEmulator::invoke_function(String p_function) {
	const auto ascii = p_function.ascii();
	const std::string_view sview{ ascii.get_data(), (size_t)ascii.length() };
	gaddr_t address = 0x0;

	machine().reset_instruction_counter();
	try {
		address = machine().address_of(sview);
		return machine().vmcall<MAX_INSTRUCTIONS>(address);
	} catch (const std::exception &e) {
		_handle_exception(address);
	}
	return -1;
}

void RiscvEmulator::_handle_exception(gaddr_t p_address) {
	auto callsite = machine().memory.lookup(p_address);
	print_line(
			"[", get_name(), "] Exception when calling:\n  ", callsite.name.c_str(), " (0x",
			String("%x").format(callsite.address), ")\n", "Backtrace:\n");
	//this->print_backtrace(address);

	try {
		throw; // re-throw
	} catch (const riscv::MachineTimeoutException &e) {
		_handle_timeout(p_address);
		return; // NOTE: might wanna stay
	} catch (const riscv::MachineException &e) {
		const String instr(machine().cpu.current_instruction_to_string().c_str());
		const String regs(machine().cpu.registers().to_string().c_str());

		print_line(
				"\nException: ", e.what(), "  (data: ", String("%x").format(e.data()), ")\n",
				">>> ", instr, "\n",
				">>> Machine registers:\n[PC\t", String("%x").format(machine().cpu.pc()),
				"] ", regs, "\n");
	} catch (const std::exception &e) {
		print_line("\nMessage: ", e.what(), "\n\n");
	}
	print_line(
			"Program page: ", machine().memory.get_page_info(machine().cpu.pc()).c_str(),
			"\n");
	print_line(
			"Stack page: ", machine().memory.get_page_info(machine().cpu.reg(2)).c_str(),
			"\n");
}

void RiscvEmulator::_handle_timeout(gaddr_t p_address) {
	this->m_budget_overruns++;
	auto callsite = machine().memory.lookup(p_address);
	print_line(
			"RiscvEmulator: Timeout for '", callsite.name.c_str(),
			"' (Timeouts: ", m_budget_overruns, "\n");
}

gaddr_t RiscvEmulator::_address_of(std::string_view p_name) const {
	return machine().address_of(p_name);
}
