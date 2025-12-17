/**************************************************************************/
/*  gdscript_stablehlo_to_riscv.h                                        */
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
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"
#include "core/variant/dictionary.h"

// Internal StableHLO to RISC-V ELF64 compiler
// Parses StableHLO MLIR and generates RISC-V ELF64 binaries directly

class StableHLOParser {
public:
	struct Operation {
		String name;
		Vector<String> operands;
		Vector<String> results;
		Dictionary attributes;
		String type_signature;
	};
	
	struct Function {
		String name;
		Vector<Pair<String, String>> arguments;  // (name, type)
		String return_type;
		Vector<Operation> operations;
	};
	
	struct Module {
		Vector<Function> functions;
	};
	
	static Module parse(const String &p_mlir_text);
	
private:
	static void skip_whitespace(const String &p_text, int &p_pos);
	static String parse_identifier(const String &p_text, int &p_pos);
	static String parse_string_literal(const String &p_text, int &p_pos);
	static String parse_type(const String &p_text, int &p_pos);
	static Operation parse_operation(const String &p_text, int &p_pos);
	static Function parse_function(const String &p_text, int &p_pos);
};

class RISCVCodeGenerator {
public:
	struct Register {
		String name;  // e.g., "fa0", "a0", "t0"
		String type;  // "f64", "i32", "i1"
		bool is_float;
	};
	
	static PackedByteArray generate_elf64(const StableHLOParser::Module &p_module, bool p_debug = true);
	
private:
	static void generate_function(const StableHLOParser::Function &p_func, Vector<String> &p_assembly, Vector<uint8_t> &p_data);
	static void generate_operation(const StableHLOParser::Operation &p_op, Vector<String> &p_assembly, Vector<uint8_t> &p_data, HashMap<String, Register> &p_registers, int &p_next_reg);
	static Register allocate_register(const String &p_type, int &p_next_reg);
	static String get_register_name(int p_reg_id, const String &p_type);
	static void emit(const String &p_instruction, Vector<String> &p_assembly);
	static void emit_data(const Vector<uint8_t> &p_bytes, Vector<uint8_t> &p_data);
};

class ELF64Generator {
public:
	static PackedByteArray create_elf64(
		const Vector<uint8_t> &p_code,
		const Vector<uint8_t> &p_data,
		const String &p_entry_symbol,
		uint64_t p_entry_address,
		const Dictionary &p_symbols = Dictionary(),
		bool p_debug = true
	);
	
private:
	static void write_elf_header(PackedByteArray &p_elf, uint64_t p_entry, uint64_t p_phoff, uint64_t p_shoff);
	static void write_program_header(PackedByteArray &p_elf, uint32_t p_type, uint64_t p_offset, uint64_t p_vaddr, uint64_t p_filesz, uint64_t p_memsz, uint32_t p_flags);
	static void write_section_header(PackedByteArray &p_elf, uint32_t p_name, uint32_t p_type, uint64_t p_addr, uint64_t p_offset, uint64_t p_size);
	static void align_to(PackedByteArray &p_elf, uint64_t p_alignment);
};
