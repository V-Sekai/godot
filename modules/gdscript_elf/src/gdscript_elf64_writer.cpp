/**************************************************************************/
/*  gdscript_elf64_writer.cpp                                             */
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

#include "gdscript_elf64_writer.h"
#include "gdscript_elf64_mode.h"

#include "gdscript_riscv_encoder.h"
#include "modules/gdscript/gdscript_function.h"

#include <elfio/elfio.hpp>
#include <elfio/elfio_symbols.hpp>
#include <sstream>

PackedByteArray GDScriptELF64Writer::write_elf64(GDScriptFunction *p_function, ELF64CompilationMode p_mode) {
	if (!p_function || p_function->_code_ptr == nullptr || p_function->_code_size == 0) {
		return PackedByteArray();
	}

	// 1. Encode RISC-V instructions from bytecode
	PackedByteArray code = GDScriptRISCVEncoder::encode_function(p_function, p_mode);
	if (code.is_empty()) {
		return PackedByteArray();
	}

	// 2. Create ELF64 file using elfio
	ELFIO::elfio writer;
	writer.create(ELFIO::ELFCLASS64, ELFIO::ELFDATA2LSB);

	// Set OS ABI for Godot sandbox
	writer.set_os_abi(ELFIO::ELFOSABI_NONE); // SYSV

	writer.set_type(ELFIO::ET_EXEC);
	writer.set_machine(ELFIO::EM_RISCV); // From elfio/elf_types.hpp

	// 3. Create .text section with code
	ELFIO::section *text_sec = writer.sections.add(".text");
	text_sec->set_type(ELFIO::SHT_PROGBITS);
	text_sec->set_flags(ELFIO::SHF_ALLOC | ELFIO::SHF_EXECINSTR);
	text_sec->set_addr_align(0x10);

	// Convert PackedByteArray to char* for elfio
	const uint8_t *code_data = code.ptr();
	text_sec->set_data(reinterpret_cast<const char *>(code_data), code.size());

	// 4. Create loadable segment
	const ELFIO::Elf64_Addr ENTRY_POINT = 0x10000;
	const ELFIO::Elf_Xword ELF_PAGE_SIZE = 0x1000;

	ELFIO::segment *text_seg = writer.segments.add();
	text_seg->set_type(ELFIO::PT_LOAD);
	text_seg->set_virtual_address(ENTRY_POINT);
	text_seg->set_physical_address(ENTRY_POINT);
	text_seg->set_flags(ELFIO::PF_X | ELFIO::PF_R); // Executable + Readable
	text_seg->set_align(ELF_PAGE_SIZE);
	text_seg->add_section(text_sec, text_sec->get_addr_align());

	// 5. Set entry point to 0 (no auto-execution)
	// ELF files are function libraries, not executables
	// Functions will be called by address, not via entry point
	writer.set_entry(0);

	// 6. Add symbol table with function name
	// Get function name
	StringName func_name = p_function->get_name();
	String func_name_str = func_name;

	// Create string table section
	ELFIO::section *str_sec = writer.sections.add(".strtab");
	str_sec->set_type(ELFIO::SHT_STRTAB);

	// Create string table accessor
	ELFIO::string_section_accessor stra(str_sec);
	// Add function name to string table
	ELFIO::Elf_Word str_index = stra.add_string(func_name_str.utf8().get_data());

	// Create symbol table section
	ELFIO::section *sym_sec = writer.sections.add(".symtab");
	sym_sec->set_type(ELFIO::SHT_SYMTAB);
	sym_sec->set_info(1); // Number of local symbols (before global symbols)
	sym_sec->set_addr_align(0x8);
	sym_sec->set_entry_size(writer.get_default_entry_size(ELFIO::SHT_SYMTAB));
	sym_sec->set_link(str_sec->get_index()); // Link to string table

	// Create symbol table accessor
	ELFIO::symbol_section_accessor syma(writer, sym_sec);
	// Add function symbol (value is ENTRY_POINT, size is code size)
	// STB_GLOBAL = 1, STT_FUNC = 2
	// Use ELF_ST_INFO macro directly (it's a macro, not in namespace)
	unsigned char sym_info = ELF_ST_INFO(ELFIO::STB_GLOBAL, ELFIO::STT_FUNC);
	syma.add_symbol(str_index, ENTRY_POINT, code.size(), sym_info, 0, text_sec->get_index());

	// 7. Save to string stream and convert to PackedByteArray
	return elfio_to_packed_byte_array(writer);
}

bool GDScriptELF64Writer::can_write_elf64(GDScriptFunction *p_function, ELF64CompilationMode p_mode) {
	if (!p_function) {
		return false;
	}

	// Check if function has bytecode
	if (p_function->_code_ptr == nullptr || p_function->_code_size == 0) {
		return false;
	}

	// Godot syscall mode: Always allowed (sandbox handles everything via ECALLs)

	return true;
}

PackedByteArray GDScriptELF64Writer::elfio_to_packed_byte_array(ELFIO::elfio &p_writer) {
	std::ostringstream oss;
	if (!p_writer.save(oss)) {
		return PackedByteArray();
	}

	std::string elf_data = oss.str();
	PackedByteArray result;
	result.resize(elf_data.size());
	memcpy(result.ptrw(), elf_data.data(), elf_data.size());
	return result;
}
