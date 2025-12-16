/**************************************************************************/
/*  gdscript_c_compiler.cpp                                               */
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

#include "gdscript_c_compiler.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "core/string/print_string.h"

GDScriptCCompiler::GDScriptCCompiler() :
		compiler_available(false) {
	detect_cross_compiler();
}

String GDScriptCCompiler::find_cross_compiler() const {
	// Try common RISC-V cross-compiler names
	const String compiler_candidates[] = {
		"riscv64-unknown-elf-gcc",
		"riscv64-linux-gnu-gcc",
		"riscv64-elf-gcc",
		"riscv-gcc"
	};

	List<String> empty_args;
	for (const String &compiler : compiler_candidates) {
		String pipe_output;
		Error err = OS::get_singleton()->execute(compiler, empty_args, &pipe_output, nullptr, false, nullptr, false);
		if (err == OK) {
			return compiler;
		}
	}

	return String(); // Not found
}

String GDScriptCCompiler::detect_cross_compiler() {
	if (!detected_compiler_path.is_empty()) {
		return detected_compiler_path;
	}

	detected_compiler_path = find_cross_compiler();
	compiler_available = !detected_compiler_path.is_empty();

	if (compiler_available) {
		print_verbose("GDScriptCCompiler: Detected RISC-V cross-compiler: " + detected_compiler_path);
	} else {
		print_verbose("GDScriptCCompiler: No RISC-V cross-compiler found in PATH");
	}

	return detected_compiler_path;
}

bool GDScriptCCompiler::is_cross_compiler_available() const {
	return compiler_available;
}

Error GDScriptCCompiler::compile_to_object_file(const String &p_c_source_path, const String &p_output_path, const Vector<String> &p_include_paths) const {
	if (!compiler_available) {
		return ERR_UNAVAILABLE;
	}

	// Build compiler arguments
	Vector<String> args;
	args.push_back("-c"); // Compile only, don't link
	args.push_back("-O0"); // No optimization for debugging
	args.push_back("-g"); // Include debug info
	args.push_back("-nostdlib"); // No standard library
	args.push_back("-static"); // Static linking
	args.push_back("-march=rv64gc"); // RISC-V 64-bit with GC extensions
	args.push_back("-mabi=lp64d"); // ABI for 64-bit
	args.push_back("-fPIC"); // Position independent code

	// Add include paths
	for (const String &include_path : p_include_paths) {
		args.push_back("-I" + include_path);
	}

	args.push_back("-o");
	args.push_back(p_output_path);
	args.push_back(p_c_source_path);

	List<String> args_list;
	for (const String &arg : args) {
		args_list.push_back(arg);
	}
	List<String> output;
	int exit_code;
	Error err = OS::get_singleton()->execute(detected_compiler_path, args_list, &output, true, false, &exit_code);

	if (err != OK || exit_code != 0) {
		print_error("GDScriptCCompiler: Compilation failed:");
		for (const String &line : output) {
			print_error("  " + line);
		}
		return FAILED;
	}

	return OK;
}

Error GDScriptCCompiler::link_to_executable(const String &p_object_path, const String &p_output_path) const {
	if (!compiler_available) {
		return ERR_UNAVAILABLE;
	}

	// Link with minimal options for ELF executable
	Vector<String> args;
	args.push_back("-nostdlib"); // No standard library
	args.push_back("-static"); // Static linking
	args.push_back("-march=rv64gc");
	args.push_back("-mabi=lp64d");
	args.push_back("-o");
	args.push_back(p_output_path);
	args.push_back(p_object_path);

	List<String> args_list;
	for (const String &arg : args) {
		args_list.push_back(arg);
	}
	List<String> output;
	int exit_code;
	Error err = OS::get_singleton()->execute(detected_compiler_path, args_list, &output, true, false, &exit_code);

	if (err != OK || exit_code != 0) {
		print_error("GDScriptCCompiler: Linking failed:");
		for (const String &line : output) {
			print_error("  " + line);
		}
		return FAILED;
	}

	return OK;
}

Error GDScriptCCompiler::compile_c_to_elf(const String &p_c_source, const Vector<String> &p_include_paths, PackedByteArray &r_elf_data) const {
	if (!compiler_available) {
		return ERR_UNAVAILABLE;
	}

	// Create temporary directory for compilation
	String temp_dir = OS::get_singleton()->get_cache_path();
	if (temp_dir.is_empty()) {
		temp_dir = OS::get_singleton()->get_user_data_dir();
	}
	temp_dir = temp_dir.path_join("godot_gdscript_tmp");

	Ref<DirAccess> dir = DirAccess::create_for_path(temp_dir);
	if (!dir.is_valid()) {
		return ERR_CANT_CREATE;
	}

	// Create temp directory if it doesn't exist
	if (!dir->dir_exists(temp_dir)) {
		Error err = dir->make_dir_recursive(temp_dir);
		if (err != OK) {
			return err;
		}
	}

	// Generate unique filenames
	uint64_t timestamp = OS::get_singleton()->get_ticks_msec();
	String base_name = vformat("gdscript_%llu", timestamp);
	String c_source_path = temp_dir.path_join(base_name + ".c");
	String object_path = temp_dir.path_join(base_name + ".o");
	String elf_path = temp_dir.path_join(base_name + ".elf");

	// Write C source to file
	Ref<FileAccess> source_file = FileAccess::open(c_source_path, FileAccess::WRITE);
	if (!source_file.is_valid()) {
		return ERR_FILE_CANT_WRITE;
	}
	source_file->store_string(p_c_source);
	source_file->close();

	// Compile to object file
	Error compile_err = compile_to_object_file(c_source_path, object_path, p_include_paths);
	if (compile_err != OK) {
		// Cleanup temp files
		dir->remove(c_source_path);
		return compile_err;
	}

	// Link to executable
	Error link_err = link_to_executable(object_path, elf_path);
	if (link_err != OK) {
		// Cleanup temp files
		dir->remove(c_source_path);
		dir->remove(object_path);
		return link_err;
	}

	// Read ELF data
	Ref<FileAccess> elf_file = FileAccess::open(elf_path, FileAccess::READ);
	if (!elf_file.is_valid()) {
		return ERR_FILE_CANT_READ;
	}

	r_elf_data.resize(elf_file->get_length());
	elf_file->get_buffer(r_elf_data.ptrw(), r_elf_data.size());
	elf_file->close();

	// Cleanup temp files
	dir->remove(c_source_path);
	dir->remove(object_path);
	dir->remove(elf_path);

	print_verbose(vformat("GDScriptCCompiler: Successfully compiled C code to ELF (%d bytes)", r_elf_data.size()));

	return OK;
}
