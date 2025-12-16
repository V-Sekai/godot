/**************************************************************************/
/*  gdscript_c_compiler.h                                                 */
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

#pragma once

#include "core/object/ref_counted.h"
#include "core/templates/vector.h"

class GDScriptCCompiler : public RefCounted {
	GDCLASS(GDScriptCCompiler, RefCounted);

private:
	String detected_compiler_path;
	bool compiler_available;

	String find_cross_compiler() const;
	Error compile_to_object_file(const String &p_c_source_path, const String &p_output_path, const Vector<String> &p_include_paths) const;
	Error link_to_executable(const String &p_object_path, const String &p_output_path) const;

public:
	GDScriptCCompiler();

	String detect_cross_compiler();
	bool is_cross_compiler_available() const;

	Error compile_c_to_elf(const String &p_c_source, const Vector<String> &p_include_paths, PackedByteArray &r_elf_data) const;

	// Native C++ compilation for testing (not RISC-V)
	Error compile_cpp_to_native(const String &p_cpp_source, const Vector<String> &p_include_paths, String &r_executable_path) const;
};
