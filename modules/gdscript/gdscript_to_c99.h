/**************************************************************************/
/*  gdscript_to_c99.h                                                     */
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

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

class GDScriptFunction;
class GDScriptDataType;
class GDScript;

// GDScript to C99 code generator
// Converts GDScript functions to C99 code that can be compiled to RISC-V ELF64
class GDScriptToC99 : public RefCounted {
	GDCLASS(GDScriptToC99, RefCounted);

public:
	// Generate C99 code from a GDScript function
	static String generate_c99(const GDScriptFunction *p_func);
	
	// Compile C99 code to RISC-V ELF64 binary
	// Uses TCC (if RISC-V enabled) or external RISC-V GCC/Clang
	static PackedByteArray compile_c99_to_elf64(const String &p_c99_code, const String &p_function_name, bool p_debug = true);
	
	// Check if a function can be converted to C99
	static bool can_convert_to_c99(const GDScriptFunction *p_func);
	
	// GDScript-accessible wrapper methods (instance methods)
	// Convert a function from a GDScript resource to C99 code
	String convert_script_function_to_c99(Ref<GDScript> p_script, const StringName &p_function_name) const;
	
	// Check if a function from a GDScript resource can be converted
	bool can_convert_script_function(Ref<GDScript> p_script, const StringName &p_function_name) const;
	
	// Static methods - exposed for direct access from GDScript
	// Convert a function from a GDScript resource to C99 code (static version)
	static String generate_c99_from_script(Ref<GDScript> p_script, const StringName &p_function_name);
	
	// Check if a function from a GDScript resource can be converted (static version)
	static bool can_convert_script(Ref<GDScript> p_script, const StringName &p_function_name);

protected:
	static void _bind_methods();

private:
	// Convert GDScript type to C99 type string
	static String gdscript_type_to_c99(const GDScriptDataType &p_type);
	
	// Generate C99 code for a single operation
	static String generate_operation_c99(const GDScriptFunction *p_func, int p_ip, HashMap<int, String> &p_stack_vars, int &p_next_var);
	
	// Generate syscall wrapper functions
	static String generate_syscall_wrappers();
	
	// Generate function signature
	static String generate_function_signature(const GDScriptFunction *p_func);
	
	// Generate function body
	static String generate_function_body(const GDScriptFunction *p_func, HashMap<int, String> &p_stack_vars);
};
