/**************************************************************************/
/*  gdscript_to_stablehlo.h                                               */
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
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFLICTING. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "modules/gdscript/gdscript_function.h"

class GDScriptToStableHLO : public RefCounted {
	GDCLASS(GDScriptToStableHLO, RefCounted);

public:
	// Convert GDScript function to StableHLO text format (MLIR text)
	// Returns the StableHLO text representation
	static String convert_function_to_stablehlo_text(const GDScriptFunction *p_function);
	
	// Write StableHLO text to file (MLIR text format)
	// Returns path to the .stablehlo file, or empty string on error
	static String convert_function_to_stablehlo_bytecode(const GDScriptFunction *p_function, const String &p_output_path);
	
	// Generate StableHLO file directly from GDScript function (bypasses C++ generation)
	// Returns path to the .stablehlo file, or empty string on error
	static String generate_mlir_file(const GDScriptFunction *p_function, const String &p_output_path);
	
	// Check if function can be converted (only basic opcodes)
	static bool can_convert_function(const GDScriptFunction *p_function);

private:
	// Generate StableHLO constant operation
	static String generate_constant(const Variant &p_value, int &p_value_id);
	
	// Generate StableHLO operation for a GDScript opcode
	static String generate_operation(int p_opcode, const int *p_code_ptr, int &p_ip, int p_code_size, int &p_value_id);
	
	// Check if opcode is basic and supported
	static bool is_basic_opcode(int p_opcode);
};

