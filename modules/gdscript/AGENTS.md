# GDScript Module - Agent Documentation

## Overview

The GDScript module provides Godot Engine's native scripting language implementation, including compilation, execution, and code generation capabilities. This document provides guidance for AI agents and developers working with the GDScript codebase.

## Key Components

### Core Classes

- **`GDScript`**: Main script class representing a GDScript resource
- **`GDScriptFunction`**: Represents a compiled GDScript function with bytecode
- **`GDScriptParser`**: Parses GDScript source code into an AST
- **`GDScriptCompiler`**: Compiles GDScript AST to bytecode
- **`GDScriptAnalyzer`**: Performs static analysis on GDScript code

### Code Generation

- **`GDScriptToC99`**: Converts GDScript functions to C99 code for RISC-V ELF64 compilation
  - Location: `gdscript_to_c99.h` / `gdscript_to_c99.cpp`
  - Key methods:
    - `generate_c99()`: Generate C99 code from a GDScript function
    - `compile_c99_to_elf64()`: Compile C99 code to RISC-V ELF64 binary
    - `can_convert_to_c99()`: Check if a function can be converted

- **`GDScriptToStableHLO`**: Converts GDScript to StableHLO MLIR dialect
  - Location: `gdscript_to_stablehlo.h` / `gdscript_to_stablehlo.cpp`
  - Provides intermediate representation for MLIR-based compilation

### Bytecode Structure

GDScript functions are compiled to bytecode stored in `GDScriptFunction::code` (PackedInt32Array). Each instruction consists of:
- Opcode (first word)
- Operands (subsequent words, opcode-dependent)

Key opcodes:
- `OPCODE_RETURN`: Return from function
- `OPCODE_ASSIGN`: Variable assignment
- `OPCODE_OPERATOR`: Arithmetic/logical operations
- `OPCODE_CALL`: Function calls
- `OPCODE_JUMP_IF`: Conditional jumps

### Type System

- **`GDScriptDataType`**: Represents GDScript types
  - `BUILTIN`: Built-in Variant types (INT, FLOAT, STRING, etc.)
  - `NATIVE`: Native Godot classes
  - `SCRIPT`: Script types
  - `GDSCRIPT`: GDScript classes

Type mapping to C99:
- `Variant::INT` → `int64_t`
- `Variant::FLOAT` → `double`
- `Variant::BOOL` → `bool`
- `Variant::STRING` → `char*`
- Complex types → `void*`

## Compilation Pipeline

### GDScript → C99 → RISC-V ELF64

```
GDScript Source
    ↓
[GDScriptParser] → AST
    ↓
[GDScriptCompiler] → Bytecode (GDScriptFunction)
    ↓
[GDScriptToC99] → C99 Code
    ↓
[C Compiler] (TCC or RISC-V GCC) → RISC-V ELF64 Binary
```

### Key Steps

1. **Parse**: `GDScript::reload()` parses source code
2. **Compile**: `GDScriptCompiler` generates bytecode
3. **Convert**: `GDScriptToC99::generate_c99()` converts bytecode to C99
4. **Compile**: `GDScriptToC99::compile_c99_to_elf64()` compiles C99 to binary

## Syscall Integration

The C99 code generation includes syscall wrappers for RISC-V using inline assembly:

- `godot_syscall_print()`: Print strings
- `godot_vcall()`: Virtual function calls
- `godot_vcreate()`: Create Variant instances
- `godot_vassign()`: Variant assignment
- `godot_obj_prop_get/set()`: Object property access
- `godot_type_test()`: Type checking

All syscalls use RISC-V `ecall` instruction with register conventions:
- `a7`: Syscall number
- `a0-a6`: Arguments
- `a0`: Return value

## Testing

### Unit Tests

- Location: `modules/gdscript/tests/`
- Key test files:
  - `test_gdscript_to_c99.cpp`: C99 conversion tests
  - `test_gdscript.cpp`: Core GDScript functionality
  - `test_gdscript_elf_e2e.h`: End-to-end ELF compilation tests

### Running Tests

```bash
# Build with tests enabled
scons -j8 target=editor tests=yes

# Run C99 conversion tests
./bin/godot.macos.editor.arm64 --headless --test --test-name="*C99*"
```

### Test Scripts

Runtime test scripts are located in:
- `modules/gdscript/tests/scripts/runtime/features/`
- Used for validating C99 code generation

## Common Patterns

### Accessing Function Bytecode

```cpp
GDScriptFunction *func = script->get_member_functions()["function_name"];
if (func && !func->code.is_empty()) {
    const int *code_ptr = func->code.ptr();
    int code_size = func->code.size();
    // Process bytecode...
}
```

### Generating C99 Code

```cpp
if (GDScriptToC99::can_convert_to_c99(func)) {
    String c99_code = GDScriptToC99::generate_c99(func);
    // Use c99_code...
}
```

### Compiling to ELF64

```cpp
PackedByteArray elf_binary = GDScriptToC99::compile_c99_to_elf64(
    c99_code, 
    function_name, 
    debug_mode
);
```

## File Structure

```
modules/gdscript/
├── gdscript.h/cpp              # Main GDScript class
├── gdscript_function.h/cpp     # Function representation
├── gdscript_parser.h/cpp       # Parser
├── gdscript_compiler.h/cpp     # Compiler
├── gdscript_to_c99.h/cpp       # C99 code generator
├── gdscript_to_stablehlo.h/cpp # StableHLO converter
├── tests/                      # Unit tests
│   ├── test_gdscript_to_c99.cpp
│   └── test_gdscript.cpp
└── AGENTS.md                   # This file
```

## Design Documents

- `mlir/GDSCRIPT_TO_C99_DESIGN.md`: Detailed design for GDScript → C99 compilation
- `mlir/STABLEHLO_TO_ELF64_INTERNAL.md`: Internal StableHLO to RISC-V compilation

## Important Notes

1. **Bytecode Format**: Bytecode uses packed integers. Use `GDScriptFunction::get_opcode_size()` to decode instruction sizes.

2. **Stack Variables**: The C99 generator uses a stack variable mapping (`HashMap<int, String>`) to track temporary variables.

3. **Type Conversion**: Always use `gdscript_type_to_c99()` to convert GDScript types to C99 types.

4. **Syscall Wrappers**: All syscall wrappers are generated once per C99 file via `generate_syscall_wrappers()`.

5. **Function Signatures**: Function names are sanitized (replace `@` with `_at_`, `.` with `_`) for C compatibility.

## Debugging Tips

- Enable debug mode in `compile_c99_to_elf64()` to include debug symbols
- Check `can_convert_to_c99()` before attempting conversion
- Use `ERR_PRINT()` to log conversion issues
- Validate generated C99 code matches expected format (see design docs)

## Future Work

- Expand opcode support in C99 generator
- Improve stack variable management
- Add support for more GDScript features (classes, inheritance, etc.)
- Integrate TCC for RISC-V cross-compilation
- Add source mapping for debugging
