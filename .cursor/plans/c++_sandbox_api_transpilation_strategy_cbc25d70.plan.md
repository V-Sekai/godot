---
name: C++ Sandbox API transpilation strategy
overview: "Switch from C99 to C++ code generation using Sandbox API types directly. Add native C++ test compilation path using SandboxDummy (no libriscv needed). Keep RISC-V ELF path for production. This follows fwsgonzo's advice: \"step 1 is always make it work - parse GDScript, make it use the Godot Sandbox API.\""
todos:
  - id: switch_to_cpp_generation
    content: Update gdscript_bytecode_c_code_generator.cpp to generate C++ code using Sandbox API types (GuestVariant, etc.) instead of C99 with Variant* pointers
    status: pending
  - id: update_compiler_cpp_support
    content: Update gdscript_c_compiler.cpp to handle .cpp files and use g++/riscv64-unknown-elf-g++ with -std=c++17 flag
    status: pending
  - id: add_native_compilation
    content: Add compile_cpp_to_native() method to GDScriptCCompiler for test compilation (native C++, not RISC-V)
    status: pending
  - id: create_sandbox_dummy_test_harness
    content: Create test helpers in test_gdscript_c_generation.h to compile C++ to native and execute with SandboxDummy (no libriscv needed)
    status: pending
  - id: update_function_signatures
    content: Update generated function signatures to use GuestVariant and Sandbox API types
    status: pending
  - id: verify_riscv_path_still_works
    content: Verify RISC-V ELF compilation path still works with C++ code generation
    status: pending
  - id: update_opcodes_to_sandbox_api
    content: Update all opcode generation to use Sandbox API types (GET_MEMBER, SET_MEMBER, CALL, etc.)
    status: pending
  - id: update_testing_plan
    content: Update testing strategy to use native C++ path with SandboxDummy for faster iteration, keep RISC-V ELF path for production verification
    status: pending
---

# C++ Sandbox API Transpilation Strategy

## Overview

Following fwsgonzo's guidance: "step 1 is always make it work - parse GDScript, make it use the Godot Sandbox API." Switch from C99 code generation to C++ code generation that uses Sandbox API types directly (GuestVariant, etc.). This enables:

1. **Simpler transpilation**: Use Sandbox API types directly instead of manual syscall marshaling
2. **Better testing**: Compile to native C++ and test with SandboxDummy (no libriscv/Godot needed)
3. **Production path**: Still compile to RISC-V ELF for actual sandbox execution

## Architecture

### Current Approach (C99 + Syscalls)

```
GDScript Bytecode → C99 Code → RISC-V ELF → Sandbox (libriscv) → Syscalls → VM
```

### New Approach (C++ + Sandbox API)

```
GDScript Bytecode → C++ Code (Sandbox API) → {
    Test Path: Native C++ → SandboxDummy (no libriscv)
    Prod Path: RISC-V ELF → Sandbox (libriscv)
}
```

## Key Changes

### 1. Code Generation: C99 → C++

**File**: `modules/gdscript_elf/src/gdscript_bytecode_c_code_generator.cpp`

**Changes**:

- Rename `generate_c_code()` → `generate_cpp_code()` (or keep name, change implementation)
- Generate C++ code using Sandbox API types:
  - Use `GuestVariant` instead of `Variant*` pointers
  - Include Sandbox headers: `#include "modules/sandbox/src/guest_datatypes.h"`
  - Use Sandbox API functions directly instead of syscalls where possible
  - Still use syscalls for host communication (ECALL_VCALL, etc.) but with C++ types

**Example transformation**:

```cpp
// Old (C99):
Variant stack[100];
stack[0] = Variant(42);
register Variant* a0 asm("a0") = &stack[0];
__asm__ volatile("ecall" : : "r"(syscall_number), "r"(a0));

// New (C++):
GuestVariant stack[100];
stack[0].v.i = 42;
stack[0].type = Variant::INT;
// Use Sandbox API or syscalls with GuestVariant
```

### 2. Compiler: Support Both Native C++ and RISC-V ELF

**File**: `modules/gdscript_elf/src/gdscript_c_compiler.cpp`

**Changes**:

- Add `compile_cpp_to_native()` method for test compilation
- Keep `compile_c_to_elf()` for production (but accept C++ source)
- Update `compile_to_object_file()` to accept `.cpp` files and use `g++` or `riscv64-unknown-elf-g++`

**New method**:

```cpp
Error GDScriptCCompiler::compile_cpp_to_native(
    const String &p_cpp_source,
    const Vector<String> &p_include_paths,
    String &r_executable_path
) const;
```

### 3. Test Infrastructure: SandboxDummy Pattern

**File**: `modules/gdscript_elf/test_gdscript_c_generation.h` (or new test file)

**Changes**:

- Add test compilation path that compiles C++ to native executable
- Link with SandboxDummy instead of full Sandbox
- Execute native code and verify results

**Test helper**:

```cpp
// Compile C++ code to native executable, link with SandboxDummy
static Error compile_cpp_to_native_test(
    const String &p_cpp_code,
    const String &p_output_path
);

// Execute native code with SandboxDummy
static Variant execute_native_with_dummy(
    const String &p_executable_path,
    const String &p_function_name,
    const Array &p_args
);
```

### 4. Function Signature Updates

**File**: `modules/gdscript_elf/src/gdscript_bytecode_c_code_generator.cpp`

**Current signature**:

```cpp
void gdscript_<name>(void* instance, Variant* args, int argcount, Variant* result, ...)
```

**New C++ signature** (for native testing):

```cpp
void gdscript_<name>(SandboxDummy* sandbox, GuestVariant* args, int argcount, GuestVariant* result, ...)
```

**For RISC-V ELF** (production):

- Keep similar signature but use syscalls to access Sandbox instance
- Or use a global/thread-local Sandbox pointer accessible via syscall

## Implementation Phases

### Phase 1: C++ Code Generation (Core)

1. **Update code generator to emit C++**

   - Change file extension from `.c` to `.cpp`
   - Include Sandbox headers
   - Use `GuestVariant` instead of `Variant*`
   - Update function signatures

2. **Update compiler to handle C++**

   - Detect C++ source files
   - Use `g++` or `riscv64-unknown-elf-g++` instead of `gcc`
   - Add `-std=c++17` flag

3. **Verify RISC-V ELF path still works**

   - Ensure C++ code compiles to RISC-V ELF
   - Test with real Sandbox

### Phase 2: Native C++ Test Path

1. **Add native compilation method**

   - `compile_cpp_to_native()` in `GDScriptCCompiler`
   - Compile with host compiler (not cross-compiler)
   - Link with SandboxDummy

2. **Create test harness**

   - Helper to compile C++ to native
   - Helper to execute with SandboxDummy
   - Update existing tests to use new path

3. **Verify test path works**

   - Test simple functions (RETURN, ASSIGN)
   - Verify no libriscv dependency needed

### Phase 3: Update All Opcodes

1. **Convert syscall-based opcodes to use Sandbox API**

   - GET_MEMBER, SET_MEMBER: Use Sandbox API directly (if possible) or keep syscalls
   - CALL: Use Sandbox API for argument marshaling
   - OPERATOR: Use Sandbox API types

2. **Update testing**

   - All 14 core opcodes tested via native path
   - Production path verified with RISC-V ELF

## Benefits

1. **Simpler transpilation**: Use Sandbox API types directly, less manual marshaling
2. **Better testing**: No need to spin up Godot or libriscv for tests
3. **Type safety**: C++ types catch errors at compile time
4. **Easier debugging**: Native code easier to debug than RISC-V ELF
5. **Faster iteration**: Native compilation faster than cross-compilation

## Files to Modify

1. `modules/gdscript_elf/src/gdscript_bytecode_c_code_generator.cpp`

   - Change to C++ code generation
   - Use Sandbox API types

2. `modules/gdscript_elf/src/gdscript_bytecode_c_code_generator.h`

   - Update method names/signatures if needed

3. `modules/gdscript_elf/src/gdscript_c_compiler.cpp`

   - Add native C++ compilation path
   - Update to handle `.cpp` files

4. `modules/gdscript_elf/src/gdscript_c_compiler.h`

   - Add `compile_cpp_to_native()` declaration

5. `modules/gdscript_elf/test_gdscript_c_generation.h`

   - Add native test helpers
   - Update tests to use SandboxDummy pattern

6. `modules/gdscript_elf/AGENTS.md`

   - Document new C++ approach
   - Update architecture diagrams

## Testing Strategy Update

The existing testing plan for 14 core opcodes remains valid, but now:

- **C++ Generation Verification**: Verify C++ code uses Sandbox API types correctly
- **Native Execution Testing**: Compile to native C++, test with SandboxDummy (no libriscv)
- **ELF Execution Testing**: Still test RISC-V ELF path for production verification

## Notes

- **Sandbox API Access**: Generated C++ code running in RISC-V ELF still needs syscalls to access Sandbox instance (can't directly call C++ methods from guest code)
- **GuestVariant in ELF**: GuestVariant is a data structure that can be used in both native and RISC-V contexts
- **Dual Path**: Maintain both test (native) and production (RISC-V ELF) compilation paths