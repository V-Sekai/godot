# StableHLO to RISC-V ELF64 Compilation Pipeline

## Overview

This document describes the design for converting StableHLO MLIR to RISC-V ELF64 binaries for expedited execution in the Godot Sandbox module. The pipeline prioritizes both performance and debugging capabilities.

## Compilation Pipeline

```
GDScript Function
    ↓
StableHLO MLIR (text)  [Current: gdscript_to_stablehlo.cpp]
    ↓
MLIR Lowering Passes
    ↓
LLVM IR
    ↓
RISC-V Object Files (via LLVM backend)
    ↓
RISC-V ELF64 Binary
    ↓
Sandbox Execution (modules/sandbox)
```

## Architecture

### Phase 1: MLIR to LLVM IR

**Tool**: `mlir-translate` or `mlir-opt` with LLVM dialect lowering
- Input: StableHLO MLIR text (`.mlir.stablehlo`)
- Output: LLVM IR (`.ll`)
- Process:
  1. Parse StableHLO MLIR
  2. Lower StableHLO ops to Standard/Arith/LLVM dialects
  3. Convert to LLVM IR

**Command**:
```bash
mlir-opt input.mlir.stablehlo \
  --convert-stablehlo-to-std \
  --convert-std-to-llvm \
  --mlir-print-ir-after-all \
  | mlir-translate --mlir-to-llvmir > output.ll
```

### Phase 2: LLVM IR to RISC-V Object

**Tool**: `llc` (LLVM compiler) with RISC-V target
- Input: LLVM IR (`.ll`)
- Output: RISC-V assembly or object file (`.s` or `.o`)
- Target: `riscv64-unknown-elf` or `riscv64-linux-gnu`

**Command**:
```bash
llc -mtriple=riscv64-unknown-elf \
    -mattr=+c,+d \
    -filetype=obj \
    -o output.o \
    input.ll
```

### Phase 3: Object to ELF64 Binary

**Tool**: `riscv64-unknown-elf-ld` (linker)
- Input: RISC-V object file (`.o`)
- Output: RISC-V ELF64 binary (`.elf`)
- Link with:
  - Minimal runtime (syscall stubs)
  - Godot syscall interface (`godot_syscalls`)
  - Debug symbols (for debugging support)

**Command**:
```bash
riscv64-unknown-elf-ld \
  -T linker.ld \
  -Ttext 0x100000 \
  --gc-sections \
  -o output.elf \
  input.o \
  syscalls.o
```

## Implementation Strategy

### Option A: External Toolchain (Expedited)

**Pros**:
- Fastest to implement
- Uses existing, well-tested tools
- Full debugging support via GDB
- Can leverage existing MLIR/LLVM toolchains

**Cons**:
- Requires external dependencies (MLIR, LLVM, RISC-V toolchain)
- Build-time dependency

**Implementation**:
1. Create wrapper class `StableHLOToELF64` in `modules/gdscript/`
2. Use `OS.execute()` to call external tools
3. Cache compiled ELF binaries
4. Integrate with `GDScriptFunction::compile_to_elf64()`

### Option B: Embedded Compiler (Integrated)

**Pros**:
- No external dependencies
- Full control over compilation
- Can optimize for Godot-specific patterns

**Cons**:
- Much more complex
- Requires embedding MLIR/LLVM libraries
- Larger binary size

**Implementation**:
1. Embed MLIR/LLVM libraries as thirdparty
2. Implement compilation pipeline in C++
3. Direct integration with GDScript compilation

## Recommended Approach: Internal Direct Compilation (Option C)

Given that the sandbox already embeds libtcc (though for native code), we can implement a simpler internal approach that directly generates RISC-V assembly from StableHLO, bypassing the need for full MLIR/LLVM toolchains.

### Internal Compilation Pipeline

```
StableHLO MLIR (text)
    ↓
Parse & Lower (internal C++ parser)
    ↓
RISC-V Assembly (text)
    ↓
Assemble & Link (simple ELF generator)
    ↓
RISC-V ELF64 Binary
```

This approach:
- **No external dependencies**: Everything is internal
- **Fast compilation**: Direct translation without heavy toolchains
- **Debuggable**: Can include source mapping and debug symbols
- **Smaller footprint**: Only implements what we need

## Recommended Approach: Hybrid (Option A with Caching) - DEPRECATED

### 1. Add Compilation Method to GDScriptToStableHLO

```cpp
// In gdscript_to_stablehlo.h
static PackedByteArray compile_stablehlo_to_elf64(
    const String &p_mlir_text,
    const String &p_function_name,
    bool p_debug = true  // Include debug symbols
);
```

### 2. Compilation Pipeline Implementation

```cpp
PackedByteArray GDScriptToStableHLO::compile_stablehlo_to_elf64(
    const String &p_mlir_text,
    const String &p_function_name,
    bool p_debug
) {
    // Step 1: Write MLIR to temp file
    String temp_dir = OS::get_singleton()->get_cache_dir() + "/stablehlo_compile/";
    String mlir_file = temp_dir + p_function_name + ".mlir";
    String llvm_file = temp_dir + p_function_name + ".ll";
    String obj_file = temp_dir + p_function_name + ".o";
    String elf_file = temp_dir + p_function_name + ".elf";
    
    // Write MLIR
    Ref<FileAccess> f = FileAccess::open(mlir_file, FileAccess::WRITE);
    f->store_string(p_mlir_text);
    f->close();
    
    // Step 2: MLIR -> LLVM IR
    Array mlir_args;
    mlir_args.push_back(mlir_file);
    mlir_args.push_back("--convert-stablehlo-to-std");
    mlir_args.push_back("--convert-std-to-llvm");
    mlir_args.push_back("|");
    mlir_args.push_back("mlir-translate");
    mlir_args.push_back("--mlir-to-llvmir");
    mlir_args.push_back(">");
    mlir_args.push_back(llvm_file);
    
    int exit_code = OS::get_singleton()->execute("mlir-opt", mlir_args);
    if (exit_code != 0) {
        ERR_PRINT("Failed to convert MLIR to LLVM IR");
        return PackedByteArray();
    }
    
    // Step 3: LLVM IR -> RISC-V Object
    Array llc_args;
    llc_args.push_back("-mtriple=riscv64-unknown-elf");
    llc_args.push_back("-mattr=+c,+d");
    llc_args.push_back("-filetype=obj");
    if (p_debug) {
        llc_args.push_back("-debug");
    }
    llc_args.push_back("-o");
    llc_args.push_back(obj_file);
    llc_args.push_back(llvm_file);
    
    exit_code = OS::get_singleton()->execute("llc", llc_args);
    if (exit_code != 0) {
        ERR_PRINT("Failed to compile LLVM IR to RISC-V");
        return PackedByteArray();
    }
    
    // Step 4: Link to ELF64
    Array ld_args;
    ld_args.push_back("-T");
    ld_args.push_back("linker.ld");  // Minimal linker script
    ld_args.push_back("-Ttext");
    ld_args.push_back("0x100000");
    if (p_debug) {
        ld_args.push_back("--debug");
    }
    ld_args.push_back("-o");
    ld_args.push_back(elf_file);
    ld_args.push_back(obj_file);
    // Link with syscall stubs
    
    exit_code = OS::get_singleton()->execute("riscv64-unknown-elf-ld", ld_args);
    if (exit_code != 0) {
        ERR_PRINT("Failed to link ELF binary");
        return PackedByteArray();
    }
    
    // Step 5: Read ELF binary
    Ref<FileAccess> elf_f = FileAccess::open(elf_file, FileAccess::READ);
    if (!elf_f.is_valid()) {
        ERR_PRINT("Failed to read ELF file");
        return PackedByteArray();
    }
    
    PackedByteArray elf_data;
    elf_data.resize(elf_f->get_length());
    elf_f->get_buffer(elf_data.ptrw(), elf_data.size());
    elf_f->close();
    
    return elf_data;
}
```

### 3. Integration with GDScriptFunction

```cpp
PackedByteArray GDScriptFunction::compile_to_elf64(int p_mode) const {
    // Generate StableHLO MLIR
    String stablehlo_text = GDScriptToStableHLO::convert_function_to_stablehlo_text(this);
    if (stablehlo_text.is_empty()) {
        return PackedByteArray();
    }
    
    // Compile StableHLO to ELF64
    bool debug = (p_mode & 0x1) != 0;  // Bit 0 = debug mode
    return GDScriptToStableHLO::compile_stablehlo_to_elf64(
        stablehlo_text,
        get_name(),
        debug
    );
}

bool GDScriptFunction::can_compile_to_elf64(int p_mode) const {
    // Can compile if we can convert to StableHLO
    return GDScriptToStableHLO::can_convert_function(this);
}
```

## Debugging Support

### Debug Symbols
- Include `-g` flag in `llc` for DWARF debug info
- Preserve function names and source locations
- Map StableHLO operations back to GDScript source

### Debugging Tools
- **GDB**: Use `riscv64-unknown-elf-gdb` with ELF binary
- **addr2line**: Map PC addresses to source locations
- **Sandbox Profiling**: Use existing `sandbox_profiling.cpp` infrastructure

### Source Mapping
```cpp
// Store mapping: PC address -> GDScript source location
HashMap<uint64_t, Pair<String, int>> pc_to_source_map;

// During compilation, track:
// - StableHLO operation -> GDScript opcode/IP
// - LLVM instruction -> StableHLO operation
// - RISC-V instruction -> LLVM instruction
```

## Linker Script

Create minimal linker script for RISC-V ELF64:

```ld
/* linker.ld */
ENTRY(_start)

MEMORY {
    RAM : ORIGIN = 0x100000, LENGTH = 1M
}

SECTIONS {
    .text : {
        *(.text.start)
        *(.text)
    } > RAM
    
    .rodata : {
        *(.rodata)
    } > RAM
    
    .data : {
        *(.data)
    } > RAM
    
    .bss : {
        *(.bss)
    } > RAM
}
```

## Syscall Integration

The compiled ELF must link with Godot syscall stubs:

```cpp
// syscalls_stub.c
extern void godot_syscall_print(const char* msg, size_t len);
extern void* godot_vcall(void* obj, const char* method, ...);
// ... other syscalls

// Minimal _start for RISC-V
void _start() {
    // Initialize minimal runtime
    // Call main() if needed
    // Exit via syscall
}
```

## Performance Optimizations

1. **Caching**: Cache compiled ELF binaries by function signature hash
2. **Parallel Compilation**: Compile multiple functions in parallel
3. **Incremental**: Only recompile when GDScript source changes
4. **LTO**: Use Link-Time Optimization for smaller binaries

## Error Handling

- Validate MLIR syntax before compilation
- Check tool availability (mlir-opt, llc, ld)
- Provide clear error messages with source locations
- Fallback to StableHLO text if compilation fails

## Testing

1. **Unit Tests**: Test each compilation phase independently
2. **Integration Tests**: Test full pipeline with sample GDScript functions
3. **Sandbox Tests**: Verify ELF execution in sandbox
4. **Debugging Tests**: Verify debug symbols and source mapping

## Future Enhancements

1. **JIT Compilation**: Compile on-demand in sandbox
2. **Optimization Passes**: Add MLIR optimization passes
3. **Custom Operations**: Optimize Godot-specific operations
4. **Multi-target**: Support other architectures (ARM, x86)

## Dependencies

### Required Tools
- MLIR/LLVM toolchain (mlir-opt, mlir-translate)
- LLVM compiler (llc) with RISC-V backend
- RISC-V toolchain (riscv64-unknown-elf-gcc, riscv64-unknown-elf-ld)

### Optional Tools
- GDB for debugging
- addr2line for source mapping
- objdump for disassembly

## Implementation Priority

1. **Phase 1**: Basic compilation pipeline (MLIR -> LLVM IR -> RISC-V ELF)
2. **Phase 2**: Integration with GDScriptFunction::compile_to_elf64()
3. **Phase 3**: Debugging support and source mapping
4. **Phase 4**: Performance optimizations and caching
5. **Phase 5**: Advanced features (JIT, custom optimizations)
