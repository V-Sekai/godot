# Internal StableHLO to RISC-V ELF64 Compilation

## Overview

This document describes an **internal, dependency-free** approach to compiling StableHLO MLIR directly to RISC-V ELF64 binaries, entirely within Godot without external tools.

## Architecture

### Compilation Pipeline

```
StableHLO MLIR (text)
    ↓
[Internal Parser] - Parse MLIR text to AST
    ↓
[Internal Lowering] - Convert StableHLO ops to RISC-V operations
    ↓
[Code Generator] - Generate RISC-V assembly
    ↓
[ELF Generator] - Create RISC-V ELF64 binary
    ↓
RISC-V ELF64 (ready for sandbox)
```

## Implementation Strategy

### Phase 1: MLIR Parser (Internal)

**Approach**: Simple recursive descent parser for StableHLO MLIR syntax
- Parse module, functions, operations
- Extract types, constants, operations
- Build internal representation (IR)

**Key Operations to Support**:
- `stablehlo.constant` → RISC-V data section
- `stablehlo.add`, `stablehlo.multiply`, etc. → RISC-V arithmetic instructions
- `stablehlo.custom_call` → RISC-V function calls to syscalls
- `stablehlo.return` → RISC-V return instruction
- `stablehlo.compare` → RISC-V comparison + branch instructions

### Phase 2: Operation Lowering

**StableHLO → RISC-V Mapping**:

| StableHLO Operation | RISC-V Implementation |
|---------------------|----------------------|
| `stablehlo.constant dense<42.0> : tensor<f64>` | Load 64-bit float constant into register |
| `stablehlo.add %a, %b : tensor<f64>` | `fadd.d rd, rs1, rs2` (double precision add) |
| `stablehlo.multiply %a, %b : tensor<f64>` | `fmul.d rd, rs1, rs2` |
| `stablehlo.subtract %a, %b : tensor<f64>` | `fsub.d rd, rs1, rs2` |
| `stablehlo.divide %a, %b : tensor<f64>` | `fdiv.d rd, rs1, rs2` |
| `stablehlo.compare EQ, %a, %b` | `feq.d rd, rs1, rs2` (set rd=1 if equal) |
| `stablehlo.compare GT, %a, %b` | `flt.d rd, rs2, rs1` (a > b means b < a) |
| `stablehlo.custom_call @godot_vcall(...)` | Call syscall wrapper function |
| `stablehlo.return %val` | Move %val to return register, jump to return |

### Phase 3: RISC-V Assembly Generation

**Register Allocation**:
- Use RISC-V calling convention (a0-a7 for args, fa0-fa7 for float args)
- Map StableHLO SSA values to RISC-V registers
- Spill to stack when needed

**Function Prologue/Epilogue**:
```asm
# Function prologue
addi sp, sp, -16    # Allocate stack frame
sd   ra, 8(sp)      # Save return address
sd   s0, 0(sp)      # Save frame pointer
mv   s0, sp         # Set frame pointer

# Function body
# ... generated code ...

# Function epilogue
ld   ra, 8(sp)      # Restore return address
ld   s0, 0(sp)      # Restore frame pointer
addi sp, sp, 16     # Deallocate stack frame
ret                 # Return
```

### Phase 4: ELF64 Generation

**Simple ELF Generator**:
- Create ELF64 header
- Create program headers (text, data sections)
- Write RISC-V machine code
- Add symbol table for debugging
- Add section headers

**Minimal ELF Structure**:
```
ELF Header
Program Header (PT_LOAD for .text)
Program Header (PT_LOAD for .data)
.text section (RISC-V code)
.data section (constants)
.symtab section (debug symbols)
```

## Implementation Details

### 1. MLIR Parser

```cpp
class StableHLOParser {
    struct Operation {
        String name;
        Vector<String> operands;
        Vector<String> results;
        Dictionary attributes;
    };
    
    struct Function {
        String name;
        Vector<Pair<String, String>> args;  // (name, type)
        String return_type;
        Vector<Operation> operations;
    };
    
    struct Module {
        Vector<Function> functions;
    };
    
    Module parse(const String &mlir_text);
    Function parse_function(String &text, int &pos);
    Operation parse_operation(String &text, int &pos);
};
```

### 2. RISC-V Code Generator

```cpp
class RISCVCodeGenerator {
    struct Register {
        int id;
        String type;  // "i32", "f64", "i1"
    };
    
    HashMap<String, Register> value_to_register;
    int next_register = 0;
    Vector<String> assembly_lines;
    
    void generate_function(const Function &func);
    void generate_operation(const Operation &op);
    String allocate_register(const String &type);
    void emit_instruction(const String &inst);
};
```

### 3. ELF Generator

```cpp
class ELF64Generator {
    PackedByteArray generate_elf(
        const Vector<uint8_t> &code,
        const Vector<uint8_t> &data,
        const Dictionary &symbols
    );
    
    void write_elf_header(PackedByteArray &elf);
    void write_program_headers(PackedByteArray &elf);
    void write_sections(PackedByteArray &elf);
    void write_symbol_table(PackedByteArray &elf);
};
```

## StableHLO Operation Implementations

### Arithmetic Operations

```cpp
// stablehlo.add %a, %b : tensor<f64>
void generate_add(const Operation &op) {
    Register a = get_register(op.operands[0]);
    Register b = get_register(op.operands[1]);
    Register result = allocate_register("f64");
    
    emit("fadd.d " + result.name + ", " + a.name + ", " + b.name);
    set_result(op.results[0], result);
}
```

### Custom Calls (Syscalls)

```cpp
// stablehlo.custom_call @godot_vcall(%obj, %method, %args...)
void generate_custom_call(const Operation &op) {
    String callee = op.attributes["callee"];
    
    if (callee == "@godot_vcall") {
        // Load syscall number
        emit("li a7, GODOT_SYSCALL_VCALL");
        // Load arguments
        for (int i = 0; i < op.operands.size(); i++) {
            Register arg = get_register(op.operands[i]);
            emit("mv a" + itos(i) + ", " + arg.name);
        }
        // Call syscall
        emit("ecall");
        // Result in a0
        Register result = allocate_register("f64");
        emit("mv " + result.name + ", a0");
        set_result(op.results[0], result);
    }
}
```

### Constants

```cpp
// stablehlo.constant dense<42.0> : tensor<f64>
void generate_constant(const Operation &op) {
    String value = extract_constant_value(op);
    String type = extract_type(op);
    
    if (type == "tensor<f64>") {
        // Store in .data section
        uint64_t data_offset = add_data_section(value);
        // Load from data section
        Register result = allocate_register("f64");
        emit("la t0, .data + " + itos(data_offset));
        emit("fld " + result.name + ", 0(t0)");
        set_result(op.results[0], result);
    }
}
```

## RISC-V Calling Convention

**Integer Arguments**: a0-a7 (x10-x17)
**Float Arguments**: fa0-fa7 (f10-f17)
**Return Values**: a0/a1 (integer), fa0/fa1 (float)
**Callee-saved**: s0-s11 (x8-x9, x18-x27), fs0-fs11 (f8-f9, f28-f31)
**Temporary**: t0-t6 (x5-x7, x28-x31), ft0-ft11 (f0-f7, f12-f27)

## ELF64 Structure

**Minimal ELF64 for RISC-V**:
- ELF Header (64 bytes)
- Program Header Table (2 entries × 56 bytes = 112 bytes)
- .text section (RISC-V code)
- .data section (constants)
- Symbol table (optional, for debugging)

**Entry Point**: Set to function start address (typically 0x100000)

## Debugging Support

### Source Mapping
- Map RISC-V PC addresses to StableHLO operations
- Map StableHLO operations to GDScript source locations
- Store in ELF symbol table or separate debug section

### Debug Symbols
- Function names
- Variable names (from StableHLO SSA values)
- Source file and line numbers

## Implementation Steps

1. **MLIR Parser** (Week 1-2)
   - Parse basic MLIR syntax
   - Extract functions, operations, types
   - Handle constants and attributes

2. **RISC-V Code Generator** (Week 2-3)
   - Implement register allocation
   - Generate basic arithmetic operations
   - Generate function prologue/epilogue

3. **ELF Generator** (Week 3-4)
   - Create ELF64 header
   - Write program headers
   - Write sections
   - Add symbol table

4. **Integration** (Week 4)
   - Integrate with `GDScriptFunction::compile_to_elf64()`
   - Test with sandbox execution
   - Add debugging support

5. **Optimization** (Week 5+)
   - Register allocation optimization
   - Dead code elimination
   - Constant folding

## Advantages of Internal Approach

1. **No External Dependencies**: Works out of the box
2. **Fast Compilation**: Direct translation, no heavy toolchains
3. **Debuggable**: Full control over debug symbols and source mapping
4. **Small Binary Size**: Only implements what we need
5. **Cross-Platform**: Works on all platforms Godot supports
6. **Incremental**: Can add features as needed

## Limitations

1. **Not Full MLIR**: Only supports StableHLO operations we use
2. **Basic Optimizations**: No advanced optimizations (can add later)
3. **Manual Maintenance**: Need to update for new StableHLO ops

## Example: Complete Function

**Input (StableHLO)**:
```mlir
module {
  func.func @add_numbers(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f64>
    stablehlo.return %0 : tensor<f64>
  }
}
```

**Generated RISC-V Assembly**:
```asm
.section .text
.global add_numbers
add_numbers:
    # Prologue
    addi sp, sp, -16
    sd   ra, 8(sp)
    sd   s0, 0(sp)
    mv   s0, sp
    
    # Body: add %arg0, %arg1
    fadd.d fa0, fa0, fa1  # fa0 = arg0, fa1 = arg1, result in fa0
    
    # Epilogue
    ld   ra, 8(sp)
    ld   s0, 0(sp)
    addi sp, sp, 16
    ret
```

**Generated ELF64**: Binary containing the above assembly as machine code
