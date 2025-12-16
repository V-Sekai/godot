---
name: Minimize Opcode Implementation via Composite Patterns
overview: Aggressively consolidate opcodes by replacing specialized variants with composite patterns using existing core opcodes, reducing from ~24 functional opcodes to ~8-10 core opcodes
todos: []
---

# Minimize Opcode Implementation via Composite Patterns

## Goal

Reduce the number of directly implemented opcodes from ~24 to ~8-10 core opcodes by using composite patterns. **ALL opcodes must be supported** - no VM fallback allowed. Complex opcodes are implemented using:

1. Composite patterns (routing variants to core implementations)
2. Syscalls (calling VM operations via ECALL_* syscalls)

This minimizes the number of core opcode implementations while ensuring complete opcode coverage.

## Consolidation Strategy

### 1. Return Opcodes: 6 → 1

**Current**: `OPCODE_RETURN`, `OPCODE_RETURN_TYPED_BUILTIN`, `OPCODE_RETURN_TYPED_ARRAY`, `OPCODE_RETURN_TYPED_DICTIONARY`, `OPCODE_RETURN_TYPED_NATIVE`, `OPCODE_RETURN_TYPED_SCRIPT`

**Consolidated**: All use `OPCODE_RETURN` pattern

- All typed return variants generate identical C code (type checking happens at bytecode level)
- Single implementation: `*result = <value>; return;`

**Files to modify**:

- `modules/gdscript_elf/src/gdscript_bytecode_c_code_generator.cpp` (lines 150-174)
- `modules/gdscript_elf/src/gdscript_bytecode_elf_compiler.cpp` (lines 64-69)

### 2. Assign Opcodes: 5 → 1

**Current**: `OPCODE_ASSIGN`, `OPCODE_ASSIGN_NULL`, `OPCODE_ASSIGN_TRUE`, `OPCODE_ASSIGN_FALSE`, `OPCODE_ASSIGN_TYPED_*`

**Consolidated**: All use `OPCODE_ASSIGN` pattern

- `ASSIGN_NULL` → `ASSIGN` with `Variant()` constant
- `ASSIGN_TRUE` → `ASSIGN` with `true` constant
- `ASSIGN_FALSE` → `ASSIGN` with `false` constant
- `ASSIGN_TYPED_*` → `ASSIGN` (type checking at bytecode level)

**Implementation**: Route all assign variants to single `ASSIGN` case that handles:

```cpp
dst = src;  // Where src can be Variant(), true, false, or any value
```

**Files to modify**:

- `modules/gdscript_elf/src/gdscript_bytecode_c_code_generator.cpp` (lines 175-229)
- `modules/gdscript_elf/src/gdscript_bytecode_elf_compiler.cpp` (lines 70-78)

### 3. Jump Opcodes: 5 → 3

**Current**: `OPCODE_JUMP`, `OPCODE_JUMP_IF`, `OPCODE_JUMP_IF_NOT`, `OPCODE_JUMP_TO_DEF_ARGUMENT`, `OPCODE_JUMP_IF_SHARED`

**Consolidated**:

- `JUMP_TO_DEF_ARGUMENT` → `JUMP` (same unconditional jump)
- `JUMP_IF_SHARED` → `JUMP_IF` (treat as conditional jump)
- Keep: `JUMP`, `JUMP_IF`, `JUMP_IF_NOT`

**Files to modify**:

- `modules/gdscript_elf/src/gdscript_bytecode_c_code_generator.cpp` (lines 230-293)
- `modules/gdscript_elf/src/gdscript_bytecode_elf_compiler.cpp` (lines 79-83)

### 4. Keep Core Opcodes (Minimal Set)

**Essential opcodes to keep**:

1. `OPCODE_RETURN` - Function return
2. `OPCODE_ASSIGN` - Assignment (handles all variants)
3. `OPCODE_JUMP` - Unconditional jump
4. `OPCODE_JUMP_IF` - Conditional jump (true)
5. `OPCODE_JUMP_IF_NOT` - Conditional jump (false)
6. `OPCODE_OPERATOR_VALIDATED` - All operators
7. `OPCODE_GET_MEMBER` - Property get
8. `OPCODE_SET_MEMBER` - Property set
9. `OPCODE_CALL` - Method call (all CALL* variants use ECALL_VCALL syscall)
10. `OPCODE_LINE` - Debug metadata (no-op comment)
11. `OPCODE_BREAKPOINT` - Debug metadata (no-op comment)
12. `OPCODE_ASSERT` - Debug metadata (no-op comment)
13. `OPCODE_END` - Function end (no-op comment)
14. All `OPCODE_TYPE_ADJUST_*` - No-op comments (already minimal)

**Total core opcodes: ~14** (down from ~24)

**All other opcodes**: Implemented via syscalls or composite patterns. Every opcode generates valid C code - no fallback, no TODO comments.

### 5. All Remaining Opcodes → Composite Patterns or Syscalls

**Requirement**: ALL opcodes must be supported. No VM fallback allowed. Use composite patterns or syscalls to implement remaining opcodes.

**Implementation Strategy**:

1. **Method Calls** → Use `ECALL_VCALL` syscall

   - All `CALL*` variants route to single implementation using `ECALL_VCALL`
   - Marshals arguments and calls VM via syscall

2. **Keyed/Indexed Operations** → Use `ECALL_VFETCH`/`ECALL_VSTORE` or array/dict syscalls

   - `GET_KEYED*`, `GET_INDEXED*` → Use `ECALL_ARRAY_AT` or `ECALL_DICTIONARY_OPS`
   - `SET_KEYED*`, `SET_INDEXED*` → Use array/dict syscalls

3. **Named Operations** → Use `ECALL_OBJ_PROP_GET`/`ECALL_OBJ_PROP_SET` (same as GET_MEMBER/SET_MEMBER)

   - `GET_NAMED*`, `SET_NAMED*` → Route to property get/set syscalls

4. **Constructors** → Use `ECALL_VCREATE` syscall

   - All `CONSTRUCT*` variants use `ECALL_VCREATE` with appropriate parameters

5. **Casts** → Use `ECALL_VASSIGN` or composite pattern

   - Cast operations can use variant assignment syscall or composite ASSIGN pattern

6. **Type Tests** → Use `ECALL_VCALL` or composite pattern

   - Type tests can be implemented via VM call or operator comparison

7. **Iterators** → Use composite pattern (JUMP + ASSIGN + OPERATOR_VALIDATED)

   - Iteration can be built from core opcodes: initialize counter, loop with JUMP_IF, increment with OPERATOR_VALIDATED

8. **Static Variables** → Use `ECALL_VFETCH`/`ECALL_VSTORE` syscalls

   - Static variable access via VM syscalls

9. **Global Operations** → Use `ECALL_VFETCH`/`ECALL_VSTORE` syscalls

   - Global access via VM syscalls

10. **Await Operations** → Use `ECALL_VCALL` syscall

    - Await operations call VM via syscall (async handling in VM)

11. **Lambda Creation** → Use `ECALL_CALLABLE_CREATE` syscall

    - Lambda creation via callable syscall

12. **Non-validated Operators** → Use `ECALL_VCALL` syscall

    - `OPCODE_OPERATOR` (non-validated) uses VM call instead of operator_funcs[]

**Key Principle**: Every opcode must generate valid C code that either:

- Uses a core opcode implementation (composite pattern)
- Calls VM via syscall (ECALL_*)
- Generates no-op comment (TYPE_ADJUST_*, debug opcodes)

**No opcode should generate empty code or TODO comments that would cause compilation failure.**

**Syscall Mapping Reference** (from `modules/sandbox/src/syscalls.h`):

- `ECALL_VCALL` (501) - Variant call (method calls)
  - Signature: `(GuestVariant* vp, gaddr_t method, unsigned mlen, gaddr_t args_ptr, gaddr_t args_size, gaddr_t vret_addr)`
  - Used for: All method calls, utility calls, builtin calls
- `ECALL_VASSIGN` (503) - Variant assignment
  - Used for: Assignment operations, casts
- `ECALL_VFETCH` (519) - Variant fetch (get operations)
  - Signature: `(unsigned index, gaddr_t gdata, int method)`
  - Used for: Getting values from scoped variants (static variables, globals)
- `ECALL_VSTORE` (520) - Variant store (set operations)
  - Signature: `(unsigned* vidx, Variant::Type type, gaddr_t gdata, gaddr_t gsize)`
  - Used for: Storing values to scoped variants (static variables, globals)
- `ECALL_VCREATE` (517) - Variant creation (constructors)
  - Signature: `(GuestVariant* vp, Variant::Type type, int method, gaddr_t gdata)`
  - Used for: All constructor operations
- `ECALL_ARRAY_OPS` (521) - Array operations
  - Used for: Array element access, array operations
- `ECALL_ARRAY_AT` (522) - Array element access
  - Used for: Indexed get/set operations on arrays
- `ECALL_DICTIONARY_OPS` (524) - Dictionary operations
  - Used for: Keyed get/set operations on dictionaries
- `ECALL_OBJ_PROP_GET` (545) - Object property get
  - Used for: GET_MEMBER, GET_NAMED operations
- `ECALL_OBJ_PROP_SET` (546) - Object property set
  - Used for: SET_MEMBER, SET_NAMED operations
- `ECALL_CALLABLE_CREATE` (538) - Callable creation (lambdas)
  - Used for: Lambda creation operations

**Note**: The current `generate_syscall()` function only supports up to 5 arguments via registers (a0-a4). For syscalls with more arguments (like ECALL_VCALL with 6 args), we need to either:

1. Extend `generate_syscall()` to support more arguments
2. Use a different approach (e.g., pass arguments via memory/stack)
3. Create specialized syscall generation functions for complex syscalls

## Implementation Steps

### Step 1: Consolidate Return Opcodes

- Merge all `RETURN_TYPED_*` cases into single `RETURN` case
- Update `is_basic_opcodes_only()` to accept all return variants but route to single implementation

### Step 2: Consolidate Assign Opcodes

- Create helper function to resolve assign source (handles NULL/TRUE/FALSE as constants)
- Route all assign variants to single `ASSIGN` case
- Update `is_basic_opcodes_only()` accordingly

### Step 3: Consolidate Jump Opcodes

- Route `JUMP_TO_DEF_ARGUMENT` to `JUMP` case
- Route `JUMP_IF_SHARED` to `JUMP_IF` case
- Update `is_basic_opcodes_only()` accordingly

### Step 4: Extend Syscall Generation for Complex Syscalls

- Extend `generate_syscall()` or create specialized functions for syscalls with >5 arguments
- For `ECALL_VCALL` (6 args): May need to pass some arguments via memory or use a different calling convention
- Ensure syscall argument marshaling matches the expected syscall signatures

### Step 5: Implement All Remaining Opcodes via Syscalls

- Implement method calls using `ECALL_VCALL` syscall (all CALL* variants)
- Implement keyed/indexed operations using `ECALL_ARRAY_AT` or `ECALL_DICTIONARY_OPS`
- Implement named operations using `ECALL_OBJ_PROP_GET`/`ECALL_OBJ_PROP_SET` (same as GET_MEMBER/SET_MEMBER)
- Implement constructors using `ECALL_VCREATE` syscall
- Implement casts using `ECALL_VASSIGN` or composite ASSIGN pattern
- Implement type tests using `ECALL_VCALL` or composite operator pattern
- Implement static variables using `ECALL_VFETCH`/`ECALL_VSTORE` syscalls
- Implement global operations using `ECALL_VFETCH`/`ECALL_VSTORE` syscalls
- Implement await operations using `ECALL_VCALL` syscall
- Implement lambda creation using `ECALL_CALLABLE_CREATE` syscall
- Implement non-validated operators using `ECALL_VCALL` syscall
- Implement iterators using composite pattern (JUMP + ASSIGN + OPERATOR_VALIDATED) or syscalls if needed
- Ensure every opcode generates valid C code (no TODO comments or empty code)

### Step 6: Update Opcode Support Checker

- Modify `is_basic_opcodes_only()` in `gdscript_bytecode_elf_compiler.cpp` to accept ALL opcodes (since all are now supported)
- Remove the default case that returns false - all opcodes should be accepted
- Add all opcode cases to the switch statement (iterators, constructors, casts, etc.)

### Step 7: Update Documentation

- Update `AGENTS.md` and `IMPLEMENTATION_STATUS.md` to reflect minimized core opcode count
- Document composite pattern approach for variants
- Document syscall-based implementation for VM operations
- Note that ALL opcodes are now supported (no fallback)

## Files to Modify

1. **`modules/gdscript_elf/src/gdscript_bytecode_c_code_generator.cpp`**

   - Consolidate return opcodes (lines 150-174)
   - Consolidate assign opcodes (lines 175-229)
   - Consolidate jump opcodes (lines 230-293)
   - Extend or create specialized syscall generation functions for complex syscalls
   - Implement all remaining opcodes via syscalls or composite patterns (lines 314-591)

2. **`modules/gdscript_elf/src/gdscript_bytecode_elf_compiler.cpp`**

   - Update `is_basic_opcodes_only()` to accept ALL opcodes (lines 49-150)
   - Add all opcode cases to the switch statement
   - Remove default case that returns false

3. **`modules/gdscript_elf/AGENTS.md`**

   - Update opcode count documentation

4. **`modules/gdscript_elf/IMPLEMENTATION_STATUS.md`**

   - Update opcode count and explain composite pattern approach

## Benefits

1. **Reduced code duplication**: Single implementation for each operation type
2. **Easier maintenance**: Changes to core opcodes automatically apply to all variants
3. **Clearer architecture**: Core opcodes are clearly distinguished from variants
4. **Future-proof**: Easy to add specialized implementations later if needed

## Testing

- Existing tests should continue to pass (same C code output)
- Verify that consolidated opcodes generate identical C code to current implementation
- Verify that all opcodes generate valid C code (no compilation errors)
- Test that syscall-based opcodes properly marshal arguments and call VM
- Test complex syscalls (ECALL_VCALL with 6 args) to ensure argument marshaling works correctly
- Ensure no opcodes generate TODO comments or empty code that would cause compilation failure
- Verify that functions with previously unsupported opcodes can now be compiled to ELF

## Implementation Notes

### Syscall Argument Marshaling

The current `generate_syscall()` function uses RISC-V calling convention with registers a0-a4 for arguments. For syscalls requiring more arguments:

1. **Option A**: Extend to use a5, a6, etc. (if RISC-V ABI supports it)
2. **Option B**: Pass additional arguments via memory (stack or heap)
3. **Option C**: Create specialized syscall generation functions that handle specific syscall signatures

For `ECALL_VCALL` specifically, the signature requires 6 arguments. We may need to:
- Pass some arguments via memory
- Use a wrapper function that marshals arguments correctly
- Create a specialized `generate_vcall_syscall()` function

### Composite Pattern Examples

- `ASSIGN_NULL`: Generate `stack[dst] = Variant();` (composite of ASSIGN with constant)
- `ASSIGN_TRUE`: Generate `stack[dst] = true;` (composite of ASSIGN with constant)
- `RETURN_TYPED_*`: Generate same code as `RETURN` (type checking at bytecode level)
- `JUMP_TO_DEF_ARGUMENT`: Generate same code as `JUMP` (unconditional jump)
- Iterators: Can be built from JUMP_IF + ASSIGN + OPERATOR_VALIDATED (increment counter)