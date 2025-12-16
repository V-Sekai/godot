# GDScript ELF64 Compiler Tool

A command-line tool to compile GDScript functions to ELF64 binaries for execution in Godot Sandbox.

## Usage

```bash
godot --headless --script tools/gdscript_elf64_compiler.gd <input.gd> [output_dir]
```

### Arguments

- `<input.gd>` - Path to the GDScript file to compile (required)
- `[output_dir]` - Directory where ELF64 binaries will be saved (optional, defaults to current directory)

### Examples

```bash
# Compile script.gd
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd

# Compile to specific output directory
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd output/

# Compile with absolute paths
godot --headless --script tools/gdscript_elf64_compiler.gd /path/to/script.gd /path/to/output/
```

## Testing ELF Files

Use the test harness to execute ELF files in Godot Sandbox:

```bash
godot --headless --script tools/test_elf64_with_sandbox.gd <elf_file> [function_name] [args...]
```

### Example

```bash
# Test an ELF file
godot --headless --script tools/test_elf64_with_sandbox.gd output_elf/test_elf64_sample_add.elf

# Call a specific function
godot --headless --script tools/test_elf64_with_sandbox.gd output_elf/test_elf64_sample_add.elf add 5 3
```

## Compilation Mode

All ELF files are compiled using **Godot syscalls** (ECALL 500+). The binaries are designed to run in the Godot Sandbox environment, which provides the necessary syscall handlers for Godot API access.

## Output

For each function in the GDScript that can be compiled to ELF64, the tool generates a separate `.elf` file:

- Format: `<script_name>_<function_name>.elf`
- Example: If `math.gd` has a function `add`, the output will be `math_add.elf`

## Requirements

- The GDScript must have at least one function that can be compiled to ELF64
- Functions must have bytecode (not empty functions)
- The output directory must be writable (will be created if it doesn't exist)

## Error Handling

The tool will exit with:
- Exit code 0: Success
- Exit code 1: Error (invalid arguments, compilation failure, I/O errors)

## Example GDScript

```gdscript
# test.gd
func add(a: int, b: int) -> int:
    return a + b

func multiply(x: int, y: int) -> int:
    return x * y
```

Compiling this script will produce:
- `test_add.elf`
- `test_multiply.elf`
