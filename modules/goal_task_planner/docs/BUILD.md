# Build Instructions

This document provides build instructions for the Goal Task Planner module.

## Prerequisites

-   Godot Engine source code
-   SCons build system
-   Platform-specific build tools (Xcode for macOS, Visual Studio for Windows, etc.)

## Build Commands

### Build Editor

Build the Godot editor with the goal_task_planner module:

```bash
godot-build-editor
```

Or manually:

```bash
scons platform=macos arch=arm64 target=editor dev_build=yes debug_symbols=yes compiledb=yes tests=yes generate_bundle=yes cache_path=/Users/ernest.lee/.scons_cache
```

**Note**: Adjust platform, arch, and cache_path for your system.

### Build and Run Tests

Build the editor and run all tests:

```bash
godot-build-editor && ./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests
```

### Run Specific Test

Run a specific test by name:

```bash
./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests --test-name="<Test Name>"
```

Example:

```bash
./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests --test-name="*Persona*"
```

## Build System

The module uses **SCons** build system. Source files are automatically discovered via `SCsub`.

### Source File Discovery

All `.cpp` files in the module directory are automatically included via:

```python
env.add_source_files(env.modules_sources, "*.cpp")
```

### Class Registration

Classes are registered in `register_types.cpp` during module initialization at `MODULE_INITIALIZATION_LEVEL_SCENE`.

## Platform-Specific Notes

### macOS (ARM64)

-   Uses `platform=macos arch=arm64`
-   Output: `bin/godot.macos.editor.dev.arm64`
-   Bundle: `bin/godot_macos_editor_dev.app`

### Other Platforms

Adjust the `platform` and `arch` parameters:

-   Windows: `platform=windows`
-   Linux: `platform=linuxbsd`
-   Android: `platform=android`

## Troubleshooting

### Build Errors

1. **Missing dependencies**: Ensure all prerequisites are installed
2. **Cache issues**: Clear SCons cache: `scons --clean`
3. **Compilation errors**: Check for syntax errors in source files

### Test Failures

1. **Stale binary**: Rebuild after code changes: `godot-build-editor`
2. **Test isolation**: Ensure tests call `plan->reset()` at the start
3. **State pollution**: Use `state.duplicate(true)` before passing state to `find_plan()`

## Development Workflow

1. Make code changes
2. Build: `godot-build-editor`
3. Run tests: `./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests`
4. Fix any failures
5. Repeat

## Related Documentation

-   [AGENTS.md](AGENTS.md) - Detailed development guide
-   [README.md](README.md) - User-facing documentation
