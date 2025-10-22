# Agents

This document describes how AI agents integrate with the Godot project development workflow using the Beads issue tracking system.

## Agent Integration

AI agents are designed to work seamlessly with the Beads dependency-aware issue tracker to manage development tasks efficiently. Agents can:

-   **Discover work**: Create new issues when identifying tasks, bugs, or features
-   **Claim work**: Take ownership of ready tasks from the `bd ready` queue
-   **Track progress**: Update issue status as work progresses
-   **Complete work**: Close issues when tasks are finished

## Agent Workflow

### 1. Work Discovery

When an agent identifies new work during code analysis, testing, or user interaction:

```bash
bd create "Fix rendering bug in OIT pipeline" -t bug -p 1
```

### 2. Claiming Tasks

Agents check for unblocked work ready to start:

```bash
bd ready  # Shows issues with no blocking dependencies
bd update issue-id --status in_progress --assignee agent-name
```

### 3. Progress Updates

As work progresses, agents update status and add notes:

```bash
bd update issue-id --notes "Investigating shader compilation issue"
```

### 4. Completion

When work is done, agents close the issue:

```bash
bd close issue-id --reason "Fixed shader compilation in gles3_builders.py"
```

## Dependencies

Agents respect dependency chains to avoid working on blocked tasks:

-   **Blocks**: Hard dependencies that must be completed first
-   **Related**: Soft connections for awareness
-   **Parent-child**: Epic/subtask relationships
-   **Discovered-from**: Auto-created related work

## Best Practices

-   Always check `bd ready` before starting new work
-   Update status frequently to reflect progress
-   Add detailed notes for complex tasks
-   Close issues promptly when work is complete
-   Create dependencies when work has prerequisites

## Integration Points

-   **Code Analysis**: Agents scan for TODOs, FIXMEs, and potential issues
-   **Testing**: Automated test failures trigger issue creation
-   **Build System**: Build failures create blocking issues
-   **Documentation**: Missing docs trigger documentation tasks
-   **Performance**: Performance regressions create optimization tasks

## Agent Types

-   **Code Review Agent**: Analyzes pull requests and creates follow-up issues
-   **Test Agent**: Runs tests and creates issues for failures
-   **Build Agent**: Monitors builds and creates issues for failures
-   **Documentation Agent**: Checks documentation coverage
-   **Performance Agent**: Monitors performance metrics
-   **Security Agent**: Scans for security vulnerabilities

## Database Extension

Agents can extend the Beads database with custom tables for:

-   Agent execution logs
-   Performance metrics
-   Code analysis results
-   Test run histories

See the Beads documentation for database extension patterns.

## Recent Debugging Tasks

### OIT Shader Compilation Debugging

**Problem:** The `godot_output.log` showed shader compilation errors related to `oit_debug_tile_visualization` in `servers/rendering/renderer_rd/shaders/forward_clustered/scene_forward_clustered.glsl`. The errors indicated that the function was not found or had an incorrect signature.

**Debugging Process:**

1.  Reviewed `godot_output.log` to identify specific errors.
2.  Examined `servers/rendering/renderer_rd/shaders/forward_clustered/scene_forward_clustered.glsl` and `servers/rendering/renderer_rd/shaders/effects/oit_dispatch.glsl.inc` to understand the function call and definition.
3.  Consulted `OIT_INTEGRATION_STEPS.md` and `OIT_PROGRESS.md` for context on OIT integration.
4.  Identified that `oit_dispatch.glsl.inc` was guarded by `#ifdef USE_OIT`, and `USE_OIT` was not defined during compilation, leading to the function being undefined.

**Solution Implemented (Previous Iteration):**

1.  Moved `#include "../effects/oit_dispatch.glsl.inc"` outside the `#ifdef USE_OIT` block in `servers/rendering/renderer_rd/shaders/forward_clustered/scene_forward_clustered.glsl`.
2.  Wrapped the OIT data structures, buffers, and functions within an `#ifdef USE_OIT` block in `servers/rendering/renderer_rd/shaders/effects/oit_dispatch.glsl.inc`.
3.  Corrected a nested `#ifdef USE_OIT` in `servers/rendering/renderer_rd/shaders/effects/oit_dispatch.glsl.inc`.

**Verification (Previous Iteration):**

-   Recompiled the Godot editor using `scons platform=macos target=editor dev_build=yes debug_symbols=yes compiledb=yes tests=yes generate_bundle=yes`.
-   The compilation completed successfully without any errors.
-   Running the editor still showed shader compilation errors related to `oit_debug_tile_visualization`, indicating a caching issue or incorrect include logic.
-   Performed a clean build (`scons --clean ...`) and then recompiled, which resolved the shader compilation errors when running the editor.

**Render Mode Integration (Current Iteration):**

**Problem:** The previous approach to OIT integration (`USE_OIT` define) was not aligned with Godot's render mode system. The user requested to integrate OIT as a render mode similar to `depth_prepass_alpha`, using a descriptive name instead of an acronym.

**Debugging Process:**

1.  Searched for `depth_prepass_alpha` in shader files and C++ rendering server code to understand its implementation.
2.  Discovered that `depth_prepass_alpha` is registered as a render mode string in C++ (`scene_shader_forward_clustered.cpp`) which then triggers a preprocessor define (`#define USE_OPAQUE_PREPASS`) in the GLSL shaders.
3.  Determined that direct string searches for render modes in GLSL files were unsuccessful because the render mode strings are translated into preprocessor defines by the C++ shader compilation system.
4.  Collaborated with the user to choose a descriptive render mode name: `alpha_order_independent`.

**Solution Implemented:**

1.  **`servers/rendering/renderer_rd/forward_clustered/scene_shader_forward_clustered.h`**: Added `bool uses_alpha_order_independent = false;` to the `ShaderData` struct.
2.  **`servers/rendering/renderer_rd/forward_mobile/scene_shader_forward_mobile.h`**: Added `bool uses_alpha_order_independent = false;` to the `ShaderData` struct.
3.  **`servers/rendering/renderer_rd/forward_clustered/scene_shader_forward_clustered.cpp`**:
    *   Initialized `uses_alpha_order_independent = false;` in `ShaderData::set_code()`.
    *   Added `actions.render_mode_flags["alpha_order_independent"] = &uses_alpha_order_independent;`.
    *   Added `actions.render_mode_defines["alpha_order_independent"] = "#define USE_ALPHA_ORDER_INDEPENDENT\n";`.
4.  **`servers/rendering/renderer_rd/forward_mobile/scene_shader_forward_mobile.cpp`**:
    *   Initialized `uses_alpha_order_independent = false;` in `ShaderData::set_code()`.
    *   Added `actions.render_mode_flags["alpha_order_independent"] = &uses_alpha_order_independent;`.
    *   Added `actions.render_mode_defines["alpha_order_independent"] = "#define USE_ALPHA_ORDER_INDEPENDENT\n";`.
5.  **`servers/rendering/shader_types.cpp`**: Added `shader_modes[RS::SHADER_SPATIAL].modes.push_back({ PNAME("alpha_order_independent") });` to register the new render mode.
6.  **`servers/rendering/renderer_rd/shaders/forward_clustered/scene_forward_clustered.glsl`**: Replaced `#ifdef USE_OIT` with `#ifdef USE_ALPHA_ORDER_INDEPENDENT` around the `oit_dispatch.glsl.inc` include and OIT-related fragment shader code.
7.  **`servers/rendering/renderer_rd/shaders/forward_mobile/scene_forward_mobile.glsl`**:
    *   Added `#ifdef USE_ALPHA_ORDER_INDEPENDENT` block and included `../effects/oit_dispatch.glsl.inc` in the vertex shader section.
    *   Wrapped the OIT related code in the fragment shader with `#ifdef USE_ALPHA_ORDER_INDEPENDENT`.
8.  **`scene/resources/material.cpp`**: Updated the line to `code += ", alpha_order_independent";` when `transparency == TRANSPARENCY_ORDER_INDEPENDENT`.

**Current OIT Integration Status:**

-   OIT is now integrated as a proper render mode (`alpha_order_independent`) in both the clustered and mobile renderers.
-   The GLSL shaders will conditionally include and execute OIT-related code based on this render mode.
-   Further testing and refinement of the OIT implementation within the shaders may be required.

### `max_fragments` Field Missing Error

**Problem:** The `godot_output.log` shows a shader compilation error: `ERROR: 0:4134: 'max_fragments' : no such field in structure 'implementation_data_block'`. This indicates that the `ImplementationData` struct, as seen by the shader compiler, does not contain the `max_fragments` field, despite it being added to `servers/rendering/renderer_rd/shaders/forward_clustered/scene_forward_clustered_inc.glsl`.

**Debugging Process:**

1.  Identified the error message in `godot_output.log`.
2.  Confirmed that `max_fragments` was added to the `ImplementationData` struct in `servers/rendering/renderer_rd/shaders/forward_clustered/scene_forward_clustered_inc.glsl`.
3.  Hypothesized that the `ImplementationData` struct definition is not consistently available across all renderers (e.g., mobile renderer) or that there's a more fundamental issue with its inclusion.

**Proposed Solution:**

1.  Move the `ImplementationData` struct definition to a new common include file: `servers/rendering/renderer_rd/shaders/implementation_data_inc.glsl`.
2.  Include this new common file in both `servers/rendering/renderer_rd/shaders/forward_clustered/scene_forward_clustered_inc.glsl` and `servers/rendering/renderer_rd/shaders/forward_mobile/scene_forward_mobile_inc.glsl`. This will ensure that `max_fragments` is available to all renderers that use `ImplementationData`.
