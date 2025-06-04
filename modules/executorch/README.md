<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Godot ExecuTorch Module](#godot-executorch-module)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage in Godot](#usage-in-godot)
    - [Basic Setup](#basic-setup)
    - [Running Inference](#running-inference)
    - [Using MCP Tools](#using-mcp-tools)
    - [Performance Monitoring](#performance-monitoring)
  - [Module Architecture](#module-architecture)
    - [Core Components](#core-components)
    - [Class Hierarchy](#class-hierarchy)
  - [Module Integration Benefits](#module-integration-benefits)
  - [Development](#development)
    - [Testing the Module](#testing-the-module)
  - [Deployment](#deployment)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Godot ExecuTorch Module

This project is currently in **early development** and is **NOT ready for
production use**.

**⚠️ IMPORTANT: NOTHING CURRENTLY WORKS ⚠️**

This is a **concept/prototype** module intended to integrate ExecuTorch machine learning
inference capabilities into Godot Engine with built-in MCP (Model Context Protocol) support.

**Current Status:**

- 🔴 **Nothing is implemented** - this is concept documentation only
- 🔴 **API is completely unstable** and subject to breaking changes
- 🔴 **No working code** - placeholder implementations only
- 🔴 **No testing** - may not even compile
- 🔴 **No official releases** - use at your own risk
- 🔴 **Documentation describes planned features** that don't exist yet

**Planned Features (NOT IMPLEMENTED):**
- ExecuTorchRuntime class for model loading and inference
- Built-in MCP (Model Context Protocol) server
- GDScript integration with native performance
- Cross-platform model execution
- Performance monitoring and benchmarking
- Linear regression and other model type support

**TODO - Build System Integration:**
- 🔧 **Compile ExecuTorch separately** and use its compilation database (`compile_commands.json`) to figure out how to properly build with SCons
- 🔧 **Extract build flags and dependencies** from ExecuTorch's CMake configuration
- 🔧 **Port CMake settings to SCons** for proper Godot module integration
- 🔧 **Handle third-party dependencies** (XNNPACK, etc.) in Godot's build system

**Do not use this in:**

- Production games or applications
- Commercial projects
- Critical systems
- Any project expecting working functionality

**This is for experimentation, planning, and development reference only!**
