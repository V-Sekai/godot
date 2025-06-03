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

**Do not use this in:**

- Production games or applications
- Commercial projects
- Critical systems
- Any project expecting working functionality

**This is for experimentation, planning, and development reference only!**

## Research goals

(1) Research the complete technical specifications and architectural components
of both the Model Context Protocol (MCP) and ExecuTorch models, including their
data formats, communication protocols, and operational lifecycles. Investigate
methods for designing the MCP Tool primitive to expose ExecuTorch model
inference capabilities, focusing on defining precise JSON Schemas for input and
output data types and exploring strategies for semantic enrichment of JSON
responses.

(2) Evaluate suitable native SDKs for MCP server development, prioritizing
options that align with native/bare-metal deployment and facilitate
self-contained packaging. Research the implementation details of the MCP Tool
handler function, focusing on efficient conversion of JSON inputs to ExecuTorch
tensor formats, invoking model inference, and transforming raw ExecuTorch
outputs back into structured JSON responses, along with robust error handling
strategies.

(3) Investigate various tools and techniques for packaging the entire system,
including the MCP server, ExecuTorch model, and all necessary runtimes and
dependencies, into a single, self-contained, and portable execution unit,
prioritizing methods that minimize external dependencies and runtime overhead.
Analyze different deployment models for the self-contained package, identify
performance optimization techniques specific to ExecuTorch and MCP, and
establish best practices for operational aspects.

(4) Synthesize all gathered information into a comprehensive integration guide
and detailed documentation for performing a one-to-one wrapping of an ExecuTorch
model into an MCP server, demonstrating the process with a practical case study
and emphasizing native/bare-metal deployment strategies. Research specific
architectural suggestions for integrating an ExecuTorch-MCP server C++ module
within a Godot Engine monolithic bundle, including methods for embedding
ExecuTorch model weights as a pck, utilizing Godot's native C++ extension
system, and implementing efficient data flow between Godot's scene system and
the ExecuTorch inference engine while maintaining the self-contained, portable
execution requirements.
