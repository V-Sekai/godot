# MCP Agents Integration

This document describes how to integrate AI agents with the Godot MCP (Model Context Protocol) server.

## Overview

The Model Context Protocol (MCP) enables AI assistants to interact with external tools and resources through a standardized JSON-RPC 2.0 interface. The Godot MCP implementation provides a **unified MCP server** that offers safe GDScript execution capabilities with configurable security restrictions.

### Server Architecture

| Server | Port | Domain | Tools | Resources |
|--------|------|--------|-------|-----------|
| **Godot MCP** | 8083 | Safe GDScript execution & scene tree manipulation | `compile_script`, `execute_script`, `execute_action_sequence` | `resource://...` |

### Key Features

- **Unified Server**: Single MCP server providing GDScript execution and scene tree manipulation
- **HTTP Transport**: JSON-RPC 2.0 over HTTP for reliable communication
- **Multiple Actions**: SafeGDScript compilation, execution, and mesh export through action sequences
- **Resources & Prompts**: Support for MCP resources and prompts for context sharing
- **Security Restrictions**: Configurable permission levels (`public` - maximum security)
- **Action Executor**: Forward executor for scene tree operations with timeout protection
- **MCP Spec Compliant**: Full adherence to MCP protocol specifications
- **Cursor Integration**: Optimized for AI assistant integration
- **JSON Schema**: Complete parameter validation and documentation

The server implements MCP protocol version 2024-11-05 and uses HTTP transport for communication.

## Getting Started

### Accessing the MCP Server

The Godot MCP server provides a unified interface for safe GDScript execution. The server runs on port 8083 and provides comprehensive tool discovery and execution capabilities.

#### Running the MCP Server

```bash
# Start the Godot MCP server
./bin/godot.macos.editor.arm64 --headless --mcp-server --path modules/mcp/test_project

# The server will be available at http://localhost:8083
# Use MCP client tools to connect and discover available tools
```

#### Cursor Integration

The Godot MCP server is designed to work seamlessly with Cursor AI assistant:

```json
// Add to your Cursor MCP configuration (~/.cursor/mcp.json)
{
  "mcpServers": {
    "godot-mcp": {
      "command": "godot.macos.editor.arm64",
      "args": ["--headless", "--mcp-server", "--path", "modules/mcp/test_project"],
      "env": {}
    }
  }
}
```

#### Server Implementation Example

The MCP server is integrated into the Godot engine and can be started via command line flags.

```gdscript
extends Node

func _ready():
    # Get the MCP server singleton
    var mcp_server = Engine.get_singleton("MCPServer")

    if mcp_server:
        # The server starts automatically if --mcp-server flag is provided
        # Default port is 8083
        
        # Register domain-specific tools
        mcp_server.register_tool("domain_specific_tool", _tool_handler)

# Poll for incoming messages
func _process(delta):
    var mcp_server = Engine.get_singleton("MCPServer")
    if mcp_server and mcp_server.is_server_running():
        mcp_server.poll()
```

## Resources and Prompts

The Godot MCP server supports MCP Resources and Prompts to provide additional context to AI agents.

### Resources

Resources allow agents to read data from the Godot environment. They are identified by URIs starting with `resource://`.

- **List Resources**: `resources/list` returns all registered resources.
- **Read Resource**: `resources/read` returns the content of a specific resource.

Example resource registration in GDScript:
```gdscript
func _ready():
    var mcp_server = Engine.get_singleton("MCPServer")
    if mcp_server:
        mcp_server.register_resource("project_info", _get_project_info)

func _get_project_info(params, request_id, client_id):
    var info = {
        "name": ProjectSettings.get_setting("application/config/name"),
        "version": ProjectSettings.get_setting("application/config/version")
    }
    return info
```

### Prompts

Prompts provide reusable templates for agent interactions.

- **List Prompts**: `prompts/list` returns all registered prompt templates.
- **Get Prompt**: `prompts/get` returns a specific prompt with its messages.

Example prompt registration:
```gdscript
func _ready():
    var mcp_server = Engine.get_singleton("MCPServer")
    if mcp_server:
        mcp_server.register_prompt("explain_scene", {
            "description": "Explains the current scene structure",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": "Please explain the structure of the current scene."
                    }
                }
            ]
        })
```

## Agent Architecture

### MCP Server Connection

AI agents connect to the unified Godot MCP server to access GDScript execution capabilities:

```python
import asyncio
import httpx
import json
from typing import Dict, List, Any

# Godot MCP Server configuration
GODOT_MCP_SERVER = {
    "url": "http://localhost:8083",
    "description": "Safe GDScript execution with security restrictions"
}

async def discover_godot_server():
    """Discover tools from the Godot MCP server"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Get tools
            tools_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            response = await client.post(GODOT_MCP_SERVER["url"], json=tools_payload)
            result = response.json()

            if "result" in result and "tools" in result["result"]:
                tools = result["result"]["tools"]
                print(f"✅ Found {len(tools)} tools on Godot MCP server")
                return tools
            else:
                print("❌ Invalid MCP response")
                return []

    except Exception as e:
        print(f"❌ Godot MCP server unavailable: {e}")
        return []

async def call_godot_tool(tool_name: str, arguments: Dict[str, Any]):
    """Call a tool on the Godot MCP server"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    async with httpx.AsyncClient(timeout=30.0) as client:  # Longer timeout for tool calls
        response = await client.post(GODOT_MCP_SERVER["url"], json=payload)
        return response.json()

# Usage
async def main():
    # Discover available tools
    tools = await discover_godot_server()

    print("Available Godot MCP tools:")
    for tool in tools:
        print(f"  • {tool['name']}: {tool.get('description', 'No description')}")

    # Example: Execute GDScript code
    result = await call_godot_tool(
        "execute_script",
        {
            "code": "func run():\n    return 2 + 3",
            "language": "safegdscript",
            "method": "run"
        }
    )
    print("Execution result:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Godot Tool Examples

### Tool Handler Signature

The Godot MCP server provides a unified tool interface with comprehensive parameter validation and security restrictions.

### SafeGDScript Execution

SafeGDScript execution is handled through the `execute_script` tool. This provides a way to run logic within the Godot environment with strict security boundaries.

#### Tool Parameters (execute_script)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `code` | string | Yes | The SafeGDScript code to execute |
| `language` | string | Yes | Language type (must be "safegdscript") |
| `method` | string | No | The method to call in the script (e.g., "run") |
| `input_properties` | object | No | Properties to set before execution |
| `permission_level` | string | No | Security level (default: "public") |

#### SafeGDScript Capabilities

SafeGDScript supports:
- Complex mathematical operations
- Array and dictionary manipulation
- Control flow (loops, conditionals)
- String processing
- Algorithm implementation
- @export variable syntax for data storage

**Note**: SafeGDScript executes for side effects and can return structured data via a specified method (e.g., `run()`). You must specify the `method` parameter in the tool call to execute a specific function.

### Compile Script Tool

The `compile_script` tool validates SafeGDScript syntax without executing the code.

#### Tool Parameters (compile_script)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `code` | string | Yes | The SafeGDScript code to validate |
| `language` | string | Yes | Language type (must be "safegdscript") |

#### Example Usage

```python
# Validate script syntax
result = await call_godot_tool(
    "compile_script",
    {
        "code": "@export var result: int = 42\nfunc run():\n    return result",
        "language": "safegdscript"
    }
)
```

### Execute Action Sequence Tool

The `execute_action_sequence` tool executes sequences of actions on the Godot scene tree with timeout protection.

#### Tool Parameters (execute_action_sequence)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `actions` | array | Yes | List of action objects to execute |
| `timeout` | number | No | Execution timeout in seconds (default: 5.0) |

#### Action Types

1. **command_call**: Call a method on a node (e.g., `create_node`, `set_property`)
2. **command_set_script**: Execute SafeGDScript on a node
3. **command_export_scene**: Export the current scene to a file

#### Example Usage

```python
# Create a node and set its name
result = await call_godot_tool(
    "execute_action_sequence",
    {
        "actions": [
            {
                "action": "command_call",
                "method": "create_node",
                "arguments": ["Node3D", "MyNode"]
            },
            {
                "action": "command_set_script",
                "script": "func run():\n    print('Hello from action sequence')",
                "method": "run"
            }
        ]
    }
)
```

#### Example Usage

```python
import httpx
import json
import asyncio

async def execute_gdscript_code():
    # Execute safe GDScript code
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "execute_script",
            "arguments": {
                "code": """func run():
    var vec1 = Vector2(10, 20)
    var vec2 = Vector2(5, 15)
    return vec1 + vec2""",
                "language": "safegdscript",
                "method": "run",
                "permission_level": "public"
            }
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8083", json=payload)
        result = response.json()
        print("Execution result:", result)

if __name__ == "__main__":
    asyncio.run(execute_gdscript_code())
```

## SafeGDScript Capabilities

The Godot MCP server provides **SafeGDScript** execution - a secure subset of GDScript designed for AI agent code execution. SafeGDScript excels at computational work and algorithmic processing while maintaining maximum security.

### Confirmed Capabilities (✅ Tested & Working)

- ✅ **Mathematical Operations**: Arithmetic, built-in functions (`abs`, `sqrt`, `sin`, etc.), complex expressions.
- ✅ **Data Structures**: Arrays and Dictionaries with standard methods.
- ✅ **Control Flow**: `if/elif/else`, `for` loops, `while` loops, `break`, `continue`.
- ✅ **String Operations**: Concatenation and standard string methods.
- ✅ **Variable Management**: Local variables and `@export` syntax for data storage.

### Security Restrictions (🔒 Blocked)

The following operations are **completely blocked** for security:
- 🔒 **System Access**: OS operations, file system access.
- 🔒 **Engine Access**: Engine singletons, expression evaluation.
- 🔒 **Network Access**: HTTP requests, socket operations.

## Best Practices

### Error Handling

Always check for both JSON-RPC errors and tool-specific execution errors:

```python
async def safe_call_tool(client, tool_name, arguments):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments}
    }
    response = await client.post("http://localhost:8083", json=payload)
    result = response.json()
    
    if "error" in result:
        print(f"RPC Error: {result['error']}")
        return None
        
    return result.get("result")
```

### Timeout Protection

The server enforces a 5-second timeout for all script executions to prevent infinite loops.

## Development & Testing

### Building Godot with MCP Support

```bash
# Build Godot editor with MCP module
scons platform=macos arch=arm64 target=editor module_mcp_enabled=yes
```

### Running the Server

```bash
# Run MCP server with auto-start flag
./bin/godot.macos.editor.arm64 --headless --mcp-server --path modules/mcp/test_project
```

## References

- [MCP Specification](https://modelcontextprotocol.io/specification/2024-11-05/)
- [Godot MCP Server Documentation](./README.md)
- [Godot Engine](https://godotengine.org/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [godot-sandbox GDExtension](https://github.com/libriscv/godot-sandbox)
