# MCP Test Project

This directory contains a minimal Godot project for testing the MCP (Model Context Protocol) module.

## Running the MCP Server Test

1. Build Godot with MCP module enabled:
   ```bash
   scons platform=macos arch=arm64 target=editor module_mcp_enabled=yes
   ```

2. Run the test from this directory:
   ```bash
   cd test_project
   ../../../bin/godot.macos.editor.arm64 --headless --path .
   ```

3. The MCP server will start on port 8080 with test tools and resources registered.

## Available Test Tools

- **echo**: Echoes back the input message with timestamp
- **math_add**: Adds two numbers
- **get_scene_tree**: Returns scene hierarchy information

## Available Test Resources

- **server_status**: Server status information
- **scene_info**: Current scene information

## Testing with curl

```bash
# List available tools
curl -X POST http://localhost:8080 \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# Test echo tool
curl -X POST http://localhost:8080 \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"message":"Hello MCP!"}}}'
```

## Running Python Compliance Tests

```bash
cd ../modules/mcp/tests
python3 test_mcp_compliance.py
```
