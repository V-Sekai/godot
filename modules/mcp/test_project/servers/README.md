# MCP Server Collection

This directory contains specialized MCP (Model Context Protocol) servers, each focused on specific functionality domains. Each server runs on a different port and provides tools and resources related to its domain.

## Available Servers

### 1. Communication Server (`communication_server.gd`)
- **Port**: 8080
- **Tools**: `echo`
- **Resources**: `server_status`
- **Purpose**: Basic communication and server status monitoring

### 2. Math Server (`math_server.gd`)
- **Port**: 8081
- **Tools**: `evaluate_expression`
- **Resources**: None
- **Purpose**: Mathematical expression evaluation and computation

### 3. Scene Server (`scene_server.gd`)
- **Port**: 8082
- **Tools**: `get_scene_tree`
- **Resources**: `scene_info`
- **Purpose**: Scene inspection and management

### 4. Script Server (`script_server.gd`)
- **Port**: 8083
- **Tools**: `execute_script`
- **Resources**: None
- **Purpose**: Safe GDScript execution and validation

### 5. 3D Server (`3d_server.gd`)
- **Port**: 8084
- **Tools**: `generate_3d_model`
- **Resources**: None
- **Purpose**: 3D model generation and scene manipulation

### 6. System Server (`system_server.gd`)
- **Port**: 8085
- **Tools**: None
- **Resources**: `engine_info`, `input_info`
- **Purpose**: System information and input configuration

## Usage

To run a specific server, update your `test_scene.tscn` to use the desired server script:

```bash
# Edit test_scene.tscn to change the script path
# Then run the project
./bin/godot.macos.editor.arm64 --headless --path modules/mcp/test_project
```

## Testing

Each server can be tested individually using curl commands:

```bash
# Test Communication Server (port 8080)
curl -X POST http://localhost:8080 -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"message":"Hello!"}}}'

# Test Math Server (port 8081)
curl -X POST http://localhost:8081 -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"evaluate_expression","arguments":{"expression":"2+3*4"}}}'

# Test Scene Server (port 8082)
curl -X POST http://localhost:8082 -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# And so on for each server...
```

## Architecture Benefits

- **Modularity**: Each server focuses on a specific domain
- **Scalability**: Can run multiple servers simultaneously on different ports
- **Specialization**: Domain-specific optimizations and error handling
- **Maintainability**: Easier to update and test individual components
- **Flexibility**: Clients can connect to only the servers they need

## Running Multiple Servers

You can run multiple servers simultaneously by starting multiple Godot instances with different scene configurations, each using a different server script.
