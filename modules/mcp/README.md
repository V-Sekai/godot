# MCP Godot Module

A native Godot C++ module that implements the Model Context Protocol (MCP) server specification. This module enables Godot to act as an MCP server, accepting connections from MCP clients and handling requests via Streamable HTTP transport.

## Features

- ✅ **Streamable HTTP Transport** - Chunked transfer encoding for continuous data streams
- ✅ **MCP Protocol Support** - Full support for MCP protocol versions 2024-11-05, 2025-03-26, and 2025-06-18
- ✅ **Request/Response Pattern** - Synchronous request handling (like Elixir GenServer `handle_call`)
- ✅ **Async Notifications** - Asynchronous notification handling (like Elixir GenServer `handle_cast`)
- ✅ **Tool Registration** - Register custom tools that can be called by MCP clients
- ✅ **Resource Registration** - Register resources that can be read by MCP clients
- ✅ **Progress Notifications** - Send progress updates to clients
- ✅ **Batch Requests** - Support for batch JSON-RPC requests
- ✅ **ex-mcp Compatible** - Works seamlessly with [ex-mcp](https://github.com/fire/ex-mcp) clients and servers

## Quick Start

### Accessing the Server

```gdscript
# Get the MCP server singleton
var mcp_server = Engine.get_singleton("MCPServer")

# Start the server on port 8080
mcp_server.start_server(8080)

# The server is now listening for MCP client connections
```

### Registering a Tool

```gdscript
# Register a tool handler
mcp_server.register_tool("my_tool", my_tool_handler)

# Handler signature: (params: Dictionary, request_id, client_id)
func my_tool_handler(params: Dictionary, request_id, client_id):
    if request_id != null:
        # Request/response pattern - must respond
        var result = do_work(params)
        # Option 1: Return result (automatically sent)
        return result
        # Option 2: Call send_response manually
        # mcp_server.send_response(client_id, request_id, result)
    else:
        # Async notification - no response needed
        do_work_async(params)
```

### Registering a Resource

```gdscript
# Register a resource handler
mcp_server.register_resource("my_resource", my_resource_handler)

func my_resource_handler(params: Dictionary, request_id, client_id):
    if request_id != null:
        var uri = params.get("uri", "")
        var content = load_resource(uri)
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": content
            }]
        }
```

### Sending Progress Updates

```gdscript
# Send a progress notification to a client
mcp_server.send_progress(client_id, "token123", {
    "percentage": 50,
    "status": "processing"
})
```

### Polling for Incoming Messages

The server uses event-driven callbacks, but you need to call `poll()` regularly to process incoming connections and messages:

```gdscript
func _process(delta):
    var mcp_server = Engine.get_singleton("MCPServer")
    if mcp_server.is_server_running():
        mcp_server.poll()
```

## API Reference

### MCPServer Methods

- `start_server(port: int) -> void` - Start the MCP server on the specified port
- `stop_server() -> void` - Stop the MCP server
- `is_server_running() -> bool` - Check if the server is currently running
- `register_tool(name: String, handler: Callable) -> void` - Register a tool handler
- `register_resource(name: String, handler: Callable) -> void` - Register a resource handler
- `send_response(client_id: int64, request_id: Variant, result: Variant) -> void` - Send a response to a client
- `send_notification(client_id: int64, method: String, params: Dictionary) -> void` - Send a notification to a client
- `send_progress(client_id: int64, progress_token: String, progress: Dictionary) -> void` - Send a progress update
- `poll() -> void` - Process incoming connections and messages (call regularly)

### Handler Signatures

All handlers receive three parameters:
- `params: Dictionary` - Request parameters
- `request_id: Variant` - Request ID (null for notifications)
- `client_id: int64` - Client connection ID

### Response Patterns

**Request/Response (Synchronous):**
- Messages with `id` field require a response
- Handler can return a result directly (auto-sent) or call `send_response()` manually

**Async/Notification (Asynchronous):**
- Messages without `id` field are notifications
- Handler receives `request_id` as `null`, no response needed

## Supported MCP Methods

- `initialize` - Handshake with client
- `tools/list` - List available tools
- `tools/call` - Call a registered tool
- `resources/list` - List available resources
- `resources/read` - Read a resource

## Protocol Versions

The module supports MCP protocol versions:
- 2024-11-05
- 2025-03-26
- 2025-06-18 (default)

## Transport

- **Streamable HTTP** with chunked transfer encoding
- Content-Type: `application/json`
- Connection: `keep-alive` for persistent connections
- Compatible with ex-mcp's HTTP/SSE transport expectations

## Error Handling

The module uses standard JSON-RPC error codes:
- `PARSE_ERROR = -32700`
- `INVALID_REQUEST = -32600`
- `METHOD_NOT_FOUND = -32601`
- `INVALID_PARAMS = -32602`
- `INTERNAL_ERROR = -32603`

## Compatibility

The module is designed to work seamlessly with:
- [ex-mcp](https://github.com/fire/ex-mcp) v0.6.0+ clients and servers
- Any MCP client/server that follows the MCP specification

## Building

The module is automatically detected by Godot's build system. No manual configuration needed.

To build with tests:
```bash
scons tests=yes
```

## Testing

Tests are located in `godot/modules/mcp/tests/` and can be run as part of Godot's test suite.

## References

- [MCP Specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
- [ex-mcp Library](https://github.com/fire/ex-mcp)
- [Godot JSONRPC Module](../jsonrpc/) - Reference implementation

## License

This module follows the same license as the Godot Engine.
