#include "mcp_runner.h"
#include "mcp_server.h"

MCPRunner::MCPRunner() {
}

MCPRunner::~MCPRunner() {
}

void MCPRunner::_bind_methods() {
}

void MCPRunner::_process(double delta) {
    MCPServer *mcp_server = Object::cast_to<MCPServer>(Engine::get_singleton()->get_singleton_object("MCPServer"));
    if (mcp_server) {
        mcp_server->poll();
    }
}
