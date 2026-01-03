#pragma once

#include "scene/main/node.h"

class MCPRunner : public Node {
    GDCLASS(MCPRunner, Node);

protected:
    static void _bind_methods();

public:
    MCPRunner();
    ~MCPRunner();

    void _process(double delta);
};
