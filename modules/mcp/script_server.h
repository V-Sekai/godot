#ifndef MCP_SCRIPT_SERVER_H
#define MCP_SCRIPT_SERVER_H

#include "core/object/ref_counted.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "core/string/ustring.h"
#include "scene/main/node.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"
#include "core/io/resource_saver.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"

class MCPScriptServer : public RefCounted {
    GDCLASS(MCPScriptServer, RefCounted);

protected:
    static void _bind_methods();

public:
    MCPScriptServer();
    ~MCPScriptServer();

    // Tool handlers
    Dictionary _tool_compile_script(const Dictionary &params, const String &request_id, const String &client_id);
    Dictionary _tool_execute_script(const Dictionary &params, const String &request_id, const String &client_id);
    Dictionary _tool_execute_action_sequence(const Dictionary &params, const String &request_id, const String &client_id);

    // Action executors
	Dictionary _execute_single_action(const String &action_name, const Dictionary &parameters);
	Dictionary _execute_call(const Dictionary &params);
	Dictionary _execute_script_action(const Dictionary &params);
	Dictionary _execute_export_scene(const Dictionary &params);

    // Security checks
    Dictionary _apply_security_checks(const String &code, const String &permission_level);
    Dictionary _apply_method_security_checks(const String &method, const Array &arguments);

    // Helper functions
    Node *_get_node(const String &path);
    Dictionary _collect_exported_properties_from_script(Object *script, const String &source_code);
    Dictionary _execute_safescript_code(const String &code, const Dictionary &input_properties, const String &method_name = "");

    // SafeGDScript availability check
    bool _is_safegdscript_available();
    
    // Callback for allowing classes in SafeGDScript
    bool _is_class_allowed(const String &class_name);
};

#endif // MCP_SCRIPT_SERVER_H
