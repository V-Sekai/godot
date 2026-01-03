#include "script_server.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"
#include "core/string/ustring.h"
#include "core/object/script_language.h"
#include "scene/resources/packed_scene.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "scene/resources/resource_format_text.h"

MCPScriptServer::MCPScriptServer() {
}

MCPScriptServer::~MCPScriptServer() {
}

void MCPScriptServer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_tool_compile_script", "params", "request_id", "client_id"), &MCPScriptServer::_tool_compile_script);
    ClassDB::bind_method(D_METHOD("_tool_execute_script", "params", "request_id", "client_id"), &MCPScriptServer::_tool_execute_script);
    ClassDB::bind_method(D_METHOD("_tool_execute_action_sequence", "params", "request_id", "client_id"), &MCPScriptServer::_tool_execute_action_sequence);
    ClassDB::bind_method(D_METHOD("_is_class_allowed", "class_name"), &MCPScriptServer::_is_class_allowed);
}

bool MCPScriptServer::_is_safegdscript_available() {
    if (ClassDB::class_exists("SafeGDScript")) {
        static bool methods_printed = false;
        if (!methods_printed) {
            methods_printed = true;
            
            String classes_to_check[] = {"SafeGDScript", "Sandbox", "ELFScript", "CPPScript"};
            for (const String &cls : classes_to_check) {
                if (ClassDB::class_exists(cls)) {
                    print_line("MCP: " + cls + " methods:");
                    List<MethodInfo> methods;
                    ClassDB::get_method_list(cls, &methods);
                    for (const MethodInfo &mi : methods) {
                        print_line("  - " + mi.name);
                    }
                    
                    print_line("MCP: " + cls + " properties:");
                    List<PropertyInfo> props;
                    ClassDB::get_property_list(cls, &props);
                    for (const PropertyInfo &pi : props) {
                        print_line("  - " + pi.name);
                    }
                }
            }
        }
        return true;
    }
	return false;
}

Dictionary MCPScriptServer::_tool_compile_script(const Dictionary &params, const String &request_id, const String &client_id) {
    String code = params.get("code", "");
    String language = params.get("language", "gdscript");

    Dictionary result;
    result["compiled"] = false;
    result["language"] = language;

    if (code.is_empty()) {
        result["error"] = "Code is required";
        return result;
    }

    if (language != "safegdscript") {
        result["error"] = "Language must be 'safegdscript' (only SafeGDScript compilation supported)";
        return result;
    }

    if (!_is_safegdscript_available()) {
        result["error"] = "SafeGDScript is not available in this Godot build";
        return result;
    }

    // Apply security checks before compilation
    Dictionary security_result = _apply_security_checks(code, "public");
    if (!security_result.is_empty()) {
        result["error"] = security_result.get("error", "Security check failed");
        return result;
    }

    // Actually try to compile it using SafeGDScript
    Object *script_obj = ClassDB::instantiate("SafeGDScript");
    if (!script_obj) {
        result["error"] = "Failed to create SafeGDScript instance for compilation check";
        return result;
    }

    // Configure SafeGDScript restrictions and callbacks
    if (script_obj->has_method("set_restrictions")) {
        script_obj->call("set_restrictions", true);
    }
    if (script_obj->has_method("set_class_allowed_callback")) {
        Callable class_callback = Callable(this, "_is_class_allowed");
        script_obj->call("set_class_allowed_callback", class_callback);
    }

    // Set source code
    String processed_code = code.replace("\\n", "\n");
    if (!processed_code.ends_with("\n")) {
        processed_code += "\n";
    }
    
    Variant set_source_result = script_obj->call("set_source_code", processed_code);
    if (set_source_result.get_type() == Variant::INT && (int)set_source_result != OK) {
        result["error"] = "Failed to set SafeGDScript source code";
        memdelete(script_obj);
        return result;
    }

    // Reload/compile the script
    Variant reload_result = script_obj->call("reload");
    if (reload_result.get_type() == Variant::INT && (int)reload_result != OK) {
        result["error"] = "SafeGDScript compilation failed: " + String::num((int)reload_result);
        memdelete(script_obj);
        return result;
    }

    result["compiled"] = true;
    result["message"] = "SafeGDScript compiled successfully";
    
    memdelete(script_obj);
    return result;
}

Dictionary MCPScriptServer::_tool_execute_script(const Dictionary &params, const String &request_id, const String &client_id) {
    String code = params.get("code", "");
    String language = params.get("language", "safegdscript");
    String method = params.get("method", "");

    Dictionary result;
    result["success"] = false;

    if (language != "safegdscript") {
        result["error"] = "Language must be 'safegdscript'";
        return result;
    }

    if (!_is_safegdscript_available()) {
        result["error"] = "SafeGDScript is not available";
        return result;
    }

    // Apply security checks before execution
    Dictionary security_result = _apply_security_checks(code, "public");
    if (!security_result.is_empty()) {
        result["error"] = security_result.get("error", "Security check failed");
        return result;
    }

    Dictionary script_result = _execute_safescript_code(code, Dictionary(), method);
    bool executed = script_result.get("executed", false);
    result["success"] = executed;
    if (executed) {
        result["message"] = "Script executed successfully";
        result["result"] = script_result.get("execution_result", Variant());
        if (script_result.has("exported_results")) {
            result["exported_results"] = script_result["exported_results"];
        }
    } else {
        result["error"] = script_result.get("error", "Execution failed");
    }

    return result;
}

Dictionary MCPScriptServer::_tool_execute_action_sequence(const Dictionary &params, const String &request_id, const String &client_id) {
    Dictionary result;
    result["success"] = false;

    // Validate input structure
    if (!params.has("actions")) {
        result["error"] = "Missing 'actions' parameter";
        return result;
    }

    Variant actions_var = params["actions"];
    if (actions_var.get_type() != Variant::ARRAY) {
        result["error"] = "'actions' must be an array";
        return result;
    }

    Array actions = actions_var;
    Array action_results;

    // Get timeout
    double timeout = params.get("timeout", 5.0);
    uint64_t start_time_msec = OS::get_singleton()->get_ticks_msec();
    uint64_t timeout_msec = (uint64_t)(timeout * 1000.0);

    for (int i = 0; i < actions.size(); i++) {
        // Check for timeout
        if (OS::get_singleton()->get_ticks_msec() - start_time_msec > timeout_msec) {
            result["error"] = "Execution timeout after " + String::num_real(timeout) + " seconds";
            result["success"] = false;
            result["executed_actions"] = i;
            result["results"] = action_results;
            return result;
        }

        Variant action_var = actions[i];
        if (action_var.get_type() != Variant::ARRAY && action_var.get_type() != Variant::DICTIONARY) {
            Dictionary action_result;
            action_result["success"] = false;
            action_result["message"] = "Action must be a dictionary";
            action_result["action_index"] = i;
            action_results.push_back(action_result);
            result["error"] = "Action failed: " + String(action_result.get("message", "Unknown error"));
            result["success"] = false;
            result["executed_actions"] = i + 1;
            result["results"] = action_results;
            return result;
        }

        Dictionary action = action_var;
        String action_name = action.get("action", "");

        Dictionary action_result = _execute_single_action(action_name, action);
        action_result["action_index"] = i;
        action_results.push_back(action_result);

        // Stop on first failure
        if (!action_result.get("success", true)) {
            result["success"] = false;
            result["executed_actions"] = i + 1;
            result["results"] = action_results;
            result["error"] = "Action failed: " + String(action_result.get("message", "Unknown error"));
            return result;
        }
    }

    result["success"] = true;
    result["executed_actions"] = actions.size();
    result["results"] = action_results;
    result["message"] = "All actions executed successfully";
    return result;
}

Dictionary MCPScriptServer::_execute_single_action(const String &action_name, const Dictionary &parameters) {
    Dictionary result;
    result["success"] = false;
    result["message"] = "";
    result["action"] = action_name;

    if (action_name == "command_call") {
        result = _execute_call(parameters);
    } else if (action_name == "command_set_script") {
        result = _execute_script_action(parameters);
    } else if (action_name == "command_export_scene") {
        result = _execute_export_scene(parameters);
    } else {
        result["message"] = "Unknown action: " + action_name;
    }

    return result;
}

Dictionary MCPScriptServer::_execute_call(const Dictionary &params) {
    Dictionary result;
    result["action"] = "command_call";

    String method = params.get("method", "");
    String node_path = params.get("node_path", "");
    Array arguments = params.get("arguments", Array());

    if (method.is_empty()) {
        result["message"] = "method parameter required";
        return result;
    }

    // Handle scene tree operations directly in C++
    if (method == "create_node") {
        String parent_path = params.get("parent_path", "/root");
        String node_type = params.get("node_type", "Node");
        String name = params.get("name", "");

        Node *parent = _get_node(parent_path);
        if (!parent) {
            result["message"] = "Parent node not found: " + parent_path;
            return result;
        }

        Object *new_obj = ClassDB::instantiate(node_type);
        if (!new_obj) {
            result["message"] = "Failed to create object of type: " + node_type;
            return result;
        }
        Node *new_node = Object::cast_to<Node>(new_obj);
        if (!new_node) {
            result["message"] = "Created object is not a Node: " + node_type;
            memdelete(new_obj);
            return result;
        }

        if (!name.is_empty()) {
            new_node->set_name(name);
        }

        parent->add_child(new_node);
        result["success"] = true;
        result["message"] = "Created " + node_type + " node at " + String(new_node->get_path());
        result["node_path"] = String(new_node->get_path());
        return result;
    }

    if (method == "set" || method == "get") {
        // Handle property operations
        if (!node_path.is_empty()) {
            Node *target = _get_node(node_path);
            if (target) {
                if (method == "set" && arguments.size() >= 2) {
                    String prop_name = arguments[0];
                    Variant prop_value = arguments[1];
                    target->set(prop_name, prop_value);
                    result["success"] = true;
                    result["message"] = "Set property " + prop_name + " on " + node_path;
                } else if (method == "get" && arguments.size() >= 1) {
                    String prop_name = arguments[0];
                    Variant prop_value = target->get(prop_name);
                    result["success"] = true;
                    result["message"] = "Got property " + prop_name + " from " + node_path;
                    result["value"] = prop_value;
                } else {
                    result["message"] = "Invalid arguments for " + method;
                }
            } else {
                result["message"] = "Node not found: " + node_path;
            }
        } else {
            result["message"] = "node_path required for " + method;
        }
        return result;
    } else if (method == "add_child" || method == "remove_child" || method == "queue_free") {
        // Handle other node operations
        if (!node_path.is_empty()) {
            Node *target = _get_node(node_path);
            if (target) {
                if (method == "queue_free") {
                    target->queue_free();
                    result["success"] = true;
                    result["message"] = "Queued node for deletion: " + node_path;
                } else {
                    result["message"] = method + " not yet implemented";
                }
            } else {
                result["message"] = "Node not found: " + node_path;
            }
        } else {
            result["message"] = "node_path required for " + method;
        }
        return result;
    }

    // For all other methods, we need a target
    Object *target_obj = nullptr;
    if (!node_path.is_empty()) {
        target_obj = _get_node(node_path);
    } else {
        // Allow calling methods on singletons or resources directly if no node_path
        // This part would need more sophisticated lookup for non-node objects/resources
        // For now, assume node_path is always provided for non-singleton calls
    }

    if (!target_obj) {
        result["message"] = "Target object/node not found: " + node_path;
        return result;
    }

    // Apply unified security checks
    Dictionary security_result = _apply_method_security_checks(method, arguments);
    if (!security_result.is_empty()) {
        result["message"] = security_result.get("error", "Security check failed");
        return result;
    }

    // Check if set_script is allowed (only if SafeGDScript is available)
    if (method == "set_script" && !_is_safegdscript_available()) {
        result["message"] = "set_script is not available - SafeGDScript not found in this build";
        return result;
    }

    // Execute the method call
    if (target_obj->has_method(method)) {
        Variant call_result = target_obj->callv(method, arguments);
        result["success"] = true;
        result["message"] = "Called " + method + " on " + node_path;
        result["call_result"] = call_result;
    } else {
        // Try to set as property - this will fail silently if property doesn't exist
        if (arguments.size() == 1) {
            target_obj->set(method, arguments[0]);
            result["success"] = true;
            result["message"] = "Set property " + method + " on " + node_path;
        } else {
            result["message"] = "Setting property requires exactly one argument";
        }
    }

    return result;
}

Dictionary MCPScriptServer::_execute_script_action(const Dictionary &params) {
    Dictionary result;
    result["action"] = "command_set_script";

    String script_code = params.get("script", "");
    if (script_code.is_empty()) {
        result["success"] = false;
        result["message"] = "Script code is required";
        return result;
    }
    
    // JSON parser converts \n escape sequences to actual newline characters
    // However, if we receive the string with literal backslash-n (two chars: \ + n),
    // we need to convert them. Check if the string contains literal backslash-n.
    // In C++ string literals, "\\n" represents a backslash followed by n (two chars)
    // We need to replace the two-character sequence backslash+n with a single newline
    
    // Count occurrences of literal backslash-n (two characters)
    int backslash_n_count = 0;
    for (int i = 0; i < script_code.length() - 1; i++) {
        if (script_code[i] == '\\' && script_code[i + 1] == 'n') {
            backslash_n_count++;
        }
    }
    
    if (backslash_n_count > 0) {
        // Replace literal backslash-n (two characters) with actual newline
        script_code = script_code.replace("\\n", "\n");
    }
    
    // Also handle tabs
    script_code = script_code.replace("\\t", "\t");
    
    // Ensure script ends with newline (GDScript requirement)
    if (!script_code.ends_with("\n")) {
        script_code += "\n";
    }

    String target_path = params.get("target_path", "");
    String method_to_call = params.get("method", "");
    Node *target_node = nullptr;
    
    // If target_path is provided, set script on that node
    if (!target_path.is_empty()) {
        target_node = _get_node(target_path);
        if (!target_node) {
            result["success"] = false;
            result["message"] = "Target node not found: " + target_path;
            return result;
        }
    }

    Dictionary input_properties;
    if (params.has("input_properties")) {
        input_properties = params["input_properties"];
    }

    // If we have a target node, set script on it and execute
    if (target_node) {
        // Create SafeGDScript instance
        if (!_is_safegdscript_available()) {
            result["success"] = false;
            result["message"] = "SafeGDScript is not available in this build";
            return result;
        }

        Object *script_obj = ClassDB::instantiate("SafeGDScript");
        if (!script_obj) {
            result["success"] = false;
            result["message"] = "Failed to create SafeGDScript instance";
            return result;
        }

        // Configure SafeGDScript restrictions and callbacks BEFORE setting source code
        if (script_obj->has_method("set_restrictions")) {
            script_obj->call("set_restrictions", true);
        } else {
            script_obj->set("restrictions", true);
        }

        if (script_obj->has_method("set_class_allowed_callback")) {
            Callable class_callback = Callable(this, "_is_class_allowed");
            script_obj->call("set_class_allowed_callback", class_callback);
        }
        
        if (script_obj->has_method("get_sandbox")) {
            Variant sandbox_variant = script_obj->call("get_sandbox");
            if (sandbox_variant.get_type() == Variant::OBJECT) {
                Object *sandbox_obj = sandbox_variant;
                if (sandbox_obj && sandbox_obj->has_method("set_class_allowed_callback")) {
                    Callable class_callback = Callable(this, "_is_class_allowed");
                    sandbox_obj->call("set_class_allowed_callback", class_callback);
                }
            }
        }
        
        // Note: If restrictions are enabled, classes need to be explicitly allowed
        // The callback should handle this, but we may need to set it before source code

        // Set source code (after all restrictions/callbacks are configured)
        Variant set_source_result = script_obj->call("set_source_code", script_code);
        if (set_source_result.get_type() == Variant::INT && (int)set_source_result != OK) {
            result["success"] = false;
            result["message"] = "Failed to set SafeGDScript source code: " + String::num((int)set_source_result);
            memdelete(script_obj);
            return result;
        }

        // Reload/compile the script
        Variant reload_result = script_obj->call("reload");
        Error reload_err = OK;
        if (reload_result.get_type() == Variant::INT) {
            reload_err = (Error)(int)reload_result;
        }
        
        // Check if compilation failed
        if (reload_err != OK) {
            result["success"] = false;
            result["message"] = "SafeGDScript compilation failed with error code: " + String::num((int)reload_err);
            memdelete(script_obj);
            return result;
        }
        
        // Validate script compilation by checking if it has valid bytecode/ELF data
        // SafeGDScript should have a method to check compilation status
        Script *script_ptr = Object::cast_to<Script>(script_obj);
        if (script_ptr) {
            // Try to validate the script by checking if it can be used
            // If the script failed to compile, it won't have valid bytecode
            if (script_obj->has_method("is_valid")) {
                Variant is_valid = script_obj->call("is_valid");
                if (is_valid.get_type() == Variant::BOOL && !(bool)is_valid) {
                    result["success"] = false;
                    result["message"] = "SafeGDScript compilation failed - script is not valid";
                    memdelete(script_obj);
                    return result;
                }
            }
        }

        // Set the script on the target node
        target_node->set_script(Ref<Script>(Object::cast_to<Script>(script_obj)));

        // Set input properties before calling _ready
        Array keys = input_properties.keys();
        for (int i = 0; i < keys.size(); i++) {
            String prop_name = keys[i];
            target_node->set(prop_name, input_properties[prop_name]);
        }

        // Call _ready() to initialize the script
        if (target_node->has_method("_ready")) {
            target_node->call("_ready");
        }

        // Call the specified method if it exists
        if (!method_to_call.is_empty() && target_node->has_method(method_to_call)) {
            Variant execution_result = target_node->call(method_to_call);
            result["execution_result"] = execution_result;
        }

        // Collect exported properties
        Dictionary exported_data = _collect_exported_properties_from_script(script_obj, script_code);
        
        result["success"] = true;
        result["message"] = "Script set and executed on " + target_path;
        if (!exported_data.is_empty()) {
            result["exported_results"] = exported_data;
        }
    } else {
        // Fallback to temporary node execution
        Dictionary script_result = _execute_safescript_code(script_code, input_properties, method_to_call);
        bool executed = script_result.get("executed", false);
        result["success"] = executed;
        result["message"] = "Script executed";
        result["data"] = script_result;
        result["result"] = script_result.get("execution_result", Variant());
    }

    return result;
}

Dictionary MCPScriptServer::_apply_security_checks(const String &code, const String &permission_level) {
    if (permission_level != "public" && permission_level != "restricted") {
        return Dictionary();
    }

    // Dangerous patterns (same as SafeGDScript)
    // Block destructive operations and information disclosure
    Vector<String> dangerous_patterns = {
        // Process execution
        "OS.execute", "OS.create_process", "OS.shell_open", "OS.move_to_trash",
        
        // File system write operations
        "DirAccess.remove", "DirAccess.rename", "DirAccess.make_dir_recursive",
        "FileAccess.open", "FileAccess.open_encrypted", "FileAccess.open_compressed",
        
        // File system read operations (information disclosure)
        "FileAccess.get_file_as_bytes", "FileAccess.get_file_as_string",
        "FileAccess.file_exists", "FileAccess.get_modified_time",
        "FileAccess.get_md5", "FileAccess.get_sha256",
        "DirAccess.open", "DirAccess.copy", "DirAccess.list_dir_begin", "DirAccess.get_next",
        "DirAccess.dir_exists", "DirAccess.get_current_dir", "DirAccess.change_dir",
        
        // Resource loading and saving
        "load(", "preload(", "ResourceLoader.load", "ResourceLoader.load_threaded_request",
        "ResourceLoader.exists", "ResourceLoader.get_recognized_extensions_for_type",
        "ResourceLoader.get_dependencies", "ResourceSaver.save",
        
        // Code execution
        "Engine.get_singleton", "ClassDB.instantiate", "ClassDB.class_call_static",
        "GDScript.new", "JavaScript.eval", "Expression.new",
        
        // OS information disclosure
        "OS.get_name", "OS.get_version", "OS.get_locale",
        "OS.get_cmdline_args", "OS.get_environment", "OS.set_environment",
        "OS.get_executable_path", "OS.get_user_data_dir", "OS.get_config_dir",
        "OS.get_cache_dir", "OS.get_temp_dir", "OS.get_unique_id",
        "OS.get_keycode_string", "OS.find_keycode_from_string",
        "OS.is_keycode_unicode", "OS.get_main_thread_id",
        "OS.has_feature", "OS.request_permissions", "OS.get_granted_permissions",
        "OS.revoke_granted_permissions",
        "OS.get_low_processor_mode_sleep_usec", "OS.set_low_processor_mode_sleep_usec",
        
        // Reflection/introspection (information disclosure)
        "ClassDB.class_exists", "ClassDB.class_get_property_list",
        "ClassDB.class_get_method_list", "ClassDB.class_get_signal_list",
        "ClassDB.class_get_constant_list", "ClassDB.class_has_method",
        "ClassDB.class_has_property", "ClassDB.class_has_signal",
        "ClassDB.class_has_constant", "ClassDB.class_get_property",
        "ClassDB.class_get_property_default_value",
        "ClassDB.class_get_method_argument_count",
        "ClassDB.class_get_method_argument_type",
        "ClassDB.class_get_method_argument_name",
        "ClassDB.class_get_method_return_type",
        
        // Networking (blocked for security)
        "HTTPClient.new", "TCPServer.new", "TCPClient.new", "UDPServer.new", "UDPClient.new",
        "WebSocketClient.new", "WebSocketServer.new", "WebRTCPeerConnection.new",
        
        // Threading (blocked for security)
        "Thread.new", "Mutex.new", "Semaphore.new"
    };

    for (const String &pattern : dangerous_patterns) {
        if (code.contains(pattern)) {
            Dictionary result;
            result["error"] = String("Code contains potentially dangerous operations: ") + pattern;
            return result;
        }
    }

    // Additional security checks
    if (code.contains("import") || code.contains("preload")) {
        Dictionary result;
        result["error"] = "Code contains import/preload statements which are not allowed";
        return result;
    }

    return Dictionary(); // No security violations
}

Dictionary MCPScriptServer::_apply_method_security_checks(const String &method, const Array &arguments) {
    // Dangerous methods
    Vector<String> dangerous_methods = {
        "execute", "create_process", "shell_open", "move_to_trash",
        "open", "open_encrypted", "open_compressed",
        "remove", "rename", "make_dir_recursive",
        "get_singleton", "instantiate", "class_call_static",
        "eval", "new"
    };

    for (const String &dangerous_method : dangerous_methods) {
        if (method == dangerous_method) {
            Dictionary result;
            result["error"] = String("Method '") + method + "' is blocked for security reasons";
            return result;
        }
    }

    // Dangerous classes
    Vector<String> dangerous_classes = {
        "HTTPClient", "TCPServer", "TCPClient", "UDPServer", "UDPClient",
        "WebSocketClient", "WebSocketServer", "WebRTCPeerConnection",
        "Thread", "Mutex", "Semaphore",
        "FileAccess", "DirAccess", "ConfigFile",
        "OS", "Engine", "ClassDB", "ResourceLoader",
        "JavaScript", "Expression"
    };

    // Check arguments for dangerous class instantiation
    for (int i = 0; i < arguments.size(); i++) {
        Variant arg = arguments[i];
        if (arg.get_type() == Variant::STRING) {
            String arg_str = arg;
            for (const String &dangerous_class : dangerous_classes) {
                if (arg_str.contains(dangerous_class + ".new") || arg_str.contains(dangerous_class + ".")) {
                    Dictionary result;
                    result["error"] = String("Access to '") + dangerous_class + "' is blocked for security reasons";
                    return result;
                }
            }
        }
    }

    // Block load/preload operations
    if (method == "load" || method == "preload") {
        Dictionary result;
        result["error"] = "Resource loading operations are blocked for security reasons";
        return result;
    }

    return Dictionary(); // No security violations
}

Node *MCPScriptServer::_get_node(const String &path) {
    SceneTree *tree = SceneTree::get_singleton();
    if (!tree) {
        return nullptr;
    }

    Window *root = tree->get_root();
    if (!root) {
        return nullptr;
    }

    if (path == "/root" || path == "/") {
        return root;
    }

    return root->get_node_or_null(path);
}

Dictionary MCPScriptServer::_collect_exported_properties_from_script(Object *script_obj, const String &source_code) {
    Dictionary result;

    if (!script_obj) {
        result["error"] = "No script object provided";
        return result;
    }

    // Parse the source code to find @export variables
    Vector<String> lines = source_code.split("\n");
    Dictionary exported_vars;

    for (const String &line : lines) {
        String trimmed = line.strip_edges();
        if (trimmed.begins_with("@export")) {
            // Extract variable name from @export var name: type = value
            Vector<String> parts = trimmed.split(" ");
            if (parts.size() >= 2 && parts[1] == "var") {
                for (int i = 2; i < parts.size(); i++) {
                    String part = parts[i];
                    if (part.contains(":")) {
                        // Found variable name
                        String var_name = part.split(":")[0].strip_edges();
                        exported_vars[var_name] = true; // Mark as exported
                        break;
                    }
                }
            }
        }
    }

    // Use string-based calls to get property values if the script supports it
    if (!exported_vars.is_empty()) {
        Dictionary actual_values;
        Array keys = exported_vars.keys();
        for (int i = 0; i < keys.size(); i++) {
            String prop_name = keys[i];
            if (script_obj->has_method("get_property_default_value")) {
                Variant default_value = script_obj->call("get_property_default_value", prop_name);
                actual_values[prop_name] = default_value;
            } else {
                // Fallback to marking as exported without value
                actual_values[prop_name] = Variant();
            }
        }
        result["exported_properties"] = actual_values;
        result["count"] = actual_values.size();
    }

    return result;
}

Dictionary MCPScriptServer::_execute_export_scene(const Dictionary &params) {
    Dictionary result;
    result["action"] = "command_export_scene";

    String node_path = params.get("node_path", "");
    if (node_path.is_empty()) {
        result["success"] = false;
        result["message"] = "node_path parameter required";
        return result;
    }
    String format = params.get("format", "tscn"); // Default to tscn

    // Find the node
    Node *target_node = _get_node(node_path);
    if (!target_node) {
        result["success"] = false;
        result["message"] = "Node not found: " + node_path;
        return result;
    }

    // Create a PackedScene
    Ref<PackedScene> packed_scene;
    packed_scene.instantiate();

    // Pack the node into the scene
    Error pack_err = packed_scene->pack(target_node);
    if (pack_err != OK) {
        result["success"] = false;
        result["message"] = "Failed to pack scene: " + itos(pack_err);
        return result;
    }

    // Determine the file extension based on format
    String extension = ".tscn";
    if (format.to_lower() == "escn") {
        extension = ".escn";
    }
    String temp_filename = "temp_export_scene_" + String::num_int64(OS::get_singleton()->get_ticks_usec()) + "_" + String::num_int64(OS::get_singleton()->get_process_id()) + "_" + String::num_int64(rand()) + extension;
    String temp_path = "res://" + temp_filename;
    
    // Use ResourceFormatSaverText directly to save the scene in text format
    if (!ResourceFormatSaverText::singleton) {
        result["success"] = false;
        result["message"] = "ResourceFormatSaverText singleton not available";
        return result;
    }
    
    Error save_err = ResourceFormatSaverText::singleton->save(packed_scene, temp_path,
                                         ResourceSaver::FLAG_RELATIVE_PATHS |
                                         ResourceSaver::FLAG_OMIT_EDITOR_PROPERTIES);
    if (save_err != OK) {
        result["success"] = false;
        result["message"] = "Failed to export scene: " + itos(save_err);
        return result;
    }

    // Read the saved file as text (since we're using text format)
    Ref<FileAccess> file = FileAccess::open(temp_path, FileAccess::READ);
    if (!file.is_valid()) {
        result["success"] = false;
        result["message"] = "Failed to read exported scene file";
        return result;
    }

    String scene_text = file->get_as_text();
    file->close();

    // Clean up the temp file
    Ref<DirAccess> dir = DirAccess::open("res://");
    if (dir.is_valid()) {
        dir->remove(temp_filename);
    }

    result["success"] = true;
    result["message"] = "Scene exported successfully";
    result["exported_scene_data"] = scene_text;
    result["mime_type"] = "text/plain";
    result["file_extension"] = extension;
    return result;
}

Dictionary MCPScriptServer::_execute_safescript_code(const String &code, const Dictionary &input_properties, const String &method_name) {
    Dictionary result;
    result["executed"] = false;

    if (!_is_safegdscript_available()) {
        result["error"] = "SafeGDScript is not available in this build";
        return result;
    }
    
    // Replace literal \n escape sequences with actual newlines
    String processed_code = code.replace("\\n", "\n");
    
    // Ensure code ends with newline (GDScript requirement)
    if (!processed_code.ends_with("\n")) {
        processed_code += "\n";
    }

    // Step 1: Create a temporary node for script execution
    Node *test_node = memnew(Node);
    test_node->set_name("SafeGDScript_Test_Node");

    // Add to scene tree
    SceneTree *tree = SceneTree::get_singleton();
    if (tree) {
        Window *root_window = tree->get_root();
        if (root_window) {
            root_window->add_child(test_node);
        } else {
            result["error"] = "No scene root available";
            memdelete(test_node);
            return result;
        }
    } else {
        result["error"] = "No scene tree available";
        memdelete(test_node);
        return result;
    }

    // Step 2: Create SafeGDScript and assign it to the node
    Object *script_obj = ClassDB::instantiate("SafeGDScript");
    if (!script_obj) {
        result["error"] = "Failed to create SafeGDScript instance";
        if (test_node->get_parent()) {
            test_node->get_parent()->remove_child(test_node);
        }
        memdelete(test_node);
        return result;
    }

    // Configure SafeGDScript restrictions and callbacks BEFORE setting source code
    if (script_obj->has_method("set_restrictions")) {
        script_obj->call("set_restrictions", true);
    }
    if (script_obj->has_method("set_class_allowed_callback")) {
        Callable class_callback = Callable(this, "_is_class_allowed");
        script_obj->call("set_class_allowed_callback", class_callback);
    }

    // Assign script to node
    test_node->set_script(Ref<Script>(Object::cast_to<Script>(script_obj)));

    // Step 4: Set the code
    // Set source code using call() method (after all restrictions/callbacks are configured)
    Variant set_source_result = script_obj->call("set_source_code", processed_code);
    if (set_source_result.get_type() == Variant::INT && (int)set_source_result != OK) {
        result["error"] = "Failed to set SafeGDScript source code";
        if (test_node->get_parent()) {
            test_node->get_parent()->remove_child(test_node);
        }
        memdelete(test_node);
        // script_obj is managed by Ref<Script>, don't manually delete
        return result;
    }

    // Reload/compile the script
    Variant reload_result = script_obj->call("reload");
    if (reload_result.get_type() == Variant::INT && (int)reload_result != OK) {
        result["error"] = "SafeGDScript compilation failed: " + String::num((int)reload_result);
        if (test_node->get_parent()) {
            test_node->get_parent()->remove_child(test_node);
        }
        memdelete(test_node);
        // script_obj is managed by Ref<Script>, don't manually delete
        return result;
    }

    // Set input properties
    Array keys = input_properties.keys();
    for (int i = 0; i < keys.size(); i++) {
        String prop_name = keys[i];
        // Try to set the property - if it doesn't exist, it will fail silently
        test_node->set(prop_name, input_properties[prop_name]);
    }

    // Force script initialization
    if (test_node->has_method("_ready")) {
        test_node->call("_ready");
    }

    // Try to call the specified method if it exists
    Variant execution_result = Variant();
    String target_method = method_name;

    if (!target_method.is_empty() && test_node->has_method(target_method)) {
        execution_result = test_node->call(target_method);
    } else if (!target_method.is_empty()) {
        result["error"] = "Method '" + target_method + "' not found in script";
        if (test_node->get_parent()) {
            test_node->get_parent()->remove_child(test_node);
        }
        memdelete(test_node);
        return result;
    }

    result["executed"] = true;
    result["execution_result"] = execution_result;

    // Collect exported properties
    Dictionary exported_data = _collect_exported_properties_from_script(script_obj, processed_code);
    if (!exported_data.is_empty()) {
        result["exported_results"] = exported_data;
    }

    // Clean up
    if (test_node->get_parent()) {
        test_node->get_parent()->remove_child(test_node);
    }
    memdelete(test_node);
    // script_obj is managed by Ref<Script>, don't manually delete

    return result;
}

bool MCPScriptServer::_is_class_allowed(const String &class_name) {
    print_line("MCP: Checking if class is allowed: " + class_name);
    // Centralized whitelist - synced across all SafeGDScript usage
    // This whitelist is used by the set_class_allowed_callback for both
    // _execute_script_action and _execute_safescript_code
    
    Vector<String> allowed_classes = {
        // Mesh classes
        "ArrayMesh", "BoxMesh", "SphereMesh", "CylinderMesh", "CapsuleMesh",
        "PlaneMesh", "PrismMesh", "TorusMesh", "Mesh", "MeshInstance3D",
        
        // Packed arrays (all variants)
        "PackedVector3Array", "PackedInt32Array", "PackedFloat32Array",
        "PackedVector2Array", "PackedVector4Array",
        "PackedInt64Array", "PackedFloat64Array", "PackedStringArray",
        "PackedByteArray", "PackedColorArray",
        
        // Basic data types
        "Dictionary", "Array", "String", "StringName",
        
        // Vector types
        "Vector2", "Vector3", "Vector4", "Vector2i", "Vector3i", "Vector4i",
        
        // Math types
        "Rect2", "Rect2i", "AABB", "Basis", "Transform2D", "Transform3D",
        "Quaternion", "Plane", "Color", "NodePath", "RID"
    };
    
    for (const String &allowed : allowed_classes) {
        if (class_name == allowed) {
            return true;
        }
    }
    
    // By default, deny unknown classes for security
    // Variant types are allowed by Sandbox by default, but we explicitly whitelist
    // the ones we need above
    return false;
}
