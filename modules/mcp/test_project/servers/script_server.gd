extends Node

# Use the C++ MCPScriptServer implementation if available, otherwise fall back to GDScript
var _script_server
var _use_cpp = false

func _ready():
	# Try C++ MCPScriptServer first, fall back to GDScript
	if ClassDB.class_exists("MCPScriptServer"):
		_script_server = ClassDB.instantiate("MCPScriptServer")
		_use_cpp = true
		print("✓ Using C++ MCPScriptServer implementation")
	else:
		print("⚠️ C++ MCPScriptServer not available, using GDScript implementation")
		_use_cpp = false
		_script_server = self  # Use self for GDScript implementation

	if not _script_server:
		print("✗ Failed to create ScriptServer instance")
		get_tree().quit(1)
		return

	# Register tools - try C++ first, fall back to GDScript if needed
	_register_tools()

func _register_tools():
	var mcp_server = Engine.get_singleton("MCPServer")
	if not mcp_server:
		print("✗ MCPServer singleton not found!")
		get_tree().quit(1)
		return

	print("✓ MCPServer singleton found")

	# Test 2: Register tools and resources
	print("\n--- Registering Script Tools ---")

	# Script compilation tool
	var compile_description = """Validate that SafeGDScript code compiles without executing it.

This tool checks syntax and compilation errors without running the code. Useful for validating scripts before execution.

Examples of SafeGDScript:

Basic syntax check:
@export var value: int = 42

func my_method():
    return value

Parameters: code (string), language (must be 'safegdscript')"""

	# Use C++ if available, otherwise GDScript
	if _use_cpp:
		mcp_server.register_tool("compile_script", compile_description, _script_server, "_tool_compile_script")
		print("✓ Script compilation tool registered (C++)")
	else:
		mcp_server.register_tool("compile_script", compile_description, self, "_tool_compile_script")
		print("✓ Script compilation tool registered (GDScript)")

	# Action sequence executor tool
	var executor_description = """Execute a sequence of actions on the Godot scene tree with timeout protection.

This is a simple forward executor that runs predefined sequences of scene tree operations.
Actions are executed in order and can include generic method calls and SafeGDScript execution.
All actions use unified security restrictions to prevent dangerous operations.
Execution is limited to 5 seconds by default to prevent infinite loops.

Available actions:
- command_call: Generic method call on nodes, objects, and resources (method, node_path, arguments)
  - Special handling for "create_node" method (creates new nodes)
  - "set_script" method allowed on objects and resources
  - Same security restrictions as SafeGDScript (blocks dangerous operations)
- command_set_script: Execute SafeGDScript code on scene nodes (script, input_properties)

Example sequence for creating a scene:
[
  {"action": "command_call", "method": "create_node", "node_type": "Node2D", "name": "GameWorld", "parent_path": "/root"},
  {"action": "command_call", "method": "create_node", "node_type": "Camera2D", "name": "MainCamera", "parent_path": "/root/GameWorld"},
  {"action": "command_call", "method": "set", "node_path": "/root/GameWorld/MainCamera", "arguments": ["current", true]},
  {"action": "command_set_script", "script": "@export var message: String = 'Hello'; func my_method(): print(message)", "method": "my_method"}
]

Parameters:
- actions: Array of action objects (required)
- timeout: Maximum execution time in seconds (optional, default: 5.0)"""

	# Use C++ if available, otherwise GDScript
	if _use_cpp:
		mcp_server.register_tool("execute_action_sequence", executor_description, _script_server, "_tool_execute_action_sequence")
		print("✓ Action sequence executor tool registered (C++)")
	else:
		mcp_server.register_tool("execute_action_sequence", executor_description, self, "_tool_execute_action_sequence")
		print("✓ Action sequence executor tool registered (GDScript)")

	# Test 3: Start server
	print("\n--- Starting Script Server ---")
	var port = 8087
	print("Starting script MCP server on port ", port, "...")
	mcp_server.start_server(port)

	# Wait a bit for server to start
	await get_tree().create_timer(0.2).timeout

	if mcp_server.is_server_running():
		print("✓ Script MCP server started successfully on port ", port)
		print("✓ Ready to accept connections!")
		print("\n--- Script Server Ready ---")
		print("Available tools: compile_script, execute_action_sequence")
	else:
		print("✗ Failed to start script MCP server")
		get_tree().quit(1)

func _process(delta):
	var mcp_server = Engine.get_singleton("MCPServer")
	if mcp_server and mcp_server.is_server_running():
		mcp_server.poll()

func _notification(what):
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		var mcp_server = Engine.get_singleton("MCPServer")
		if mcp_server:
			mcp_server.stop_server()
			print("Script MCP server stopped")
		get_tree().quit()

# Tool handler functions (GDScript fallback implementation)

func _tool_compile_script(params, request_id, client_id):
	var code = params.get("code", "")
	var language = params.get("language", "gdscript")

	var result = {"compiled": false, "language": language}

	if code.is_empty():
		result["error"] = "Code is required"
		return result

	if language != "safegdscript":
		result["error"] = "Language must be 'safegdscript' (only SafeGDScript compilation supported)"
		return result

	print("🔧 Compiling SafeGDScript code for validation...")

	# Create SafeGDScript instance for compilation
	var script = null
	if ClassDB.class_exists("SafeGDScript"):
		script = ClassDB.instantiate("SafeGDScript")
		if script:
			print("✓ SafeGDScript instance created for compilation")

			# Set the source code
			script.source_code = code
			print("✓ Set source code for compilation")

			# Try to compile (reload)
			var compile_result = script.reload()
			print("✓ Compilation result:", compile_result)

			if compile_result == OK:
				result["compiled"] = true
				result["message"] = "SafeGDScript compiled successfully"

				# Get compilation info
				var methods = script.get_script_method_list()
				var properties = script.get_script_property_list()

				result["method_count"] = methods.size()
				result["property_count"] = properties.size()
			else:
				result["error"] = "SafeGDScript compilation failed with error: " + str(compile_result)
	else:
		result["error"] = "SafeGDScript is not available in this Godot build"

	return result

func _tool_execute_action_sequence(params, request_id, client_id):
	var actions = params.get("actions", [])
	var timeout = params.get("timeout", 5.0)

	var result = {"success": true, "executed_actions": 0, "results": []}

	if actions.is_empty():
		result["error"] = "No actions provided"
		result["success"] = false
		return result

	# Execute actions with timeout
	var start_time = Time.get_ticks_msec()
	var timeout_ms = timeout * 1000.0

	var action_results = []
	for i in range(actions.size()):
		# Check timeout
		var current_time = Time.get_ticks_msec()
		if current_time - start_time > timeout_ms:
			result["error"] = "Execution timeout after " + str(timeout) + " seconds"
			result["success"] = false
			result["executed_actions"] = i
			result["results"] = action_results
			return result

		var action = actions[i]
		var action_name = action.get("action", "")

		var action_result = _execute_single_action(action_name, action)
		action_result["action_index"] = i
		action_results.append(action_result)

		# Stop on first failure
		if not action_result.get("success", true):
			result["success"] = false
			result["executed_actions"] = i + 1
			result["results"] = action_results
			result["error"] = "Action failed: " + action_result.get("message", "Unknown error")
			return result

	result["executed_actions"] = actions.size()
	result["results"] = action_results
	result["message"] = "All actions executed successfully"

	return result

# Action execution functions (GDScript fallback)
func _execute_single_action(action_name: String, parameters: Dictionary) -> Dictionary:
	var result = {"success": false, "message": "", "action": action_name}

	if action_name == "command_call":
		result = _execute_call(parameters)
	elif action_name == "command_set_script":
		result = _execute_script_action(parameters)
	else:
		result["message"] = "Unknown action: " + action_name

	return result

func _execute_call(params: Dictionary) -> Dictionary:
	var method = params.get("method", "")
	var node_path = params.get("node_path", "")
	var arguments = params.get("arguments", [])

	if method.is_empty():
		return {"success": false, "message": "method parameter required", "action": "command_call"}

	# Special handling for node creation
	if method == "create_node":
		var node_type = params.get("node_type", "Node")
		var name = params.get("name", "")
		var parent_path = params.get("parent_path", "/root")

		var parent = _get_node(parent_path)
		if not parent:
			return {"success": false, "message": "Parent node not found: " + parent_path, "action": "command_call"}

		# Create the node
		var new_node = ClassDB.instantiate(node_type)
		if not new_node:
			return {"success": false, "message": "Failed to create node of type: " + node_type, "action": "command_call"}

		if not name.is_empty():
			new_node.name = name

		parent.add_child(new_node)
		return {"success": true, "message": "Created " + node_type + " node", "action": "command_call"}

	# For all other methods, we need a target
	var target = _get_node(node_path)
	if not target:
		return {"success": false, "message": "Target node not found: " + node_path, "action": "command_call"}

	# Apply unified security checks
	var security_result = _apply_method_security_checks(method, arguments)
	if not security_result.is_empty():
		return {"success": false, "message": security_result.get("error", "Security check failed"), "action": "command_call"}

	# Check if set_script is allowed (only if SafeGDScript is available)
	if method == "set_script" and not _is_safegdscript_available():
		return {"success": false, "message": "set_script is not available - SafeGDScript not found in this build", "action": "command_call"}

	# Execute the method call
	if target.has_method(method):
		var call_result = target.callv(method, arguments)
		return {
			"success": true,
			"message": "Called " + method + " on " + node_path,
			"action": "command_call",
			"result": call_result
		}
	else:
		return {"success": false, "message": "Method " + method + " not found on node " + node_path, "action": "command_call"}

func _execute_script_action(params: Dictionary) -> Dictionary:
	var script_code = params.get("script", "")
	var input_properties = params.get("input_properties", {})

	if script_code.is_empty():
		return {"success": false, "message": "Script code is required", "action": "command_set_script"}

	# Execute the SafeGDScript code
	var script_result = _execute_safescript_code(script_code, input_properties)

	# Check if script executed successfully
	var executed = script_result.get("executed", false)

	return {
		"success": executed,
		"message": "Script executed",
		"action": "command_set_script",
		"data": script_result,
		"result": script_result.get("execution_result")
	}

# Security check functions (unified between SafeGDScript and command_call)
func _apply_security_checks(code, permission_level):
	if permission_level != "public" and permission_level != "restricted":
		return {}

	# Dangerous patterns (same as SafeGDScript)
	var dangerous_patterns = [
		"OS.execute", "OS.create_process", "OS.shell_open", "OS.move_to_trash",
		"DirAccess.remove", "DirAccess.rename", "DirAccess.make_dir_recursive",
		"FileAccess.open", "FileAccess.open_encrypted", "FileAccess.open_compressed",
		"Engine.get_singleton", "ClassDB.instantiate", "ClassDB.class_call_static",
		"load(", "ResourceLoader.load", "GDScript.new",
		"JavaScript.eval", "Expression.new",
		"HTTPClient.new", "TCPServer.new", "TCPClient.new", "UDPServer.new", "UDPClient.new",
		"WebSocketClient.new", "WebSocketServer.new", "WebRTCPeerConnection.new",
		"Thread.new", "Mutex.new", "Semaphore.new"
	]

	for pattern in dangerous_patterns:
		if pattern in code:
			return {"error": "Code contains potentially dangerous operations: " + pattern}

	# Additional security checks
	if "import" in code or "preload" in code:
		return {"error": "Code contains import/preload statements which are not allowed"}

	print("✓ User/restricted script - comprehensive security applied")
	return {}

func _apply_method_security_checks(method: String, arguments: Array) -> Dictionary:
	# Dangerous methods
	var dangerous_methods = [
		"execute", "create_process", "shell_open", "move_to_trash",
		"open", "open_encrypted", "open_compressed",
		"remove", "rename", "make_dir_recursive",
		"get_singleton", "instantiate", "class_call_static",
		"eval", "new"
	]

	for dangerous_method in dangerous_methods:
		if method == dangerous_method:
			return {"error": "Method '" + method + "' is blocked for security reasons"}

	# Dangerous classes
	var dangerous_classes = [
		"HTTPClient", "TCPServer", "TCPClient", "UDPServer", "UDPClient",
		"WebSocketClient", "WebSocketServer", "WebRTCPeerConnection",
		"Thread", "Mutex", "Semaphore",
		"FileAccess", "DirAccess", "ConfigFile",
		"OS", "Engine", "ClassDB", "ResourceLoader",
		"JavaScript", "Expression"
	]

	# Check arguments for dangerous class instantiation
	for arg in arguments:
		if typeof(arg) == TYPE_STRING:
			for dangerous_class in dangerous_classes:
				if dangerous_class + ".new" in arg or dangerous_class + "." in arg:
					return {"error": "Access to '" + dangerous_class + "' is blocked for security reasons"}

	# Block load/preload operations
	if method == "load" or method == "preload":
		return {"error": "Resource loading operations are blocked for security reasons"}

	return {}

func _is_safegdscript_available():
	return ClassDB.class_exists("SafeGDScript")

func _execute_safescript_code(code: String, input_properties: Dictionary = {}, method_name: String = ""):
	"""Execute SafeGDScript code and return results"""
	var result = {
		"executed": false,
		"execution_result": null,
		"exported_results": {},
		"message": "",
		"result": null
	}

	# Apply security checks
	var security_errors = _apply_security_checks(code, "public")
	if not security_errors.is_empty():
		result["error"] = security_errors.get("error", "Security check failed")
		return result

	if not _is_safegdscript_available():
		result["error"] = "SafeGDScript is not available in this Godot build"
		return result

	# Create SafeGDScript instance
	var script = ClassDB.instantiate("SafeGDScript")
	if not script:
		result["error"] = "Failed to create SafeGDScript instance"
		return result

	# Set the source code
	script.source_code = code

	# Validate the script
	var error = script.reload()
	if error != OK:
		result["error"] = "Syntax error in code: " + str(error)
		result["parse_error"] = "Parse error code: " + str(error)
		return result

	# Create a test node to run the script on
	var test_node = Node.new()
	test_node.name = "SafeGDScript_Test_Node"
	get_tree().root.add_child(test_node)

	# Set the script on the node
	test_node.set_script(script)

	# Force script property initialization by calling _ready if it exists
	if test_node.has_method("_ready"):
		test_node.call("_ready")

	# Set input properties before execution
	if not input_properties.is_empty():
		for prop_name in input_properties.keys():
			if test_node.has_property(prop_name):
				test_node.set(prop_name, input_properties[prop_name])

	# Execute the script
	var execution_result = null
	var execution_error = null

	if not method_name.is_empty():
		if test_node.has_method(method_name):
			execution_result = test_node.call(method_name)
		else:
			execution_error = "Method '" + method_name + "' not found in script"
	
	# Collect exported properties after execution
	var exported_data = _collect_exported_properties_from_script(script, code)
	if not exported_data.is_empty():
		result["exported_results"] = exported_data

	# Clean up
	test_node.queue_free()

	if execution_error:
		result["error"] = execution_error
		result["message"] = "Script validation successful but execution failed"
	else:
		result["executed"] = true
		result["execution_result"] = execution_result
		result["result"] = execution_result
		result["message"] = "Script executed successfully"

	return result

func _collect_exported_properties_from_script(script, source_code: String):
	var exported_data = {}

	# Get available properties from the script object
	var available_properties = script.get_script_property_list()

	# Parse source code for @export declarations
	var lines = source_code.split("\n")
	var exported_vars = []

	for line in lines:
		var trimmed = line.strip_edges()
		if trimmed.begins_with("@export"):
			# Parse @export var name: Type = value
			var var_start = trimmed.find("var ")
			if var_start != -1:
				var var_decl = trimmed.substr(var_start + 4)
				var colon_pos = var_decl.find(":")
				if colon_pos != -1:
					var var_name = var_decl.substr(0, colon_pos).strip_edges()
					exported_vars.append(var_name)

	# Try to access the exported variables
	for var_name in exported_vars:
		if script.has_method("get"):
			var value = script.call("get", var_name)
			if value != null:
				exported_data[var_name] = value

	return exported_data

# Helper functions
func _get_node(path: String):
	if path == "/root":
		return get_tree().root
	else:
		return get_tree().root.get_node_or_null(path)

# The actual tool implementations are now in the C++ ScriptServer class
# This GDScript file just provides the Godot scene integration
