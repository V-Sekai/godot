/**************************************************************************/
/*  beam_server.cpp                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifdef LINUXBSD_ENABLED

#define TIMEOUT 5000

#include "beam_server.h"

#include <arpa/inet.h>
#include <ei.h>
#include <ei_connect.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

#include "core/error/error_macros.h"
#include "core/object/class_db.h"
#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "scene/main/node.h"

BeamServer *BeamServer::singleton = nullptr;

void BeamServer::_bind_methods() {
	// No methods to bind for now
}

BeamServer *BeamServer::get_singleton() {
	return singleton;
}

BeamServer::BeamServer() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

BeamServer::~BeamServer() {
	finish();
	singleton = nullptr;
}

void BeamServer::init() {
	print_line("Initializing Godot BEAM server...");

	// Initialize Erlang Interface library
	if (ei_init() < 0) {
		WARN_PRINT("Failed to initialize Erlang Interface library - BEAM server disabled");
		return;
	}

	print_line("Erlang Interface library initialized");

	// Seed random number generator
	srand(time(NULL));

	// Create node name
	const char *node_name = "beam_server";
	const char *cookie = "godot_beam";

	print_line("Setting up Erlang node: " + String(node_name));

	struct in_addr ipaddr;
	ipaddr.s_addr = htonl(INADDR_LOOPBACK);

	// Try to initialize the Erlang cnode
	int result = ei_connect_xinit(&cnode, node_name, "localhost", NULL, &ipaddr, cookie, 1U);
	if (result < 0) {
		WARN_PRINT("Failed to initialize Erlang cnode (error: " + itos(result) + ") - BEAM server disabled");
		return;
	}

	print_line("Erlang C-node initialized: " + String(node_name));

	// Listen for RPC calls
	listen_fd = ei_publish(&cnode, 0);
	if (listen_fd == -1) {
		WARN_PRINT("Failed to publish Erlang node - BEAM server disabled");
		return;
	}

	print_line("Erlang node published and listening for connections");
	print_line("Godot BEAM server started on " + String(node_name) + " with cookie: " + String(cookie));
}

void BeamServer::finish() {
	if (client_fd > 0) {
		close(client_fd);
		client_fd = -1;
	}
	if (listen_fd > 0) {
		close(listen_fd);
		listen_fd = -1;
	}
}

void BeamServer::update() {
	// Check for new connections (non-blocking)
	if (listen_fd > 0 && client_fd <= 0) {
		ErlConnect conn;
		int fd = ei_accept(&cnode, listen_fd, &conn);
		if (fd > 0) {
			client_fd = fd;
		}
	}

	// Process messages from existing connection
	if (client_fd > 0) {
		ei_x_buff x;
		erlang_msg msg;

		ei_x_new(&x);

		int size = ei_receive_msg_tmo(client_fd, &msg, &x, 0); // Non-blocking

		if (size > 0) {
			// Message received, process it
			process_rpc_message(client_fd, &msg, &x);
		} else if (size < 0) {
			// Connection closed or error
			close(client_fd);
			client_fd = -1;
		}

		ei_x_free(&x);
	}
}

void BeamServer::process_rpc_message(int fd, erlang_msg *msg, ei_x_buff *x) {
	ei_x_buff response;
	int index = 1; // Skip version byte
	int arity;
	char atom[256];

	// Object scope for this RPC call
	BeamObjectScope object_scope;

	// Decode the message
	if (ei_decode_tuple_header(x->buff, &index, &arity) < 0) {
		std::cerr << "Failed to decode message tuple" << std::endl;
		return;
	}

	if (arity < 2) { // {Function, Args...}
		std::cerr << "Invalid message arity: " << arity << std::endl;
		return;
	}

	// Get the function name
	if (ei_decode_atom(x->buff, &index, atom) < 0) {
		std::cerr << "Failed to decode function atom" << std::endl;
		return;
	}

	// Initialize response buffer
	ei_x_new_with_version(&response);

	Variant result;

	// Handle different RPC functions
	if (strcmp(atom, "call_method") == 0) {
		if (arity != 4) {
			std::cerr << "call_method requires 3 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, method_name, args_array
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant method_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant args_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || method_var.get_type() != Variant::STRING) {
			std::cerr << "call_method: first arg must be object, second must be string" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String method_name = method_var.operator String();

		if (args_var.get_type() == Variant::ARRAY) {
			Array args = args_var.operator Array();
			result = obj->call(method_name, args);
		} else {
			result = obj->call(method_name);
		}

	} else if (strcmp(atom, "get_property") == 0) {
		if (arity != 3) {
			std::cerr << "get_property requires 2 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, property_name
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant prop_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || prop_var.get_type() != Variant::STRING) {
			std::cerr << "get_property: first arg must be object, second must be string" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String prop_name = prop_var.operator String();
		result = obj->get(prop_name);

	} else if (strcmp(atom, "set_property") == 0) {
		if (arity != 4) {
			std::cerr << "set_property requires 3 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, property_name, value
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant prop_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant value_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || prop_var.get_type() != Variant::STRING) {
			std::cerr << "set_property: first arg must be object, second must be string" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String prop_name = prop_var.operator String();
		obj->set(prop_name, value_var);
		result = Variant(true); // Success

	} else if (strcmp(atom, "create_node") == 0) {
		if (arity != 2) {
			std::cerr << "create_node requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: class_name
		Variant class_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (class_var.get_type() != Variant::STRING) {
			std::cerr << "create_node: argument must be string" << std::endl;
			ei_x_free(&response);
			return;
		}

		String class_name = class_var.operator String();
		Object *node = ClassDB::instantiate(class_name);
		if (node) {
			object_scope.add_scoped_object(node);
			result = Variant(node);
		} else {
			result = Variant(); // nil
		}

	} else if (strcmp(atom, "free_object") == 0) {
		if (arity != 2) {
			std::cerr << "free_object requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: object
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT) {
			std::cerr << "free_object: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		if (obj) {
			memdelete(obj);
			result = Variant(true);
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "get_class") == 0) {
		if (arity != 2) {
			std::cerr << "get_class requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: object
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT) {
			std::cerr << "get_class: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		if (obj) {
			result = Variant(obj->get_class());
		} else {
			result = Variant();
		}

	} else if (strcmp(atom, "is_class") == 0) {
		if (arity != 3) {
			std::cerr << "is_class requires 2 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, class_name
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant class_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || class_var.get_type() != Variant::STRING) {
			std::cerr << "is_class: first arg must be object, second must be string" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String class_name = class_var.operator String();
		if (obj) {
			result = Variant(obj->is_class(class_name));
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "inherits_from") == 0) {
		if (arity != 3) {
			std::cerr << "inherits_from requires 2 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, class_name
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant class_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || class_var.get_type() != Variant::STRING) {
			std::cerr << "inherits_from: first arg must be object, second must be string" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String class_name = class_var.operator String();
		if (obj) {
			result = Variant(ClassDB::is_parent_class(obj->get_class(), class_name));
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "has_method") == 0) {
		if (arity != 3) {
			std::cerr << "has_method requires 2 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, method_name
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant method_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || method_var.get_type() != Variant::STRING) {
			std::cerr << "has_method: first arg must be object, second must be string" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String method_name = method_var.operator String();
		if (obj) {
			result = Variant(obj->has_method(method_name));
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "get_method_list") == 0) {
		if (arity != 2) {
			std::cerr << "get_method_list requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: object
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT) {
			std::cerr << "get_method_list: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		if (obj) {
			List<MethodInfo> methods;
			obj->get_method_list(&methods);
			Array method_names;
			for (const MethodInfo &mi : methods) {
				method_names.push_back(mi.name);
			}
			result = Variant(method_names);
		} else {
			result = Variant(Array());
		}

	} else if (strcmp(atom, "has_property") == 0) {
		if (arity != 3) {
			std::cerr << "has_property requires 2 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, property_name
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant prop_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || prop_var.get_type() != Variant::STRING) {
			std::cerr << "has_property: first arg must be object, second must be string" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String prop_name = prop_var.operator String();
		if (obj) {
			bool valid = false;
			obj->get(prop_name, &valid);
			result = Variant(valid);
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "get_property_list") == 0) {
		if (arity != 2) {
			std::cerr << "get_property_list requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: object
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT) {
			std::cerr << "get_property_list: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		if (obj) {
			List<PropertyInfo> properties;
			obj->get_property_list(&properties);
			Array property_names;
			for (const PropertyInfo &pi : properties) {
				property_names.push_back(pi.name);
			}
			result = Variant(property_names);
		} else {
			result = Variant(Array());
		}

	} else if (strcmp(atom, "connect_signal") == 0) {
		if (arity != 5) {
			std::cerr << "connect_signal requires 4 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, signal_name, target_object, method_name
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant signal_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant target_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant method_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || signal_var.get_type() != Variant::STRING ||
			target_var.get_type() != Variant::OBJECT || method_var.get_type() != Variant::STRING) {
			std::cerr << "connect_signal: invalid argument types" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String signal_name = signal_var.operator String();
		Object *target = target_var.operator Object *();
		String method_name = method_var.operator String();

		if (obj && target) {
			Error err = obj->connect(signal_name, Callable(target, method_name));
			result = Variant(err == OK);
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "disconnect_signal") == 0) {
		if (arity != 5) {
			std::cerr << "disconnect_signal requires 4 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, signal_name, target_object, method_name
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant signal_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant target_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant method_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || signal_var.get_type() != Variant::STRING ||
			target_var.get_type() != Variant::OBJECT || method_var.get_type() != Variant::STRING) {
			std::cerr << "disconnect_signal: invalid argument types" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String signal_name = signal_var.operator String();
		Object *target = target_var.operator Object *();
		String method_name = method_var.operator String();

		if (obj && target) {
			obj->disconnect(signal_name, Callable(target, method_name));
			result = Variant(true);
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "emit_signal") == 0) {
		if (arity < 3) {
			std::cerr << "emit_signal requires at least 2 arguments (object and signal_name)" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: object, signal_name, [args...]
		Variant obj_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant signal_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (obj_var.get_type() != Variant::OBJECT || signal_var.get_type() != Variant::STRING) {
			std::cerr << "emit_signal: first arg must be object, second must be string" << std::endl;
			ei_x_free(&response);
			return;
		}

		Object *obj = obj_var.operator Object *();
		String signal_name = signal_var.operator String();

		if (obj) {
			// Decode additional arguments
			int num_args = arity - 3; // -1 for function name, -2 for object and signal
			if (num_args > 0) {
				// Create array of Variants for emit_signal
				Variant *args = new Variant[num_args];
				for (int i = 0; i < num_args; i++) {
					args[i] = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
				}
				// Use the variadic template version by calling with individual arguments
				// This is a workaround since direct casting is problematic
				switch (num_args) {
					case 1: obj->emit_signal(signal_name, args[0]); break;
					case 2: obj->emit_signal(signal_name, args[0], args[1]); break;
					case 3: obj->emit_signal(signal_name, args[0], args[1], args[2]); break;
					case 4: obj->emit_signal(signal_name, args[0], args[1], args[2], args[3]); break;
					case 5: obj->emit_signal(signal_name, args[0], args[1], args[2], args[3], args[4]); break;
					default:
						// Too many arguments, not supported
						result = Variant(false);
						delete[] args;
						ei_x_free(&response);
						return;
				}
				delete[] args;
			} else {
				obj->emit_signal(signal_name);
			}
			result = Variant(true);
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "add_child") == 0) {
		if (arity != 4) {
			std::cerr << "add_child requires 3 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: parent_node, child_node, name
		Variant parent_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant child_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant name_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (parent_var.get_type() != Variant::OBJECT || child_var.get_type() != Variant::OBJECT ||
			name_var.get_type() != Variant::STRING) {
			std::cerr << "add_child: invalid argument types" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *parent = Object::cast_to<Node>(parent_var.operator Object *());
		Node *child = Object::cast_to<Node>(child_var.operator Object *());
		String name = name_var.operator String();

		if (parent && child) {
			parent->add_child(child);
			if (!name.is_empty()) {
				child->set_name(name);
			}
			result = Variant(true);
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "remove_child") == 0) {
		if (arity != 3) {
			std::cerr << "remove_child requires 2 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: parent_node, child_node
		Variant parent_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant child_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (parent_var.get_type() != Variant::OBJECT || child_var.get_type() != Variant::OBJECT) {
			std::cerr << "remove_child: invalid argument types" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *parent = Object::cast_to<Node>(parent_var.operator Object *());
		Node *child = Object::cast_to<Node>(child_var.operator Object *());

		if (parent && child && child->get_parent() == parent) {
			parent->remove_child(child);
			result = Variant(true);
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "get_parent") == 0) {
		if (arity != 2) {
			std::cerr << "get_parent requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: node
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT) {
			std::cerr << "get_parent: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		if (node) {
			Node *parent = node->get_parent();
			result = Variant(parent);
		} else {
			result = Variant();
		}

	} else if (strcmp(atom, "get_child_count") == 0) {
		if (arity != 2) {
			std::cerr << "get_child_count requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: node
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT) {
			std::cerr << "get_child_count: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		if (node) {
			result = Variant(node->get_child_count());
		} else {
			result = Variant(0);
		}

	} else if (strcmp(atom, "get_child") == 0) {
		if (arity != 3) {
			std::cerr << "get_child requires 2 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: node, index
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant index_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT || index_var.get_type() != Variant::INT) {
			std::cerr << "get_child: invalid argument types" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		int idx = index_var.operator int();

		if (node && idx >= 0 && idx < node->get_child_count()) {
			Node *child = node->get_child(idx);
			result = Variant(child);
		} else {
			result = Variant();
		}

	} else if (strcmp(atom, "find_child") == 0) {
		if (arity != 4) {
			std::cerr << "find_child requires 3 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: node, pattern, recursive
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant pattern_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant recursive_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT || pattern_var.get_type() != Variant::STRING ||
			recursive_var.get_type() != Variant::BOOL) {
			std::cerr << "find_child: invalid argument types" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		String pattern = pattern_var.operator String();
		bool recursive = recursive_var.operator bool();

		if (node) {
			Node *child = node->find_child(pattern, recursive);
			result = Variant(child);
		} else {
			result = Variant();
		}

	} else if (strcmp(atom, "get_node") == 0) {
		if (arity != 3) {
			std::cerr << "get_node requires 2 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: node, path
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant path_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT || path_var.get_type() != Variant::NODE_PATH) {
			std::cerr << "get_node: invalid argument types" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		NodePath path = path_var.operator NodePath();

		if (node) {
			Node *target = node->get_node(path);
			result = Variant(target);
		} else {
			result = Variant();
		}

	} else if (strcmp(atom, "get_path") == 0) {
		if (arity != 2) {
			std::cerr << "get_path requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: node
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT) {
			std::cerr << "get_path: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		if (node) {
			NodePath path = node->get_path();
			result = Variant(path);
		} else {
			result = Variant();
		}

	} else if (strcmp(atom, "set_name") == 0) {
		if (arity != 3) {
			std::cerr << "set_name requires 2 arguments" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode arguments: node, name
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);
		Variant name_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT || name_var.get_type() != Variant::STRING) {
			std::cerr << "set_name: invalid argument types" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		String name = name_var.operator String();

		if (node) {
			node->set_name(name);
			result = Variant(true);
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "get_name") == 0) {
		if (arity != 2) {
			std::cerr << "get_name requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: node
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT) {
			std::cerr << "get_name: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		if (node) {
			String name = node->get_name();
			result = Variant(name);
		} else {
			result = Variant();
		}

	} else if (strcmp(atom, "queue_free") == 0) {
		if (arity != 2) {
			std::cerr << "queue_free requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: node
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT) {
			std::cerr << "queue_free: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		if (node) {
			node->queue_free();
			result = Variant(true);
		} else {
			result = Variant(false);
		}

	} else if (strcmp(atom, "is_inside_tree") == 0) {
		if (arity != 2) {
			std::cerr << "is_inside_tree requires 1 argument" << std::endl;
			ei_x_free(&response);
			return;
		}

		// Decode argument: node
		Variant node_var = ErlangVariantDecoder::decode_variant(x, &index, &object_scope);

		if (node_var.get_type() != Variant::OBJECT) {
			std::cerr << "is_inside_tree: argument must be object" << std::endl;
			ei_x_free(&response);
			return;
		}

		Node *node = Object::cast_to<Node>(node_var.operator Object *());
		if (node) {
			result = Variant(node->is_inside_tree());
		} else {
			result = Variant(false);
		}

	} else {
		std::cerr << "Unknown RPC function: " << atom << std::endl;
		ei_x_free(&response);
		return;
	}

	// Encode the result
	ErlangVariantEncoder::encode_variant(&response, result, &object_scope);

	// Send the response back
	if (ei_send(fd, &msg->from, response.buff, response.index) < 0) {
		std::cerr << "Failed to send response" << std::endl;
	}

	ei_x_free(&response);
}

#endif // LINUXBSD_ENABLED
