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
