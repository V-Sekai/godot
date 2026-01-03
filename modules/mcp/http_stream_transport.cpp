/**************************************************************************/
/*  http_stream_transport.cpp                                             */
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

#include "http_stream_transport.h"

#include "core/io/json.h"
#include "core/variant/variant_utility.h"

HTTPStreamTransport::HTTPStreamTransport() {
	server.instantiate();
}

HTTPStreamTransport::~HTTPStreamTransport() {
	stop();
}

void HTTPStreamTransport::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start", "port"), &HTTPStreamTransport::start);
	ClassDB::bind_method(D_METHOD("stop"), &HTTPStreamTransport::stop);
	ClassDB::bind_method(D_METHOD("is_listening"), &HTTPStreamTransport::is_listening);
	ClassDB::bind_method(D_METHOD("get_port"), &HTTPStreamTransport::get_port);
	ClassDB::bind_method(D_METHOD("poll"), &HTTPStreamTransport::poll);
	ClassDB::bind_method(D_METHOD("send_response", "client_id", "response"), &HTTPStreamTransport::send_response);
	ClassDB::bind_method(D_METHOD("send_notification", "client_id", "notification"), &HTTPStreamTransport::send_notification);
	ClassDB::bind_method(D_METHOD("send_202_accepted", "client_id"), &HTTPStreamTransport::send_202_accepted);
}

Error HTTPStreamTransport::start(int p_port) {
	if (listening) {
		stop();
	}

	Error err = server->listen(p_port);
	if (err != OK) {
		return err;
	}

	port = p_port;
	listening = true;
	clients.clear();
	next_client_id = 1;

	return OK;
}

void HTTPStreamTransport::stop() {
	if (!listening) {
		return;
	}

	server->stop();
	for (KeyValue<int64_t, ClientConnection> &E : clients) {
		if (E.value.tcp.is_valid()) {
			E.value.tcp->disconnect_from_host();
		}
	}
	clients.clear();
	listening = false;
	port = 0;
}

String HTTPStreamTransport::parse_chunked_data(const String &p_data, String &p_remaining) {
	String result;
	p_remaining = p_data;

	while (true) {
		int newline_pos = p_remaining.find("\r\n");
		if (newline_pos == -1) {
			break; // Need more data
		}

		String chunk_size_str = p_remaining.substr(0, newline_pos).strip_edges();
		if (chunk_size_str.is_empty()) {
			p_remaining = p_remaining.substr(newline_pos + 2);
			continue;
		}

		// Parse hex chunk size
		int chunk_size = 0;
		if (chunk_size_str.get_slice(";", 0).is_valid_hex_number(false)) {
			chunk_size = chunk_size_str.get_slice(";", 0).hex_to_int();
		}

		if (chunk_size == 0) {
			// Last chunk
			p_remaining = p_remaining.substr(newline_pos + 2);
			break;
		}

		// Check if we have enough data for this chunk
		int data_start = newline_pos + 2;
		if (p_remaining.length() < data_start + chunk_size + 2) {
			break; // Need more data
		}

		// Extract chunk data
		String chunk_data = p_remaining.substr(data_start, chunk_size);
		result += chunk_data;
		p_remaining = p_remaining.substr(data_start + chunk_size + 2);
	}

	return result;
}

String HTTPStreamTransport::extract_json_messages(const String &p_buffer, String &p_remaining) {
	// Try to find complete JSON objects
	int brace_count = 0;
	int start_pos = -1;
	p_remaining = p_buffer;

	for (int i = 0; i < p_buffer.length(); i++) {
		char c = p_buffer[i];
		if (c == '{') {
			if (start_pos == -1) {
				start_pos = i;
			}
			brace_count++;
		} else if (c == '}') {
			brace_count--;
			if (brace_count == 0 && start_pos != -1) {
				// Found complete JSON object
				String json_str = p_buffer.substr(start_pos, i - start_pos + 1);
				p_remaining = p_buffer.substr(i + 1);
				return json_str;
			}
		}
	}

	// No complete JSON object found
	p_remaining = p_buffer;
	return String();
}

void HTTPStreamTransport::process_client(ClientConnection &p_client) {
	if (!p_client.tcp.is_valid() || p_client.tcp->get_status() != StreamPeerTCP::STATUS_CONNECTED) {
		return;
	}

	// Read available data
	uint8_t buffer[4096];
	int available = p_client.tcp->get_available_bytes();
	if (available <= 0) {
		return;
	}

	int to_read = available > 4096 ? 4096 : available;
	int bytes_read = 0;
	Error err = p_client.tcp->get_partial_data(buffer, to_read, bytes_read);
	if (err != OK || bytes_read <= 0) {
		return;
	}

	String new_data = String::utf8((const char *)buffer, bytes_read);

	// Check if we have HTTP headers and haven't processed them yet
	if (!p_client.headers_received) {
		p_client.request_buffer += new_data;

		int header_end = p_client.request_buffer.find("\r\n\r\n");
		if (header_end == -1) {
			return; // Need more headers
		}

		// Parse headers
		String headers = p_client.request_buffer.substr(0, header_end);
		Vector<String> header_lines = headers.split("\r\n");
		p_client.chunked_request = false;
		p_client.sse_mode = false;
		p_client.is_get_request = false;

		// Parse request line to detect GET vs POST
		if (header_lines.size() > 0) {
			String request_line = header_lines[0];
			if (request_line.to_lower().begins_with("get ")) {
				p_client.is_get_request = true;
			}
		}

		// Check for SSE request (Accept: text/event-stream)
		for (const String &line : header_lines) {
			if (line.to_lower().begins_with("accept:")) {
				if (line.to_lower().contains("text/event-stream")) {
					p_client.sse_mode = true;
				}
			}
			if (line.to_lower().begins_with("transfer-encoding:")) {
				if (line.to_lower().contains("chunked")) {
					p_client.chunked_request = true;
				}
			}
		}

		// Send HTTP response headers immediately only for GET requests (SSE)
		if (p_client.is_get_request) {
			String response_headers = "HTTP/1.1 200 OK\r\n";
			if (p_client.sse_mode) {
				// SSE mode: use text/event-stream
				response_headers += "Content-Type: text/event-stream\r\n";
				response_headers += "Cache-Control: no-cache\r\n";
				response_headers += "Connection: keep-alive\r\n";
			} else {
				// Regular mode: use application/json with chunked encoding
				response_headers += "Content-Type: application/json\r\n";
				response_headers += "Transfer-Encoding: chunked\r\n";
				response_headers += "Connection: keep-alive\r\n";
				p_client.chunked_response = true;
			}
			response_headers += "\r\n";

			CharString response_headers_utf8 = response_headers.utf8();
			p_client.tcp->put_data((const uint8_t *)response_headers_utf8.get_data(), response_headers_utf8.length());
			p_client.response_headers_sent = true;
		}

		p_client.headers_received = true;

		// For GET requests with SSE, keep connection open for server-initiated messages
		if (p_client.is_get_request && p_client.sse_mode) {
			// GET requests: server can send notifications/requests, no priming event needed
			// Keep connection open - don't process body
			return;
		}

		// Remove headers from buffer, keep body
		p_client.body_buffer = p_client.request_buffer.substr(header_end + 4);
		p_client.request_buffer = String();
	} else {
		// Headers already received, add to body buffer
		p_client.body_buffer += new_data;
	}

	// Process chunked data if needed
	String json_data;
	if (p_client.chunked_request && p_client.headers_received) {
		String remaining;
		json_data = parse_chunked_data(p_client.body_buffer, remaining);
		p_client.body_buffer = remaining;
	} else if (!p_client.body_buffer.is_empty() && p_client.headers_received) {
		// Non-chunked: use body buffer directly
		json_data = p_client.body_buffer;
		// We will update body_buffer after extracting messages
	}

	// Extract JSON-RPC messages
	String remaining_json = json_data;
	while (true) {
		String next_remaining;
		String json_message = extract_json_messages(remaining_json, next_remaining);
		if (json_message.is_empty()) {
			break;
		}
		remaining_json = next_remaining;

		// Parse JSON
		Variant parsed = JSON::parse_string(json_message);
		if (parsed != Variant()) {
			// Support both Dictionary (single message) and Array (batch request)
			if (parsed.get_type() == Variant::DICTIONARY || parsed.get_type() == Variant::ARRAY) {
				if (message_callback.is_valid()) {
					message_callback.call(p_client.client_id, parsed);
				} else {
					print_line("HTTPStreamTransport: message_callback is not valid!");
				}
			}
		}
	}

	// Update body buffer with what's left
	if (!p_client.chunked_request) {
		p_client.body_buffer = remaining_json;
	}
}

void HTTPStreamTransport::send_chunked_response(Ref<StreamPeerTCP> p_peer, const String &p_data) {
	if (!p_peer.is_valid()) {
		return;
	}

	// Convert data to chunked format
	CharString utf8_data = p_data.utf8();
	int data_len = utf8_data.length();

	// Send chunk size (hex) + CRLF
	String chunk_header = String::num_int64(data_len, 16).to_upper() + "\r\n";
	CharString chunk_header_utf8 = chunk_header.utf8();
	p_peer->put_data((const uint8_t *)chunk_header_utf8.get_data(), chunk_header_utf8.length());

	// Send chunk data
	p_peer->put_data((const uint8_t *)utf8_data.get_data(), data_len);

	// Send CRLF after chunk
	const char *crlf = "\r\n";
	p_peer->put_data((const uint8_t *)crlf, 2);
}

void HTTPStreamTransport::poll() {
	if (!listening) {
		return;
	}

	// Accept new connections
	if (server->is_connection_available()) {
		Ref<StreamPeerTCP> tcp = server->take_connection();
		if (tcp.is_valid()) {
			ClientConnection client;
			client.tcp = tcp;
			client.client_id = next_client_id++;
			clients[client.client_id] = client;
		}
	}

	// Process existing clients
	Vector<int64_t> to_remove;
	for (KeyValue<int64_t, ClientConnection> &E : clients) {
		ClientConnection &client = E.value;
		if (client.tcp.is_valid()) {
			StreamPeerTCP::Status status = client.tcp->get_status();
			if (status == StreamPeerTCP::STATUS_CONNECTED) {
				process_client(client);
			} else {
				// Connection is not connected - mark for removal
				to_remove.push_back(E.key);
			}
		} else {
			// Invalid TCP connection - mark for removal
			to_remove.push_back(E.key);
		}
	}

	// Remove disconnected clients
	for (int64_t client_id : to_remove) {
		clients.erase(client_id);
	}
}

void HTTPStreamTransport::send_response(int64_t p_client_id, const Dictionary &p_response) {
	if (!clients.has(p_client_id)) {
		return;
	}

	ClientConnection &client = clients[p_client_id];
	if (!client.tcp.is_valid() || client.tcp->get_status() != StreamPeerTCP::STATUS_CONNECTED) {
		return;
	}

	// Send headers if not sent yet
	if (!client.response_headers_sent) {
		String response_headers = "HTTP/1.1 200 OK\r\n";
		if (client.sse_mode) {
			response_headers += "Content-Type: text/event-stream\r\n";
			response_headers += "Cache-Control: no-cache\r\n";
			response_headers += "Connection: keep-alive\r\n";
			response_headers += "\r\n";
			CharString response_headers_utf8 = response_headers.utf8();
			client.tcp->put_data((const uint8_t *)response_headers_utf8.get_data(), response_headers_utf8.length());
		} else {
			// Regular HTTP mode: use Content-Length and Connection: close for simplicity
			JSON json;
			String json_str = json.stringify(p_response);
			CharString json_utf8 = json_str.utf8();

			response_headers += "Content-Type: application/json\r\n";
			response_headers += "Content-Length: " + String::num_int64(json_utf8.length()) + "\r\n";
			response_headers += "Connection: close\r\n";
			response_headers += "\r\n";

			CharString response_headers_utf8 = response_headers.utf8();
			client.tcp->put_data((const uint8_t *)response_headers_utf8.get_data(), response_headers_utf8.length());
			client.tcp->put_data((const uint8_t *)json_utf8.get_data(), json_utf8.length());
			
			// Don't disconnect immediately, let the client close or handle it in poll
			client.response_headers_sent = true;
			return;
		}
		client.response_headers_sent = true;
	}

	// Convert response to JSON for SSE
	JSON json;
	String json_str = json.stringify(p_response);

	if (client.sse_mode) {
		// SSE format: data: <json>\n\n
		// Per MCP spec: "The server SHOULD immediately send an SSE event consisting of 
		// an event ID and an empty data field" when initiating an SSE stream for POST requests.
		// However, the MCP SDK doesn't handle empty data fields, so we skip the priming event
		// and start event IDs at 1 for the actual response.
		if (client.event_id == 0) {
			client.event_id = 1;  // Start event IDs at 1
		}
		
		// Send the JSON-RPC response with event ID
		String sse_event = "id: " + String::num_int64(client.event_id) + "\n";
		sse_event += "data: " + json_str + "\n\n";
		
		CharString sse_utf8 = sse_event.utf8();
		client.tcp->put_data((const uint8_t *)sse_utf8.get_data(), sse_utf8.length());
		client.event_id++;
		
		// According to MCP spec: "After the JSON-RPC _response_ has been sent, 
		// the server **SHOULD** close the SSE stream."
		// For POST requests, close after response. For GET requests, keep open for streaming.
		if (!client.is_get_request) {
			// POST request: close connection after sending response
			client.tcp->disconnect_from_host();
		}
	} else {
		// Regular HTTP mode: already handled in the headers block above
	}
}

void HTTPStreamTransport::send_notification(int64_t p_client_id, const Dictionary &p_notification) {
	send_response(p_client_id, p_notification);
}

void HTTPStreamTransport::send_202_accepted(int64_t p_client_id) {
	if (!clients.has(p_client_id)) {
		return;
	}

	ClientConnection &client = clients[p_client_id];
	if (!client.tcp.is_valid() || client.tcp->get_status() != StreamPeerTCP::STATUS_CONNECTED) {
		return;
	}

	// For notifications, return HTTP 202 Accepted with no body (per MCP spec)
	// Only send if headers haven't been sent yet
	if (!client.response_headers_sent) {
		String response = "HTTP/1.1 202 Accepted\r\n";
		if (client.sse_mode) {
			response += "Content-Type: text/event-stream\r\n";
			response += "Connection: keep-alive\r\n";
		} else {
			response += "Content-Type: application/json\r\n";
			response += "Content-Length: 0\r\n";
			response += "Connection: close\r\n";
		}
		response += "\r\n";

		CharString response_utf8 = response.utf8();
		client.tcp->put_data((const uint8_t *)response_utf8.get_data(), response_utf8.length());
		client.response_headers_sent = true;
	}
}
