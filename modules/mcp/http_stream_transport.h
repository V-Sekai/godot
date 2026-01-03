/**************************************************************************/
/*  http_stream_transport.h                                               */
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

#pragma once

#include "core/io/stream_peer.h"
#include "core/io/tcp_server.h"
#include "core/object/ref_counted.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

class HTTPStreamTransport : public RefCounted {
	GDCLASS(HTTPStreamTransport, RefCounted);

public:
	struct ClientConnection {
		Ref<StreamPeerTCP> tcp;
		String request_buffer;
		String response_buffer;
		bool headers_received = false;
		bool response_headers_sent = false;
		bool chunked_request = false;
		bool chunked_response = false;
		bool sse_mode = false;  // Server-Sent Events mode
		bool is_get_request = false;  // True if this is a GET request
		int64_t event_id = 0;     // SSE event ID counter
		String chunked_buffer;
		String body_buffer;
		int64_t client_id;
	};

private:
	Ref<TCPServer> server;
	HashMap<int64_t, ClientConnection> clients;
	int64_t next_client_id = 1;
	int port = 0;
	bool listening = false;

	String parse_chunked_data(const String &p_data, String &p_remaining);
	String extract_json_messages(const String &p_buffer, String &p_remaining);
	void process_client(ClientConnection &p_client);
	void send_chunked_response(Ref<StreamPeerTCP> p_peer, const String &p_data);

protected:
	static void _bind_methods();

public:
	HTTPStreamTransport();
	~HTTPStreamTransport();

	Error start(int p_port);
	void stop();
	bool is_listening() const { return listening; }
	int get_port() const { return port; }

	void poll();
	void send_response(int64_t p_client_id, const Dictionary &p_response);
	void send_notification(int64_t p_client_id, const Dictionary &p_notification);
	void send_202_accepted(int64_t p_client_id);  // For JSON-RPC notifications

	// Callback for when a complete JSON-RPC message is received
	Callable message_callback;
};
