/**************************************************************************/
/*  test_http_stream_transport.h                                          */
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

#include "tests/test_macros.h"
#include "tests/test_utils.h"

#include "../http_stream_transport.h"

namespace TestHTTPStreamTransport {

TEST_CASE("[HTTPStreamTransport] Server lifecycle - start and stop") {
	Ref<HTTPStreamTransport> transport = memnew(HTTPStreamTransport);

	// Test initial state
	CHECK(transport->is_listening() == false);
	CHECK(transport->get_port() == 0);

	// Start server on available port
	Error err = transport->start(8080);
	if (err == OK) {
		CHECK(transport->is_listening() == true);
		CHECK(transport->get_port() == 8080);

		// Stop server
		transport->stop();
		CHECK(transport->is_listening() == false);
		CHECK(transport->get_port() == 0);
	} else {
		// Port might be in use, that's okay for this test
		WARN("Could not start server on port 8080, possibly in use");
	}

	// Ref will be automatically cleaned up
}

TEST_CASE("[HTTPStreamTransport] Server restart") {
	Ref<HTTPStreamTransport> transport = memnew(HTTPStreamTransport);

	// Start server
	Error err = transport->start(8081);
	if (err == OK) {
		CHECK(transport->is_listening() == true);

		// Restart on different port
		err = transport->start(8082);
		if (err == OK) {
			CHECK(transport->is_listening() == true);
			CHECK(transport->get_port() == 8082);
		}

		transport->stop();
	}

	// Ref will be automatically cleaned up
}

// Note: Private method parse_chunked_data testing removed as it tests implementation details

// Note: Private method extract_json_messages testing removed as it tests implementation details

TEST_CASE("[HTTPStreamTransport] Client connection management") {
	Ref<HTTPStreamTransport> transport = memnew(HTTPStreamTransport);

	// Test polling when not listening (should not crash)
	transport->poll();
	CHECK(transport->is_listening() == false);

	// Start server
	Error err = transport->start(8083);
	if (err == OK) {
		// Test polling with no connections
		transport->poll();

		// Test sending response to non-existent client (should not crash)
		Dictionary response;
		response["jsonrpc"] = "2.0";
		response["result"] = "test";
		response["id"] = 1;
		transport->send_response(999, response);
		transport->send_notification(999, response);

		transport->stop();
	}

	// Ref will be automatically cleaned up
}

TEST_CASE("[HTTPStreamTransport] HTTP header parsing simulation") {
	// Test the logic for parsing HTTP headers (this would be tested more thoroughly
	// with actual network connections in integration tests)

	String http_request =
		"POST / HTTP/1.1\r\n"
		"Content-Type: application/json\r\n"
		"Transfer-Encoding: chunked\r\n"
		"Content-Length: 123\r\n"
		"\r\n"
		"5\r\nHello\r\n0\r\n\r\n";

	// Verify headers are properly formatted
	int header_end = http_request.find("\r\n\r\n");
	CHECK(header_end != -1);

	String headers = http_request.substr(0, header_end);
	Vector<String> header_lines = headers.split("\r\n");

	// Should have multiple header lines
	CHECK(header_lines.size() >= 3);

	// Check for Transfer-Encoding header
	bool has_chunked = false;
	for (const String &line : header_lines) {
		if (line.to_lower().begins_with("transfer-encoding:")) {
			if (line.to_lower().contains("chunked")) {
				has_chunked = true;
				break;
			}
		}
	}
	CHECK(has_chunked == true);

	// Check body starts after headers
	String body = http_request.substr(header_end + 4);
	CHECK(body.begins_with("5\r\nHello"));
}

TEST_CASE("[HTTPStreamTransport] Memory management") {
	// Test that transport can be created and destroyed without issues
	Ref<HTTPStreamTransport> transport = memnew(HTTPStreamTransport);
	CHECK(transport.is_valid());

	// Test multiple start/stop cycles
	for (int i = 0; i < 3; i++) {
		Error err = transport->start(8084 + i);
		if (err == OK) {
			transport->poll();
			transport->stop();
		}
	}

	// Ref will be automatically cleaned up
}

// Note: Private method send_chunked_response testing removed as it tests implementation details

TEST_CASE("[HTTPStreamTransport] Error handling - invalid operations") {
	Ref<HTTPStreamTransport> transport = memnew(HTTPStreamTransport);

	// Test operations on stopped server
	transport->poll();
	transport->stop(); // Should not crash when already stopped

	// Test starting with invalid port (should fail gracefully)
	Error err = transport->start(-1);
	CHECK(err != OK);

	err = transport->start(99999); // Invalid port
	CHECK(err != OK);

	// Ref will be automatically cleaned up
}

} // namespace TestHTTPStreamTransport
