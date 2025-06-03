/**************************************************************************/
/*  test_linear_regression.h                                              */
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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../executorch_model.h"
#include "../executorch_runtime.h"
#include "../mcp_server_internal.h"

#include "tests/test_macros.h"

/**
 * Comprehensive doctest suite for ExecuTorch Godot Module
 * Tests linear regression functionality: y = 2x + 3
 *
 * https://openstax.org/books/introductory-statistics-2e/pages/12-3-the-regression-equation
 */

TEST_CASE("ExecuTorchModel Basic Functionality") {
	ExecuTorchModel model;

	SUBCASE("Initial state");
	CHECK_FALSE(model.is_loaded());
	CHECK(model.get_input_names().size() > 0); // Should have default names
	CHECK(model.get_output_names().size() > 0);

	SUBCASE("Load from buffer");
	std::vector<uint8_t> dummy_data = { 0x01, 0x02, 0x03, 0x04, 0x05 };

	CHECK(model.load_from_buffer(dummy_data));
	CHECK(model.is_loaded());

	// Test input/output names
	auto input_names = model.get_input_names();
	auto output_names = model.get_output_names();

	CHECK(input_names.size() > 0);
	CHECK(output_names.size() > 0);
	CHECK(input_names[0] == "input_0");
	CHECK(output_names[0] == "output_0");

	SUBCASE("Load from empty buffer should fail");
	std::vector<uint8_t> empty_data;
	CHECK_FALSE(model.load_from_buffer(empty_data));
	CHECK_FALSE(model.is_loaded());
}

TEST_CASE("Linear Regression Core Tests") {
	ExecuTorchModel model;
	std::vector<uint8_t> dummy_data = { 0x01, 0x02, 0x03, 0x04, 0x05 };
	REQUIRE(model.load_from_buffer(dummy_data));
	REQUIRE(model.is_loaded());

	SUBCASE("Zero input test");
	std::map<std::string, std::vector<float>> inputs;
	inputs["input_0"] = { 0.0f };

	auto outputs = model.forward(inputs);

	REQUIRE(outputs.find("output_0") != outputs.end());
	REQUIRE(outputs["output_0"].size() > 0);

	float result = outputs["output_0"][0];
	float expected = 3.0f; // y = 2*0 + 3 = 3

	CHECK(std::abs(result - expected) < 0.001f);

	SUBCASE("Unit input test");
	inputs["input_0"] = { 1.0f };
	outputs = model.forward(inputs);

	REQUIRE(outputs.find("output_0") != outputs.end());
	result = outputs["output_0"][0];
	expected = 5.0f; // y = 2*1 + 3 = 5

	CHECK(std::abs(result - expected) < 0.001f);

	SUBCASE("Negative input test");
	inputs["input_0"] = { -1.0f };
	outputs = model.forward(inputs);

	REQUIRE(outputs.find("output_0") != outputs.end());
	result = outputs["output_0"][0];
	expected = 1.0f; // y = 2*(-1) + 3 = 1

	CHECK(std::abs(result - expected) < 0.001f);
}

TEST_CASE("Multiple Linear Regression Test Cases") {
	ExecuTorchModel model;
	std::vector<uint8_t> dummy_data = { 0x01, 0x02, 0x03, 0x04, 0x05 };
	REQUIRE(model.load_from_buffer(dummy_data));

	struct TestCase {
		float input;
		float expected;
		std::string name;
	};

	std::vector<TestCase> test_cases = {
		{ 0.0f, 3.0f, "Zero" },
		{ 1.0f, 5.0f, "Unit" },
		{ 2.0f, 7.0f, "Double" },
		{ -1.0f, 1.0f, "Negative" },
		{ 0.5f, 4.0f, "Half" },
		{ -2.0f, -1.0f, "Negative Double" },
		{ 10.0f, 23.0f, "Large Positive" }
	};

	for (const auto &test_case : test_cases) {
		INFO("Testing case: " << test_case.name << " with input: " << test_case.input);

		std::map<std::string, std::vector<float>> inputs;
		inputs["input_0"] = { test_case.input };

		auto outputs = model.forward(inputs);

		REQUIRE(outputs.find("output_0") != outputs.end());
		REQUIRE(outputs["output_0"].size() > 0);

		float result = outputs["output_0"][0];
		float error = std::abs(result - test_case.expected);

		CHECK(error < 0.001f);
	}
}

TEST_CASE("ExecuTorchRuntime Integration") {
	ExecuTorchRuntime runtime;

	SUBCASE("Initial state check");
	CHECK_FALSE(runtime.is_model_loaded());
	CHECK(runtime.get_total_inferences() == 0);
	CHECK(runtime.get_last_inference_time() == 0.0);

	SUBCASE("Load and test model");
	std::vector<uint8_t> dummy_data = { 0x01, 0x02, 0x03, 0x04, 0x05 };

	CHECK(runtime._load_model_from_buffer(dummy_data));
	CHECK(runtime.is_model_loaded());

	// Test linear regression through runtime
	std::map<std::string, std::vector<float>> inputs;
	inputs["input_0"] = { 2.5f };

	auto result = runtime.run_inference(inputs);

	REQUIRE(result.find("output_0") != result.end());
	REQUIRE(result["output_0"].size() > 0);

	float output = result["output_0"][0];
	float expected = 8.0f; // y = 2*2.5 + 3 = 8

	CHECK(std::abs(output - expected) < 0.001f);

	SUBCASE("Performance monitoring");
	runtime.reset_performance_stats();
	CHECK(runtime.get_total_inferences() == 0);

	// Run multiple inferences
	for (int i = 0; i < 3; ++i) {
		std::map<std::string, std::vector<float>> test_inputs;
		test_inputs["input_0"] = { static_cast<float>(i) };
		runtime.run_inference(test_inputs);
	}

	CHECK(runtime.get_total_inferences() == 3);
	CHECK(runtime.get_last_inference_time() > 0.0);
}

TEST_CASE("MCP Server Linear Regression") {
	MCPServerInternal server;

	SUBCASE("Server initialization");
	CHECK(server.initialize("TestServer", "1.0.0"));
	CHECK(server.is_initialized());

	auto tools = server.list_tools();
	CHECK(tools.size() > 0);

	SUBCASE("Linear regression through MCP");
	// Create and set model
	auto model = std::make_shared<ExecuTorchModel>();
	std::vector<uint8_t> dummy_data = { 0x01, 0x02, 0x03, 0x04, 0x05 };
	REQUIRE(model->load_from_buffer(dummy_data));
	REQUIRE(server.set_model(model));

	// Test inference through MCP tool
	std::map<std::string, std::vector<float>> params;
	params["input_0"] = { 3.5f };

	auto response = server.call_tool("run_inference", params);

	CHECK(response.success);
	REQUIRE(response.result.find("output_0") != response.result.end());
	REQUIRE(response.result["output_0"].size() > 0);

	float output = response.result["output_0"][0];
	float expected = 10.0f; // y = 2*3.5 + 3 = 10

	CHECK(std::abs(output - expected) < 0.001f);
}

TEST_CASE("Error Handling Tests") {
	SUBCASE("Model not loaded inference");
	ExecuTorchModel model;

	std::map<std::string, std::vector<float>> inputs;
	inputs["input_0"] = { 1.0f };

	CHECK_THROWS(model.forward(inputs));

	SUBCASE("Runtime without model");
	ExecuTorchRuntime runtime;

	CHECK_THROWS(runtime.run_inference(inputs));

	SUBCASE("MCP server without model");
	MCPServerInternal server;
	REQUIRE(server.initialize("TestServer", "1.0.0"));

	std::map<std::string, std::vector<float>> params;
	params["input_0"] = { 1.0f };

	auto response = server.call_tool("run_inference", params);
	CHECK_FALSE(response.success);
}

TEST_CASE("Mathematical Precision Tests") {
	ExecuTorchModel model;
	std::vector<uint8_t> dummy_data = { 0x01, 0x02, 0x03, 0x04, 0x05 };
	REQUIRE(model.load_from_buffer(dummy_data));

	SUBCASE("Decimal precision test");
	std::vector<float> test_inputs = {
		0.1f, 0.33f, 0.666f, 1.234f, -0.5f, -1.777f
	};

	for (float input : test_inputs) {
		std::map<std::string, std::vector<float>> inputs;
		inputs["input_0"] = { input };

		auto outputs = model.forward(inputs);

		REQUIRE(outputs.find("output_0") != outputs.end());
		REQUIRE(outputs["output_0"].size() > 0);

		float result = outputs["output_0"][0];
		float expected = 2.0f * input + 3.0f;

		INFO("Input: " << input << ", Expected: " << expected << ", Got: " << result);
		CHECK(std::abs(result - expected) < 0.001f);
	}
}
