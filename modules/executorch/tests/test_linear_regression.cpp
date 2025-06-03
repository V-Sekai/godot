#include "../executorch_runtime.h"
#include "../executorch_model.h"
#include "../mcp_server_internal.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <memory>

/**
 * Simple C++ test suite for ExecuTorch Godot Module
 * Tests linear regression functionality: y = 2x + 3
 */

class LinearRegressionTest {
private:
    int tests_passed = 0;
    int tests_total = 0;

    void assert_test(bool condition, const std::string& test_name, const std::string& message = "") {
        tests_total++;
        if (condition) {
            tests_passed++;
            std::cout << "✓ " << test_name << std::endl;
        } else {
            std::cout << "✗ " << test_name;
            if (!message.empty()) {
                std::cout << " - " << message;
            }
            std::cout << std::endl;
        }
    }

    void assert_near(float actual, float expected, float tolerance, const std::string& test_name) {
        float error = std::abs(actual - expected);
        bool passed = error < tolerance;

        std::string message = "Expected: " + std::to_string(expected) +
                             ", Got: " + std::to_string(actual) +
                             ", Error: " + std::to_string(error);

        assert_test(passed, test_name, message);
    }

public:
    void test_executorch_model_basic() {
        std::cout << "\n=== ExecuTorchModel Basic Tests ===" << std::endl;

        ExecuTorchModel model;

        // Test initial state
        assert_test(!model.is_loaded(), "Model initially not loaded");
        assert_test(model.get_input_names().size() > 0, "Has default input names");
        assert_test(model.get_output_names().size() > 0, "Has default output names");

        // Test loading from buffer
        std::vector<uint8_t> dummy_data = {0x01, 0x02, 0x03, 0x04, 0x05};
        assert_test(model.load_from_buffer(dummy_data), "Load from valid buffer");
        assert_test(model.is_loaded(), "Model loaded after load_from_buffer");

        // Test input/output names
        auto input_names = model.get_input_names();
        auto output_names = model.get_output_names();
        assert_test(input_names[0] == "input_0", "Default input name");
        assert_test(output_names[0] == "output_0", "Default output name");

        // Test empty buffer
        std::vector<uint8_t> empty_data;
        ExecuTorchModel model2;
        assert_test(!model2.load_from_buffer(empty_data), "Empty buffer should fail");
    }

    void test_linear_regression_core() {
        std::cout << "\n=== Linear Regression Core Tests ===" << std::endl;

        ExecuTorchModel model;
        std::vector<uint8_t> dummy_data = {0x01, 0x02, 0x03, 0x04, 0x05};

        if (!model.load_from_buffer(dummy_data)) {
            std::cout << "✗ Failed to load model for regression tests" << std::endl;
            return;
        }

        // Test y = 2x + 3 with various inputs
        struct TestCase {
            float input;
            float expected;
            std::string name;
        };

        std::vector<TestCase> test_cases = {
            {0.0f, 3.0f, "Zero input (y = 2*0 + 3 = 3)"},
            {1.0f, 5.0f, "Unit input (y = 2*1 + 3 = 5)"},
            {2.0f, 7.0f, "Double input (y = 2*2 + 3 = 7)"},
            {-1.0f, 1.0f, "Negative input (y = 2*(-1) + 3 = 1)"},
            {0.5f, 4.0f, "Half input (y = 2*0.5 + 3 = 4)"},
            {-2.0f, -1.0f, "Negative double (y = 2*(-2) + 3 = -1)"},
            {10.0f, 23.0f, "Large positive (y = 2*10 + 3 = 23)"}
        };

        for (const auto& test_case : test_cases) {
            std::map<std::string, std::vector<float>> inputs;
            inputs["input_0"] = {test_case.input};

            try {
                auto outputs = model.forward(inputs);

                if (outputs.find("output_0") != outputs.end() && outputs["output_0"].size() > 0) {
                    float result = outputs["output_0"][0];
                    assert_near(result, test_case.expected, 0.001f, test_case.name);
                } else {
                    assert_test(false, test_case.name, "No output produced");
                }
            } catch (const std::exception& e) {
                assert_test(false, test_case.name, "Exception: " + std::string(e.what()));
            }
        }
    }

    void test_executorch_runtime() {
        std::cout << "\n=== ExecuTorchRuntime Tests ===" << std::endl;

        ExecuTorchRuntime runtime;

        // Test initial state
        assert_test(!runtime.is_model_loaded(), "Runtime initially no model");
        assert_test(runtime.get_total_inferences() == 0, "Initial inference count zero");
        assert_test(runtime.get_last_inference_time() == 0.0, "Initial inference time zero");

        // Load model through runtime (using public method)
        std::vector<uint8_t> dummy_data = {0x01, 0x02, 0x03, 0x04, 0x05};

        // Create a temporary file for testing
        std::string temp_file = "/tmp/test_model.pte";
        std::ofstream file(temp_file, std::ios::binary);
        file.write(reinterpret_cast<const char*>(dummy_data.data()), dummy_data.size());
        file.close();

        bool loaded = runtime.load_model_from_file(temp_file);
        assert_test(loaded, "Load model from file");
        assert_test(runtime.is_model_loaded(), "Runtime reports model loaded");

        if (loaded) {
            // Test inference through runtime
            std::map<std::string, std::vector<float>> inputs;
            inputs["input_0"] = {2.5f};

            try {
                auto result = runtime.run_inference(inputs);

                if (result.find("output_0") != result.end() && result["output_0"].size() > 0) {
                    float output = result["output_0"][0];
                    float expected = 8.0f; // y = 2*2.5 + 3 = 8
                    assert_near(output, expected, 0.001f, "Runtime linear regression");
                } else {
                    assert_test(false, "Runtime linear regression", "No output");
                }

                // Test performance monitoring
                assert_test(runtime.get_total_inferences() > 0, "Inference count updated");
                assert_test(runtime.get_last_inference_time() > 0.0, "Inference time recorded");

            } catch (const std::exception& e) {
                assert_test(false, "Runtime inference", "Exception: " + std::string(e.what()));
            }
        }

        // Clean up
        std::remove(temp_file.c_str());
    }

    void test_mcp_server() {
        std::cout << "\n=== MCP Server Tests ===" << std::endl;

        MCPServerInternal server;

        // Test initialization
        assert_test(server.initialize("TestServer", "1.0.0"), "MCP server initialization");
        assert_test(server.is_initialized(), "MCP server initialized state");

        auto tools = server.list_tools();
        assert_test(tools.size() > 0, "MCP server has tools");

        // Check for expected tools
        bool has_inference = false;
        bool has_health = false;
        for (const auto& tool : tools) {
            if (tool == "run_inference") has_inference = true;
            if (tool == "health_check") has_health = true;
        }
        assert_test(has_inference, "Has run_inference tool");
        assert_test(has_health, "Has health_check tool");

        // Test with model
        auto model = std::make_shared<ExecuTorchModel>();
        std::vector<uint8_t> dummy_data = {0x01, 0x02, 0x03, 0x04, 0x05};

        if (model->load_from_buffer(dummy_data) && server.set_model(model)) {
            assert_test(true, "Set model in MCP server");

            // Test inference through MCP
            std::map<std::string, std::vector<float>> params;
            params["input_0"] = {3.5f};

            auto response = server.call_tool("run_inference", params);
            assert_test(response.success, "MCP inference tool success");

            if (response.success && response.result.find("output_0") != response.result.end()) {
                float output = response.result["output_0"][0];
                float expected = 10.0f; // y = 2*3.5 + 3 = 10
                assert_near(output, expected, 0.001f, "MCP linear regression");
            }

            // Test health check
            auto health_response = server.call_tool("health_check", {});
            assert_test(health_response.success, "MCP health check");

        } else {
            assert_test(false, "Set model in MCP server");
        }
    }

    void test_error_handling() {
        std::cout << "\n=== Error Handling Tests ===" << std::endl;

        // Test model without loading
        ExecuTorchModel model;
        std::map<std::string, std::vector<float>> inputs;
        inputs["input_0"] = {1.0f};

        bool threw_exception = false;
        try {
            model.forward(inputs);
        } catch (const std::exception&) {
            threw_exception = true;
        }
        assert_test(threw_exception, "Model throws when not loaded");

        // Test runtime without model
        ExecuTorchRuntime runtime;
        threw_exception = false;
        try {
            runtime.run_inference(inputs);
        } catch (const std::exception&) {
            threw_exception = true;
        }
        assert_test(threw_exception, "Runtime throws when no model");

        // Test MCP server without model
        MCPServerInternal server;
        server.initialize("TestServer", "1.0.0");

        auto response = server.call_tool("run_inference", inputs);
        assert_test(!response.success, "MCP fails gracefully without model");

        // Test invalid MCP tool
        auto invalid_response = server.call_tool("nonexistent_tool", {});
        assert_test(!invalid_response.success, "MCP handles invalid tool");
    }

    void run_all_tests() {
        std::cout << "=== ExecuTorch Godot Module Linear Regression Tests ===" << std::endl;

        test_executorch_model_basic();
        test_linear_regression_core();
        test_executorch_runtime();
        test_mcp_server();
        test_error_handling();

        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << tests_total << std::endl;
        std::cout << "Passed: " << tests_passed << std::endl;
        std::cout << "Failed: " << (tests_total - tests_passed) << std::endl;

        float success_rate = (float)tests_passed / (float)tests_total * 100.0f;
        std::cout << "Success rate: " << success_rate << "%" << std::endl;

        if (tests_passed == tests_total) {
            std::cout << "🎉 All tests passed!" << std::endl;
        } else {
            std::cout << "❌ Some tests failed" << std::endl;
        }
    }

    int get_failed_count() const {
        return tests_total - tests_passed;
    }
};

int main() {
    LinearRegressionTest test;
    test.run_all_tests();
    return test.get_failed_count();
}
