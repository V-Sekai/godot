/**************************************************************************/
/*  test_matmul.cpp                                                       */
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

#include "tests/test_macros.h"

#include "../litrt_compiled_model.h"
#include "../litrt_environment.h"
#include "../litrt_model.h"
#include "../litrt_tensor_buffer.h"

#include "core/io/file_access.h"
#include "core/os/os.h"
#include "tests/test_utils.h"

// NOTE: This test uses a pre-generated TFLite model file (matmul_model.tflite).
// To generate the model, use the Python script:
//   python3 modules/litert/tests/generate_matmul_model.py
// Then copy the generated matmul_model.tflite to tests/data/
//
// Programmatic C++ model generation would require TFLite model building APIs
// with complex dependencies (tflite/converter/schema/mutable/schema_generated.h),
// so the Python script approach is preferred for simplicity.

TEST_CASE("[Litrt][MatMul] Hello World Matrix Multiplication") {
	// Create environment
	Ref<LiteRtEnvironmentRef> environment = memnew(LiteRtEnvironmentRef);
	Error err = environment->create();
	REQUIRE_MESSAGE(err == OK, "Environment should be created successfully");

	// Load model from file
	Ref<LiteRtModelRef> model = memnew(LiteRtModelRef);
	// Use test data path (tests/data/) instead of res://
	String model_path = TestUtils::get_data_path("matmul_model.tflite");

	err = model->load_from_file(model_path);

	// If model file doesn't exist, skip test with helpful message
	if (err != OK) {
		INFO("Model not found at: ", model_path);
		INFO("To generate the model, run: python3 modules/litert/tests/generate_matmul_model.py");
		INFO("Then copy the generated matmul_model.tflite to tests/data/");
		return;
	}

	// Create compiled model
	Ref<LiteRtCompiledModelRef> compiled_model = memnew(LiteRtCompiledModelRef);
	err = compiled_model->create(environment, model);
	REQUIRE_MESSAGE(err == OK, "Compiled model should be created successfully");

	// Create input buffer (2x3 matrix: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
	PackedFloat32Array input_data;
	input_data.push_back(1.0f);
	input_data.push_back(2.0f);
	input_data.push_back(3.0f);
	input_data.push_back(4.0f);
	input_data.push_back(5.0f);
	input_data.push_back(6.0f);

	PackedInt32Array input_shape;
	input_shape.push_back(2);
	input_shape.push_back(3);

	Ref<LiteRtTensorBufferRef> input_buffer = memnew(LiteRtTensorBufferRef);
	err = input_buffer->create_from_array(input_data, input_shape);
	REQUIRE_MESSAGE(err == OK, "Input buffer should be created successfully");

	// Create output buffer (2x2 matrix: [[0.0, 0.0], [0.0, 0.0]])
	PackedFloat32Array output_data;
	output_data.push_back(0.0f);
	output_data.push_back(0.0f);
	output_data.push_back(0.0f);
	output_data.push_back(0.0f);

	PackedInt32Array output_shape;
	output_shape.push_back(2);
	output_shape.push_back(2);

	Ref<LiteRtTensorBufferRef> output_buffer = memnew(LiteRtTensorBufferRef);
	err = output_buffer->create_from_array(output_data, output_shape);
	REQUIRE_MESSAGE(err == OK, "Output buffer should be created successfully");

	// Run inference
	// Use Array instead of TypedArray since TypedArray requires type registration
	Array inputs;
	inputs.push_back(input_buffer);

	Array outputs;
	outputs.push_back(output_buffer);

	err = compiled_model->run(0, inputs, outputs);
	REQUIRE_MESSAGE(err == OK, "Inference should run successfully");

	// Get output
	PackedFloat32Array result = output_buffer->get_data();
	CHECK_MESSAGE(result.size() == 4, "Output should have 4 elements (2x2 matrix)");
	INFO("Inference completed successfully!");
	INFO("Output values: ", result);
}
