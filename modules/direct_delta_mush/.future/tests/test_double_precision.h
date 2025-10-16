/**************************************************************************/
/*  test_double_precision.h                                               */
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

#include "../ddm_math.h"

namespace DDMTests {

TEST_SUITE("[Modules][DirectDeltaMush][DoublePrecision]") {
	TEST_CASE("[DoublePrecision] Float to double conversion") {
		using namespace DDMMath;

		DoublePrecision d = DoublePrecision(3.14159f);
		CHECK_MESSAGE(Math::is_equal_approx(d.hi, 3.14159f), "High part should equal input");
		CHECK_MESSAGE(Math::is_equal_approx(d.lo, 0.0f), "Low part should be zero");
	}

	TEST_CASE("[DoublePrecision] Addition - basic") {
		using namespace DDMMath;

		DoublePrecision a(1.0f, 0.0f);
		DoublePrecision b(2.0f, 0.0f);
		DoublePrecision result = double_add(a, b);

		CHECK_MESSAGE(Math::is_equal_approx(result.hi, 3.0f), "Addition result should be 3.0");
	}

	TEST_CASE("[DoublePrecision] Addition - high precision") {
		using namespace DDMMath;

		// Test case where single precision loses accuracy
		DoublePrecision a(1.0f, 1e-7f);
		DoublePrecision b(1.0f, 1e-7f);
		DoublePrecision result = double_add(a, b);

		float single_precision = (1.0f + 1e-7f) + (1.0f + 1e-7f);
		float double_precision = result.hi + result.lo;

		CHECK_MESSAGE(double_precision > single_precision ||
						Math::is_equal_approx(double_precision, 2.0f + 2e-7f, 1e-8f),
				"Double precision should preserve low-order bits");
	}

	TEST_CASE("[DoublePrecision] Multiplication - basic") {
		using namespace DDMMath;

		DoublePrecision a(3.0f, 0.0f);
		DoublePrecision b(4.0f, 0.0f);
		DoublePrecision result = double_mul(a, b);

		CHECK_MESSAGE(Math::is_equal_approx(result.hi, 12.0f), "Multiplication result should be 12.0");
	}

	TEST_CASE("[DoublePrecision] Multiplication - Dekker precision") {
		using namespace DDMMath;

		// Test Dekker multiplication precision
		DoublePrecision a(1.0f + 1e-6f, 0.0f);
		DoublePrecision b(1.0f + 1e-6f, 0.0f);
		DoublePrecision result = double_mul(a, b);

		float expected = (1.0f + 1e-6f) * (1.0f + 1e-6f);
		float actual = result.hi + result.lo;

		CHECK_MESSAGE(Math::is_equal_approx(actual, expected, 1e-12f),
				"Dekker multiplication should maintain precision");
	}

	TEST_CASE("[DoublePrecision] Division - basic") {
		using namespace DDMMath;

		DoublePrecision a(10.0f, 0.0f);
		DoublePrecision b(2.0f, 0.0f);
		DoublePrecision result = double_div(a, b);

		CHECK_MESSAGE(Math::is_equal_approx(result.hi, 5.0f), "Division result should be 5.0");
	}

	TEST_CASE("[DoublePrecision] Division - precision") {
		using namespace DDMMath;

		DoublePrecision a(1.0f, 0.0f);
		DoublePrecision b(3.0f, 0.0f);
		DoublePrecision result = double_div(a, b);

		float actual = result.hi + result.lo;
		float expected = 1.0f / 3.0f;

		CHECK_MESSAGE(Math::is_equal_approx(actual, expected, 1e-7f),
				"Division should maintain reasonable precision");
	}

	TEST_CASE("[DoublePrecision] Square root - basic") {
		using namespace DDMMath;

		DoublePrecision x(4.0f, 0.0f);
		DoublePrecision result = double_sqrt(x);

		CHECK_MESSAGE(Math::is_equal_approx(result.hi, 2.0f), "Square root of 4 should be 2");
	}

	TEST_CASE("[DoublePrecision] Square root - precision") {
		using namespace DDMMath;

		DoublePrecision x(2.0f, 0.0f);
		DoublePrecision result = double_sqrt(x);

		float actual = result.hi + result.lo;
		float expected = std::sqrt(2.0f);

		CHECK_MESSAGE(Math::is_equal_approx(actual, expected, 1e-6f),
				"Square root should maintain precision");
	}

	TEST_CASE("[DoublePrecision] Reciprocal") {
		using namespace DDMMath;

		DoublePrecision x(4.0f, 0.0f);
		DoublePrecision result = double_recip(x);

		CHECK_MESSAGE(Math::is_equal_approx(result.hi, 0.25f), "Reciprocal of 4 should be 0.25");
	}

	TEST_CASE("[DoublePrecision] Cotangent - 45 degrees") {
		using namespace DDMMath;

		float angle_45 = Math_PI / 4.0f;
		DoublePrecision result = double_cot(DoublePrecision(angle_45));

		float actual = result.hi + result.lo;
		CHECK_MESSAGE(Math::is_equal_approx(actual, 1.0f, 0.01f),
				"cot(45°) should be approximately 1.0");
	}

	TEST_CASE("[DoublePrecision] Cotangent weight computation") {
		using namespace DDMMath;

		float alpha = Math_PI / 3.0f; // 60 degrees
		float beta = Math_PI / 6.0f; // 30 degrees

		DoublePrecision weight = double_cotangent_weight(alpha, beta);
		float actual = weight.hi + weight.lo;

		// Expected: (cot(60°) + cot(30°)) / 2 = (0.577 + 1.732) / 2 ≈ 1.155
		CHECK_MESSAGE(actual > 0.0f && actual < 2.0f,
				"Cotangent weight should be in reasonable range");
	}

	TEST_CASE("[DoublePrecision] Operator overloads") {
		using namespace DDMMath;

		DoublePrecision a(3.0f, 0.0f);
		DoublePrecision b(2.0f, 0.0f);

		DoublePrecision sum = a + b;
		CHECK_MESSAGE(Math::is_equal_approx(sum.hi, 5.0f), "Operator+ should work");

		DoublePrecision diff = a - b;
		CHECK_MESSAGE(Math::is_equal_approx(diff.hi, 1.0f), "Operator- should work");

		DoublePrecision prod = a * b;
		CHECK_MESSAGE(Math::is_equal_approx(prod.hi, 6.0f), "Operator* should work");

		DoublePrecision quot = a / b;
		CHECK_MESSAGE(Math::is_equal_approx(quot.hi, 1.5f), "Operator/ should work");
	}

	TEST_CASE("[DoublePrecision] Edge cases - zero") {
		using namespace DDMMath;

		DoublePrecision zero(0.0f, 0.0f);
		DoublePrecision one(1.0f, 0.0f);

		DoublePrecision result = double_add(zero, one);
		CHECK_MESSAGE(Math::is_equal_approx(result.hi, 1.0f), "Adding zero should not change value");

		result = double_mul(zero, one);
		CHECK_MESSAGE(Math::is_equal_approx(result.hi, 0.0f), "Multiplying by zero gives zero");
	}

	TEST_CASE("[DoublePrecision] Edge cases - very small numbers") {
		using namespace DDMMath;

		DoublePrecision tiny(1e-7f, 0.0f);
		DoublePrecision result = double_add(tiny, tiny);

		float actual = result.hi + result.lo;
		CHECK_MESSAGE(actual > 0.0f, "Adding tiny numbers should not underflow");
	}

	TEST_CASE("[DoublePrecision] Accuracy test - sum of many small values") {
		using namespace DDMMath;

		DoublePrecision sum(0.0f, 0.0f);
		DoublePrecision small(0.1f, 0.0f);

		// Add 0.1 ten times
		for (int i = 0; i < 10; i++) {
			sum = double_add(sum, small);
		}

		float actual = sum.hi + sum.lo;
		CHECK_MESSAGE(Math::is_equal_approx(actual, 1.0f, 1e-6f),
				"Sum of 10 * 0.1 should be close to 1.0");
	}
}

} // namespace DDMTests
