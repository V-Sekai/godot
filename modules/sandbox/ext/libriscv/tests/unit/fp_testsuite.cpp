/**************************************************************************/
/*  fp_testsuite.cpp                                                      */
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

#include <catch2/catch_test_macros.hpp>
#include <libriscv/machine.hpp>
extern std::vector<uint8_t> build_and_load(const std::string &code,
		const std::string &args = "-O2 -static", bool cpp = false);
static const uint64_t MAX_MEMORY = 8ul << 20; /* 8MB */
static const uint64_t MAX_INSTRUCTIONS = 10'000'000ul;
static const std::string cwd{ SRCDIR };
using namespace riscv;

TEST_CASE("Verify floating point instructions", "[Verification]") {
	const auto fpfuncfile = cwd + "/fptest/floating-point.cpp";
	const auto binary = build_and_load(R"M(
	#include "fptest/fptest.cpp"
)M",
			"-O2 -static -I" + cwd + " " + fpfuncfile, true);

	riscv::Machine<RISCV64> machine{ binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls();
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
			{ "compute_pi" },
			{ "LC_TYPE=C", "LC_ALL=C", "USER=root" });
	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 0);
}

TEST_CASE("Compute PI slowly", "[Verification]") {
	const auto binary = build_and_load(R"M(
	#include <assert.h>
	#include <float.h>

	inline int kinda64(float val, double expectation) {
		return val >= expectation-FLT_EPSILON
			&& val < expectation+FLT_EPSILON;
	}

	static struct {
		double sum;
		int counter;
		int sign;
	} pi;

	static double compute_more_pi()
	{
	    pi.sum += pi.sign / (2.0 * pi.counter + 1.0);
		pi.counter ++;
		pi.sign = -pi.sign;
	    return 4.0 * pi.sum;
	}
	int main() {
		pi.sign = 1;
		assert(kinda64(compute_more_pi(), 4.0));
		assert(kinda64(compute_more_pi(), 2.66666666666));
		assert(kinda64(compute_more_pi(), 3.46666666666));
	})M");

	riscv::Machine<RISCV64> machine{ binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls();
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
			{ "compute_pi" },
			{ "LC_TYPE=C", "LC_ALL=C", "USER=root" });
	// Run for at most X instructions before giving up
	machine.simulate(MAX_INSTRUCTIONS);

	REQUIRE(machine.return_value() == 0);
}
