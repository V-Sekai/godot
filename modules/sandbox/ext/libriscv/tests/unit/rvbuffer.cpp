/**************************************************************************/
/*  rvbuffer.cpp                                                          */
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
#include <catch2/matchers/catch_matchers_string.hpp>

#include <libriscv/machine.hpp>
extern std::vector<uint8_t> build_and_load(const std::string &code,
		const std::string &args = "-O2 -static", bool cpp = false);
static const uint64_t MAX_MEMORY = 8ul << 20; /* 8MB */
static const uint64_t MAX_INSTRUCTIONS = 10'000'000ul;
using namespace riscv;

TEST_CASE("Sequential buffer", "[Buffer]") {
	const auto binary = build_and_load(R"M(
	int main() {
		return 666;
	})M");

	riscv::Machine<RISCV64> machine{ binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls();
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
			{ "vmcall" },
			{ "LC_TYPE=C", "LC_ALL=C", "USER=root" });

	auto origin = machine.memory.mmap_allocate(1);

	static const char hello[] = "hello world!";

	for (auto addr = origin; addr < origin + 4096 - 12; addr++) {
		machine.memory.memcpy(addr, hello, sizeof(hello));

		auto buf = machine.memory.rvbuffer(addr, 12);
		REQUIRE(buf.is_sequential());
		REQUIRE(buf.size() == 12);
		REQUIRE(buf.strview() == "hello world!");
		REQUIRE(buf.to_string() == "hello world!");
	}

	// maxlen works
	REQUIRE_THROWS_WITH([&] {
		machine.memory.rvbuffer(origin, 128, 127);
	}(),
			Catch::Matchers::ContainsSubstring("Protection fault"));
}

TEST_CASE("Boundary buffer", "[Buffer]") {
	const auto binary = build_and_load(R"M(
	int main() {
		return 666;
	})M");

	riscv::Machine<RISCV64> machine{ binary, { .memory_max = MAX_MEMORY } };
	// We need to install Linux system calls for maximum gucciness
	machine.setup_linux_syscalls();
	// We need to create a Linux environment for runtimes to work well
	machine.setup_linux(
			{ "vmcall" },
			{ "LC_TYPE=C", "LC_ALL=C", "USER=root" });

	auto origin = machine.memory.mmap_allocate(1);

	static const char hello[] = "hello world!";
	char buffer[13];

	for (auto addr = origin + 4096 - 11; addr < origin + 4095; addr++) {
		machine.memory.memcpy(addr, hello, sizeof(hello));
		auto buf = machine.memory.rvbuffer(addr, 12);
		REQUIRE(buf.is_sequential() == false);
		REQUIRE(buf.size() == 12);
		REQUIRE(buf.to_string() == "hello world!");
		// String view is no longer possible
		buf.copy_to(buffer, 12);
		buffer[12] = 0;
		REQUIRE(std::string(buffer) == "hello world!");
	}
}
