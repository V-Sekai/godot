/**************************************************************************/
/*  service.cpp                                                           */
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

#include <cassert>
#include <net/interfaces>
#include <os>
#include <smp>

static const uint16_t PORT = 1234;
// avoid endless loops, code that takes too long and excessive memory usage
static const size_t MAX_INSTRUCTIONS = 80 * 1000000;
static const size_t MAX_MEMORY = 1024 * 1024 * 32;

static const std::vector<std::string> env = {
	"LC_CTYPE=C", "LC_ALL=C", "USER=groot"
};
static const std::vector<std::string> args = {
	"hello_world", "test!"
};

#include "server.hpp"
#include <thread>
// RISC-V system call stuff
#include <linux.hpp>
#include <syscalls.hpp>
#include <threads.hpp>
static uint64_t micros_now();

static void multiprocess_task(buffer_t binary) {
	SMP::global_lock();
	printf("CPU %d executing %zu bytes binary\n",
			SMP::cpu_id(), binary.size());
	SMP::global_unlock();

	State<4> state;
	// go-time: create machine, execute code
	const uint64_t t0 = micros_now();
	riscv::Machine<riscv::RISCV32> machine{ binary, MAX_MEMORY };

	prepare_linux<riscv::RISCV32>(machine, args, env);
	setup_linux_syscalls(state, machine);
	setup_multithreading(state, machine);

	try {
		machine.simulate(MAX_INSTRUCTIONS);
	} catch (std::exception &e) {
		printf("Received exception from machine: %s\n", e.what());
	}
	const uint64_t t1 = micros_now();
	const auto instructions = machine.cpu.registers().counter;
	const int64_t micros = t1 - t0;
	// ...
	SMP::add_bsp_task(
			SMP::task_func::make_packed(
					[state, instructions, micros]() {
						if (instructions == MAX_INSTRUCTIONS) {
							printf("WARNING: Maximum instructions reached (%lu)\n", MAX_INSTRUCTIONS);
							printf("--> Program did not complete\n");
						}
						printf("* Executed %lu instructions in %ld micros\n", instructions, micros);
						printf("* Machine output:\n%s\n", state.output.c_str());
					}));
}

void Service::start() {
	// threads will now be migrated to free CPUs
	SMP::migrate_threads();

	auto &inet = net::Interfaces::get(0);
	file_server(inet, PORT,
			[](buffer_t buffer) {
				new std::thread(&multiprocess_task, std::move(buffer));
			});
	printf("Listening on port %u\n", PORT);
}

#include <sys/time.h>
uint64_t micros_now() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000ul + tv.tv_usec;
}
