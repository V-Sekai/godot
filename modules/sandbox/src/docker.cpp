/**************************************************************************/
/*  docker.cpp                                                            */
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

#include "docker.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "sandbox_project_settings.h"
//#define ENABLE_TIMINGS 1
#ifdef ENABLE_TIMINGS
#include <time.h>
#endif

static constexpr bool VERBOSE_CMD = true;

static bool ContainerIsAlreadyRunning(String container_name) {
	OS *os = OS::get_singleton();
	List<String> arguments;
	arguments.push_back("container");
	arguments.push_back("inspect");
	arguments.push_back("-f");
	arguments.push_back("{{.State.Running}}");
	arguments.push_back(container_name);
	String pipe;
	int exitcode = 0;
	Error err = os->execute(SandboxProjectSettings::get_docker_path(), arguments, &pipe, &exitcode);
	const int res = (err == OK) ? exitcode : -1;
	Array output;
	if (err == OK) {
		output.push_back(pipe);
	}
	if constexpr (VERBOSE_CMD) {
		PackedStringArray args_array;
		for (const String &arg : arguments) {
			args_array.push_back(arg);
		}
		print_line(SandboxProjectSettings::get_docker_path() + " " + String(", ").join(args_array));
	}
	if (res != 0) {
		return false;
	}
	const String running = output[0];
	return running.contains("true");
}

bool Docker::ContainerPullLatest(String image_name, Array &output) {
	OS *os = OS::get_singleton();
	List<String> arguments;
	arguments.push_back("pull");
	arguments.push_back(image_name);
	String pipe;
	int exitcode = 0;
	Error err = os->execute(SandboxProjectSettings::get_docker_path(), arguments, &pipe, &exitcode);
	const int res = (err == OK) ? exitcode : -1;
	if (err == OK) {
		output.push_back(pipe);
	}
	if constexpr (VERBOSE_CMD) {
		PackedStringArray args_array;
		for (const String &arg : arguments) {
			args_array.push_back(arg);
		}
		print_line(SandboxProjectSettings::get_docker_path() + " " + String(", ").join(args_array));
	}
	return res == 0;
}

String Docker::ContainerGetMountPath(String container_name) {
	OS *os = OS::get_singleton();
	List<String> arguments;
	arguments.push_back("inspect");
	arguments.push_back("-f");
	arguments.push_back("{{ (index .Mounts 0).Source }}");
	arguments.push_back(container_name);
	String pipe;
	int exitcode = 0;
	Error err = os->execute(SandboxProjectSettings::get_docker_path(), arguments, &pipe, &exitcode);
	const int res = (err == OK) ? exitcode : -1;
	if constexpr (VERBOSE_CMD) {
		PackedStringArray args_array;
		for (const String &arg : arguments) {
			args_array.push_back(arg);
		}
		print_line(SandboxProjectSettings::get_docker_path() + " " + String(", ").join(args_array));
	}
	if (res != 0) {
		return "";
	}
	return pipe.replace("\n", "");
}

bool Docker::ContainerStart(String container_name, String image_name, Array &output) {
	if (!SandboxProjectSettings::get_docker_enabled()) {
		return true;
	}
	if (ContainerIsAlreadyRunning(container_name)) {
		ProjectSettings *project_settings = ProjectSettings::get_singleton();
		// If the container mount path does not match the current project path, stop the container.
		String path = ContainerGetMountPath(container_name);
		String project_path = project_settings->globalize_path("res://");
		//printf("Container mount path: %s\n", path.utf8().get_data());
		//printf("Current project path: %s\n", project_path.utf8().get_data());
		if (!path.is_empty() && !project_path.begins_with(path)) {
			print_line("Container mount path (" + path + ") does not match the current project path (" + project_path + "). Stopping the container.");
			Docker::ContainerStop(container_name);
		} else {
			// The container is already running and the mount path matches the current project path.
			print_line("Container " + container_name + " was already running.");
			return true;
		}
	}
	// The container is not running. Try to pull the latest image.
	Array dont_care; // We don't care about the output of the image pull (for now).
	if (ContainerPullLatest(image_name, dont_care)) {
		// Delete the container if it exists. It's not running, but it might be stopped.
		ContainerDelete(container_name, dont_care);
	} else {
		WARN_PRINT("Sandbox: Failed to pull the latest container image: " + image_name);
	}
	// Start the container, even if the image pull failed. It might be locally available.
	OS *os = OS::get_singleton();
	List<String> arguments;
	arguments.push_back("run");
	arguments.push_back("--name");
	arguments.push_back(container_name);
	arguments.push_back("-dv");
	arguments.push_back(".:/usr/src");
	arguments.push_back(image_name);
	String pipe;
	int exitcode = 0;
	Error err = os->execute(SandboxProjectSettings::get_docker_path(), arguments, &pipe, &exitcode);
	const int res = (err == OK) ? exitcode : -1;
	if (err == OK) {
		output.push_back(pipe);
	}
	if constexpr (VERBOSE_CMD) {
		PackedStringArray args_array;
		for (const String &arg : arguments) {
			args_array.push_back(arg);
		}
		print_line(SandboxProjectSettings::get_docker_path() + " " + String(", ").join(args_array));
	}
	return res == 0;
}

Array Docker::ContainerStop(String container_name) {
	if (!SandboxProjectSettings::get_docker_enabled()) {
		return Array();
	}
	OS *os = OS::get_singleton();
	List<String> arguments;
	arguments.push_back("stop");
	arguments.push_back(container_name);
	arguments.push_back("--time");
	arguments.push_back("0");
	String pipe;
	int exitcode = 0;
	Error err = os->execute(SandboxProjectSettings::get_docker_path(), arguments, &pipe, &exitcode);
	Array output;
	if (err == OK) {
		output.push_back(pipe);
	}
	if constexpr (VERBOSE_CMD) {
		PackedStringArray args_array;
		for (const String &arg : arguments) {
			args_array.push_back(arg);
		}
		print_line(SandboxProjectSettings::get_docker_path() + " " + String(", ").join(args_array));
	}
	return output;
}

bool Docker::ContainerExecute(String container_name, const PackedStringArray &p_arguments, Array &output, bool verbose) {
	if (!SandboxProjectSettings::get_docker_enabled()) {
		return false;
	}
#ifdef ENABLE_TIMINGS
	timespec start;
	clock_gettime(CLOCK_MONOTONIC, &start);
#endif

	OS *os = OS::get_singleton();
	List<String> arguments;
	arguments.push_back("exec");
	arguments.push_back("-t");
	arguments.push_back(container_name);
	arguments.push_back("bash");
	for (int i = 0; i < p_arguments.size(); i++) {
		arguments.push_back(p_arguments[i]);
	}
	String pipe;
	int exitcode = 0;
	Error err = os->execute(SandboxProjectSettings::get_docker_path(), arguments, &pipe, &exitcode);
	const int res = (err == OK) ? exitcode : -1;
	if (err == OK) {
		output.push_back(pipe);
	}
	if (VERBOSE_CMD && verbose) {
		PackedStringArray args_array;
		for (const String &arg : arguments) {
			args_array.push_back(arg);
		}
		print_line(SandboxProjectSettings::get_docker_path() + " " + String(", ").join(args_array));
	}

#ifdef ENABLE_TIMINGS
	timespec end;
	clock_gettime(CLOCK_MONOTONIC, &end);
	const double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
	fprintf(stderr, "Docker::ContainerExecute: %f seconds\n", elapsed);
#endif

	return res == 0;
}

int Docker::ContainerVersion(String container_name, const PackedStringArray &p_arguments) {
	// Execute --version in the container.
	Array output;
	if (ContainerExecute(container_name, p_arguments, output)) {
		// Docker container responds with a number, eg "1" (ASCII)
		return String(output[0]).to_int();
	}
	return -1;
}

bool Docker::ContainerDelete(String container_name, Array &output) {
	OS *os = OS::get_singleton();
	List<String> arguments;
	arguments.push_back("rm");
	arguments.push_back(container_name);
	String pipe;
	int exitcode = 0;
	Error err = os->execute(SandboxProjectSettings::get_docker_path(), arguments, &pipe, &exitcode);
	const int res = (err == OK) ? exitcode : -1;
	if (err == OK) {
		output.push_back(pipe);
	}
	if constexpr (VERBOSE_CMD) {
		PackedStringArray args_array;
		for (const String &arg : arguments) {
			args_array.push_back(arg);
		}
		print_line(SandboxProjectSettings::get_docker_path() + " " + String(", ").join(args_array));
	}
	return res == 0;
}
