/**************************************************************************/
/*  godot_runtime_api.cpp                                                 */
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

#include "core/extension/godot_runtime_api.h"

#include "core/extension/godot_instance.h"
#include "main/main.h"

#include "os_windows.h"

static OS_Windows *os = nullptr;

class GodotInstanceLinuxBSD : public GodotInstance {
public:
	bool start() override {
		const bool result = GodotInstance::start();
		os->get_main_loop()->initialize();
		return result;
	}

	void shutdown() override {
		os->get_main_loop()->finalize();
		GodotInstance::shutdown();
	}
};

GDExtensionObjectPtr create_godot_instance(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func) {
	os = new OS_Windows();

	Error err = Main::setup(p_argv[0], p_argc - 1, &p_argv[1], false, p_init_func);
	if (err != OK) {
		return nullptr;
	}

	GodotInstance *godot_instance = memnew(GodotInstanceLinuxBSD);

	return (GDExtensionObjectPtr)godot_instance;
}

void destroy_godot_instance(GDExtensionObjectPtr p_godot_instance) {
	memdelete((GodotInstance *)p_godot_instance);
}
