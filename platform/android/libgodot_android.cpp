/**************************************************************************/
/*  libgodot_android.cpp                                                  */
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

#include <jni.h>

#include "core/extension/godot_instance.h"
#include "core/profiling.h"
#include "libgodot_android.h"
#include "main/main.h"

#include "dir_access_jandroid.h"
#include "file_access_android.h"
#include "file_access_filesystem_jandroid.h"
#include "java_godot_io_wrapper.h"
#include "net_socket_android.h"
#include "os_android.h"
#include "thread_jandroid.h"
#include "java_godot_wrapper.h"
#include "api/java_class_wrapper.h"
#include "plugin/godot_plugin_jni.h"

static OS_Android *os = nullptr;

static GodotInstance *instance = nullptr;

class GodotInstanceCallbacksAndroid : public GodotInstanceCallbacks {
public:
	void focus_out(GodotInstance *p_instance) override {
		os->main_loop_focusout();
	}
	void focus_in(GodotInstance *p_instance) override {
		os->main_loop_focusin();
	}
	void pause(GodotInstance *p_instance) override {
	}
	void resume(GodotInstance *p_instance) override {
	}
};

static GodotInstanceCallbacksAndroid callbacks;

static GodotIOJavaWrapper *godot_io_wrapper = nullptr;
static GodotJavaWrapper *godot_wrapper = nullptr;
static JavaClassWrapper *java_class_wrapper = nullptr;
static jobject class_loader = nullptr;

extern LIBGODOT_API GDExtensionObjectPtr libgodot_create_godot_instance_android(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func, JNIEnv* env, jobject p_asset_manager, jobject p_net_utils, jobject p_directory_access_handler, jobject p_file_access_handler, jobject p_godot_io_wrapper, jobject p_godot_wrapper, jobject p_class_loader) {
	ERR_FAIL_COND_V_MSG(instance != nullptr, nullptr, "Only one Godot Instance may be created.");
	
	godot_init_profiler();

	JavaVM *jvm;
	env->GetJavaVM(&jvm);

	class_loader = env->NewGlobalRef(p_class_loader);
	init_thread_jandroid(jvm, env);
	setup_android_class_loader(class_loader);

	FileAccessAndroid::setup(p_asset_manager);
	DirAccessJAndroid::setup(p_directory_access_handler);
	FileAccessFilesystemJAndroid::setup(p_file_access_handler);
	NetSocketAndroid::setup(p_net_utils);
	godot_io_wrapper = new GodotIOJavaWrapper(env, p_godot_io_wrapper);
	godot_wrapper = new GodotJavaWrapper(env, p_godot_wrapper);

	os = new OS_Android(godot_wrapper, godot_io_wrapper, false);

	Error err = Main::setup(p_argv[0], p_argc - 1, &p_argv[1], false);
	if (err != OK) {
		return nullptr;
	}

	java_class_wrapper = memnew(JavaClassWrapper);

	instance = memnew(GodotInstance);
	if (!instance->initialize(p_init_func, &callbacks)) {
		memdelete(instance);
		instance = nullptr;
		delete godot_io_wrapper;
		delete godot_wrapper;
		memdelete(java_class_wrapper);
		return nullptr;
	}

	return (GDExtensionObjectPtr)instance;
}

extern LIBGODOT_API void libgodot_destroy_godot_instance(GDExtensionObjectPtr p_godot_instance) {
	GodotInstance *godot_instance = (GodotInstance *)p_godot_instance;
	JNIEnv *env = get_jni_env();
	env->DeleteGlobalRef(class_loader);
	if (instance == godot_instance) {
		godot_instance->stop();
		memdelete(godot_instance);
		instance = nullptr;
		if (java_class_wrapper) {
			unregister_plugins_singletons();
        	memdelete(java_class_wrapper);
    	}
		Main::cleanup(true);
		if (godot_io_wrapper) {
        	delete godot_io_wrapper;
    	}
		if (godot_wrapper) {
        	delete godot_wrapper;
    	}
		delete os;
		AudioDriverManager::cleanup();
		os = nullptr;
	}
}
