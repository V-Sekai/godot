/**************************************************************************/
/*  rendering_native_surface_android.cpp                                  */
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

#include "rendering_native_surface_android.h"
#include <android/native_window.h>

#include "modules/regex/regex.h"

#if defined(VULKAN_ENABLED)
#include "rendering_context_driver_vulkan_android.h"
#endif

#if defined(GLES3_ENABLED)
#include <vector>
#include "servers/rendering/gl_manager.h"
#include "servers/rendering/rendering_server.h"
#include "drivers/egl/gl_manager_embedded_angle.h"
#include <android/native_window.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <dlfcn.h>

#define GL_ERR(expr) { expr; GLenum err = glGetError(); if (err) { print_line(vformat("%s:%s: %x error", __FUNCTION__, #expr, err)); } }

#ifndef EGL_KHR_platform_android
#define EGL_KHR_platform_android 1
#define EGL_PLATFORM_ANDROID_KHR          0x3141
#endif /* EGL_KHR_platform_android */

struct WindowData {
	EGLSurface surface = EGL_NO_SURFACE;
	ANativeWindow *window = nullptr;
	uint32_t width;
	uint32_t height;
};

class GLManagerAndroid : public EGLManager {
private:
	virtual const char *_get_platform_extension_name() const override;
	virtual EGLenum _get_platform_extension_enum() const override;
	virtual EGLenum _get_platform_api_enum() const override;
	virtual Vector<EGLAttrib> _get_platform_display_attributes() const override;
	virtual Vector<EGLint> _get_platform_context_attribs() const override;

public:
	void window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) override {}
	bool validate_driver() const override;

	GLManagerAndroid() {}
	~GLManagerAndroid() {}
};

const char *GLManagerAndroid::_get_platform_extension_name() const {
	return "EGL_KHR_platform_android";
}

EGLenum GLManagerAndroid::_get_platform_extension_enum() const {
	return EGL_PLATFORM_ANDROID_KHR;
}

Vector<EGLAttrib> GLManagerAndroid::_get_platform_display_attributes() const {
	Vector<EGLAttrib> ret;
	return ret;
}

EGLenum GLManagerAndroid::_get_platform_api_enum() const {
	return EGL_OPENGL_ES_API;
}

Vector<EGLint> GLManagerAndroid::_get_platform_context_attribs() const {
	Vector<EGLint> ret;
	ret.push_back(EGL_CONTEXT_CLIENT_VERSION);
	ret.push_back(3);
	ret.push_back(EGL_NONE);

	return ret;
}

#endif // GLES3_ENABLED



void RenderingNativeSurfaceAndroid::_bind_methods() {
	ClassDB::bind_static_method("RenderingNativeSurfaceAndroid", D_METHOD("create", "window", "width", "height"), &RenderingNativeSurfaceAndroid::create_api);
	ClassDB::bind_method(D_METHOD("get_window"), &RenderingNativeSurfaceAndroid::get_window_api);
	ClassDB::bind_method(D_METHOD("get_width"), &RenderingNativeSurfaceAndroid::get_width);
	ClassDB::bind_method(D_METHOD("get_height"), &RenderingNativeSurfaceAndroid::get_height);
}

Ref<RenderingNativeSurfaceAndroid> RenderingNativeSurfaceAndroid::create_api(uint64_t p_window, uint32_t p_width, uint32_t p_height) {
	return RenderingNativeSurfaceAndroid::create((ANativeWindow *)p_window, p_width, p_height);
}

Ref<RenderingNativeSurfaceAndroid> RenderingNativeSurfaceAndroid::create(ANativeWindow *p_window, uint32_t p_width, uint32_t p_height) {
	Ref<RenderingNativeSurfaceAndroid> result = memnew(RenderingNativeSurfaceAndroid);
	result->window = p_window;
	result->width = p_width;
	result->height = p_height;
	return result;
}

RenderingContextDriver *RenderingNativeSurfaceAndroid::create_rendering_context(const String &p_driver_name) {
#if defined(VULKAN_ENABLED)
	if (p_driver_name == "vulkan") {
		return memnew(RenderingContextDriverVulkanAndroid);
	}
#endif
	return nullptr;
}

bool GLManagerAndroid::validate_driver() const {
	void *handle = dlopen("libGLESv3.so", RTLD_LOCAL);
	if (handle == nullptr) {
		CRASH_NOW_MSG("Unable to open libGLESv3.so");
	}
	PFNGLGETSTRINGPROC getStringProc = (PFNGLGETSTRINGPROC) dlsym(handle, "glGetString");
	ERR_FAIL_COND_V_MSG(getStringProc == nullptr, false, "Unable to load glGetString symbol");

	const String rendering_device_name = String::utf8((const char *)getStringProc(GL_RENDERER));
	const String rendering_device_vendor = String::utf8((const char *)getStringProc(GL_VENDOR));
	print_line(vformat("Device name: %s", rendering_device_name));
	print_line(vformat("Vendor: %s", rendering_device_vendor));
	dlclose(handle);
	if (rendering_device_name.contains("PowerVR") || rendering_device_vendor.contains("Imagination")) {
		print_line("Detected Imagination GPU");
	}
	return true;
}


GLManager *RenderingNativeSurfaceAndroid::create_gl_manager(const String &p_driver_name) {
#if defined(GLES3_ENABLED)
	if (p_driver_name == "opengl3") {
		#ifdef GLAD_ENABLED
			static const char *EGL_NAMES[] = {"libEGL.so"};
			static const char *GL_NAMES[] = {"libGLESv3.so"};
			gladSetupEGL(1, EGL_NAMES);
			gladSetupGLES2(1, GL_NAMES);
        #endif
        return memnew(GLManagerAndroid);
	}
	#if defined(ANGLE_ENABLED)
	if (p_driver_name == "opengl3_angle") {
		setenv("ANGLE_FEATURE_OVERRIDES_DISABLED", "supportsSwapchainMaintenance1", 1);
		#ifdef GLAD_ENABLED
			static const char *EGL_NAMES[] = {"libEGL_angle.so"};
			static const char *GL_NAMES[] = {"libGLESv2_angle.so"};
			gladSetupEGL(1, EGL_NAMES);
			gladSetupGLES2(1, GL_NAMES);
		#endif
		return memnew(GLManagerANGLE_Embedded);
	}
	#endif
#endif
	return nullptr;
}

void *RenderingNativeSurfaceAndroid::get_native_id() const {
	return (void *)window;
}


RenderingNativeSurfaceAndroid::RenderingNativeSurfaceAndroid() {
	// Does nothing.
}

RenderingNativeSurfaceAndroid::~RenderingNativeSurfaceAndroid() {
	// Does nothing.
}
