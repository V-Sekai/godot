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

#import "rendering_context_driver_vulkan_macos.h"
#include "rendering_native_surface_android.h"
#include <android/native_window.h>

#if defined(VULKAN_ENABLED)
#include "rendering_context_driver_vulkan_android.h"
#endif

#if defined(GLES3_ENABLED)
#include <vector>
#include "servers/rendering/gl_manager.h"

#include <android/native_window.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>

struct WindowData {
	GLint backingWidth = 0;
	GLint backingHeight = 0;
	GLuint viewRenderbuffer = 0;
	GLuint viewFramebuffer = 0;
	GLuint depthRenderbuffer = 0;

	EGLSurface surface = EGL_NO_SURFACE;
	ANativeWindow *window = nullptr;
};

class GLManagerAndroid : public GLManager {
public:
	virtual Error initialize(void *p_native_display = nullptr) override;
	virtual Error open_display(void *p_native_display = nullptr) override { return OK; }
	virtual Error window_create(DisplayServer::WindowID p_id, Ref<RenderingNativeSurface> p_native_surface, int p_width, int p_height) override;
	virtual void window_resize(DisplayServer::WindowID p_id, int p_width, int p_height) override;
	virtual void window_make_current(DisplayServer::WindowID p_id) override;
	virtual void release_current() override {}
	virtual void swap_buffers() override;
	virtual void window_destroy(DisplayServer::WindowID p_id) override;
	virtual Size2i window_get_size(DisplayServer::WindowID p_id) override;
	void deinitialize();
	virtual int window_get_render_target(DisplayServer::WindowID p_id) const override;
	virtual int window_get_color_texture(DisplayServer::WindowID p_id) const override;

	virtual void set_use_vsync(bool p_use) override {}
	virtual bool is_using_vsync() const override { return false; }

	~GLManagerAndroid() {
		deinitialize();
	}

protected:
	bool create_framebuffer(DisplayServer::WindowID p_id, void *p_layer);

private:
	HashMap<DisplayServer::WindowID, WindowData> windows;

	EGLDisplay display = EGL_NO_DISPLAY;
	EGLContext context = EGL_NO_CONTEXT;
	EGLConfig config;
};

typedef struct EGLConfigRequirements {
	EGLint red_size;
	EGLint green_size;
	EGLint blue_size;
	EGLint alpha_size;
	EGLint depth_size;
	EGLint stencil_size;
} EGLConfigRequirements;


const EGLConfigRequirements config_reqs[] = {
	{
        .red_size = 8,
        .green_size = 8,
        .blue_size = 8,
		.alpha_size = 8,
		.depth_size = 24,
		.stencil_size = 0
	},
	{
        .red_size = 8,
        .green_size = 8,
        .blue_size = 8,
		.alpha_size = 8,
		.depth_size = 16,
		.stencil_size = 0
	},
};

EGLint config_attribs[] = {
	EGL_RED_SIZE, 4,
	EGL_GREEN_SIZE, 4,
	EGL_BLUE_SIZE, 4,
	EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
	EGL_NONE
};

static EGLint findConfigAttrib(EGLDisplay p_display, EGLConfig p_config, int p_attribute, int p_default_value) {
	EGLint value;
	if (eglGetConfigAttrib(p_display, p_config, p_attribute, &value)) {
		return value;
	}
	return p_default_value;
}

static int findMatchingConfig(EGLDisplay p_display, EGLConfigRequirements req, std::vector<EGLConfig> configs) {
	int result = -1;
	for (EGLConfig cfg : configs) {
		result ++;
		
		EGLint d = findConfigAttrib(p_display, cfg, EGL_DEPTH_SIZE, 0);
		EGLint s = findConfigAttrib(p_display, cfg, EGL_STENCIL_SIZE, 0);

		if (d < req.depth_size || s < req.stencil_size) {
			continue;
		}

		EGLint r = findConfigAttrib(p_display, cfg, EGL_RED_SIZE, 0);
		EGLint g = findConfigAttrib(p_display, cfg, EGL_GREEN_SIZE, 0);
		EGLint b = findConfigAttrib(p_display, cfg, EGL_BLUE_SIZE, 0);
		EGLint a = findConfigAttrib(p_display, cfg, EGL_ALPHA_SIZE, 0);

		if (r == req.red_size && g == req.green_size && b == req.blue_size && a == req.alpha_size) {
			return result;
		}
	}
	return -1;
}

Error GLManagerAndroid::initialize(void *p_native_display) {
	// Create GL ES 3 context
	if (OS::get_singleton()->get_current_rendering_method() == "gl_compatibility" && context == nullptr) {
		display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
		ERR_FAIL_COND_V(display == EGL_NO_DISPLAY, FAILED);
		
		if (!eglInitialize(display, 0, 0)) {
			int error = eglGetError();
			deinitialize();
			ERR_FAIL_V_MSG(FAILED, vformat("Unable to initialize EGL Display: %d", error));
		}

		{
			EGLint numConfigs;
			if (!eglChooseConfig(display, config_attribs, nullptr, 0, &numConfigs)) {
				int error = eglGetError();
				deinitialize();
        		ERR_FAIL_V_MSG(FAILED, vformat("Unable to retrieve the count of matching configs: %d", error));
		    }
			if (numConfigs < 1) {
				deinitialize();
				ERR_FAIL_V_MSG(FAILED, "No matching configs retrieved");
			}
			std::vector<EGLConfig> configs(numConfigs);
			if (!eglChooseConfig(display, config_attribs, configs.data(), numConfigs, &numConfigs)) {
				int error = eglGetError();
				deinitialize();
        		ERR_FAIL_V_MSG(FAILED, vformat("Unable to retrieve the matching configs: %d", error));
		    }

			int matchingConfig = -1;
			for (EGLConfigRequirements req : config_reqs) {
				matchingConfig = findMatchingConfig(display, req, configs);
				if (matchingConfig >= 0) {
					break;
				}
			}
			if (matchingConfig < 0) {
				deinitialize();
				ERR_FAIL_V_MSG(FAILED, "Unable to find a matching configuration.");
			}
			config = configs[matchingConfig];
		}

		context = eglCreateContext(display, config, 0, 0);
		if (!context) {
			int error = eglGetError();
			deinitialize();
			ERR_FAIL_V_MSG(FAILED, vformat("Unable to create EGL Context: %d", error));
		}
	}

	return OK;
}

void GLManagerAndroid::window_resize(DisplayServer::WindowID p_id, int p_width, int p_height) {
	ERR_FAIL_COND(!windows.has(p_id));
	WindowData &gles_data = windows[p_id];

	if (!eglMakeCurrent(display, gles_data.surface, gles_data.surface, context)) {
		ERR_FAIL_MSG(vformat("eglMakeCurrent() returned error %d", eglGetError()));
	}

	window_destroy(p_id);
	window_create(p_id, (void *) gles_data.window, p_width, p_height);
}

Size2i GLESContextAndroid::window_get_size(DisplayServer::WindowID p_id) {
	ERR_FAIL_COND_V(!windows.has(p_id), Size2i());
	WindowData &gles_data = windows[p_id];
	return Size2i();
}

void GLManagerAndroid::window_make_current(DisplayServer::WindowID p_id) {
	ERR_FAIL_COND(!windows.has(p_id));
	WindowData &gles_data = windows[p_id];
	if (!eglMakeCurrent(display, gles_data.surface, gles_data.surface, context)) {
		ERR_FAIL_MSG(vformat("eglMakeCurrent() returned error %d", eglGetError()));
	}
	glBindFramebuffer(GL_FRAMEBUFFER, gles_data.viewFramebuffer);
	current_window = p_id;
}

void GLManagerAndroid::swap_buffers() {
	ERR_FAIL_COND(!windows.has(current_window));
	WindowData &gles_data = windows[current_window];
	if (!eglMakeCurrent(display, gles_data.surface, gles_data.surface, context)) {
		ERR_FAIL_MSG(vformat("eglMakeCurrent() returned error %d", eglGetError()));
	}
	glBindRenderbuffer(GL_RENDERBUFFER, gles_data.viewRenderbuffer);
	eglSwapBuffers(display, gles_data.surface);
}

void GLESContextAndroid::deinitialize() {
	if (display) {
	    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		if (context) {
			eglDestroyContext(display, context);
		}
		for (KeyValue<DisplayServer::WindowID, WindowData> kv : windows) {
			WindowData& w = kv.value;
			eglDestroySurface(display, w.surface);
		}
		windows.clear();
       	eglTerminate(display);
	}
    
    display = EGL_NO_DISPLAY;
    context = EGL_NO_CONTEXT;
}

Error GLManagerAndroid::window_create(DisplayServer::WindowID p_id, Ref<RenderingNativeSurface> p_native_surface, int p_width, int p_height) {
	Ref<RenderingNativeSurfaceAndroid> android_surface = Object::cast_to<RenderingNativeSurfaceAndroid>(*p_native_surface);
	if (create_framebuffer(p_id, (void *)android_surface->get_window())) {
		return OK;
	} else {
		return FAILED;
	}
}

bool GLESContextAndroid::create_framebuffer(DisplayServer::WindowID p_id, void *p_layer) {
	WindowData &gles_data = windows[p_id];
	gles_data.window = (ANativeWindow *) p_layer;

	gles_data.surface = eglCreateWindowSurface(display, config, gles_data.window, 0);
	ERR_FAIL_COND_V_MSG(!gles_data.surface, false, vformat("eglCreateWindowSurface() returned error %d", eglGetError()));

	if (!eglMakeCurrent(display, gles_data.surface, gles_data.surface, context)) {
		ERR_FAIL_V_MSG(false, vformat("eglMakeCurrent() returned error %d", eglGetError()));
	}

	glGenFramebuffers(1, &gles_data.viewFramebuffer);
	glGenRenderbuffers(1, &gles_data.viewRenderbuffer);

	glBindFramebuffer(GL_FRAMEBUFFER, gles_data.viewFramebuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, gles_data.viewRenderbuffer);
	
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, gles_data.viewRenderbuffer);

	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &gles_data.backingWidth);
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &gles_data.backingHeight);

	// For this sample, we also need a depth buffer, so we'll create and attach one via another renderbuffer.
	glGenRenderbuffers(1, &gles_data.depthRenderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, gles_data.depthRenderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, gles_data.backingWidth, gles_data.backingHeight);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gles_data.depthRenderbuffer);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		ERR_FAIL_V_MSG(false, vformat("Failed to make complete framebuffer object %x", glCheckFramebufferStatus(GL_FRAMEBUFFER)));
	}

	return true;
}

// Clean up any buffers we have allocated.
void GLManagerAndroid::window_destroy(DisplayServer::WindowID p_id) {
	ERR_FAIL_COND(!windows.has(p_id));
	WindowData &gles_data = windows[p_id];

	if (!eglMakeCurrent(display, gles_data.surface, gles_data.surface, context)) {
		ERR_FAIL_MSG(vformat("eglMakeCurrent() returned error %d", eglGetError()));
	}

	glDeleteFramebuffers(1, &gles_data.viewFramebuffer);
	gles_data.viewFramebuffer = 0;
	glDeleteRenderbuffers(1, &gles_data.viewRenderbuffer);
	gles_data.viewRenderbuffer = 0;

	if (gles_data.depthRenderbuffer) {
		glDeleteRenderbuffers(1, &gles_data.depthRenderbuffer);
		gles_data.depthRenderbuffer = 0;
	}

	windows.erase(p_id);
}

int GLManagerAndroid::window_get_render_target(DisplayServer::WindowID p_id) const {
	ERR_FAIL_COND_V(!windows.has(p_id), 0);
	const WindowData &gles_data = windows[p_id];
	return gles_data.viewFramebuffer;
}

int GLManagerAndroid::window_get_color_texture(DisplayServer::WindowID p_id) const {
	return -1;
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

void *RenderingNativeSurfaceAndroid::get_native_id() const {
	return (void *)window;
}

RenderingContextDriver *RenderingNativeSurfaceAndroid::create_rendering_context(const String &p_driver_name) {
#if defined(VULKAN_ENABLED)
	if (p_driver_name == "vulkan") {
		return memnew(RenderingContextDriverVulkanAndroid);
	}
#endif
	return nullptr;
}

GLManager *RenderingNativeSurfaceAndroid::create_gl_manager(const String &p_driver_name) {
#if defined(GLES3_ENABLED)
	return memnew(GLManagedAndroid);
#else
	return nullptr;
#endif
}

RenderingNativeSurfaceAndroid::RenderingNativeSurfaceAndroid() {
	// Does nothing.
}

RenderingNativeSurfaceAndroid::~RenderingNativeSurfaceAndroid() {
	// Does nothing.
}
