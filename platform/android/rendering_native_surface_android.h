/**************************************************************************/
/*  rendering_native_surface_android.h                                    */
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

#ifndef RENDERING_NATIVE_SURFACE_ANDROID_H
#define RENDERING_NATIVE_SURFACE_ANDROID_H

#include "core/variant/native_ptr.h"
#include "servers/rendering/rendering_native_surface.h"

struct ANativeWindow;

class RenderingNativeSurfaceAndroid : public RenderingNativeSurface {
	GDCLASS(RenderingNativeSurfaceAndroid, RenderingNativeSurface);

	static void _bind_methods();

	ANativeWindow *window;
	uint32_t width;
	uint32_t height;

public:
	static Ref<RenderingNativeSurfaceAndroid> create_api(uint64_t p_window, uint32_t p_width, uint32_t p_height);

	static Ref<RenderingNativeSurfaceAndroid> create(ANativeWindow *p_window, uint32_t p_width, uint32_t p_height);

	ANativeWindow *get_window() const {
		return window;
	}

	uint64_t get_window_api() const {
		return (uint64_t)window;
	}

	uint32_t get_width() const {
		return width;
	}

	uint32_t get_height() const {
		return height;
	}

	RenderingContextDriver *create_rendering_context(const String &p_driver_name) override;
	GLManager *create_gl_manager(const String &p_driver_name) override;

	void *get_native_id() const override;

	RenderingNativeSurfaceAndroid();
	~RenderingNativeSurfaceAndroid();
};

#endif // RENDERING_NATIVE_SURFACE_ANDROID_H
