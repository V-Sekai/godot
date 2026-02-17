/**************************************************************************/
/*  api.cpp                                                               */
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

#include "api.h"

#include "java_class_wrapper.h"
#include "jni_singleton.h"

#include "core/config/engine.h"
#include "platform/android/rendering_native_surface_android.h"

#if !defined(ANDROID_ENABLED)
static JavaClassWrapper *java_class_wrapper = nullptr;
#endif

#if !defined(ANDROID_ENABLED)
// Dummy RenderingNativeSurfaceAndroid implementation

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
	Ref<RenderingNativeSurfaceAndroid> result;
	return result;
}

RenderingContextDriver *RenderingNativeSurfaceAndroid::create_rendering_context(const String &p_rendering_driver) {
	return nullptr;
}

GLManager *RenderingNativeSurfaceAndroid::create_gl_manager(const String &p_rendering_driver) {
	return nullptr;
}

void *RenderingNativeSurfaceAndroid::get_native_id() const {
	return nullptr;
}


RenderingNativeSurfaceAndroid::RenderingNativeSurfaceAndroid() {
	// Does nothing.
}

RenderingNativeSurfaceAndroid::~RenderingNativeSurfaceAndroid() {
	// Does nothing.
}

#endif

void register_core_android_api() {
	GDREGISTER_ABSTRACT_CLASS(RenderingNativeSurfaceAndroid);
}

void unregister_core_android_api() {
}

void register_android_api() {
#if !defined(ANDROID_ENABLED)
	// On Android platforms, the `java_class_wrapper` instantiation occurs in
	// `platform/android/java_godot_lib_jni.cpp#Java_org_godotengine_godot_GodotLib_setup`
	java_class_wrapper = memnew(JavaClassWrapper);
#endif
	GDREGISTER_CLASS(JNISingleton);
	GDREGISTER_CLASS(JavaClass);
	GDREGISTER_CLASS(JavaObject);
	GDREGISTER_CLASS(JavaClassWrapper);
	Engine::get_singleton()->add_singleton(Engine::Singleton("JavaClassWrapper", JavaClassWrapper::get_singleton()));
}

void unregister_android_api() {
#if !defined(ANDROID_ENABLED)
	if (java_class_wrapper) {
		memdelete(java_class_wrapper);
		java_class_wrapper = nullptr;
	}
#endif
}

void JavaClass::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_java_class_name"), &JavaClass::get_java_class_name);
	ClassDB::bind_method(D_METHOD("get_java_method_list"), &JavaClass::get_java_method_list);
	ClassDB::bind_method(D_METHOD("get_java_parent_class"), &JavaClass::get_java_parent_class);
	ClassDB::bind_method(D_METHOD("has_java_method", "method"), &JavaClass::has_java_method);
}

void JavaObject::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_java_class"), &JavaObject::get_java_class);
	ClassDB::bind_method(D_METHOD("has_java_method", "method"), &JavaObject::has_java_method);
}

void JavaClassWrapper::_bind_methods() {
	ClassDB::bind_method(D_METHOD("wrap", "name"), &JavaClassWrapper::wrap);
	ClassDB::bind_method(D_METHOD("get_exception"), &JavaClassWrapper::get_exception);
}

#if !defined(ANDROID_ENABLED)
bool JavaClass::_get(const StringName &p_name, Variant &r_ret) const {
	return false;
}

Variant JavaClass::callp(const StringName &, const Variant **, int, Callable::CallError &) {
	return Variant();
}

String JavaClass::get_java_class_name() const {
	return "";
}

TypedArray<Dictionary> JavaClass::get_java_method_list() const {
	return TypedArray<Dictionary>();
}

Ref<JavaClass> JavaClass::get_java_parent_class() const {
	return Ref<JavaClass>();
}

bool JavaClass::has_java_method(const StringName &) const {
	return false;
}

JavaClass::JavaClass() {
}

JavaClass::~JavaClass() {
}

Variant JavaObject::callp(const StringName &, const Variant **, int, Callable::CallError &) {
	return Variant();
}

Ref<JavaClass> JavaObject::get_java_class() const {
	return Ref<JavaClass>();
}

bool JavaObject::has_java_method(const StringName &) const {
	return false;
}

JavaClassWrapper *JavaClassWrapper::singleton = nullptr;

Ref<JavaClass> JavaClassWrapper::_wrap(const String &, bool) {
	return Ref<JavaClass>();
}

JavaClassWrapper::JavaClassWrapper() {
	singleton = this;
}

JavaClassWrapper::~JavaClassWrapper() {
	ERR_FAIL_COND(singleton != this);
	singleton = nullptr;
}

#endif
