/**************************************************************************/
/*  resource_saver_safegdscript.cpp                                       */
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

#include "resource_saver_safegdscript.h"
#include "core/io/file_access.h"
#include "script_safegdscript.h"

static Ref<ResourceFormatSaverSafeGDScript> saver;

void ResourceFormatSaverSafeGDScript::init() {
	saver.instantiate();
	ResourceSaver::add_resource_format_saver(saver, true);
}

void ResourceFormatSaverSafeGDScript::deinit() {
	if (saver.is_valid()) {
		ResourceSaver::remove_resource_format_saver(saver);
		saver.unref();
	}
}

Error ResourceFormatSaverSafeGDScript::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	SafeGDScript *script = Object::cast_to<SafeGDScript>(*p_resource);
	if (script == nullptr) {
		ERR_PRINT("ResourceFormatSaverSafeGDScript::save: Resource is not a SafeGDScript.");
		return ERR_FILE_CANT_WRITE;
	}

	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
	if (file.is_null()) {
		ERR_PRINT("ResourceFormatSaverSafeGDScript::save: Failed to open file for writing: " + p_path);
		return ERR_FILE_CANT_OPEN;
	}

	file->store_string(script->get_source_code());
	return OK;
}

Error ResourceFormatSaverSafeGDScript::set_uid(const String &p_path, ResourceUID::ID p_uid) {
	return OK;
}

bool ResourceFormatSaverSafeGDScript::recognize(const Ref<Resource> &p_resource) const {
	return Object::cast_to<SafeGDScript>(*p_resource) != nullptr;
}

void ResourceFormatSaverSafeGDScript::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<SafeGDScript>(*p_resource) == nullptr) {
		return;
	}
	p_extensions->push_back("safegd");
}

bool ResourceFormatSaverSafeGDScript::recognize_path(const Ref<Resource> &p_resource, const String &p_path) const {
	return Object::cast_to<SafeGDScript>(*p_resource) != nullptr;
}
