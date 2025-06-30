/**************************************************************************/
/*  resource_loader.cpp                                                   */
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

#include "resource_loader.h"
#include "resource.h"

namespace zylann::godot {

PackedStringArray get_recognized_extensions_for_type(const String &type_name) {
#if defined(ZN_GODOT)
	List<String> extensions_list;
	ResourceLoader::get_recognized_extensions_for_type(type_name, &extensions_list);
	PackedStringArray extensions_array;
	for (const String &extension : extensions_list) {
		extensions_array.push_back(extension);
	}
	return extensions_array;

#elif defined(ZN_GODOT_EXTENSION)
	return ResourceLoader::get_singleton()->get_recognized_extensions_for_type(type_name);
#endif
}

Ref<Resource> load_resource(const String &path) {
#if defined(ZN_GODOT)
	return ResourceLoader::load(path);
#elif defined(ZN_GODOT_EXTENSION)
	return ResourceLoader::get_singleton()->load(path);
#endif
}

} // namespace zylann::godot
