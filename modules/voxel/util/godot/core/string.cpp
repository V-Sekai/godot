/**************************************************************************/
/*  string.cpp                                                            */
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

#include "string.h"
#include <sstream>

namespace zylann::godot {

#ifdef TOOLS_ENABLED

PackedStringArray to_godot(const StdVector<std::string_view> &svv) {
	PackedStringArray psa;
	// Not resizing up-front, because in Godot core writing elements uses different code than GDExtension.
	for (unsigned int i = 0; i < svv.size(); ++i) {
		psa.append(to_godot(svv[i]));
	}
	return psa;
}

PackedStringArray to_godot(const StdVector<StdString> &sv) {
	PackedStringArray psa;
	// Not resizing up-front, because in Godot core writing elements uses different code than GDExtension.
	for (unsigned int i = 0; i < sv.size(); ++i) {
		psa.append(to_godot(sv[i]));
	}
	return psa;
}

#endif

} // namespace zylann::godot

ZN_GODOT_NAMESPACE_BEGIN

zylann::StdStringStream &operator<<(zylann::StdStringStream &ss, GodotStringWrapper s) {
	const CharString cs = s.s.utf8();
	// String has non-explicit constructors from various types making this ambiguous
	const char *ca = cs.get_data();
	ss << ca;
	return ss;
}

ZN_GODOT_NAMESPACE_END
