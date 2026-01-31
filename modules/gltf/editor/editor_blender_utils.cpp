/**************************************************************************/
/*  editor_blender_utils.cpp                                              */
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

#include "editor_blender_utils.h"

#include "core/io/file_access.h"
#include "core/os/os.h"
#include "core/string/ustring.h"

bool EditorBlenderUtils::get_blender_version(const String &p_path, int &r_major, int &r_minor, String *r_err) {
	if (!FileAccess::exists(p_path)) {
		if (r_err) {
			*r_err = TTR("Path does not point to a valid executable.");
		}
		return false;
	}
	List<String> args;
	args.push_back("--version");
	String pipe;
	Error err = OS::get_singleton()->execute(p_path, args, &pipe);
	if (err != OK) {
		if (r_err) {
			*r_err = TTR("Couldn't run Blender executable.");
		}
		return false;
	}
	int bl = pipe.find("Blender ");
	if (bl == -1) {
		if (r_err) {
			*r_err = vformat(TTR("Unexpected --version output from Blender executable at: %s."), p_path);
		}
		return false;
	}
	pipe = pipe.substr(bl);
	pipe = pipe.replace_first("Blender ", "");
	int pp = pipe.find_char('.');
	if (pp == -1) {
		if (r_err) {
			*r_err = vformat(TTR("Couldn't extract version information from Blender executable at: %s."), p_path);
		}
		return false;
	}
	String v = pipe.substr(0, pp);
	r_major = v.to_int();
	if (r_major < 3) {
		if (r_err) {
			*r_err = vformat(TTR("Found Blender version %d.x, which is too old for this importer (3.0+ is required)."), r_major);
		}
		return false;
	}

	int pp2 = pipe.find_char('.', pp + 1);
	r_minor = pp2 > pp ? pipe.substr(pp + 1, pp2 - pp - 1).to_int() : 0;

	return true;
}

String EditorBlenderUtils::fix_windows_network_share_path(const String &p_path) {
#ifdef WINDOWS_ENABLED
	// On Windows, when using a network share path, the above will return a path starting with "//"
	// which once handed to Blender will be treated like a relative path. So we need to replace the
	// first two characters with "\\" to make it absolute again.
	if (p_path.is_network_share_path()) {
		return "\\\\" + p_path.substr(2);
	}
#endif
	return p_path;
}

bool EditorBlenderUtils::validate_blender_path(const String &p_path, String *r_err) {
	if (p_path.is_empty()) {
		if (r_err) {
			*r_err = TTR("Blender path is empty, check your Editor Settings.");
		}
		return false;
	}
	if (!FileAccess::exists(p_path)) {
		if (r_err) {
			*r_err = vformat(TTR("Invalid Blender path: %s, check your Editor Settings."), p_path);
		}
		return false;
	}
	int major, minor;
	if (!get_blender_version(p_path, major, minor, r_err)) {
		return false;
	}
	return true;
}
