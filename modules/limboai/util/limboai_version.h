/**************************************************************************/
/*  limboai_version.h                                                     */
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

#pragma once

/**
 * limboai_version.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBOAI_VERSION_H
#define LIMBOAI_VERSION_H

#include "limboai_version.gen.h"

#ifdef LIMBOAI_MODULE
#include "core/string/ustring.h"
#endif

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/variant/string.hpp>
using namespace godot;
#endif

inline String GET_LIMBOAI_VERSION() {
	String version = itos(LIMBOAI_VERSION_MAJOR) + "." + itos(LIMBOAI_VERSION_MINOR);
	if (LIMBOAI_VERSION_PATCH != 0 || strlen(LIMBOAI_VERSION_STATUS) == 0) {
		version += "." + itos(LIMBOAI_VERSION_PATCH);
	}
	if (strlen(LIMBOAI_VERSION_STATUS) > 0) {
		version += "-" + String(LIMBOAI_VERSION_STATUS);
	}
	return version;
}

#define GET_LIMBOAI_FULL_VERSION() GET_LIMBOAI_VERSION() + " [" + LIMBOAI_VERSION_SHORT_HASH + "]"

#endif // LIMBOAI_VERSION_H
