/**************************************************************************/
/*  version.h                                                             */
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

#if defined(ZN_GODOT)

#include <core/version.h>

// In Godot versions prior to 4.5, version macros were mismatching with GodotCpp.
// We define them so we can use the same macros as in extension builds,
// and I prefer when it's prefixed so it's clear what the macro is referring to.

#ifndef GODOT_VERSION_MAJOR
#define GODOT_VERSION_MAJOR VERSION_MAJOR
#endif

#ifndef GODOT_VERSION_MINOR
#define GODOT_VERSION_MINOR VERSION_MINOR
#endif

#elif defined(ZN_GODOT_EXTENSION)

// Note, in early versions of GodotCpp, this header might not exist
#include <godot_cpp/core/version.hpp>

#if !defined(GODOT_VERSION_MAJOR)

// We are prior to the version of GodotCpp that had version macros, which was during development of Godot 4.2.
// Assume Godot 4.1, though it's not guaranteed.
#define GODOT_VERSION_MAJOR 4
#define GODOT_VERSION_MINOR 1

#endif

#endif // ZN_GODOT_EXTENSION
