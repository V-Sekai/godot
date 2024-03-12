/**************************************************************************/
/*  Library.h                                                             */
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

//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LIBRARY_H
#define MATERIALX_LIBRARY_H

/// @file
/// Library-wide includes and types.  This file should be the first include for
/// any public header in the MaterialX library.

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <MaterialXCore/Generated.h>

/// Platform-specific macros for declaring imported and exported symbols.
#if defined(MATERIALX_BUILD_SHARED_LIBS)
    #if defined(_WIN32)
        #pragma warning(disable : 4251)
        #pragma warning(disable : 4275)
        #pragma warning(disable : 4661)
        #define MATERIALX_SYMBOL_EXPORT __declspec(dllexport)
        #define MATERIALX_SYMBOL_IMPORT __declspec(dllimport)
        #define MATERIALX_EXPORT_EXTERN_TEMPLATE(...) template class MATERIALX_SYMBOL_EXPORT __VA_ARGS__
        #define MATERIALX_IMPORT_EXTERN_TEMPLATE(...) extern template class MATERIALX_SYMBOL_IMPORT __VA_ARGS__
    #else
        // Presently non-Windows platforms just export all symbols from
        // shared libraries rather than using the explicit declarations.
        #define MATERIALX_SYMBOL_EXPORT
        #define MATERIALX_SYMBOL_IMPORT
        #define MATERIALX_EXPORT_EXTERN_TEMPLATE(...)
        #define MATERIALX_IMPORT_EXTERN_TEMPLATE(...)
    #endif
#else
    #define MATERIALX_SYMBOL_EXPORT
    #define MATERIALX_SYMBOL_IMPORT
    #define MATERIALX_EXPORT_EXTERN_TEMPLATE(...)
    #define MATERIALX_IMPORT_EXTERN_TEMPLATE(...)
#endif

MATERIALX_NAMESPACE_BEGIN

using std::shared_ptr;
using std::string;
using std::vector;
using std::weak_ptr;

/// A vector of strings.
using StringVec = vector<string>;
/// An unordered map with strings as both keys and values.
using StringMap = std::unordered_map<string, string>;
/// A set of strings.
using StringSet = std::set<string>;

MATERIALX_NAMESPACE_END

#endif
