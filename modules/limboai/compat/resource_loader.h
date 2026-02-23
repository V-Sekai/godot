/**************************************************************************/
/*  resource_loader.h                                                     */
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
 * resource_loader.h
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef COMPAT_RESOURCE_LOADER_H
#define COMPAT_RESOURCE_LOADER_H

#ifdef LIMBOAI_MODULE

#include "core/io/resource_loader.h"

#define RESOURCE_LOAD(m_path, m_hint) ResourceLoader::load(m_path, m_hint)
#define RESOURCE_LOAD_NO_CACHE(m_path, m_hint) ResourceLoader::load(m_path, m_hint, ResourceFormatLoader::CACHE_MODE_IGNORE)
#define RESOURCE_EXISTS(m_path, m_type_hint) (ResourceLoader::exists(m_path, m_type_hint))
#define RESOURCE_IS_CACHED(m_path) (ResourceCache::has(m_path))
#define RESOURCE_IS_SCENE_FILE(m_path) (ResourceLoader::get_resource_type(m_path) == "PackedScene")

#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION

#include <godot_cpp/classes/resource_loader.hpp>

#define RESOURCE_LOAD(m_path, m_hint) ResourceLoader::get_singleton()->load(m_path, m_hint)
#define RESOURCE_LOAD_NO_CACHE(m_path, m_hint) ResourceLoader::get_singleton()->load(m_path, m_hint, ResourceLoader::CACHE_MODE_IGNORE)
#define RESOURCE_EXISTS(m_path, m_type_hint) (ResourceLoader::get_singleton()->exists(m_path, m_type_hint))
#define RESOURCE_IS_CACHED(m_path) (ResourceLoader::get_singleton()->has_cached(res_path))
#define RESOURCE_IS_SCENE_FILE(m_path) (ResourceLoader::get_singleton()->get_recognized_extensions_for_type("PackedScene").has(m_path.get_extension()))

#endif // LIMBOAI_GDEXTENSION

#endif // COMPAT_RESOURCE_LOADER_H
