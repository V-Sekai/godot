/**************************************************************************/
/*  settings.hpp                                                          */
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

#ifndef SETTINGS_HPP
#define SETTINGS_HPP
#include <string>
#include <unordered_set>
template <int W>
static std::vector<riscv::address_type<W>> load_jump_hints(const std::string &filename, bool verbose = false);
template <int W>
static void store_jump_hints(const std::string &filename, const std::vector<riscv::address_type<W>> &hints);

#if defined(EMULATOR_MODE_LINUX)
static constexpr bool full_linux_guest = true;
#else
static constexpr bool full_linux_guest = false;
#endif
#if defined(EMULATOR_MODE_NEWLIB)
static constexpr bool newlib_mini_guest = true;
#else
static constexpr bool newlib_mini_guest = false;
#endif
#if defined(EMULATOR_MODE_MICRO)
static constexpr bool micro_guest = true;
#else
static constexpr bool micro_guest = false;
#endif

#endif // SETTINGS_HPP
