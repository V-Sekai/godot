/**************************************************************************/
/*  type_name.hpp                                                         */
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

#ifndef TYPE_NAME_HPP
#define TYPE_NAME_HPP
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#include <cstddef>
#include <cstring>
#include <stdexcept>

class static_string {
	const char *const p_;
	const std::size_t sz_;

public:
	typedef const char *const_iterator;

	template <std::size_t N>
	constexpr static_string(const char (&a)[N]) noexcept
			:
			p_(a), sz_(N - 1) {}

	constexpr static_string(const char *p, std::size_t N) noexcept
			:
			p_(p), sz_(N) {}

	constexpr const char *data() const noexcept { return p_; }
	constexpr std::size_t size() const noexcept { return sz_; }

	std::string to_string() const { return std::string(p_, sz_); }

	constexpr const_iterator begin() const noexcept { return p_; }
	constexpr const_iterator end() const noexcept { return p_ + sz_; }

	constexpr char operator[](std::size_t n) const {
		return n < sz_ ? p_[n] : throw std::out_of_range("static_string");
	}
};

template <class T>
constexpr inline static_string type_name() {
#ifdef __clang__
	static_string p = __PRETTY_FUNCTION__;
	return static_string(p.data() + 31, p.size() - 31 - 1);
#elif defined(__GNUC__)
	static_string p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
	return static_string(p.data() + 36, p.size() - 36 - 1);
#else
	return static_string(p.data() + 46, p.size() - 46 - 1);
#endif
#elif defined(_MSC_VER)
	static_string p = __FUNCSIG__;
	return static_string(p.data() + 38, p.size() - 38 - 7);
#endif
}

#define TYPE_NAME(x) type_name<decltype(x)>()

#endif // TYPE_NAME_HPP
