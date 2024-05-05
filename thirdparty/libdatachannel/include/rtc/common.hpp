/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_COMMON_H
#define RTC_COMMON_H

#ifdef RTC_STATIC
#define RTC_CPP_EXPORT
#else // dynamic library
#ifdef _WIN32
#ifdef RTC_EXPORTS
#define RTC_CPP_EXPORT __declspec(dllexport) // building the library
#else
#define RTC_CPP_EXPORT __declspec(dllimport) // using the library
#endif
#else // not WIN32
#define RTC_CPP_EXPORT
#endif
#endif

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602 // Windows 8
#endif
#ifdef _MSC_VER
#pragma warning(disable : 4251) // disable "X needs to have dll-interface..."
#endif
#endif

#ifndef RTC_ENABLE_WEBSOCKET
#define RTC_ENABLE_WEBSOCKET 1
#endif

#ifndef RTC_ENABLE_MEDIA
#define RTC_ENABLE_MEDIA 1
#endif

#include "rtc.h" // for C API defines

#include "utils.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <tuple>

namespace rtc {

#ifdef RTC_USE_CPP_EXCEPTIONS

#define RTC_WHAT what
#define RTC_EXCEPTION std::exception
#define RTC_RUNTIME_ERROR std::runtime_error
#define RTC_LOGIC_ERROR std::logic_error
#define RTC_INVALID_ARGUMENT std::invalid_argument
#define RTC_OUT_OF_RANGE std::out_of_range

#define RTC_THROW throw
#define RTC_THROW_WITHIN(exception) throw exception
#define RTC_BEGIN
#define RTC_TRY try
#define RTC_CATCH catch
#define RTC_WRAPPED(T) T
#define RTC_VOID
#define RTC_RET return

#define RTC_UNWRAP_EXPR(func) (func)
#define RTC_CHECK_EXCEPTION
#define RTC_RETHROW throw

#define RTC_UNWRAP_CATCH(func) (func)
#define RTC_UNWRAP_RETHROW(func) (func)
#define RTC_UNWRAP_CATCH_DECL(typ, var, func) typ var = (func)
#define RTC_UNWRAP_RETHROW_DECL(typ, var, func) typ var = (func)
#define RTC_UNWRAP_CATCH_VAR(var, func) var = (func)
#define RTC_UNWRAP_RETHROW_VAR(var, func) var = (func)
#define RTC_UNWRAP_CATCH_ARG(wrap, func) wrap((func))
#define RTC_UNWRAP_RETHROW_ARG(wrap, func) wrap((func))
#define RTC_COMMA ,

#else






#define RTC_WHAT c_str
#define RTC_EXCEPTION std::string
#define RTC_RUNTIME_ERROR std::string
#define RTC_LOGIC_ERROR std::string
#define RTC_INVALID_ARGUMENT std::string
#define RTC_OUT_OF_RANGE std::string

#ifdef _MSC_VER
#pragma warning(error : 4715)
#pragma warning(error : 4716)
#pragma warning(error : 4834)
#endif

class Void {};

class [[nodiscard]] ExceptionCast {
	RTC_EXCEPTION &&rtc_exception;

public:
	ExceptionCast(RTC_EXCEPTION &&p_rtc_exception) :
			rtc_exception(std::move(p_rtc_exception)) {
	}
	RTC_EXCEPTION &&exception() {
		return std::move(rtc_exception);
	}
};
template <class T>
struct [[nodiscard]] WrappedResult {
	using ExT = RTC_EXCEPTION;
	T rtc_value;
	ExT rtc_exception;
	bool rtc_is_exception;

	WrappedResult(T &&value) :
			rtc_value(std::move(value)), rtc_exception(ExT()), rtc_is_exception(false) {
	}
	WrappedResult(const T &value) :
			rtc_value(value), rtc_exception(ExT()), rtc_is_exception(false) {
	}
	WrappedResult(ExceptionCast exception_cast) :
			rtc_value(T()), rtc_exception(std::move(exception_cast.exception())), rtc_is_exception(true) {
	}
};

template<>
struct [[nodiscard]] WrappedResult<void> {
	using ExT = RTC_EXCEPTION;
	Void rtc_value;
	ExT rtc_exception;
	bool rtc_is_exception;

	WrappedResult() :
			rtc_value(Void()), rtc_exception(ExT()), rtc_is_exception(false) {
	}
	WrappedResult(Void) :
			rtc_value(Void()), rtc_exception(ExT()), rtc_is_exception(false) {
	}
	WrappedResult(ExceptionCast exception_cast) :
			rtc_value(Void()), rtc_exception(std::move(exception_cast.exception())), rtc_is_exception(true) {
	}
};

#define RTC_WRAPPED_DEFAULT_CONSTRUCTABLE(clsname) friend struct WrappedResult<clsname>
#define RTC_THROW return (::rtc::ExceptionCast)
#define RTC_TRY RTC_EXCEPTION e; bool rtcexc_was_thrown;
#define RTC_BEGIN RTC_EXCEPTION e; bool rtcexc_was_thrown
#define RTC_CATCH(extype) while (false) rtcexc_catch_label: 
#define RTC_WRAPPED(T) ::rtc::WrappedResult<T>
#define RTC_THROW_WITHIN(exception) while ((e = (exception)), (rtcexc_was_thrown = true)) goto rtcexc_catch_label
#define RTC_VOID ::rtc::WrappedResult<void>()
#define RTC_RET return RTC_VOID

#define RTC_UNWRAP_EXPR(func) ([&](){ \
		auto _rtcexc_tmp = (func); \
		e = _rtcexc_tmp.rtc_exception; \
		rtcexc_was_thrown = _rtcexc_tmp.rtc_is_exception || rtcexc_was_thrown; \
		return _rtcexc_tmp.rtc_value; \
})()
#define RTC_CATCH_EXCEPTION while (rtcexc_was_thrown) goto rtcexc_catch_label
#define RTC_RETHROW while (rtcexc_was_thrown) return (::rtc::ExceptionCast)std::move(e)

#define RTC_UNWRAP_CATCH(func) do { RTC_UNWRAP_EXPR(func); RTC_CATCH_EXCEPTION; } while (0)
#define RTC_UNWRAP_RETHROW(func) do { RTC_UNWRAP_EXPR(func); RTC_RETHROW; } while (0)
#define RTC_UNWRAP_CATCH_ARG(wrap, func) do { wrap(RTC_UNWRAP_EXPR(func)); RTC_CATCH_EXCEPTION; } while (0)
#define RTC_UNWRAP_RETHROW_ARG(wrap, func) do { wrap(RTC_UNWRAP_EXPR(func)); RTC_RETHROW; } while (0)
// Note: in a single-line if statement, this would fall out.
// This should only be used for variable declarations which cannot occur in a single line.
#define RTC_UNWRAP_CATCH_DECL(typ, var, func) typ var = RTC_UNWRAP_EXPR(func); RTC_CATCH_EXCEPTION;
#define RTC_UNWRAP_RETHROW_DECL(typ, var, func) typ var = RTC_UNWRAP_EXPR(func); RTC_RETHROW;
#define RTC_UNWRAP_CATCH_VAR(var, func) do { var = RTC_UNWRAP_EXPR(func); RTC_CATCH_EXCEPTION; } while (0)
#define RTC_UNWRAP_RETHROW_VAR(var, func) do { var = RTC_UNWRAP_EXPR(func); RTC_RETHROW; } while (0)
#define RTC_COMMA ,

#endif

using std::byte;
using std::nullopt;
using std::optional;
using std::shared_ptr;
using std::string;
using std::string_view;
using std::unique_ptr;
using std::variant;
using std::weak_ptr;

using binary = std::vector<byte>;
using binary_ptr = shared_ptr<binary>;
using message_variant = variant<binary, string>;

using std::int16_t;
using std::int32_t;
using std::int64_t;
using std::int8_t;
using std::ptrdiff_t;
using std::size_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::uint8_t;

} // namespace rtc

#endif
