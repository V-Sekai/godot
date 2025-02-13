/**************************************************************************/
/*  event.hpp                                                             */
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

#ifndef EVENT_HPP
#define EVENT_HPP
#include "script.hpp"

enum EventUsagePattern : int {
	SharedScript = 0,
	PerThread = 1,
};

/// @brief Create a wrapper for a function call, matching the
/// function type F into a given Script instance, for the given function.
/// @tparam Usage The usage pattern of the event, either SharedScript or PerThread
/// @tparam F A function type, eg. void(int)
template <typename F = void(), EventUsagePattern Usage = SharedScript>
struct Event {
	Event() = default;
	Event(Script &, const std::string &func);
	Event(Script &, Script::gaddr_t address);

	/// @brief Call the function with the given arguments
	/// @tparam Args The event function argument types
	/// @param args The event function arguments
	/// @return The std::optional return value of the function, unless void
	template <typename... Args>
	auto call(Args &&...args);

	/// @brief Call the function with the given arguments
	/// @tparam Args The event function argument types
	/// @param args The event function arguments
	/// @return The std::optional return value of the function, unless void
	template <typename... Args>
	auto operator()(Args &&...args) {
		return this->call(std::forward<Args>(args)...);
	}

	bool is_callable() const noexcept {
		return m_script != nullptr && m_addr != 0x0;
	}

	auto &script() noexcept {
		assert(m_script != nullptr);
		if constexpr (Usage == EventUsagePattern::PerThread)
			return m_script->get_fork();
		else
			return *m_script;
	}

	const auto &script() const noexcept {
		assert(m_script != nullptr);
		if constexpr (Usage == EventUsagePattern::PerThread)
			return m_script->get_fork();
		else
			return *m_script;
	}

	auto address() const noexcept {
		return m_addr;
	}

	// Turn address into function name (as long as it's visible)
	auto function() const {
		return script().symbol_name(address());
	}

private:
	Script *m_script = nullptr;
	Script::gaddr_t m_addr = 0;
};

template <typename F, EventUsagePattern Usage>
inline Event<F, Usage>::Event(Script &script, const std::string &func) :
		m_script(&script), m_addr(script.address_of(func)) {
}

template <typename F, EventUsagePattern Usage>
inline Event<F, Usage>::Event(Script &script, Script::gaddr_t address) :
		m_script(&script), m_addr(address) {
}

template <typename F, EventUsagePattern Usage>
template <typename... Args>
inline auto Event<F, Usage>::call(Args &&...args) {
	static_assert(std::is_invocable_v<F, Args...>);
	using Ret = decltype((F *){}(args...));

	if (is_callable()) {
		auto &script = this->script();
		if (auto res = script.call(address(), std::forward<Args>(args)...)) {
			if constexpr (std::is_same_v<void, Ret>)
				return true;
			else if constexpr (std::is_same_v<Script::gaddr_t, Ret>)
				return res;
			else
				return std::optional<Ret>(res.value());
		}
	}
	if constexpr (std::is_same_v<void, Ret>)
		return false;
	else
		return std::optional<Ret>{ std::nullopt };
}

#endif // EVENT_HPP
