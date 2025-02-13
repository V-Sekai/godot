/**************************************************************************/
/*  ringbuffer.hpp                                                        */
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

#ifndef RINGBUFFER_HPP
#define RINGBUFFER_HPP

#include <cassert>
#include <cstdint>
#include <cstring>

template <size_t N, typename T>
class FixedRingBuffer {
public:
	bool write(T item) {
		if (!full()) {
			buffer_at(this->m_writer++) = std::move(item);
			return true;
		}
		return false;
	}

	const T *read() {
		if (!empty()) {
			return &buffer_at(this->m_reader++);
		}
		return nullptr;
	}

	bool discard() {
		if (!this->empty()) {
			this->m_reader++;
			return true;
		}
		return false;
	}
	void clear() {
		this->m_reader = this->m_writer = 0;
	}

	size_t size() const {
		return this->m_writer - this->m_reader;
	}
	constexpr size_t capacity() const {
		return N;
	}

	bool full() const {
		return size() == capacity();
	}
	bool empty() const {
		return size() == 0;
	}

	const T *data() const {
		return this->m_buffer;
	}

private:
	T &buffer_at(size_t idx) {
		return this->m_buffer[idx % capacity()];
	}

	size_t m_reader = 0;
	size_t m_writer = 0;
	T m_buffer[N];
};

#endif // RINGBUFFER_HPP
