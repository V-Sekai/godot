/**************************************************************************/
/*  filedesc.hpp                                                          */
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

#ifndef FILEDESC_HPP
#define FILEDESC_HPP
#include "../types.hpp"
#include <functional>
#include <map>
#include <string>

#if defined(__APPLE__) || defined(__LINUX__)
#include <errno.h>
#endif

namespace riscv {

struct FileDescriptors {
#ifdef _WIN32
	typedef uint64_t real_fd_type; // SOCKET is uint64_t
#else
	typedef int real_fd_type;
#endif

	// Insert and manage real FDs, return virtual FD
	int assign_file(real_fd_type fd) { return assign(fd, false); }
	int assign_socket(real_fd_type fd) { return assign(fd, true); }
	int assign(real_fd_type fd, bool socket);
	// Get real FD from virtual FD
	real_fd_type get(int vfd);
	real_fd_type translate(int vfd);
	// Remove virtual FD and return real FD
	real_fd_type erase(int vfd);

	bool is_socket(int) const;
	bool permit_write(int vfd) {
		if (is_socket(vfd))
			return true;
		else
			return proxy_mode;
	}

	~FileDescriptors();

	std::map<int, real_fd_type> translation;

	// Default working directory (fake root)
	std::string cwd = "/home";

	static constexpr int FILE_D_BASE = 0x1000;
	static constexpr int SOCKET_D_BASE = 0x40001000;
	int file_counter = FILE_D_BASE;
	int socket_counter = SOCKET_D_BASE;

	bool permit_filesystem = false;
	bool permit_sockets = false;
	bool proxy_mode = false;

	std::function<bool(void *, std::string &)> filter_open = nullptr; /* NOTE: Can modify path */
	std::function<bool(void *, std::string &)> filter_readlink = nullptr; /* NOTE: Can modify path */
	std::function<bool(void *, const std::string &)> filter_stat = nullptr;
	std::function<bool(void *, uint64_t)> filter_ioctl = nullptr;
};

inline int FileDescriptors::assign(FileDescriptors::real_fd_type real_fd, bool socket) {
	int virtfd;
	if (!socket)
		virtfd = file_counter++;
	else
		virtfd = socket_counter++;

	translation.emplace(virtfd, real_fd);
	return virtfd;
}
inline FileDescriptors::real_fd_type FileDescriptors::get(int virtfd) {
	auto it = translation.find(virtfd);
	if (it != translation.end())
		return it->second;
	return -EBADF;
}
inline FileDescriptors::real_fd_type FileDescriptors::translate(int virtfd) {
	auto it = translation.find(virtfd);
	if (it != translation.end())
		return it->second;
	// Only allow direct access to standard pipes and errors
	return (virtfd <= 2) ? virtfd : -1;
}
inline FileDescriptors::real_fd_type FileDescriptors::erase(int virtfd) {
	auto it = translation.find(virtfd);
	if (it != translation.end()) {
		FileDescriptors::real_fd_type real_fd = it->second;
		// Remove the virt FD
		translation.erase(it);
		return real_fd;
	}
	return -EBADF;
}

inline bool FileDescriptors::is_socket(int virtfd) const {
	return virtfd >= SOCKET_D_BASE;
}

} //namespace riscv

#endif // FILEDESC_HPP
