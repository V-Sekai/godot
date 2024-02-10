/**************************************************************************/
/*  vector3i.h                                                            */
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

#ifndef VOXEL_VECTOR3I_H
#define VOXEL_VECTOR3I_H

#include "../funcs.h"
#include "funcs.h"
#include <core/math/vector3.h>
#include <core/templates/hashfuncs.h>
#include <functional>

struct VoxelVector3i {
	static const unsigned int AXIS_X = 0;
	static const unsigned int AXIS_Y = 1;
	static const unsigned int AXIS_Z = 2;
	static const unsigned int AXIS_COUNT = 3;

	union {
		struct {
			int x;
			int y;
			int z;
		};
		int coords[3];
	};

	_FORCE_INLINE_ VoxelVector3i() :
			x(0), y(0), z(0) {}

	explicit _FORCE_INLINE_ VoxelVector3i(int xyz) :
			x(xyz), y(xyz), z(xyz) {}

	_FORCE_INLINE_ VoxelVector3i(int px, int py, int pz) :
			x(px), y(py), z(pz) {}

	_FORCE_INLINE_ VoxelVector3i(const VoxelVector3i &other) { *this = other; }

	// TODO Deprecate this constructor, it is ambiguous because there are multiple
	// ways to convert a float to an int
	_FORCE_INLINE_ VoxelVector3i(const Vector3 &f) {
		x = Math::floor(f.x);
		y = Math::floor(f.y);
		z = Math::floor(f.z);
	}

	static inline VoxelVector3i from_floored(const Vector3 &f) {
		return VoxelVector3i(Math::floor(f.x), Math::floor(f.y), Math::floor(f.z));
	}

	_FORCE_INLINE_ Vector3 to_vec3() const { return Vector3(x, y, z); }

	_FORCE_INLINE_ int volume() const { return x * y * z; }

	_FORCE_INLINE_ int length_sq() const { return x * x + y * y + z * z; }

	_FORCE_INLINE_ real_t length() const {
		return Math::sqrt((real_t)length_sq());
	}

	_FORCE_INLINE_ int distance_sq(const VoxelVector3i &other) const;

	_FORCE_INLINE_ VoxelVector3i &operator=(const VoxelVector3i &other) {
		x = other.x;
		y = other.y;
		z = other.z;
		return *this;
	}

	_FORCE_INLINE_ void operator+=(const VoxelVector3i &other) {
		x += other.x;
		y += other.y;
		z += other.z;
	}

	_FORCE_INLINE_ void operator-=(const VoxelVector3i &other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
	}

	_FORCE_INLINE_ void operator*=(const int s) {
		x *= s;
		y *= s;
		z *= s;
	}

	_FORCE_INLINE_ VoxelVector3i operator-() const {
		return VoxelVector3i(-x, -y, -z);
	}

	_FORCE_INLINE_ int &operator[](unsigned int i) {
#if TOOLS_ENABLED
		CRASH_COND(i >= AXIS_COUNT);
#endif
		return coords[i];
	}

	_FORCE_INLINE_ const int &operator[](unsigned int i) const {
#if TOOLS_ENABLED
		CRASH_COND(i >= AXIS_COUNT);
#endif
		return coords[i];
	}

	// TODO Change this to be consistent with `::clamp`
	// Clamps between min and max, where max is excluded
	void clamp_to(const VoxelVector3i min, const VoxelVector3i max) {
		if (x < min.x) {
			x = min.x;
		}
		if (y < min.y) {
			y = min.y;
		}
		if (z < min.z) {
			z = min.z;
		}

		if (x >= max.x) {
			x = max.x - 1;
		}
		if (y >= max.y) {
			y = max.y - 1;
		}
		if (z >= max.z) {
			z = max.z - 1;
		}
	}

	// TODO Deprecate
	_FORCE_INLINE_ bool is_contained_in(const VoxelVector3i &min,
			const VoxelVector3i &max) {
		return x >= min.x && y >= min.y && z >= min.z && x < max.x && y < max.y &&
				z < max.z;
	}

	static void sort_min_max(VoxelVector3i &a, VoxelVector3i &b) {
		::sort(a.x, b.x);
		::sort(a.y, b.y);
		::sort(a.z, b.z);
	}

	inline VoxelVector3i floordiv(const VoxelVector3i d) const {
		return VoxelVector3i(::floordiv(x, d.x), ::floordiv(y, d.y),
				::floordiv(z, d.z));
	}

	inline VoxelVector3i floordiv(const int d) const {
		return VoxelVector3i(::floordiv(x, d), ::floordiv(y, d), ::floordiv(z, d));
	}

	inline VoxelVector3i ceildiv(const int d) const {
		return VoxelVector3i(::ceildiv(x, d), ::ceildiv(y, d), ::ceildiv(z, d));
	}

	inline VoxelVector3i wrap(const VoxelVector3i d) const {
		return VoxelVector3i(::wrap(x, d.x), ::wrap(y, d.y), ::wrap(z, d.z));
	}

	inline unsigned int get_zxy_index(const VoxelVector3i area_size) const {
		return y + area_size.y * (x + area_size.x * z); // ZXY
	}

	static inline VoxelVector3i from_zxy_index(unsigned int i,
			const VoxelVector3i area_size) {
		VoxelVector3i pos;
		pos.y = i % area_size.y;
		pos.x = (i / area_size.y) % area_size.x;
		pos.z = i / (area_size.y * area_size.x);
		return pos;
	}

	bool all_members_equal() const {
		return x == y && y == z;
	}

	static inline VoxelVector3i min(const VoxelVector3i a,
			const VoxelVector3i b) {
		return VoxelVector3i(::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z));
	}
};

_FORCE_INLINE_ VoxelVector3i operator+(const VoxelVector3i &a,
		const VoxelVector3i &b) {
	return VoxelVector3i(a.x + b.x, a.y + b.y, a.z + b.z);
}

_FORCE_INLINE_ VoxelVector3i operator-(const VoxelVector3i &a,
		const VoxelVector3i &b) {
	return VoxelVector3i(a.x - b.x, a.y - b.y, a.z - b.z);
}

_FORCE_INLINE_ VoxelVector3i operator*(const VoxelVector3i &a,
		const VoxelVector3i &b) {
	return VoxelVector3i(a.x * b.x, a.y * b.y, a.z * b.z);
}

_FORCE_INLINE_ VoxelVector3i operator/(const VoxelVector3i &a,
		const VoxelVector3i &d) {
	return VoxelVector3i(a.x / d.x, a.y / d.y, a.z / d.z);
}

_FORCE_INLINE_ bool operator==(const VoxelVector3i &a, const VoxelVector3i &b) {
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

_FORCE_INLINE_ bool operator!=(const VoxelVector3i &a, const VoxelVector3i &b) {
	return a.x != b.x || a.y != b.y || a.z != b.z;
}

_FORCE_INLINE_ VoxelVector3i operator<<(const VoxelVector3i &a, int b) {
#ifdef DEBUG_ENABLED
	CRASH_COND(b < 0);
#endif
	return VoxelVector3i(a.x << b, a.y << b, a.z << b);
}

_FORCE_INLINE_ VoxelVector3i operator>>(const VoxelVector3i &a, int b) {
#ifdef DEBUG_ENABLED
	CRASH_COND(b < 0);
#endif
	return VoxelVector3i(a.x >> b, a.y >> b, a.z >> b);
}

_FORCE_INLINE_ VoxelVector3i operator&(const VoxelVector3i &a, int b) {
	return VoxelVector3i(a.x & b, a.y & b, a.z & b);
}

inline VoxelVector3i operator%(const VoxelVector3i &a, const VoxelVector3i &b) {
	return VoxelVector3i(a.x % b.x, a.y % b.y, a.z % b.z);
}

_FORCE_INLINE_ bool operator<(const VoxelVector3i &a, const VoxelVector3i &b) {
	if (a.x == b.x) {
		if (a.y == b.y) {
			return a.z < b.z;
		} else {
			return a.y < b.y;
		}
	} else {
		return a.x < b.x;
	}
}

_FORCE_INLINE_ int
VoxelVector3i::distance_sq(const VoxelVector3i &other) const {
	return (other - *this).length_sq();
}

// For Godot
struct Vector3iHasher {
	static _FORCE_INLINE_ uint32_t hash(const VoxelVector3i &v) {
		uint32_t hash = hash_murmur3_one_32(v.x);
		hash = hash_murmur3_one_32(v.y, hash);
		return hash_murmur3_one_32(v.z, hash);
	}
};

// For STL
namespace std {
template <>
struct hash<VoxelVector3i> {
	size_t operator()(const VoxelVector3i &v) const {
		return Vector3iHasher::hash(v);
	}
};
} // namespace std

#endif // VOXEL_VECTOR3I_H
