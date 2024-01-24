/**************************************************************************/
/*  voxel_constants.h                                                     */
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

#ifndef VOXEL_CONSTANTS_H
#define VOXEL_CONSTANTS_H

#include <core/math/math_defs.h>

namespace VoxelConstants {
// These constants are chosen so you don't accidentally blow up resource usage
static const float MINIMUM_LOD_DISTANCE = 16.f;
static const float MAXIMUM_LOD_DISTANCE = 128.f;

static const unsigned int MIN_BLOCK_SIZE = 16;
static const unsigned int MAX_BLOCK_SIZE = 32;

static const unsigned int MAX_BLOCK_COUNT_PER_REQUEST = 4 * 4 * 4;

static const unsigned int MAX_LOD = 32;
static const unsigned int MAX_VOLUME_EXTENT = 0x1fffffff;
static const unsigned int MAX_VOLUME_SIZE =
		2 * MAX_VOLUME_EXTENT; // 1,073,741,822 voxels
static const unsigned int MAIN_THREAD_MESHING_BUDGET_MS = 8;

static const float INV_0x7f = 1.f / 0x7f;
static const float INV_0x7fff = 1.f / 0x7fff;
static const float INV_TAU = 1.f / Math_TAU;

// Below 32 bits, channels are normalized in -1..1, and can represent a limited
// number of values. For storing SDF, we need a range of values that extends
// beyond that, in particular for better LOD. So we can scale it to better fit
// the resolution.
static const float QUANTIZED_SDF_8_BITS_SCALE = 0.1f;
static const float QUANTIZED_SDF_8_BITS_SCALE_INV = 1.f / 0.1f;
static const float QUANTIZED_SDF_16_BITS_SCALE = 0.002f;
static const float QUANTIZED_SDF_16_BITS_SCALE_INV = 1.f / 0.002f;

static const unsigned int DEFAULT_BLOCK_SIZE_PO2 = 4;

static const float DEFAULT_COLLISION_MARGIN = 0.04f;
} // namespace VoxelConstants

#endif // VOXEL_CONSTANTS_H
