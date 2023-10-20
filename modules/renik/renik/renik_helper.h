/**************************************************************************/
/*  renik_helper.h                                                        */
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

#ifndef RENIK_HELPER_H
#define RENIK_HELPER_H

#ifndef _3D_DISABLED

#include <scene/main/node.h>

namespace RenIKHelper {
Quaternion align_vectors(Vector3 a, Vector3 b, float influence = 1);
float smoothCurve(float number, float modifier = 0.5);
Vector3 vector_rejection(Vector3 v, Vector3 normal);
float safe_acos(float f);
float safe_asin(float f);
Vector3 get_perpendicular_vector(Vector3 v);

float log_clamp(float value, float target, float looseness);
Vector3 log_clamp(Vector3 vector, Vector3 target, float looseness);
Quaternion log_clamp(Quaternion quat, Quaternion target, float looseness);
Basis log_clamp(Basis basis, Basis target, float looseness);
} // namespace RenIKHelper

#endif

#endif // RENIK_HELPER_H
